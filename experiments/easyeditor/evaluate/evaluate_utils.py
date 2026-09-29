import os
import re
import sys
import warnings

import torch
import numpy as np
import scipy
import nltk
import wandb
import typing
from ..util.generate import generate_fast, generate_fast_vllm
import torch.nn.functional as F
from ..trainer import *
from sklearn.metrics import f1_score
import openai
from ptflops import get_model_complexity_info
from nltk.corpus import stopwords

# Suppress cosmetic warnings that clutter SLURM logs
warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='invalid value encountered in scalar divide',
                        category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*autocast.*', category=FutureWarning)


# ── Refusal detection ────────────────────────────────────────────────────────
# Regex heuristic for detecting when a model declines to answer a medical
# question due to safety alignment. Matched case-insensitively against the raw
# generated text (after <think> stripping, before label extraction).
_REFUSAL_PATTERNS = re.compile(
    r"("
    r"i(?:'m| am)\s+(?:sorry|afraid|not\s+able|unable)(?:[^.]*?)"
    r"(?:cannot|can'?t|unable|not\s+able|won'?t|refuse)"
    r"|i\s+(?:cannot|can'?t|am\s+not\s+able|am\s+unable|won'?t|refuse)\b"
    r"[^.]{0,60}?(?:provide|recommend|advise|give|answer|help|assist|make|offer)"
    r"|i\s+(?:do\s+not|don'?t)\s+(?:feel\s+comfortable|have\s+the\s+ability|provide\s+medical)"
    r"|(?:consult|speak\s+(?:to|with)|seek|talk\s+to|ask)\s+(?:a|your|an?)\s+"
    r"(?:doctor|healthcare\s+(?:provider|professional)|physician|medical\s+professional|oncologist|clinician)"
    r"|as\s+an\s+ai(?:\s+language)?\s+model"
    r"|i'?m\s+just\s+an\s+ai"
    r"|not\s+(?:qualified|a\s+substitute)\s+(?:to\s+(?:provide|give)\s+medical|for\s+(?:professional\s+)?medical)"
    r"|i\s+(?:cannot|can'?t)\s+(?:and\s+(?:will|should)\s+not|in\s+good\s+conscience)"
    r"|unable\s+to\s+provide\s+(?:medical|clinical|treatment)"
    r")",
    flags=re.IGNORECASE,
)


def _detect_refusal(text):
    """Return True when the generated text looks like a safety-alignment refusal."""
    if not text or not isinstance(text, str):
        return False
    # Strip <think>...</think> so a refusal inside a reasoning trace doesn't count.
    stripped = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return bool(_REFUSAL_PATTERNS.search(stripped))


def _vanilla_hf_truncated(gen_token_row, n_input, max_out_len, tok):
    """Detect whether a HF model.generate() output hit the max_new_tokens cap.

    Truncated iff the number of new tokens equals max_out_len AND the last new
    token is not an EOS / end-of-turn token.
    """
    new_tokens = gen_token_row[n_input:]
    if len(new_tokens) < max_out_len:
        return False
    stop_ids = set()
    eos = getattr(tok, 'eos_token_id', None)
    if isinstance(eos, list):
        stop_ids.update(eos)
    elif eos is not None:
        stop_ids.add(int(eos))
    eot = getattr(tok, 'convert_tokens_to_ids', lambda *_: None)('<end_of_turn>')
    if eot is not None and eot != getattr(tok, 'unk_token_id', None):
        stop_ids.add(int(eot))
    last_id = int(new_tokens[-1].item() if hasattr(new_tokens[-1], 'item') else new_tokens[-1])
    return last_id not in stop_ids


def generate_openrouter(client, model, prompts, max_out_len=256, thinking_budget_tokens=0):
    """Generate responses via OpenRouter API with exponential-backoff retry.

    Returns (texts, truncated_flags) — parallel lists.  A flag is True when the
    OpenRouter response reports finish_reason == 'length' (max_tokens hit).
    """
    import concurrent.futures
    import time as _time
    import openai as _openai

    max_workers = 1 if (model or '').endswith(':free') else 4

    extra_kwargs = {}
    effective_max_tokens = max_out_len
    if thinking_budget_tokens > 0:
        extra_kwargs['extra_body'] = {"thinking": {"type": "enabled", "budget_tokens": thinking_budget_tokens}}
        effective_max_tokens = thinking_budget_tokens + max_out_len

    def _generate_one(prompt):
        n_try = 0
        max_tries = 10
        while n_try < max_tries:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=effective_max_tokens,
                    **extra_kwargs,
                )
                content = response.choices[0].message.content or ''
                # Strip thinking blocks if OpenRouter returns them inline
                content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL).strip()
                finish = getattr(response.choices[0], 'finish_reason', None)
                return content, (finish == 'length')
            except _openai.RateLimitError as e:
                retry_after = 0.0
                try:
                    retry_after = float(e.response.headers.get('Retry-After', 0))
                except Exception:
                    pass
                wait = max(retry_after, 5.0 * (2 ** n_try))
                wait = min(wait, 300)
                _time.sleep(wait)
                n_try += 1
            except Exception as e:
                wait = min(5.0 * (2 ** n_try), 300)
                _time.sleep(wait)
                n_try += 1
        return '', False

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_generate_one, prompts))
    texts = [r[0] for r in results]
    truncated = [r[1] for r in results]
    return texts, truncated


def test_batch_prediction_acc(model, tok, hparams, prompts, target, device, locality=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

        ans = ans.squeeze().detach().cpu().numpy().tolist()

        if locality:
            return ans

        return np.mean(np.equal(ans, target))

def test_seq2seq_batch_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        targets,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['decoder_input_ids'] = trg_tok['input_ids']
    prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            answers = ans.squeeze().detach().cpu().numpy().tolist()
            return answers if type(answers[0]) is list else [answers,]
        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()


def test_prediction_acc(model, tok, hparams, prompts, targets, device, locality=False, vanilla_generation=False, record_flops=False, vllm_model=None, lora_request=None, judge_model=None, judge_workers=4, judge_enabled=True, openrouter_client=None, openrouter_model=None, openrouter_thinking_budget=0, max_out_len_closed=20, rag_model=None):
    # Augment prompts with retrieved evidence/target when a RAG model is active.
    # Done up front — BEFORE the OpenRouter branch and before tokenization — so
    # that retrieval works for API generators too (needed by the agentic /
    # BM25 / Dense RAG-on-OpenRouter experiments), and the chat template is
    # applied to the already-augmented text on the local paths. When rag_model
    # is None (e.g. oracle-on-API, which injects evidence at data-load), this is
    # a no-op and behaviour is unchanged.
    if rag_model is not None and hasattr(rag_model, 'augment_texts'):
        if isinstance(prompts, str):
            prompts, targets = [prompts], [targets]
        prompts = rag_model.augment_texts(list(prompts))

    if openrouter_client is not None:
        if isinstance(prompts, str):
            prompts, targets = [prompts], [targets]
        gen_texts, trunc_flags = generate_openrouter(openrouter_client, openrouter_model, prompts, max_out_len=max_out_len_closed, thinking_budget_tokens=openrouter_thinking_budget)
        results = [None] * len(gen_texts)
        fallback_indices, fallback_responses, fallback_prompts, fallback_targets = [], [], [], []
        for i, (gen, tgt) in enumerate(zip(gen_texts, targets)):
            gen_lower = gen.strip().lower()
            tgt_lower = str(tgt).strip().lower()
            m = re.search(r'\b(superior|inferior|no difference)\b', gen_lower)
            if m is not None:
                acc = 1.0 if m.group(1) == tgt_lower else 0.0
                results[i] = {'acc': [acc], 'f1': [acc], 'ppl': [0],
                              'refused': _detect_refusal(gen), 'truncated': bool(trunc_flags[i])}
            else:
                fallback_indices.append(i)
                fallback_responses.append(gen)
                fallback_prompts.append(prompts[i])
                fallback_targets.append(tgt)
        if fallback_indices:
            if judge_model and judge_enabled:
                gt_dicts = [{'question': p, 'target': str(t).strip(), 'condition': '', 'context': ''}
                            for p, t in zip(fallback_prompts, fallback_targets)]
                scores, _, _ = llm_as_judge(mode='closed_qa_fallback', answers=fallback_responses,
                                         ground_truth=gt_dicts, judge_model=judge_model,
                                         max_workers=judge_workers)
                for fi, score in zip(fallback_indices, scores):
                    acc = float(score) if score is not None else 0.0
                    results[fi] = {'acc': [acc], 'f1': [acc], 'ppl': [0],
                                   'refused': _detect_refusal(gen_texts[fi]),
                                   'truncated': bool(trunc_flags[fi])}
            else:
                for fi in fallback_indices:
                    results[fi] = {'acc': [0.0], 'f1': [0.0], 'ppl': [0],
                                   'refused': _detect_refusal(gen_texts[fi]),
                                   'truncated': bool(trunc_flags[fi])}
        if locality:
            return [[int(r['acc'][0])] for r in results], [r['acc'][0] for r in results]
        return results, gen_texts, [0.0]

    #print(f'Input prompts: {prompts[0]}')

    if vllm_model is not None:
        if isinstance(prompts, str):
            prompts, targets = [prompts], [targets]
        # Stage 1: generate up to max_out_len_closed tokens (greedy), extract label with regex.
        # This avoids the prompt_logprobs memory overhead (1.2 GB/seq for 4k-token evidence).
        gen_texts, trunc_flags = generate_fast_vllm(vllm_model, prompts, lora_request, max_out_len=max_out_len_closed, tok=tok)
        if all(not g.strip() for g in gen_texts):
            import logging as _log
            _log.getLogger(__name__).warning(
                "generate_fast_vllm returned empty strings for all %d prompts in test_prediction_acc. "
                "Check model/vLLM compatibility (e.g. vLLM V1 + prompt_token_ids).", len(gen_texts)
            )
        results = [None] * len(gen_texts)
        fallback_indices, fallback_responses, fallback_prompts, fallback_targets = [], [], [], []
        for i, (gen, tgt) in enumerate(zip(gen_texts, targets)):
            gen_lower = gen.strip().lower()
            tgt_lower = str(tgt).strip().lower()
            m = re.search(r'\b(superior|inferior|no difference)\b', gen_lower)
            if m is not None:
                acc = 1.0 if m.group(1) == tgt_lower else 0.0
                results[i] = {'acc': [acc], 'f1': [acc], 'ppl': [0],
                              'refused': _detect_refusal(gen), 'truncated': bool(trunc_flags[i])}
            else:
                fallback_indices.append(i)
                fallback_responses.append(gen)
                fallback_prompts.append(prompts[i])
                fallback_targets.append(tgt)
        # Stage 2: LLM judge fallback for responses with no extractable digit
        if fallback_indices:
            if judge_model and judge_enabled:
                gt_dicts = [{'question': p, 'target': str(t).strip(), 'condition': '', 'context': ''}
                            for p, t in zip(fallback_prompts, fallback_targets)]
                print('calling llm_as_judge with fallback responses:')
                scores, _, _ = llm_as_judge(mode='closed_qa_fallback', answers=fallback_responses,
                                         ground_truth=gt_dicts, judge_model=judge_model,
                                         max_workers=judge_workers)
                for fi, score in zip(fallback_indices, scores):
                    acc = float(score) if score is not None else 0.0
                    results[fi] = {'acc': [acc], 'f1': [acc], 'ppl': [0],
                                   'refused': _detect_refusal(gen_texts[fi]),
                                   'truncated': bool(trunc_flags[fi])}
            else:
                # No judge available — non-extractable responses count as incorrect
                for fi in fallback_indices:
                    results[fi] = {'acc': [0.0], 'f1': [0.0], 'ppl': [0],
                                   'refused': _detect_refusal(gen_texts[fi]),
                                   'truncated': bool(trunc_flags[fi])}
        if locality:
            return [[int(r['acc'][0])] for r in results], [r['acc'][0] for r in results]
        return results, gen_texts, [0.0]

    if vanilla_generation:
        if isinstance(prompts, str):
            prompts, targets = [prompts, ], [targets, ]
        gen_texts = []
        trunc_flags = []
        locality_results = []

        # ── Batched HF .generate() path ─────────────────────────────────
        # Previously each prompt in `prompts` went through its own call.
        # With ~25 closed-QA prompts per probe, that cost dominates the
        # non-vLLM eval time.  We batch with left-padding (required for
        # correct causal-LM generation) and decode per row.  Batch size is
        # controlled by hparams.eval_gen_batch_size (default 8) — tune
        # down on tight VRAM.  grace_debug still emits per-prompt traces
        # but only for the FIRST prompt of each batch to avoid spam.
        _batch_size = int(getattr(hparams, 'eval_gen_batch_size', 8))
        if _batch_size < 1:
            _batch_size = 1
        _orig_pad_side = getattr(tok, 'padding_side', 'right')
        tok.padding_side = 'left'
        _pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        try:
            for _bstart in range(0, len(prompts), _batch_size):
                _bend = min(_bstart + _batch_size, len(prompts))
                _batch_prompts = prompts[_bstart:_bend]
                _batch_targets = targets[_bstart:_bend]

                # Encode the whole batch at once; chat template applied
                # per-prompt as text, then a single batched tokenizer call
                # does the left-padding.
                if getattr(tok, 'chat_template', None) is not None:
                    _templated = [
                        tok.apply_chat_template(
                            [{"role": "user", "content": p}],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        for p in _batch_prompts
                    ]
                    # The tokenizer re-tokenizes already-templated strings;
                    # add_special_tokens=False prevents a duplicate BOS.
                    enc = tok(
                        _templated,
                        padding=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    ).to(f'cuda:{device}')
                else:
                    enc = tok(
                        list(_batch_prompts),
                        padding=True,
                        return_tensors="pt",
                    ).to(f'cuda:{device}')

                if getattr(hparams, 'grace_debug', False) and len(_batch_prompts) > 0:
                    _ids0 = enc['input_ids'][0].detach().cpu().tolist()
                    print(
                        f"[EVAL-IDS] target={_batch_targets[0]!r} total_len={len(_ids0)} "
                        f"first5={_ids0[:5]} last5={_ids0[-5:]} (batch head)",
                        flush=True,
                    )

                gen_token = model.generate(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    max_new_tokens=max_out_len_closed,
                    pad_token_id=_pad_id,
                    use_cache=True,
                    do_sample=False,
                    num_beams=1,
                )
                n_input = enc['input_ids'].shape[1]  # same for every row due to left-pad

                for _row in range(gen_token.shape[0]):
                    _target_new = _batch_targets[_row]
                    _target_new_tokens = tok.encode(_target_new, add_special_tokens=False)
                    _row_tokens = gen_token[_row]
                    _gen_text = tok.decode(_row_tokens[n_input:], skip_special_tokens=True)
                    gen_texts.append(_gen_text)
                    trunc_flags.append(_vanilla_hf_truncated(_row_tokens, n_input, max_out_len_closed, tok))
                    if locality:
                        locality_results.append(
                            _row_tokens.detach().cpu().numpy().tolist()[-len(_target_new_tokens):]
                        )

                # grace_debug trace for the first row of this batch.
                if getattr(hparams, 'grace_debug', False) and gen_token.shape[0] > 0:
                    _target_new = _batch_targets[0]
                    _target_new_tokens = tok.encode(_target_new, add_special_tokens=False)
                    try:
                        _first_gen_id = int(gen_token[0][n_input].item())
                    except Exception:
                        _first_gen_id = -1
                    _first_target_id = int(_target_new_tokens[0]) if _target_new_tokens else -1
                    with torch.no_grad():
                        out = model(input_ids=enc['input_ids'][:1],
                                    attention_mask=enc['attention_mask'][:1])
                        _direct_argmax = int(out.logits[0, -1, :].argmax().item())
                        _direct_top5 = torch.topk(out.logits[0, -1, :], 5).indices.cpu().tolist()
                    print(
                        f"[EVAL-PRED] target={_target_new!r} "
                        f"first_gen={_first_gen_id} first_target={_first_target_id} "
                        f"direct_argmax={_direct_argmax} direct_top5={_direct_top5} "
                        f"gen_text={gen_texts[_bstart]!r} (batch head)",
                        flush=True,
                    )
        finally:
            tok.padding_side = _orig_pad_side

        if locality:
            return locality_results
        # Mirror vllm path: regex extraction → judge fallback
        results = [None] * len(gen_texts)
        fallback_indices, fallback_responses, fallback_prompts, fallback_targets = [], [], [], []
        for i, (gen, tgt) in enumerate(zip(gen_texts, targets)):
            # Strip thinking blocks whose tags were not removed as special tokens
            # (e.g. Qwen3 <think>…</think> when skip_special_tokens=False).
            gen_clean = re.sub(r'<think>.*?</think>', '', gen, flags=re.DOTALL).strip()
            gen_lower = gen_clean.lower()
            tgt_lower = str(tgt).strip().lower()
            # Use the LAST match so that a reasoning model's final answer takes
            # precedence over any label mentioned during its chain of thought.
            matches = re.findall(r'\b(superior|inferior|no difference)\b', gen_lower)
            if matches:
                acc = 1.0 if matches[-1] == tgt_lower else 0.0
                results[i] = {'acc': [acc], 'f1': [acc], 'ppl': [0],
                              'refused': _detect_refusal(gen), 'truncated': bool(trunc_flags[i])}
            else:
                fallback_indices.append(i)
                fallback_responses.append(gen_clean)
                fallback_prompts.append(prompts[i])
                fallback_targets.append(tgt)
        if fallback_indices:
            if judge_model and judge_enabled:
                gt_dicts = [{'question': p, 'target': str(t).strip(), 'condition': '', 'context': ''}
                            for p, t in zip(fallback_prompts, fallback_targets)]
                scores, _, _ = llm_as_judge(mode='closed_qa_fallback', answers=fallback_responses,
                                         ground_truth=gt_dicts, judge_model=judge_model,
                                         max_workers=judge_workers)
                for fi, score in zip(fallback_indices, scores):
                    acc = float(score) if score is not None else 0.0
                    results[fi] = {'acc': [acc], 'f1': [acc], 'ppl': [0],
                                   'refused': _detect_refusal(gen_texts[fi]),
                                   'truncated': bool(trunc_flags[fi])}
            else:
                for fi in fallback_indices:
                    results[fi] = {'acc': [0.0], 'f1': [0.0], 'ppl': [0],
                                   'refused': _detect_refusal(gen_texts[fi]),
                                   'truncated': bool(trunc_flags[fi])}
        return results, gen_texts, [0.0]

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    try:
        prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts, targets)]
    except:
        prompts = [p if isinstance(p, str) else '' for p in prompts]
        prompt_target = [prompt + target for prompt, target in zip(prompts, targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    if max_prompt_len > hparams.max_length:
        import logging as _log
        _log.getLogger(__name__).debug('Prompt length exceeds max_length: %d', max_prompt_len)
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=min(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        [prompt for prompt in prompts],
        padding=True,
        truncation=True,
        max_length=min(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x, y in zip(num_pad_toks, num_prompt_toks)]
    with torch.no_grad():
        start = time.time()
        if hasattr(hparams, 'fp16') and hparams.fp16:
            with torch.cuda.amp.autocast():
                outputs = model(**prompt_target_tok)
        else:
            outputs = model(**prompt_target_tok)
        forward_pass_time = time.time()-start
        #print(f'Input size: {prompt_target_tok["input_ids"].shape}')
        #print(f'Forward pass time: {time.time()-start}')
        #if record_flops:
            #start = time.time()
            #wrapped_model = ModelForFlopsWrapper(model)
            #sequence_length = prompt_tok['input_ids'].shape[1]
            #flops, params = get_model_complexity_info(wrapped_model, (2 * sequence_length,), as_strings=False, print_per_layer_stat=False, verbose=False)
            #flops = flops / 1e9
            #params = params / 1e6
            #print(f'Flops time: {time.time()-start}')
        #else:
        #    flops, params = None, None
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers_full = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels_full = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        if not isinstance(answers_full[0], list):
            answers_full = [answers_full,]
            labels_full = [labels_full,]

        # check if the padding is on the right
        right_pad = [l_f[-1] == tok.pad_token_id for l_f in labels_full]
        if any(right_pad):
            #answers_full, labels_full = answers_full[:2], labels_full[:2]
            answers_full = [a_f[:l_f.index(tok.pad_token_id)] if rp else a_f for a_f, l_f, rp in zip(answers_full, labels_full, right_pad)]
            labels_full = [l_f[:l_f.index(tok.pad_token_id)] if rp else l_f for l_f, rp in zip(labels_full, right_pad)]
            answers = slice_list(answers_full, num_prompt_toks, left=True)
            labels = slice_list(labels_full, num_prompt_toks, left=False)
        else:
            answers = slice_list(answers_full, prompt_len, left=True)
            labels = slice_list(labels_full, prompt_len, left=False)

        # discard white space tokens in the answers and labels
        #whitespace_token = tok.encode(' ')[0]
        #answers = [a[a != whitespace_token] for a in answers]
        #labels = [l[l != whitespace_token] for l in labels]

        # turn answer into text
        text_answers_full = [tok.decode(a_f) for a_f in answers_full]
        text_labels_full = [tok.decode(l_f) for l_f in labels_full]
        text_answers = [tok.decode(a) for a in answers]
        text_labels = [tok.decode(l) for l in labels]
        #print(f'Pad token id: {tok.pad_token_id}')
        #print(f'Num pad toks: {num_pad_toks}')
        #sys.exit()
        text_answers = text_answers_full

        if locality:
            acc = [np.mean(np.equal(ans, label)) for ans, label in zip(answers, labels)]
            return answers if type(answers[0]) is list else [answers,], acc
        if isinstance(answers[0], list):
            res = []
            for ans, label, label_full in zip(answers, labels, labels_full):
                temp_acc = [np.mean(np.equal(ans, label))]
                #temp_ppl = compute_perplexity_from_output(outputs, prompt_target_tok['input_ids'], len(label_full)-len(label))
                temp_ppl = [0]
                temp_f1 = [f1_score(ans, label, average='macro')]
                if np.isnan(temp_acc):
                    continue
                res.append({
                    'acc': temp_acc,
                    'f1': temp_f1,
                    'ppl': temp_ppl,
                    'refused': False,
                    'truncated': False,
                })
            return res, text_answers, [forward_pass_time]
        else:
            acc = [np.mean(np.equal(answers, labels))]
            #ppl = [compute_perplexity_from_output(outputs, prompt_target_tok['input_ids'], len(labels_full)-len(labels))]
            ppl = [0]
            f1 = [f1_score(answers, labels, average='macro')]
            res = [{
                'acc': acc,
                'f1': f1,
                'ppl': ppl,
                'refused': False,
                'truncated': False,
            }]

        if False:
            print(f'Length of answer_full: {len(answers_full)}, Length of label_full: {len(labels_full)}')
            print(f'prompt: {prompts[0]} \n target: {targets[0]} \n '
                  #f'answer_full: {text_answers_full} \n label_full: {text_labels_full} \n '
                  f'answer: {text_answers} \n label: {text_labels} \n '
                  f'acc: {acc[0]}',
                  f'f1: {f1[0]}',
                  f'ppl: {ppl[0]}')

        return res, text_answers, [forward_pass_time]


def test_generation_acc(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int = None,
    vanilla_generation: bool = False,
    vllm_model=None,
    lora_request=None,
    openrouter_client=None,
    openrouter_model=None,
    openrouter_thinking_budget=0,
    rag_model=None,
):
    """Generate open-ended responses.

    Returns (texts, truncated_flags, generation_time).  truncated_flags is a
    parallel list of booleans (True when the backend hit max_tokens).  The
    generate_fast HF path does not expose a finish_reason, so its flags default
    to False.
    """
    # Augment prompts with retrieved evidence/target at text level (before tokenization)
    # so that the chat template is preserved correctly. Done before the OpenRouter,
    # vLLM, and generate_fast paths so retrieval works for API generators too
    # (agentic / BM25 / Dense RAG-on-OpenRouter). No-op when rag_model is None.
    if rag_model is not None and hasattr(rag_model, 'augment_texts'):
        prefixes = rag_model.augment_texts(list(prefixes))

    if openrouter_client is not None:
        responses, trunc = generate_openrouter(openrouter_client, openrouter_model, prefixes, max_out_len=max_out_len, thinking_budget_tokens=openrouter_thinking_budget)
        return responses, trunc, 0.0

    if vllm_model is not None:
        responses, trunc = generate_fast_vllm(vllm_model, prefixes, lora_request, max_out_len=max_out_len, tok=tok)
        return responses, trunc, 0.0

    # get the number of tokens in the prefix
    prompt_tok = tok(
        prefixes,
        padding=True,
        return_tensors="pt",
    )
    if max_out_len is None:
        max_out_len = prompt_tok['input_ids'].shape[1] + 50
    start = time.time()
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=max_out_len,
        vanilla_generation=vanilla_generation,
    )
    generation_time = time.time() - start
    if isinstance(gen_texts, str):
        gen_texts = [gen_texts]

    # generate_fast does not expose finish_reason; default to False.
    trunc = [False] * len(gen_texts)
    return gen_texts, trunc, generation_time

def get_keyword_acc(responses, targets):
    keywords = []
    for t, target in enumerate(targets):
        or_components = target.split(" or ")
        kw_or = []
        for component in or_components:
            kw_and = []
            and_components = component.split(" and ")
            for sub_component in and_components:
                sub_component = sub_component.strip()
                sub_component = ''.join([c for c in sub_component if c.isalnum() or c.isspace()])
                sub_component = sub_component.split(' ')
                kw_and.append(sub_component)
            kw_or.append(kw_and)
        keywords.append(kw_or)
    ret = []
    for response, target in zip(responses, keywords):
        m = []
        for _or in target:
            n_or = 0
            for _and in _or:
                n_and = 0
                for word in _and:
                    if word.lower() in response.lower():
                        n_and += 1
                n_or = n_and / len(_and)
            m.append(n_or / len(_or))
        key_word_acc = max(m)
        ret.append(key_word_acc)
    return ret


def llm_as_judge(
        mode: str,
        answers: typing.List[str],
        ground_truth: typing.List[dict],
        judge_model: str = 'gpt-4o-mini-hemonc',
        max_workers: int = 4,
):
    instructions_open_qa = (
        """
        You are a medical expert tasked with evaluating how well a model’s response, comparing two treatment options, aligns with a ground truth. 
        • A patient’s condition and context
        • The ground truth comparison of two treatment options
        • The model’s response comparing two treatment options

       Your goal is to carefully assess whether the model’s response is consistent with the ground truth in terms of which treatment is preferable, equivalent, or inferior. Please also provide a brief justification for your rating based on the information in the response and the ground truth.

       Rate the alignment using the following 5-point Likert scale:
        1 – Completely Misaligned: The model’s response contradicts the ground truth.
        2 – Somewhat Misaligned: The response mostly contradicts the ground truth but mentions some correct elements.
        3 – Neutral / Unclear: The response includes both correct and incorrect elements, with no clear overall consistency.
        4 – Somewhat Aligned: The response aligns well with the ground truth but has minor inaccuracies or omissions.
        5 – Fully Aligned: The response fully aligns with the ground truth without any inaccuracies.

       Format your answer as follows:

       Score: <1–5>
       Explanation: <Brief justification based on the model’s response and the ground truth>
       """
    )

    instructions_open_gen = (
        """
You are a medical expert evaluating whether a model’s free-text treatment recommendation is consistent with a known pairwise clinical comparison.

You are provided with:
• Condition and clinical context
• Ground truth comparison:
  "[Treatment A] superior/inferior/no difference to [Treatment B] for [Condition] ([Context]) [endpoint: X]"
• A model-generated treatment recommendation

Your task is to assess whether the response is CONSISTENT with the relationship between Treatment A (A) and Treatment B (B).

Important:
This is not a recall task. The model is not required to mention A or B.
Evaluate whether the response respects the A–B relationship.

---

Relationship definitions:
• A superior to B → A should be preferred over B
• A inferior to B → B should be preferred over A
• No difference → A and B are interchangeable (no preference)

---

Evaluation guidelines:
• Do not penalize additional treatments beyond A and B.
• Track whether A and/or B are mentioned.
• If both are mentioned → check if the correct preference (or neutrality) is expressed.
• If only one is mentioned → check if this is consistent with the relationship.
• If neither is mentioned → this is not incorrect, but provides no evidence.
• Recommending the inferior treatment as preferred → strong error.

---

Scoring (1–5):

5 – Fully consistent
    • Correct preference clearly expressed OR inferior option explicitly de-emphasized
    • No-difference: both treated as equivalent

4 – Mostly consistent
    • Correct preference implied or weakly stated
    • OR only the consistent treatment is mentioned without contradiction

3 – Neutral / no evidence
    • Neither A nor B mentioned
    • OR insufficient information to assess preference

2 – Weak inconsistency
    • Incorrect treatment mentioned without clear preference
    • OR preference unclear or slightly misleading

1 – Clear inconsistency
    • Relationship contradicted:
        - Superior: B preferred over A
        - Inferior: A preferred over B
        - No difference: one clearly preferred

---

Additional flags:

• mentions_A: YES / NO
• mentions_B: YES / NO
• preference:
    - A preferred
    - B preferred
    - No clear preference
    - Neither mentioned

---

Output format:

Score: <1–5>

Flags:
- mentions_A: <YES/NO>
- mentions_B: <YES/NO>
- preference: <A preferred / B preferred / No clear preference / Neither mentioned>

Explanation:
Briefly state:
• the A–B relationship
• whether A/B appear
• the implied preference
• justification for the score
(Do not penalize additional treatments.)
        """
    )
    instructions_closed_fallback = (
        """
        You are evaluating whether a model's free-text response correctly answers a closed multiple-choice medical question.

        You are given:
        • The closed question (which asks the model to choose option 1, 2, or 3)
        • The correct answer (the option number)
        • The model's response (free text — the model may not have stated a digit explicitly)

        Determine whether the model's response implies or selects the correct option number.

        Answer with exactly one word: YES if the response conveys the correct choice, NO otherwise.
        """
    )

    if mode == 'open_qa':
        instructions = instructions_open_qa
    elif mode == 'open_gen':
        instructions = instructions_open_gen
    else:  # closed_qa_fallback
        instructions = instructions_closed_fallback

    task_template = "Condition: {}\nContext: {}\nGround Truth: {}\nResponse: {}"

    # Map internal model aliases to OpenRouter model names
    MODEL_ALIASES = {
        'gpt-4o-hemonc':      'openai/gpt-4o',
        'gpt-4o-mini-hemonc': 'openai/gpt-4o-mini',
        'gpt-4o':             'openai/gpt-4o',
        'gpt-4o-mini':        'openai/gpt-4o-mini',
    }
    openrouter_model = MODEL_ALIASES.get(judge_model, judge_model)

    from openai import OpenAI
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get('OPENROUTER_API_KEY'),
        max_retries=0,  # disable SDK retries; our loop handles all back-off
    )

    # Free-tier models are rate-limited to effectively 1 concurrent request;
    # force serial execution to avoid hammering the upstream provider.
    if openrouter_model.endswith(":free"):
        max_workers = 1

    import concurrent.futures

    import logging as _log
    import openai as _openai

    def _judge_one(args):
        idx, resp_text, gt = args
        n_try = 0
        max_tries = 10
        while n_try < max_tries:
            try:
                if mode == 'closed_qa_fallback':
                    # Binary YES/NO judge: does the free-text response imply the correct option?
                    task = (f"Question: {gt['question']}\n"
                            f"Correct Answer: Option {gt['target']}\n"
                            f"Model Response: {resp_text}")
                    api_resp = client.chat.completions.create(
                        model=openrouter_model,
                        messages=[{"role": "user",
                                   "content": f"Instructions:\n{instructions}\n\nTask:\n{task}\n\n"}],
                    )
                    answer_text = api_resp.choices[0].message.content.strip().upper()
                    score = 1.0 if 'YES' in answer_text else 0.0
                    return idx, score, answer_text, None
                else:
                    task = task_template.format(gt['condition'], gt['context'], gt['target'], resp_text)
                    api_resp = client.chat.completions.create(
                        model=openrouter_model,
                        messages=[
                            {
                                "role": 'user',
                                "content": f"Instructions:\n{instructions}\n\nTask:\n{task}\n\n"
                            },
                        ],
                    )
                    raw = api_resp.choices[0].message.content
                    out = raw.split('Explanation')
                    if len(out) != 2:
                        raise ValueError(f"Output format is incorrect:\n{out}")
                    score_str, explanation = out
                    score = int(re.sub(r'\D', '', score_str.split('Score')[-1]))
                    if score > 5 and score > 10 and score <= 15:
                        score -= 10
                    assert score >= 1 and score <= 5, f"Score is out of range: {score}"
                    flags = None
                    if mode == 'open_gen':
                        flags = {'mentions_A': None, 'mentions_B': None, 'preference': None}
                        ma = re.search(r'mentions_A\s*:\s*(YES|NO)', raw, re.IGNORECASE)
                        mb = re.search(r'mentions_B\s*:\s*(YES|NO)', raw, re.IGNORECASE)
                        pref = re.search(
                            r'preference\s*:\s*(A preferred|B preferred|No clear preference|Neither mentioned)',
                            raw, re.IGNORECASE)
                        if ma:
                            flags['mentions_A'] = ma.group(1).upper()
                        if mb:
                            flags['mentions_B'] = mb.group(1).upper()
                        if pref:
                            flags['preference'] = pref.group(1)
                    return idx, score, explanation, flags
            except _openai.RateLimitError as e:
                # Parse Retry-After header, or retryDelay / "retry in Xs" hints
                # from the body (OpenRouter forwards Google's 429 payload that way),
                # falling back to exponential backoff. The API typically requires
                # 30–120 s before the next request will succeed.
                retry_after = 0.0
                try:
                    retry_after = float(e.response.headers.get('Retry-After', 0))
                except Exception:
                    pass
                if retry_after <= 0:
                    body_txt = f"{getattr(e, 'body', '') or ''}\n{e}"
                    m = re.search(r'retry (?:in|after)\s+(\d+(?:\.\d+)?)\s*s', body_txt, re.IGNORECASE)
                    if not m:
                        m = re.search(r'retryDelay"?\s*:\s*"?\s*(\d+(?:\.\d+)?)\s*s', body_txt, re.IGNORECASE)
                    if m:
                        retry_after = float(m.group(1))
                wait = max(retry_after, 5.0 * (2 ** n_try))
                wait = min(wait, 300.0)
                _log.getLogger(__name__).warning(
                    "llm_as_judge rate limit (try %d/%d): sleeping %.0fs — %s",
                    n_try + 1, max_tries, wait, e)
                time.sleep(wait)
                n_try += 1
            except Exception as e:
                # 504 / 502 / timeouts / connection errors: exponential backoff
                # so gateway outages don't burn through max_tries in ~50 s.
                wait = min(5.0 * (2 ** n_try), 120.0)
                _log.getLogger(__name__).warning(
                    "llm_as_judge error (try %d/%d): sleeping %.0fs — %s",
                    n_try + 1, max_tries, wait, e)
                time.sleep(wait)
                n_try += 1
        _log.getLogger(__name__).warning("llm_as_judge: all %d retries exhausted for idx=%d", max_tries, idx)
        return idx, None, None, None  # all retries exhausted

    args_list = [(i, resp, gt) for i, (resp, gt) in enumerate(zip(answers, ground_truth))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_judge_one, args_list))

    results.sort(key=lambda x: x[0])
    scores = [r[1] for r in results]
    explanations = [r[2] for r in results]
    flags_list = [r[3] for r in results]

    return scores, explanations, flags_list

def compute_perplexity_from_output(output, labels, answer_start_idx):
    logits = output.logits  # Shape: (batch_size, sequence_length, vocab_size)
    batch_size, sequence_length, vocab_size = logits.size()
    logits = logits.view(-1, vocab_size)  # Shape: (batch_size * sequence_length, vocab_size)
    labels = labels.view(-1)  # Shape: (batch_size * sequence_length)

    log_probs = F.log_softmax(logits, dim=-1)  # Shape: (batch_size * sequence_length, vocab_size)

    log_probs_correct_tokens = log_probs[torch.arange(log_probs.size(0)), labels]

    mask = torch.zeros_like(labels)
    mask[answer_start_idx:] = 1
    log_probs_correct_tokens = log_probs_correct_tokens * mask.view(-1)

    negative_log_likelihood = -log_probs_correct_tokens.sum(dim=0)
    average_nll = negative_log_likelihood / mask.sum()
    perplexity = torch.exp(average_nll)

    return perplexity.item()

def test_generation_quality_serac(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,       
):
    #only single case
    prompt_tok = tok(
        prefixes,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    prompt_tok_length=len(prompt_tok['input_ids'])
    gen_texts=model.generate(**prompt_tok,max_new_tokens=256)
    if isinstance(model,SERAC):
        gen_texts=tok.decode(gen_texts[prompt_tok_length:])
        gen_texts=[gen_texts]
        print(len(gen_texts))
    else:
        gen_texts=tok.decode(gen_texts[prompt_tok_length:])
        gen_texts=[gen_texts]
        print(len(gen_texts))      
    ngram_entropy = n_gram_entropy(gen_texts, return_list=True)


    ret = {
        "ngram_entropy": ngram_entropy
    }
    return ret

def test_generation_quality(
    model,
    tok,
    prefixes: typing.List[str],
    max_out_len: int,
    vanilla_generation: bool = False,
):
    gen_texts = generate_fast(
        model,
        tok,
        prefixes,
        n_gen_per_prompt=1,
        max_out_len=max_out_len,
        vanilla_generation=vanilla_generation,
    )
    if isinstance(gen_texts, str):
        gen_texts = [gen_texts]
    ret = []
    for gen_text in gen_texts:
        ngram_entropy = n_gram_entropy(gen_text)
        ret.append({
            "ngram_entropy": ngram_entropy,
            # "reference_score": consistency_tfidf,
            "text": gen_text,
        })
    return ret

def n_gram_entropy(gen_texts, agg="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()

def compute_n_gram_entropy(sentence, ns=None, weights=None, agg="arith"):
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()

        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))

    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)

def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)

def PPL(
    model,
    tok,
    prompt: typing.Union[str, typing.List[str]],
    target_new: typing.Union[str, typing.List[str]],
    device,
):
    if isinstance(prompt, str):
        prompt,target_new = [prompt,], [target_new,]
    full_prompt = [f"{p} {l}" for p, l in zip(prompt, target_new)]
    prompt_ids = tok(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
    tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()
    for i in range(len(prompt)):
        tokens["labels"][i][:num_prompt_toks[i]] = -100
    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = -100 # What is this doing?
    batch = {f"{k1}" : v1 for k1, v1 in tokens.items()}
    input_ids = batch["input_ids"][:, :1024]#.to(device)
    if "labels" not in batch:
        target_ids = batch["input_ids"][:, :1024].clone()
    else:
        target_ids = batch["labels"][:, :1024].clone()
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(device), labels=target_ids.to(device))
        nll = outputs.loss
    ppl = torch.exp(nll)#.clip(0, 100)
    return ppl.cpu().numpy().tolist()

def verify_answer(model_answer, correct_answer):
    if type(correct_answer) is str:
        correct_answer = [[correct_answer]]
    for answer in correct_answer:
        if True not in [possible_answer in model_answer for possible_answer in answer]:
            return False
    return True

def answer_match(
    model,
    tok,
    prompt: str,
    target_new: str,
    device,
):
    inputs = tok.encode(prompt, return_tensors='pt').to(device)
    outputs = model.generate(inputs, do_sample=False, max_new_tokens=30)
    predict = tok.decode(outputs[0], skip_special_tokens=True)

    return verify_answer(predict,target_new)

def slice_list(matrix,start_indices,left):
    if isinstance(matrix[0], list):
        if left:
            return [row[start_index-1:-1] for row, start_index in zip(matrix, start_indices)]
        else:
            return [row[start_index:] for row, start_index in zip(matrix, start_indices)]
    else:
        if left:
            return matrix[start_indices[0]-1:-1]
        else:
            return matrix[start_indices[0]:]

def gather_log_probs(logits, labels):
    # print(f"labels.shape: {labels.shape} , logits.shape[:-1] :{logits.shape[:-1]}")
    assert labels.dim() == logits.dim() - 1
    assert labels.shape == logits.shape[:-1]
    return logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()

def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels

def es(pre_logits, edit_logits, q_mask, labels, same_mask):
    
    _, targ = mask_hf_labels(labels)

    pos_mask = same_mask.unsqueeze(-1) * q_mask 
    neg_mask = (~same_mask).unsqueeze(-1) * q_mask 
        
    pre_token_log_probs = gather_log_probs(pre_logits, targ)
    edit_token_log_probs = gather_log_probs(edit_logits, targ)

    mean_pos_pre = masked_mean(pre_token_log_probs, pos_mask)
    mean_pos_edit = masked_mean(edit_token_log_probs, pos_mask)
    mean_neg_edit = masked_mean(edit_token_log_probs, neg_mask)

    z_sent = (mean_pos_edit - mean_neg_edit).sigmoid()
    z_topic_raw = (mean_pos_edit - mean_pos_pre).exp()
    z_topic = min(1, z_topic_raw)

    es_sent = z_sent * z_topic
    return es_sent

def es_per_icl(example, pre_logits, edit_logits):
    with torch.no_grad():
        
        pre_q_mask = example["outer_pre"]["q_mask"]
        edit_q_mask = example["outer_edit"]["q_mask"]
        
        pre_labels = example["outer_pre"]["labels"]
        edit_labels = example["outer_edit"]["labels"]
        
        pre_mask, pre_targ = mask_hf_labels(pre_labels)
        edit_mask, edit_targ = mask_hf_labels(edit_labels)
        
        same_per_mask = example["same_per_mask"]

        pre_pos_mask = same_per_mask.unsqueeze(-1) * pre_q_mask 
        pre_neg_mask = (~same_per_mask).unsqueeze(-1) * pre_q_mask 
        edit_pos_mask = same_per_mask.unsqueeze(-1) * edit_q_mask 
        edit_neg_mask = (~same_per_mask).unsqueeze(-1) * edit_q_mask 
        
        pre_token_log_probs = gather_log_probs(pre_logits, pre_targ)
        edit_token_log_probs = gather_log_probs(edit_logits, edit_targ)

        mean_pos_pre = masked_mean(pre_token_log_probs, pre_pos_mask)
        mean_pos_edit = masked_mean(edit_token_log_probs, edit_pos_mask)
        mean_neg_edit = masked_mean(edit_token_log_probs, edit_neg_mask)

        z_per = (mean_pos_edit - mean_neg_edit).sigmoid()
        z_topic_raw = (mean_pos_edit - mean_pos_pre).exp()
        z_topic = min(1, z_topic_raw)

        es_per = z_per * z_topic
        return {
            "acc_per": es_per,
            "z_per": z_per,
            "z_topic": z_topic,
            "z_topic_raw": z_topic_raw,
            "correct_probs": mean_pos_edit,
            "wrong_probs": mean_neg_edit,
        }

def per_generation(
    model,
    tok,
    max_out_len: int,
    target_per, 
    device,
    edited_model=None,
    IKE=False,
    **kwargs
    ):
    def generate_text(query, model, tokenizer):
        input_text = query
        generation_config = {
            "max_new_tokens": max_out_len,
            "do_sample": False,
            "eos_token_id": tokenizer.eos_token_id,
        }
        src_input_ids = tokenizer(input_text).input_ids
        input_ids = torch.tensor([src_input_ids], dtype=torch.long, device=device)
        outputs = model.generate(input_ids, **generation_config)
        response = tokenizer.decode(outputs[0][len(src_input_ids) :], skip_special_tokens=True)
        return response
    
    def clean_text(text):
        return text.strip().split("\n")[0]
    
    if IKE:
        pre_text = clean_text(generate_text(kwargs["pre_q"], model, tok))
        edit_text = clean_text(generate_text(kwargs["edit_q"], model, tok))

    else:
        assert edited_model is not None
        pre_text = clean_text(generate_text(kwargs["inner_q"], model, tok))
        edit_text = clean_text(generate_text(kwargs["inner_q"], edited_model.model, tok))

    ngram_pre_text = n_gram_entropy([pre_text])
    ngram_edit_text = n_gram_entropy([edit_text])
    coherent = ngram_pre_text >= 3.5 and ngram_edit_text >= 3.5
    
    result = {
        "pre_text": pre_text,
        "edit_text": edit_text,
        "ngram_pre_text": ngram_pre_text,
        "ngram_edit_text": ngram_edit_text,
        "coherent": coherent,
        "target_per": target_per,
    }

    return result

def kl_loc_loss(pre, post, mask=None):
    
    pre = pre.to(torch.float32).contiguous()
    post = post[:,-pre.shape[1]:,:].to(torch.float32).contiguous()
    
    sequence = pre.dim() == 3
    pre_ = pre.view(-1, pre.shape[-1])
    post_ = post.view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        # print("sequence")
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError

def F1(model, tok, hparams, prompts, targets, device, locality=False, vanilla_generation=True):
    if vanilla_generation:
        if isinstance(prompts, str):
            prompts, targets = [prompts], [targets]
        results = []
        for prompt, target in zip(prompts, targets):
            target_new_tokens = tok.encode(target, add_special_tokens=False)
            prompt_tok = tok(
                prompt,
                return_tensors="pt",
            ).to(f'cuda:{device}')
            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=len(target_new_tokens),
                pad_token_id=tok.eos_token_id,
                use_cache=True,
            )
            results.append(f1_score(target_new_tokens, gen_token.detach().cpu().numpy().tolist()[0][-len(target_new_tokens):], average='macro'))
        return results
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_target_tok = tok(
        prompt_target,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    ).to(f"cuda:{device}")
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(hparams.max_length, max_prompt_len),
        return_tensors="pt",
    )
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_tok['input_ids']]
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in prompt_target_tok['input_ids'].cpu()]
    prompt_len = [x+y for x,y in zip(num_pad_toks,num_prompt_toks)]
    with torch.no_grad():
        outputs = model(**prompt_target_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits
        answers = torch.argmax(logits, dim=-1).squeeze().detach().cpu().numpy().tolist()
        labels = prompt_target_tok['input_ids'].squeeze().detach().cpu().numpy().tolist()
        answers = slice_list(answers,prompt_len,left=True)
        labels = slice_list(labels,prompt_len,left=False)

        return f1_score(answers, labels, average='macro')

def test_instance_change(model, tok, max_length, prompts, targets, device, P = None):
    demo1_str = "Whether FrancoAngeli belongs to category publisher? Yes\nWhether And Other Stories belongs to category people? No\n"
    if P is None:
        prompts = demo1_str +prompts
    else:
        prompts = P + demo1_str + prompts

    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    )
    with torch.no_grad():
        pre_edit_outputs = model.generate(
            input_ids=prompt_tok['input_ids'].to(f"cuda:{device}"),
            attention_mask=prompt_tok['attention_mask'].to(f"cuda:{device}"),
            max_new_tokens=2,
            pad_token_id=tok.eos_token_id
        )

        model_response = [tok.decode(x, skip_special_tokens=True) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        answer = model_response[0][model_response[0].rfind('?')+2:]
        # print(model_response[0], answer)

        if "yes" in answer.lower():
            return np.ones(1)
        else:
            if "no" not in answer.lower():
                print(f"entity error in define yes or no: {answer}")
                return np.array([-1.0])
            return np.zeros(1)

def test_concept_gen(model, tok, max_length, prompts, targets, device):
    if isinstance(prompts, str):
        prompts,targets = [prompts,], [targets,]
    prompts = [prompt + ' ' for prompt in prompts]
    prompt_target = [prompt + ' ' + target for prompt, target in zip(prompts,targets)]
    max_prompt_len = max([len(tok.encode(_)) for _ in prompt_target]) + 1
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=max(max_length, max_prompt_len),
        return_tensors="pt",
    )
    with torch.no_grad():
        pre_edit_outputs = model.generate(
            input_ids=prompt_tok['input_ids'].to(f"cuda:{device}"),
            attention_mask=prompt_tok['attention_mask'].to(f"cuda:{device}"),
            max_new_tokens=40,
            pad_token_id=tok.eos_token_id
        )

        model_response = [tok.decode(x, skip_special_tokens=True) for x in pre_edit_outputs.detach().cpu().numpy().tolist()]
        answer = model_response[0][len(prompts[0]):]
        return answer


def test_safety_gen(
        model, 
        tokenizer, 
        test_prompt, 
        cuda,
        max_tokens = 1624,
        max_output_tokens=600):
    tokenizer.padding_side = 'left'
    # if input_tokens (at least 1024) + output_tokens (at least 600) < 1624, truncate the input length (from right to left, as harmful questions typically appear on the right)
    if max_tokens < 1624:
        only_response = []
        for item in test_prompt:
            input = tokenizer([item,], return_tensors="pt", padding=True, truncation=True).to(f"cuda:{cuda}")
            if input["input_ids"].size(-1) > max_tokens-max_output_tokens:
                input = {k: v[:, -(max_tokens - max_output_tokens):] for k, v in input.items()}
            with torch.no_grad():
                outputs = model.generate(**input, max_new_tokens=max_output_tokens)
                texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
                texts = texts[0]
            if input["input_ids"].size(-1) > max_tokens-max_output_tokens:
                max_overlap_len = min(len(item), len(texts))
                overlap = next((item[-i:] for i in range(max_overlap_len, 0, -1) if item[-i:] == texts[:i]), "")
            else:
                overlap = item
            only_response.append(texts[len(overlap)+1:].lstrip())
        return only_response
    else:
        input = tokenizer(test_prompt, return_tensors="pt", padding=True, truncation=True).to(f"cuda:{cuda}")
        with torch.no_grad():
            outputs = model.generate(**input, max_new_tokens=max_output_tokens)
            texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            only_response = [out[len(test_prompt[index])+1:] for index, out in enumerate(texts)]
        return only_response


def extract_metric(m, metric_name):
    for component in metric_name:
        try:
            if isinstance(m, list):
                assert len(m) == 1
                m = m[0]
            m = m[component]
        except TypeError:
            print(f"m: {m}, component: {component}, metric_name: {metric_name}")
            raise
    return m[0] if isinstance(m, list) else m


def metrics_to_wandb(metrics, i, avg_edit_decay=None, compute_mean=True):
    metric_names = [['pre', 'rewrite_acc'], ['post', 'rewrite_acc'],
                    ['pre', 'locality', 'neighborhood_acc'],
                    ['post', 'locality', 'neighborhood_acc'], ['post', 'locality', 'neighborhood_score'],
                    ['pre', 'portability', 'mhop', 'performance',  'acc'], ['post', 'portability', 'mhop', 'performance',  'acc'],
                    ['pre', 'portability', 'genv2_mixed', 'performance',  'acc'], ['post', 'portability', 'genv2_mixed', 'performance',  'acc'],
                    ['pre', 'rephrase_acc'], ['post', 'rephrase_acc'],
                    ['pre', 'rewrite_forward_time'], ['post', 'rewrite_forward_time'],
                    ['post', 'retrieval_acc']]

    if compute_mean:
        mean_log = {f'{"_".join(k)}_mean': [] for k in metric_names}

        for j, m in enumerate(metrics):
            if j > i:
                break
            #print(f'j: {j}, i: {i}, {m}')
            for k in metric_names:
                try:
                    mean_log[f'{"_".join(k)}_mean'].append(extract_metric(m, k))
                except KeyError:
                    pass

        mean_log = {k: np.mean(v) for k, v in mean_log.items()}
    else:
        mean_log = {}

    current_log, current_log_mode = {}, {}
    for k in metric_names:
        try:
            current_log["_".join(k)] = extract_metric(metrics[i], k)
            if metrics[i]['post']['mode'] == 'open':
                current_log_mode[f'{"_".join(k)}_open'] = extract_metric(metrics[i], k)
            elif metrics[i]['post']['mode'] == 'closed':
                current_log_mode[f'{"_".join(k)}_closed'] = extract_metric(metrics[i], k)
            else:
                raise ValueError(f"mode {metrics[i]['post']['mode']} not recognized")
        except KeyError as e:
            #print(f'KeyError: {e}')
            #raise e
            pass
    if avg_edit_decay is not None:
        modes = [m['pre']['mode'] for m in metrics]
        aed_open, aed_closedd, aed = [], [], []
        for j in range(i):
            for k, acc in enumerate(avg_edit_decay[j]):
                if len(aed) <= k:
                    aed.append([])
                    aed_open.append([])
                    aed_closedd.append([])
                aed[k].append(acc)
                if modes[j] == 'open':
                    aed_open[k].append(acc)
                elif modes[j] == 'closed':
                    aed_closedd[k].append(acc)
                else:
                    raise ValueError(f"mode {modes[j]} not recognized")
        aed_metric = {
            f'avg_edit_decay': [np.mean(v) for v in aed],
            f'avg_edit_decay_open': [np.mean(v) for v in aed_open],
            f'avg_edit_decay_closed': [np.mean(v) for v in aed_closedd],
        }
    else:
        aed_metric = {}
    past_metrics = metrics[i]['past'] if 'past' in metrics[i].keys() else {}
    past_log = {f'past_{k}_mean': v for k, v in past_metrics.items()}
    general_metrics = metrics[i]['general'] if 'general' in metrics[i].keys() else {}
    general_log = {f'general_{k}_acc': v for k, v in general_metrics.items()}
    log = {**mean_log,
           #**mean_open_log, **mean_closed_log,
           #**mean_open_new_log, **mean_open_update_log, **mean_closed_new_log, **mean_closed_update_log,
           **current_log, **current_log_mode, **past_log, **general_log, **aed_metric}
    try:
        wandb.log(log)
    except:
        if compute_mean:
            print(mean_log)
        pass


# Define a wrapper for the model's forward pass
class ModelForFlopsWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelForFlopsWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        # Split input tensor into input_ids and attention_mask
        sequence_length = x.shape[1] // 2
        input_ids = x[:, :sequence_length].long()
        attention_mask = x[:, sequence_length:].long()
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )