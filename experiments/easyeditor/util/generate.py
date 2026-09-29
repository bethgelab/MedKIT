import re
import unicodedata
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .logit_lens import LogitLens


def generate_fast_vllm(vllm_model, prompts, lora_request=None, max_out_len=256, stop=None, tok=None):
    """
    Fast batched generation using vLLM (with optional LoRA adapter).
    Returns (texts, truncated_flags) — continuations only (no prompt), plus a
    parallel list of booleans where True means the output hit max_tokens.
    If tok is provided and has a chat_template, prompts are wrapped with
    apply_chat_template before generation (required for IT/instruct models
    such as medgemma-4b-it, LLaMA-3-Instruct, Qwen-Instruct).
    """
    from vllm import SamplingParams
    if tok is not None and getattr(tok, 'chat_template', None) is not None:
        # Use tokenize=True and pass token IDs directly to vLLM to avoid the
        # double-BOS issue: apply_chat_template(tokenize=False) embeds <bos> as
        # text, and vLLM then adds a second BOS when re-tokenizing the string
        # (because add_bos_token=true in Gemma/Qwen tokenizer configs).
        # Passing prompt_token_ids bypasses vLLM's tokenization entirely.
        prompts = [
            {"prompt_token_ids": tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
            )}
            for p in prompts
        ]
    # Collect stop token IDs from the tokenizer so vLLM stops on the model's
    # natural end tokens (e.g. Gemma 2 IT uses <end_of_turn> in addition to <eos>).
    # For non-Gemma models, convert_tokens_to_ids('<end_of_turn>') returns unk_token_id
    # so the guard prevents it from being added — no behavioural change for those models.
    stop_ids = []
    if tok is not None:
        eos = tok.eos_token_id
        if isinstance(eos, list):
            stop_ids.extend(eos)
        elif eos is not None:
            stop_ids.append(eos)
        eot_id = tok.convert_tokens_to_ids('<end_of_turn>')
        if eot_id is not None and eot_id != tok.unk_token_id:
            stop_ids.append(eot_id)
        stop_ids = list(set(stop_ids))
    # Left-truncate prompts that exceed the model's context window. Required
    # for bio_medical_llama3_8b (max_position_embeddings=8192) on Dense ABS /
    # BM25 ABS where prepended abstracts push the prompt past the limit.
    # vLLM v1 validates prompt_token_ids length BEFORE applying SamplingParams
    # truncate_prompt_tokens, so for pre-tokenized inputs we must truncate
    # the IDs ourselves. Keeping the LAST N tokens preserves the question at
    # the end; oldest retrieved context / system instructions are dropped.
    try:
        _max_model_len = vllm_model.llm_engine.model_config.max_model_len
    except AttributeError:
        _max_model_len = None
    _truncate_at = (_max_model_len - max_out_len - 64) if _max_model_len else None
    if _truncate_at and prompts and isinstance(prompts[0], dict):
        n_over = 0
        new_prompts = []
        for p in prompts:
            ids = p.get("prompt_token_ids", [])
            if len(ids) > _truncate_at:
                n_over += 1
                new_prompts.append({"prompt_token_ids": ids[-_truncate_at:]})
            else:
                new_prompts.append(p)
        prompts = new_prompts
        if n_over:
            print(f"[generate_fast_vllm] left-truncated {n_over}/{len(prompts)} "
                  f"prompts to {_truncate_at} tokens (max_model_len={_max_model_len})")
    sampling = SamplingParams(
        temperature=0,
        max_tokens=max_out_len,
        stop=stop or [],
        stop_token_ids=stop_ids if stop_ids else None,
        truncate_prompt_tokens=_truncate_at,  # also covers raw-string inputs
    )
    outputs = vllm_model.generate(prompts, sampling, lora_request=lora_request)
    truncated = [getattr(o.outputs[0], 'finish_reason', None) == 'length' for o in outputs]
    # Use tok.decode when available: vLLM V1's o.outputs[0].text is empty for
    # prompt_token_ids inputs due to an incremental-detokenizer limitation.
    # Decoding from token_ids directly always yields the correct text.
    if tok is not None:
        texts = [tok.decode(o.outputs[0].token_ids, skip_special_tokens=True) for o in outputs]
    else:
        texts = [o.outputs[0].text for o in outputs]
    return texts, truncated


def generate_interactive(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    top_k: int = 5,
    max_out_len: int = 200,
    compare_against: Optional[AutoModelForCausalLM] = None,
    use_logit_lens: bool = False,
    layer_module_tmp: str = "transformer.h.{}",
    ln_f_module: str = "transformer.ln_f",
    lm_head_module: str = "lm_head",
):
    """
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    if use_logit_lens:
        llens_gen = LogitLens(
            model,
            tok,
            layer_module_tmp,
            ln_f_module,
            lm_head_module,
            disabled=not use_logit_lens,
        )
        if compare_against:
            llens_vanilla = LogitLens(
                compare_against,
                tok,
                layer_module_tmp,
                ln_f_module,
                lm_head_module,
                disabled=not use_logit_lens,
            )

    while True:
        prompt = input("Enter a prompt: ").strip(" \r\t\n")

        print(
            f"Argument Model: "
            f"{generate_fast(model, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
        )
        if compare_against:
            print(
                f"Baseline Model: "
                f"{generate_fast(compare_against, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
            )

        if use_logit_lens:
            inp_prompt = tok([prompt], padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

            with llens_gen:
                model(**inp_prompt)
            print("\n--- Argument Model Logit Lens ---")
            llens_gen.pprint()

            if compare_against:
                with llens_vanilla:
                    compare_against(**inp_prompt)
                print("--- Baseline Model Logit Lens ---")
                llens_vanilla.pprint()

        print()


def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
    vanilla_generation=False,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Instruct models (Gemma 3 IT, LLaMA 3 IT, Qwen IT, …) require the chat
    # template to be applied before generation.  Without it, the model sees a
    # raw prompt and immediately predicts EOS/pad tokens, producing empty or
    # all-<pad> outputs.  Mirror the same logic used in generate_fast_vllm:
    # apply the template, run model.generate() for the continuation only, and
    # decode just the new tokens so callers receive a clean response string.
    if getattr(tok, 'chat_template', None) is not None:
        device = next(model.parameters()).device
        results = []
        for prompt in prompts:
            # enable_thinking=False suppresses chain-of-thought for reasoning
            # models (e.g. Qwen3); ignored by non-reasoning tokenizers.
            try:
                input_ids = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_tensors="pt",
                ).to(device)
            except TypeError:
                input_ids = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to(device)
            prompt_len = input_ids.shape[1]
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_out_len,
                    pad_token_id=tok.eos_token_id,
                    do_sample=False,
                )
            new_tokens = out[0][prompt_len:]
            text = tok.decode(new_tokens, skip_special_tokens=True)
            # Strip any residual <think>…</think> blocks whose tags were not
            # removed as special tokens (safety net for future model variants).
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            results.append(text)
        return [t for t in results for _ in range(n_gen_per_prompt)]

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    if vanilla_generation:
        n_input = input_ids.shape[1]  # prompt length (padded); decode only new tokens
        gen_txt = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_out_len
        )
        if isinstance(gen_txt, torch.Tensor):
            txt = [tok.decode(x[n_input:], skip_special_tokens=True) for x in gen_txt.detach().cpu().numpy().tolist()]
            txt = [
                unicodedata.normalize("NFKD", x)
                .replace("\n\n", " ")
                .replace("<|endoftext|>", "")
                for x in txt
            ]
        else:
            txt = gen_txt
            txt = [
                unicodedata.normalize("NFKD", x)
                .replace("\n\n", " ")
                .replace("<|endoftext|>", "")
                .replace("<s> ", "")
                for x in txt
            ]
        return txt
    batch_size = input_ids.size(0)
    original_input_len = input_ids.shape[1]  # save before generation extends the tensor

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=None if 'llama'or'baichuan' in model.name_or_path.lower() else attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            if type(model_out) is torch.Tensor:
                logits = model_out
            else:
                logits = model_out.logits
            past_key_values = model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)
    txt = [tok.decode(x[original_input_len:], skip_special_tokens=True) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt
