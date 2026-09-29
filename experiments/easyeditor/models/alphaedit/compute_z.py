from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .AlphaEdit_hparams import AlphaEditHyperParams


def wrap_with_chat_template(content: str, tok: AutoTokenizer) -> str:
    """Wrap `content` as a single user turn in the tokenizer's chat template.

    Instruct models (Gemma-3-IT in particular) predict EOS/pad on raw prompts,
    which makes v-optimization fail — the model has no prior for continuing
    bare text. Evaluation already applies the chat template (see
    easyeditor.util.generate.generate_fast), so wrapping here makes the
    editing distribution match the evaluation distribution.

    '{}' placeholders for the subject survive the Jinja template (Jinja uses
    {{ }} for substitution, not {}). The leading BOS token emitted by some
    templates is stripped to avoid a double-BOS when the tokenizer later
    auto-adds BOS during tok(...).
    """
    if getattr(tok, "chat_template", None) is None:
        return content
    out = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    bos = getattr(tok, "bos_token", None)
    if bos and out.startswith(bos):
        out = out[len(bos):]
    return out


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: AlphaEditHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        # Gemma3Config (MLLM) nests vocab_size inside text_config
        vocab_size = (
            getattr(model.config, "vocab_size", None)
            or getattr(getattr(model.config, "text_config", None), "vocab_size", None)
        )
        lm_b = next(model.parameters()).new_zeros(vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs.
    # For instruct models with a chat template (Gemma-3-IT, Llama-3-IT, Qwen-IT)
    # we wrap each prompt as a user turn so the optimization distribution
    # matches the evaluation distribution. Varied context_templates are placed
    # INSIDE the user content so the template boundaries stay intact.
    rewriting_prompts = [
        wrap_with_chat_template(context.format(request["prompt"]), tok)
        + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ]
    kl_prompts = [wrap_with_chat_template("{} is a", tok)]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )

    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    # For MLLMs (e.g. Gemma3ForConditionalGeneration), hidden_size may be
    # nested inside text_config rather than on the top-level config.
    _cfg = model.config
    _text_cfg = getattr(_cfg, "text_config", _cfg)
    if hasattr(_cfg, 'n_embd'):
        _hidden = _cfg.n_embd
    elif hasattr(_cfg, 'hidden_size'):
        _hidden = _cfg.hidden_size
    elif hasattr(_text_cfg, 'hidden_size'):
        _hidden = _text_cfg.hidden_size
    else:
        raise NotImplementedError(f"Cannot determine hidden size from model config: {_cfg}")
    delta = torch.zeros((_hidden,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                if isinstance(cur_out, tuple):
                    target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                else:
                    target_init = cur_out[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                if isinstance(cur_out, tuple):
                    if len(lookup_idxs) != len(cur_out[0]):
                        cur_out[0][idx, i, :] += delta
                    else:
                        cur_out[0][i, idx, :] += delta
                else:
                    if len(lookup_idxs) != len(cur_out):
                        cur_out[idx, i, :] += delta
                    else:
                        cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Disable lm_head once for the entire optimization loop — we never need the
    # full [batch × seq × vocab] logits tensor; all log-probs are computed
    # manually from the traced loss_layer output at specific token positions.
    lm_head_mod = nethook.get_module(model, hparams.lm_head_module)
    _orig_lm_head_fwd = lm_head_mod.forward
    lm_head_mod.forward = lambda *args, **kwargs: args[0].new_empty(0)

    loss_layer_key = hparams.layer_module_tmp.format(loss_layer)
    edit_layer_key = hparams.layer_module_tmp.format(layer)

    try:
        # Execute optimization
        for it in range(hparams.v_num_grad_steps):
            opt.zero_grad()

            # ------------------------------------------------------------------
            # Gradient accumulation: forward + backward one prompt at a time.
            # Peak activation memory is proportional to batch_size=1 instead of
            # the full batch (num_rewriting_prompts + num_kl_prompts), which
            # avoids OOM on large models with long medical prompts.
            # ------------------------------------------------------------------
            nll_vals = []   # per-rewriting-prompt NLL (Python floats, for logging)
            kl_val   = 0.0

            for pi in range(len(all_prompts)):
                single_input  = {k: v[pi : pi + 1] for k, v in input_tok.items()}
                pi_lookup_idx = lookup_idxs[pi]

                # Per-prompt edit hook: adds delta at this prompt's lookup position.
                def _edit_fn(cur_out, cur_layer,
                             _idx=pi_lookup_idx):
                    nonlocal target_init
                    if cur_layer == edit_layer_key:
                        src = cur_out[0] if isinstance(cur_out, tuple) else cur_out
                        if target_init is None:
                            print("Recording initial value of v*")
                            target_init = src[0, _idx].detach().clone()
                        # Cast to float32 before adding delta: bf16 has ~1-unit precision
                        # at typical hidden-state magnitudes (~130/element), so small delta
                        # values get rounded to zero and gradients vanish.
                        orig_dtype = src.dtype
                        src_f32 = src.float().clone()
                        src_f32[0, _idx, :] = src_f32[0, _idx, :] + delta
                        out_tensor = src_f32.to(orig_dtype)
                        if isinstance(cur_out, tuple):
                            return (out_tensor,) + cur_out[1:]
                        return out_tensor
                    return cur_out

                with nethook.TraceDict(
                    module=model,
                    layers=[loss_layer_key, edit_layer_key],
                    retain_input=False,
                    retain_output=True,
                    edit_output=_edit_fn,
                ) as tr:
                    model(**single_input)

                # Extract loss_layer hidden state for this single prompt
                raw = tr[loss_layer_key].output
                loss_out = raw[0] if isinstance(raw, tuple) else raw
                if loss_out.shape[1] != single_input["input_ids"].shape[1]:
                    loss_out = loss_out.transpose(0, 1)
                # loss_out: [1, seq_len, hidden_size]

                if pi >= len(rewriting_prompts):
                    # ---- KL prompt ----
                    # Cast to float32: bf16 matmul backward triggers CUBLAS errors.
                    kl_h  = loss_out[0, pi_lookup_idx, :].float()  # [hidden]
                    kl_lg = (ln_f(kl_h.unsqueeze(0)).float()
                             @ lm_w.to(device=kl_h.device, dtype=kl_h.dtype)
                             + lm_b.to(device=kl_h.device, dtype=kl_h.dtype))  # [1, vocab]
                    kl_lp = torch.nn.functional.log_softmax(kl_lg, dim=1)
                    if kl_distr_init is None:
                        kl_distr_init = kl_lp.detach().clone()
                    kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                        kl_distr_init, kl_lp, log_target=True, reduction="batchmean"
                    )
                    kl_val = kl_loss.item()
                    (kl_loss / len(kl_prompts)).backward()

                else:
                    # ---- Rewriting prompt ----
                    s_targets = rewriting_targets[pi : pi + 1]      # [1, seq_len]
                    tmask     = (s_targets != -100)[0]               # [seq_len]
                    # Cast to float32: bf16 matmul backward triggers CUBLAS errors.
                    th        = loss_out[0, tmask, :].float()        # [n_tgt, hidden]
                    tid       = s_targets[0, tmask]                  # [n_tgt]
                    tlogits   = (ln_f(th).float()
                                 @ lm_w.to(device=th.device, dtype=th.dtype)
                                 + lm_b.to(device=th.device, dtype=th.dtype))  # [n_tgt, vocab]
                    tlp       = torch.nn.functional.log_softmax(tlogits, dim=1)
                    tok_lp    = tlp.gather(
                        1, tid.unsqueeze(1).to(tlp.device)
                    ).squeeze(1)                                     # [n_tgt]
                    nll_i     = -tok_lp.sum() / target_ids.size(0)
                    nll_vals.append(nll_i.item())
                    (nll_i / len(rewriting_prompts)).backward()

            # Weight-decay gradient (depends only on delta, no forward pass)
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            weight_decay.backward()

            nll_mean = sum(nll_vals) / max(len(nll_vals), 1)
            total_loss = nll_mean + kl_val + weight_decay.item()
            avg_prob   = np.exp(-nll_mean)
            print(
                f"loss {np.round(total_loss, 3)} = "
                f"{np.round(nll_mean, 3)} + "
                f"{np.round(kl_val, 3)} + "
                f"{np.round(weight_decay.item(), 3)} "
                f"avg prob of [{request['target_new']}] {avg_prob}"
            )

            if total_loss < 5e-2:
                break
            if it == hparams.v_num_grad_steps - 1:
                break

            opt.step()

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()

    finally:
        lm_head_mod.forward = _orig_lm_head_fwd

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_"):]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_"):],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
