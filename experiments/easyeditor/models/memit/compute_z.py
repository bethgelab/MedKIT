import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .memit_hparams import MEMITHyperParams


def wrap_with_chat_template(content: str, tok: AutoTokenizer) -> str:
    """
    Wrap a prompt (which may contain a '{}' placeholder for the subject) in
    the tokenizer's chat template as a single user turn, returning a string
    primed for the model to continue (add_generation_prompt=True).

    Instruct models (Gemma-3-IT in particular) predict EOS/pad on raw prompts,
    which makes MEMIT's v-optimization fail — the model has no prior for
    continuing bare text.  Evaluation already applies the chat template
    (see easyeditor.util.generate.generate_fast), so wrapping here makes the
    editing distribution match the evaluation distribution.

    The '{}' placeholder inside `content` is preserved through the Jinja
    template (Jinja uses {{ }} for substitution, not {}).  We strip the
    leading BOS token that Gemma's template emits, because subsequent
    tokenization auto-adds BOS — leaving it would produce a double-BOS
    and misalign all token indices.
    """
    if getattr(tok, 'chat_template', None) is None:
        return content
    out = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    bos = getattr(tok, 'bos_token', None)
    if bos and out.startswith(bos):
        out = out[len(bos):]
    return out


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    # Some models (e.g. Qwen3) tie lm_head.weight to the token embedding weight.
    # PyTorch's named_parameters() deduplicates tied weights, so lm_head.weight
    # is not listed separately — get_parameter raises LookupError.  Fall back to
    # accessing the weight directly from the module instead.
    try:
        lm_w = nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T
    except LookupError:
        lm_w = nethook.get_module(model, hparams.lm_head_module).weight.T
    ln_f = nethook.get_module(model, hparams.ln_f_module)
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError:
        # Gemma3Config (MLLM) nests vocab_size inside text_config
        vocab_size = (
            getattr(model.config, "vocab_size", None)
            or getattr(getattr(model.config, "text_config", None), "vocab_size", None)
        )
        lm_b = next(model.parameters()).new_zeros(vocab_size)

    #print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs.
    # For instruct models with a chat template (Gemma-3-IT, Llama-3-IT, Qwen-IT)
    # we wrap each prompt as a user turn so the optimization distribution matches
    # the evaluation distribution.  Varied context_templates are placed INSIDE
    # the user content so the template boundaries stay intact.
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
    #print(f"Rewrite layer is {layer}")
    #print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    # For MLLMs (e.g. Gemma3ForConditionalGeneration), n_embd / hidden_size may
    # be nested inside a sub-config (text_config) rather than on the top-level.
    _cfg = model.config
    _text_cfg = getattr(_cfg, "text_config", _cfg)
    if hasattr(_cfg, 'n_embd'):
        _hidden = _cfg.n_embd
    elif hasattr(_cfg, 'hidden_size'):
        _hidden = _cfg.hidden_size
    elif hasattr(_text_cfg, 'hidden_size'):
        _hidden = _text_cfg.hidden_size
    else:
        raise NotImplementedError(
            f"Cannot determine hidden size from model config: {_cfg}"
        )
    delta = torch.zeros((_hidden,), requires_grad=True, device=f"cuda:{hparams.device}")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                #print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs)!=len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta.to(cur_out[0].device)
                else:
                    cur_out[0][i, idx, :] += delta.to(cur_out[0].device)

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    _verbose = bool(int(os.environ.get("MEMIT_VERBOSE", "0") or 0))

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets

        output=tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1]!=rewriting_targets.shape[1]:
            output=torch.transpose(output, 0, 1)
        full_repr = output[:len(rewriting_prompts)]

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init).to(delta.device) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        if _verbose:
            print(
                f"[MEMIT][v_opt] it={it:02d} loss={loss.item():.3f} "
                f"nll={nll_loss.item():.3f} kl={kl_loss.item():.3f} "
                f"wd={weight_decay.item():.3f} p(target)={torch.exp(-nll_loss_each).mean().item():.4f} "
                f"||delta||={delta.norm().item():.3f}"
            )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        max_norm = max_norm.to(delta.device)
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta.to(target_init.device)
    if _verbose:
        print(
            f"[MEMIT][v_opt] done — ||target_init||={target_init.norm().item():.3f} "
            f"||delta||={delta.norm().item():.3f} ||target||={target.norm().item():.3f}"
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
    track=None,
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
        subtoken = fact_token_strategy[len("subject_") :]
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=False,
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
            subtoken=fact_token_strategy[len("subject_") :],
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
