"""
Contains utilities for extracting token representations and indices
from string templates. Used in computing the left and right vectors for ROME.
"""

import contextlib
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook


# ── lm_head bypass for repr extraction ──────────────────────────────────────
# `get_reprs_at_idxs` only needs a hidden state captured via a forward hook on
# a transformer layer; the full forward's lm_head materialisation
# (hidden → vocab) is unused but allocates O(batch × seq × vocab) bf16 bytes.
# On Gemma-3 family models that's catastrophic — vocab=262144 means a single
# batch=128, seq=262 forward needs a 17 GB contiguous allocation just to throw
# the result away.  The OOM cliff hits AlphaEdit / MEMIT / ROME / EMMET / R-ROME
# the moment per-edit context-template counts push that allocation past the
# largest free contiguous block.
#
# Stubbing lm_head with an Identity-shaped module preserves the forward path
# (model still returns a CausalLMOutput-shaped object; downstream code in
# repr_tools never reads `.logits`) while reducing the allocation to
# O(batch × seq × hidden) — a ~100× shrink on Gemma-3.
class _LMHeadBypass(nn.Module):
    """Drop-in replacement for an LM head that returns the input unchanged.

    The output's last dim is `hidden_size` instead of `vocab_size`, but
    `repr_tools` and its callers don't read `.logits`, so the shape mismatch
    is harmless.  A `weight` attribute is exposed so that any downstream code
    inspecting `lm_head.weight.dtype/device` (e.g. tied-embedding sanity
    checks) doesn't AttributeError.
    """

    def __init__(self, ref_param: torch.Tensor):
        super().__init__()
        # Borrow dtype/device from the original lm_head's weight without
        # copying its values — we never use this for matmul.
        self.weight = ref_param

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def _find_lm_head(model: nn.Module) -> Tuple[Optional[nn.Module], Optional[str]]:
    """Locate the lm_head module under common HF wrapper layouts.

    Returns (parent_module, attr_name) so the caller can swap it out with
    setattr.  Try the deeper, canonical locations FIRST — on multimodal
    wrappers (Gemma3ForConditionalGeneration, Llava…) the top-level `lm_head`
    is often a `@property` that delegates to `language_model.lm_head`, and
    setattr on a property without a setter raises AttributeError.  Setting
    the underlying `language_model.lm_head` instead avoids that.
    """
    # 1. Multimodal wrapper: `model.language_model.lm_head` is the real Module.
    lm = getattr(model, 'language_model', None)
    if lm is not None and hasattr(lm, 'lm_head') and isinstance(lm.lm_head, nn.Module):
        return lm, 'lm_head'
    # 2. PEFT / nested wrappers: `model.model.lm_head`.
    inner = getattr(model, 'model', None)
    if inner is not None and hasattr(inner, 'lm_head') and isinstance(inner.lm_head, nn.Module):
        return inner, 'lm_head'
    # 3. Plain CausalLM: `model.lm_head` directly.  Last because some wrappers
    #    expose this as a read-only property.
    if hasattr(model, 'lm_head') and isinstance(getattr(model, 'lm_head'), nn.Module):
        # Confirm we can actually setattr (catches @property without setter).
        try:
            current = model.lm_head
            model.lm_head = current  # idempotent reassignment
            return model, 'lm_head'
        except (AttributeError, TypeError):
            pass
    return None, None


@contextlib.contextmanager
def _stub_lm_head(model: nn.Module):
    """Temporarily replace `model`'s lm_head with `_LMHeadBypass` for the
    duration of the with-block.  Restored on exit (and on exception)."""
    parent, attr = _find_lm_head(model)
    if parent is None:
        # Couldn't find lm_head — fall back to the original forward.  Slower
        # but correct; no behaviour change vs. before this patch.
        yield
        return
    original = getattr(parent, attr)
    # Pull a tensor we can re-export as `bypass.weight` to keep dtype/device
    # introspection working downstream.
    ref = getattr(original, 'weight', None)
    if not isinstance(ref, torch.Tensor):
        # Some lm_heads are not nn.Linear (e.g. tied-embedding wrappers).
        # Fall back to a zero scalar param of the model's main dtype.
        ref = next(model.parameters()).new_zeros(1)
    bypass = _LMHeadBypass(ref).to(device=ref.device, dtype=ref.dtype)
    setattr(parent, attr, bypass)
    try:
        yield
    finally:
        setattr(parent, attr, original)

def get_reprs_at_word_tokens(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    context_templates: List[str],
    words: List[str],
    layer: int,
    module_template: str,
    subtoken: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Retrieves the last token representation of `word` in `context_template`
    when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
    for more details.
    """

    idxs = get_words_idxs_in_templates(tok, context_templates, words, subtoken)
    return get_reprs_at_idxs(
        model,
        tok,
        [context_templates[i].format(words[i]) for i in range(len(words))],
        idxs,
        layer,
        module_template,
        track,
    )

def get_words_idxs_in_templates(
    tok: AutoTokenizer, context_templates: str, words: str, subtoken: str
) -> int:
    """
    Given list of template strings, each with *one* format specifier
    (e.g. "{} plays basketball"), and words to be substituted into the
    template, computes the post-tokenization index of their last tokens.
    """

    assert all(
        tmp.count("{}") == 1 for tmp in context_templates
    ), "We currently do not support multiple fill-ins for context"


    prefixes_len, words_len, suffixes_len, inputs_len = [], [], [], []
    for i, context in enumerate(context_templates):
        prefix, suffix = context.split("{}")
        prefix_len = len(tok.encode(prefix))
        prompt_len = len(tok.encode(prefix + words[i]))
        input_len = len(tok.encode(prefix + words[i] + suffix))
        prefixes_len.append(prefix_len)
        words_len.append(prompt_len - prefix_len)
        suffixes_len.append(input_len - prompt_len)
        inputs_len.append(input_len)

    # Compute prefixes and suffixes of the tokenized context
    # fill_idxs = [tmp.index("{}") for tmp in context_templates]
    # prefixes, suffixes = [
    #     tmp[: fill_idxs[i]] for i, tmp in enumerate(context_templates)
    # ], [tmp[fill_idxs[i] + 2 :] for i, tmp in enumerate(context_templates)]
    # words = deepcopy(words)
    #
    # # Pre-process tokens
    # for i, prefix in enumerate(prefixes):
    #     if len(prefix) > 0:
    #         assert prefix[-1] == " "
    #         prefix = prefix[:-1]
    #
    #         prefixes[i] = prefix
    #         words[i] = f" {words[i].strip()}"
    #
    # # Tokenize to determine lengths
    # assert len(prefixes) == len(words) == len(suffixes)
    # n = len(prefixes)
    # batch_tok = tok([*prefixes, *words, *suffixes])
    # if 'input_ids' in batch_tok:
    #     batch_tok = batch_tok['input_ids']
    # prefixes_tok, words_tok, suffixes_tok = [
    #     batch_tok[i : i + n] for i in range(0, n * 3, n)
    # ]
    # prefixes_len, words_len, suffixes_len = [
    #     [len(el) for el in tok_list]
    #     for tok_list in [prefixes_tok, words_tok, suffixes_tok]
    # ]

    # Compute indices of last tokens
    if subtoken == "last" or subtoken == "first_after_last":
        return [
            [
                prefixes_len[i]
                + words_len[i]
                - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
            ]
            # If suffix is empty, there is no "first token after the last".
            # So, just return the last token of the word.
            for i in range(len(context_templates))
        ]
    elif subtoken == "first":
        return [[prefixes_len[i] - inputs_len[i]] for i in range(len(context_templates))]
    else:
        raise ValueError(f"Unknown subtoken type: {subtoken}")


def get_reprs_at_idxs(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    contexts: List[str],#表示该知识的完整句子
    idxs: List[List[int]],#被填入词的位置
    layer: int,
    module_template: str,
    track: str = "in",
) -> torch.Tensor:
    """
    Runs input through model and returns averaged representations of the tokens
    at each index in `idxs`.
    """

    def _batch(n):
        for i in range(0, len(contexts), n):
            yield contexts[i : i + n], idxs[i : i + n]#将句子和被填词位置分块

    assert track in {"in", "out", "both"}
    both = track == "both"
    tin, tout = (
        (track == "in" or both),
        (track == "out" or both),
    )#tin tout都是bool结构
    module_name = module_template.format(layer)
    to_return = {"in": [], "out": []}

    def _process(cur_repr, batch_idxs, key):
        nonlocal to_return
        cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
        if cur_repr.shape[0]!=len(batch_idxs):
            cur_repr=cur_repr.transpose(0,1)
        for i, idx_list in enumerate(batch_idxs):
            to_return[key].append(cur_repr[i][idx_list].mean(0))

    for batch_contexts, batch_idxs in _batch(n=128):
        #contexts_tok:[21 19]
        contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )

        # `_stub_lm_head` swaps the model's lm_head with an Identity for the
        # forward pass.  We never read `.logits` here — only the layer
        # input/output captured by `nethook.Trace` — so skipping the
        # vocab-projection saves O(batch × seq × vocab) memory.  On Gemma-3
        # (vocab=262144) this is the difference between OOM and success on a
        # 40 GB A100 once context-template count × edit count grows.
        with torch.no_grad():
            with _stub_lm_head(model), nethook.Trace(
                module=model,
                layer=module_name,
                retain_input=tin,
                retain_output=tout,
            ) as tr:
                model(**contexts_tok)

        if tin:
            _process(tr.input, batch_idxs, "in")
        if tout:
            _process(tr.output, batch_idxs, "out")

    to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

    if len(to_return) == 1:
        return to_return["in"] if tin else to_return["out"]
    else:
        return to_return["in"], to_return["out"]
