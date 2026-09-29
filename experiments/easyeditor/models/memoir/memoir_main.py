from typing import Any, Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .MEMOIR import MEMOIR
from .utils import tokenize, get_context_templates
from .memoir_hparams import MEMOIRHyperParams

MEMOIRload = True


def apply_memoir_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MEMOIRHyperParams,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)
    device = f'cuda:{hparams.device}'
    # MEMOIR does not use act_mask/deact_mask, so context templates add no signal
    # and only inflate the training batch (11 templates × N edits + N loc_prompts
    # sequential passes per iteration = ~6× overhead).  Use a single identity
    # template to keep the batch at 2×N sequences instead.
    context_templates = get_context_templates(model, tok, length_params=[], device=device)
    editor = MEMOIR(model=model, config=hparams, device=device)
    import os
    global MEMOIRload
    if hasattr(hparams, 'load_path') and hparams.load_path and os.path.exists(hparams.load_path) and MEMOIRload:
        print("Start loading the MEMOIR model!")
        editor.load(hparams.load_path)
        MEMOIRload = False
    print(f"Executing MEMOIR algorithm for the update: ")
    for request in requests:
        print(
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    tokens, act_mask, deact_mask = tokenize(requests, tokenizer=tok, device=device, context_templates=context_templates, hparams=hparams)
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    # MEMOIR manages its own state via the in-place MEMOIRAdapter; there is no
    # dict of original weights to restore.  Return {} so BaseEditor code paths
    # that iterate weights_copy.items() don't raise AttributeError.
    weights_copy = {}

    return editor, weights_copy


def load_memoir_into_model(
        model: AutoModelForCausalLM,
        hparams: MEMOIRHyperParams,
        path: str,
) -> MEMOIR:
    """Resume entry point: build a MEMOIR wrapper around a fresh HF model and
    load a saved checkpoint into it.

    Used by run_medkit.load_checkpoint when resuming a MEMOIR run.  The
    `setattr` wrap inside `MEMOIR.__init__` is in-place on `model`, so after
    this call returns the underlying HF model is wrapped (with the correct
    `hasher.permutation` and `masks_for_edited_samples` restored) and the next
    call to `apply_memoir_to_model` is a no-op for the wrapping step.
    """
    wrapper = MEMOIR(model=model, config=hparams, device=f'cuda:{hparams.device}')
    wrapper.load_model(path)
    return wrapper
