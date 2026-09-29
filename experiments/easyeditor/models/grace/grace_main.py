from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .GRACE import GRACE
from .grace_hparams import GraceHyperParams
from .utils import tokenize, parent_module, brackets_to_periods
from ...util import nethook


def apply_grace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: GraceHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)
    device = torch.device(f'cuda:{hparams.device}')
    editor = GRACE(model=model, config=hparams, device=device)
    for request in requests:
        tokens = tokenize(request, tokenizer=tok, device=device)
        editor.edit(config=hparams, tokens=tokens, edit_id=request['target_new'])
            
    weights_copy = editor.reset_layer

    # Reset key_id so inference queries the last token via GRACEAdapter.forward's
    # key_id==-1 branch. editor.py unwraps the GRACE object and calls model.generate()
    # directly, bypassing GRACE.generate() which normally does this reset.
    adapter = getattr(parent_module(editor.model, brackets_to_periods(editor.layer)),
                      editor.layer.rsplit(".", 1)[-1])
    adapter.key_id = -1

    return editor, weights_copy


def load_grace_into_model(
        model: AutoModelForCausalLM,
        hparams: GraceHyperParams,
        path: str,
) -> GRACE:
    """Resume entry point: build a GRACE wrapper around a fresh HF model and
    load a saved codebook checkpoint into it.

    Used by run_medkit.load_checkpoint when resuming a GRACE run.  After
    this call, the underlying HF model has the GRACEAdapter wrapping its
    target layer and the codebook is restored.
    """
    device = torch.device(f'cuda:{hparams.device}')
    wrapper = GRACE(model=model, config=hparams, device=device)
    wrapper.load_model(path)
    return wrapper


