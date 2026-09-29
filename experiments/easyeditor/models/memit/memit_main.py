import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx, wrap_with_chat_template
from .memit_hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_memit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(f"cuda:{hparams.device}"), val_mat.to(f"cuda:{hparams.device}")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    #print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def _resolve_module_prefix(model, hparams):
    """
    Return a (possibly updated) copy of hparams whose module-template strings
    are adjusted to match the model's actual hookable modules.

    Needed for MLLMs such as Gemma3ForConditionalGeneration where the language
    model weights are nested under a sub-module rather than at the root.

    Key design decisions
    --------------------
    * We search named_modules() (not named_parameters()) because that is what
      nethook.get_module / TraceDict use for hooks.
    * The ORIGINAL template strings are preserved in hparams._stats_* so that
      layer_stats (which already does `model = model.language_model`) continues
      to receive paths relative to the language-model sub-module, and stats
      files on disk remain findable under their original names.
    * Matching uses dot-component comparison to avoid substring false positives
      (e.g. 'language_model.layers' must not match suffix 'model.layers').
    """
    import copy

    probe_layer = hparams.layers[0]
    probe_module = hparams.rewrite_module_tmp.format(probe_layer)

    module_names = [n for n, _ in model.named_modules() if n]
    module_set = set(module_names)

    # Fast path: the configured path already exists as a hookable module
    if probe_module in module_set:
        return hparams

    # Locate 'layers' as a stable component boundary across all architectures
    template_parts = probe_module.split(".")
    try:
        layers_idx = next(i for i, p in enumerate(template_parts) if p == "layers")
    except StopIteration:
        raise LookupError(
            f"Cannot auto-detect module prefix: 'layers' not found in template "
            f"'{hparams.rewrite_module_tmp}'. Sample modules:\n"
            + "\n".join(sorted(module_names)[:20])
        )

    # Suffix from 'layers' to end of template — stable across text-only / MLLM
    suffix_parts = template_parts[layers_idx:]          # e.g. ['layers','4','mlp','down_proj']
    template_prefix_parts = template_parts[:layers_idx] # e.g. ['model']

    # Find a module whose tail components match suffix_parts
    match_parts = None
    for n in module_names:
        n_parts = n.split(".")
        if n_parts[-len(suffix_parts):] == suffix_parts:
            match_parts = n_parts
            break

    if match_parts is None:
        raise LookupError(
            f"No hookable module ending with {suffix_parts} found in model. "
            f"Sample modules:\n" + "\n".join(sorted(module_names)[:20])
        )

    # Everything before 'layers…' in the matched path is the actual prefix
    actual_prefix_parts = match_parts[: -len(suffix_parts)]

    # Compute the extra components that the MLLM wrapper inserts
    if actual_prefix_parts[: len(template_prefix_parts)] == template_prefix_parts:
        extra_parts = actual_prefix_parts[len(template_prefix_parts):]
    else:
        extra_parts = actual_prefix_parts
        template_prefix_parts = []

    if not extra_parts:
        return hparams

    print(
        f"[MEMIT] MLLM wrapper detected — adjusting hook paths "
        f"(inserting '{'.'.join(extra_parts)}' after '{'.'.join(template_prefix_parts)}')"
    )

    hparams = copy.copy(hparams)

    # Save originals so layer_stats can use them for stats-file naming and for
    # its own hooks (layer_stats already extracts model.language_model itself)
    hparams._stats_rewrite_module_tmp = hparams.rewrite_module_tmp
    hparams._stats_layer_module_tmp   = hparams.layer_module_tmp

    def _insert_prefix(tmpl):
        parts = tmpl.split(".")
        if parts[: len(template_prefix_parts)] == template_prefix_parts:
            new_parts = (
                parts[: len(template_prefix_parts)]
                + extra_parts
                + parts[len(template_prefix_parts):]
            )
        else:
            new_parts = actual_prefix_parts + parts
        return ".".join(new_parts)

    hparams.rewrite_module_tmp = _insert_prefix(hparams.rewrite_module_tmp)
    hparams.layer_module_tmp   = _insert_prefix(hparams.layer_module_tmp)
    hparams.mlp_module_tmp     = _insert_prefix(hparams.mlp_module_tmp)
    hparams.attn_module_tmp    = _insert_prefix(hparams.attn_module_tmp)

    # lm_head and ln_f may live at a different depth than the decoder layers.
    # Find them by leaf name, preferring the candidate closest to the layer prefix.
    layer_prefix_parts = actual_prefix_parts

    def _find_module_path(leaf_name, fallback_tmpl):
        candidates = [n for n in module_names if n.split(".")[-1] == leaf_name]
        if not candidates:
            return _insert_prefix(fallback_tmpl)
        def _shared(path):
            p, lp = path.split("."), layer_prefix_parts
            return sum(1 for a, b in zip(p, lp) if a == b)
        return max(candidates, key=_shared)

    hparams.lm_head_module = _find_module_path(
        hparams.lm_head_module.split(".")[-1], hparams.lm_head_module
    )
    hparams.ln_f_module = _find_module_path(
        hparams.ln_f_module.split(".")[-1], hparams.ln_f_module
    )

    print(
        f"[MEMIT] Hook paths: rewrite='{hparams.rewrite_module_tmp}', "
        f"lm_head='{hparams.lm_head_module}', ln_f='{hparams.ln_f_module}'"
    )
    return hparams


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    print(
        f"[MEMIT] padding_side={tok.padding_side} "
        f"chat_template={'yes' if getattr(tok, 'chat_template', None) else 'no'}"
    )

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]

        if '{}' not in request['prompt']:
            found_in_prompt = request['subject'] in request['prompt']
            if not found_in_prompt:
                s = ''.join(e for e in request['subject'] if e.isalnum())
                p = ''.join(e for e in request['prompt'] if e.isalnum())
                if s in p:
                    requests[i]['prompt'] = p
                    requests[i]['subject'] = s
                    found_in_prompt = True
                else:
                    # Subject not found in prompt (e.g. HemOnc drug-pair IDs);
                    # append {} placeholder so fact_token="last" can locate the edit target
                    print(f"Warning: Subject '{request['subject']}' not found in prompt — using last-token fallback.")
                    requests[i]['prompt'] = request['prompt'] + ' {}'
                    found_in_prompt = False

            if found_in_prompt:
                requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')

    #for request in requests[:10]:
    #    print(
    #        f"MEMIT request sample: "
    #        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
    #    )

    # For MLLMs (e.g. Gemma 3 loaded as Gemma3ForConditionalGeneration), the
    # language-model weights live under a sub-module prefix that differs from
    # the bare paths used by decoder-only models.  Auto-detect this prefix so
    # the same YAML works regardless of how the model was instantiated.
    hparams = _resolve_module_prefix(model, hparams)

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(f"cuda:{hparams.device}"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                #print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        #print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        #print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error — wrap prompts with chat template to match the
        # distribution used in compute_z/compute_ks (see wrap_with_chat_template).
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[wrap_with_chat_template(request["prompt"], tok) for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
            track='out'
        ).T
        targets = zs - cur_zs
        #print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        # Use the original (pre-MLLM-adjustment) template for the stats file
        # path so that cached .npz files remain findable, and so that
        # layer_stats (which does its own model.language_model extraction)
        # receives module paths relative to the language-model sub-module.
        _stats_tmpl = getattr(hparams, "_stats_rewrite_module_tmp", hparams.rewrite_module_tmp)
        force_recompute = False
        # force_recompute = layer != hparams.layers[0]
        cov = get_cov(
            model,
            tok,
            _stats_tmpl.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
            hparams=hparams
        )

        # Compute update in double precision
        layer_ks, targets = (
            layer_ks.double(),
            targets.double(),
        )

        #if torch.isnan(cov).any():
        #    print('nan:', torch.isnan(cov).sum(), 'not nan', cov.numel(), 'nan %', torch.isnan(cov).sum() / cov.numel())
        # check for zero rows
        #row_sum = torch.sum(cov, dim=1)
        #print('number of zero rows:', torch.sum(row_sum == 0), 'total rows:', row_sum.size(0))

        # counter matrix singularity by adding a small value to the diagonal
        cov += torch.eye(cov.size(0)).to(cov.device) * 1e-6

        adj_k = torch.linalg.solve(
            hparams.mom2_update_weight * cov.double() + layer_ks @ layer_ks.T,
            layer_ks,
        )

        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        #print("orig norm", torch.linalg.norm(weights[weight_name]))
        #print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    #print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    #print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")

    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")
        # check cov for nan
        #if torch.isnan(COV_CACHE[key]).any():
        #    print('nan:', torch.isnan(COV_CACHE[key]).sum(), 'not nan', COV_CACHE[key].numel(), 'nan %', torch.isnan(COV_CACHE[key]).sum() / COV_CACHE[key].numel())
        # check for zero rows
        #row_sum = torch.sum(COV_CACHE[key], dim=1)
        #print('zero rows:', torch.where(row_sum == 0))
        #print('number of zero rows:', torch.sum(row_sum == 0), 'total rows:', row_sum.size(0))

    return (
        torch.inverse(COV_CACHE[key].to(f"cuda:{hparams.device}")) if inv else COV_CACHE[key].to(f"cuda:{hparams.device}")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        #print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
