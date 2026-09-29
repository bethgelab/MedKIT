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
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P_loaded = False
cache_c_new = False


def _resolve_nullspace_threshold(spec, S: torch.Tensor) -> float:
    """Resolve a ``nullspace_threshold`` spec into an absolute cutoff for this
    layer's singular values.

    Accepted forms:
        float / "abs:X"     → X                          (absolute cutoff)
        "rel_max:X"         → X * float(S.max())         (per-layer relative)
        "keep_top_frac:X"   → cutoff s.t. top ceil(X·d) SVs pass as column-space

    The absolute form is retained for backwards compatibility but is brittle:
    its meaning depends on the scale of the covariance matrix, which varies
    drastically across models (float16 cov → max SV ≈ 0.3; float32 cov →
    max SV ≈ 80). Prefer a relative spec.
    """
    if isinstance(spec, (int, float)):
        kind, val = "abs", float(spec)
    elif isinstance(spec, str):
        if ":" not in spec:
            raise ValueError(
                f"[AlphaEdit] nullspace_threshold spec '{spec}' must be 'abs:X', "
                f"'rel_max:X', 'keep_top_frac:X', or a bare float."
            )
        kind, val_s = spec.split(":", 1)
        val = float(val_s)
    else:
        raise TypeError(
            f"[AlphaEdit] nullspace_threshold must be float or str, got {type(spec).__name__}"
        )

    if kind == "abs":
        return val
    if kind == "rel_max":
        return val * float(S.max().item())
    if kind == "keep_top_frac":
        if not 0.0 < val <= 1.0:
            raise ValueError(f"[AlphaEdit] keep_top_frac value must be in (0, 1], got {val}")
        d = int(S.numel())
        k = max(1, int(round(val * d)))
        # torch.linalg.svd returns S descending; pick a cutoff just below the
        # k-th largest SV so exactly k SVs are preserved as column-space.
        S_desc, _ = torch.sort(S, descending=True)
        kth = float(S_desc[k - 1].item())
        # Nudge down by an epsilon scaled to kth to handle ties at the cutoff.
        return kth * (1.0 - 1e-9) if kth > 0 else -1.0
    raise ValueError(f"[AlphaEdit] unknown nullspace_threshold kind '{kind}'")


def _threshold_cache_tag(spec) -> str:
    """Filesystem-safe tag for a threshold spec, used to namespace the cached P."""
    if isinstance(spec, (int, float)):
        s = f"abs_{float(spec):g}"
    else:
        s = str(spec).replace(":", "_")
    return "thr-" + s.replace("+", "").replace(".", "p").replace("/", "_")


def _derive_P_loc(base_path: str, tag: str) -> str:
    """Inject a threshold tag into the cached P filename so different thresholds
    don't silently reuse each other's cached matrices. Idempotent."""
    p = Path(base_path)
    if f"__{tag}" in p.stem:
        return str(p)
    return str(p.with_name(f"{p.stem}__{tag}{p.suffix}"))


def _is_llama_like(model_name: str) -> bool:
    """Return True for all Llama-architecture models (down_proj shape: [hidden, intermediate])."""
    name = model_name.lower()
    return any(k in name for k in [
        "llama", "qwen", "gemma", "mistral", "gpt-j",
        "bio-medical", "contactdoctor", "adaptllm", "medicine",
    ])


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
        f"[AlphaEdit] MLLM wrapper detected — adjusting hook paths "
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
        f"[AlphaEdit] Hook paths: rewrite='{hparams.rewrite_module_tmp}', "
        f"lm_head='{hparams.lm_head_module}', ln_f='{hparams.ln_f_module}'"
    )
    return hparams


def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    reset_cache=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :param reset_cache: If true, will reset cache_c_new to False, forcing re-initialization of cache_c.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    global P, P_loaded, cache_c, cache_c_new

    # Reset cache if requested
    if reset_cache:
        cache_c_new = False

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    # For MLLMs (e.g. Gemma 3 loaded as Gemma3ForConditionalGeneration), the
    # language-model weights live under a sub-module prefix that differs from
    # the bare paths used by decoder-only models.  Auto-detect this prefix so
    # the same YAML works regardless of how the model was instantiated.
    hparams = _resolve_module_prefix(model, hparams)

    # Calculate the null-space projection matrix P.
    # The cache path is namespaced by the threshold spec so that swapping
    # `nullspace_threshold` in hparams (e.g. "abs:5.0" → "rel_max:1e-2") does
    # NOT silently reuse a P computed under the old spec.
    P_loc_resolved = _derive_P_loc(hparams.P_loc, _threshold_cache_tag(hparams.nullspace_threshold))
    if not os.path.exists(P_loc_resolved):
        print(os.path.abspath(P_loc_resolved))
        print(f"The null-space projection matrix P does not exist and now calculate.")
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if _is_llama_like(hparams.model_name):
            P = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        else:
            P = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        del W_out
        os.makedirs(os.path.dirname(os.path.abspath(P_loc_resolved)), exist_ok=True)
        for i, layer in enumerate(hparams.layers):
            P[i, :, :] = get_project(model, tok, layer, hparams)
        torch.save(P, P_loc_resolved)
        P_loaded = True
    elif P_loaded == False:
        P = torch.load(P_loc_resolved)
        P_loaded = True

    # Maintain the global variable cache_c to avoid redundant computations.
    if not cache_c_new:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(hparams.layers[-1])}.weight")
        if _is_llama_like(hparams.model_name):
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[1], W_out.shape[1]), device="cpu")
        else:
            cache_c = torch.zeros((len(hparams.layers), W_out.shape[0], W_out.shape[0]), device="cpu")
        del W_out
        cache_c_new = True

    deltas = execute_AlphaEdit(model, tok, requests, hparams, cache_template=cache_template)

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            upd_matrix = upd_m.to(f"cuda:{hparams.device}")
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the AlphaEdit update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"] = " " + request["target_new"]
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject:{request['subject']} do not exist in prompt: {request['prompt']}")
        requests[i]['prompt'] = requests[i]['prompt'].replace(requests[i]['subject'], '{}')
        print(
            f"Executing AlphaEdit algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

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
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers

        # Solve on CPU: P and cache_c are already there, and the intermediate
        # [hidden × hidden] matrices (up to ~800 MiB each for Llama 8B) would OOM
        # the GPU if moved there simultaneously.  Only the small result is sent back.
        ks_cpu    = layer_ks.cpu().float()
        resid_cpu = resid.T.cpu().float()
        P_i       = P[i, :, :].float()          # already on CPU
        c_i       = cache_c[i, :, :].float()    # already on CPU
        A = P_i @ (ks_cpu @ ks_cpu.T + c_i) + hparams.L2 * torch.eye(
            ks_cpu.shape[0], dtype=torch.float
        )
        b = P_i @ ks_cpu @ resid_cpu
        upd_matrix = torch.linalg.solve(A, b).to(f"cuda:{hparams.device}")

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        orig_norm = torch.linalg.norm(weights[weight_name]).float()
        upd_norm = torch.linalg.norm(upd_matrix).float()
        print("orig norm", orig_norm)
        print("upd norm", upd_norm)

        # Guard: compute_z can diverge (flat loss, saturated activations) and
        # emit an upd_matrix whose norm dwarfs the weight. Unchecked, one such
        # delta poisons hidden-state stats for all subsequent edits. If the
        # ratio exceeds `max_update_ratio`, scale the delta down so the
        # effective ratio is capped at that value.
        max_ratio = getattr(hparams, "max_update_ratio", 0.5) or 0.0
        if max_ratio > 0 and upd_norm > 0:
            ratio = float((upd_norm / orig_norm).item())
            if ratio > max_ratio:
                scale = max_ratio / ratio
                print(
                    f"[AlphaEdit] layer {layer}: clamping upd (ratio {ratio:.3g} "
                    f"> max {max_ratio}); scale={scale:.3g}"
                )
                upd_matrix = upd_matrix * scale

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
            deltas[weight_name] = upd_matrix.detach().cpu()

        # Clear GPU memory
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i, :, :] += layer_ks.cpu() @ layer_ks.cpu().T

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

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

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
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
            "Update matrix computed by AlphaEdit does not match original weight shape. "
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
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE


def get_project(model, tok, layer, hparams):
    force_recompute = False
    # Use the original (pre-MLLM-adjustment) template for stats file naming so
    # that cached .npz files remain findable under their original names.
    _stats_tmpl = getattr(hparams, "_stats_rewrite_module_tmp", hparams.rewrite_module_tmp)
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
    ).cpu()
    # Upcast to float64 before SVD: the covariance matrix accumulated in
    # float16 is numerically ill-conditioned in float32 (many near-equal
    # singular values), which causes torch.linalg.svd / LAPACK SLASCL to
    # fail with error code 17. float64 gives sufficient precision.
    cov_d = cov.double()
    U, S, _ = torch.linalg.svd(cov_d, full_matrices=False)
    threshold = _resolve_nullspace_threshold(hparams.nullspace_threshold, S)
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    # Diagnostic: show SV range so the threshold can be tuned if needed.
    # `spec → t=...` makes it obvious when a relative spec resolves differently
    # across layers (which it should, on a per-layer max-SV).
    print(
        f"[AlphaEdit] SV stats: min={S.min():.4g}, p1={S.kthvalue(max(1, len(S)//100)).values:.4g}, "
        f"median={S.median():.4g}, max={S.max():.4g} | "
        f"null-space: {len(small_singular_indices)}/{len(S)} "
        f"(spec='{hparams.nullspace_threshold}' → t={threshold:.4g})"
    )
    # Return as float32 — callers expect float32 projection matrices
    return (U[:, small_singular_indices] @ U[:, small_singular_indices].T).float()
