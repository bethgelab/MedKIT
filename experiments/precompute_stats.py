#!/usr/bin/env python3
"""
Precompute second-moment (covariance) statistics for MEMIT / AlphaEdit.

Run this ONCE per model on the GPU partition BEFORE editing experiments.
The resulting .npz files are cached under --stats_dir and loaded automatically
during editing (no recomputation needed).

Stats are stored at:
  <stats_dir>/<model_name_sanitised>/<ds_name>_stats/<layer>_<dtype>_mom2_<n>.npz

Example — Llama-3.1-8B (layers used by MEMIT and AlphaEdit):
    python precompute_stats.py \\
        --model_name meta-llama/Llama-3.1-8B-Instruct \\
        --layers 4,5,6,7,8 \\
        --stats_dir ./data/stats \\
        --n_samples 100 \\
        --batch_tokens 4096
"""
import argparse
import os
import sys

# Must be set before any HuggingFace tokenizer is imported to avoid
# fork deadlocks inside the DataLoader used by layer_stats.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from easyeditor.models.rome.layer_stats import layer_stats
from easyeditor.util.nethook import set_requires_grad


class _Hparams:
    """Minimal hparams object that layer_stats needs (only .device)."""
    def __init__(self, device: int):
        self.device = device


def main():
    parser = argparse.ArgumentParser(description="Precompute MEMIT/AlphaEdit covariance stats")
    parser.add_argument("--model_name", required=True,
                        help="HuggingFace model ID, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--layers", default="4,5,6,7,8",
                        help="Comma-separated layer indices to compute stats for")
    parser.add_argument("--rewrite_module_tmp", default="model.layers.{}.mlp.down_proj",
                        help="Layer name template; {} is replaced with the layer index")
    parser.add_argument("--stats_dir", default="./data/stats",
                        help="Directory to save/cache the .npz stat files")
    parser.add_argument("--ds_name", default="wikipedia",
                        choices=["wikipedia", "wikitext"],
                        help="Reference corpus for covariance estimation")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of text samples to process (100 = fast debug; >=100000 for production)")
    parser.add_argument("--batch_tokens", type=int, default=4096,
                        help="Max tokens per forward-pass batch. Keep <=4096 on 40GB GPUs.")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model (and stat accumulation) dtype")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index")
    args = parser.parse_args()

    layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    torch_dtype = {"float16": torch.float16,
                   "bfloat16": torch.bfloat16,
                   "float32": torch.float32}[args.dtype]
    hparams = _Hparams(device=args.device)

    print(f"[precompute_stats] model  : {args.model_name}")
    print(f"[precompute_stats] layers : {layer_indices}")
    print(f"[precompute_stats] corpus : {args.ds_name}  n_samples={args.n_samples}")
    print(f"[precompute_stats] dtype  : {args.dtype}   batch_tokens={args.batch_tokens}")
    print(f"[precompute_stats] output : {args.stats_dir}")
    print()

    print("Loading tokenizer …")
    # Compute nodes may have no internet.  Newer transformers calls list_repo_templates()
    # on HuggingFace Hub even for locally-cached models, which raises ConnectionError.
    # We try the normal load first; if a network error occurs we fall back to
    # local_files_only=True (no hub lookups at all).
    def _load_tokenizer(name):
        try:
            return AutoTokenizer.from_pretrained(name)
        except Exception as e:
            _estr = str(e) + str(type(e))
            if any(k in _estr for k in ("ConnectionError", "Failed to resolve",
                                         "list_repo_templates", "NoneType")):
                print(f"  Network unavailable or offline cache miss — retrying with local_files_only=True")
                # local_files_only searches all configured cache dirs (TRANSFORMERS_CACHE
                # and the default ~/.cache/huggingface) without any hub requests.
                return AutoTokenizer.from_pretrained(name, local_files_only=True)
            raise

    tok = _load_tokenizer(args.model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model in {args.dtype} …")
    def _load_model(name, torch_dtype, device):
        try:
            return AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=torch_dtype, device_map=f"cuda:{device}"
            ).eval()
        except Exception as e:
            _estr = str(e) + str(type(e))
            if any(k in _estr for k in ("ConnectionError", "Failed to resolve",
                                         "NoneType", "endswith")):
                print(f"  Network unavailable or offline cache miss — retrying with local_files_only=True")
                return AutoModelForCausalLM.from_pretrained(
                    name, torch_dtype=torch_dtype, device_map=f"cuda:{device}",
                    local_files_only=True
                ).eval()
            raise

    model = _load_model(args.model_name, torch_dtype, args.device)
    set_requires_grad(False, model)
    print(f"Model loaded on cuda:{args.device}  "
          f"(dtype={next(model.parameters()).dtype})")
    if hasattr(model, 'language_model'):
        print(f"VLM detected ({type(model).__name__}) — "
              f"layer_stats will use .language_model for hooks and forward passes")
    print()

    for layer_idx in layer_indices:
        layer_name = args.rewrite_module_tmp.format(layer_idx)
        print(f"─── Layer {layer_idx}: {layer_name}")
        layer_stats(
            model,
            tok,
            layer_name,
            args.stats_dir,
            args.ds_name,
            to_collect=["mom2"],
            sample_size=args.n_samples,
            precision=args.dtype,
            batch_tokens=args.batch_tokens,
            hparams=hparams,
        )
        print(f"    ✓ Layer {layer_idx} done\n")

    print(f"[precompute_stats] All layers done.  Stats saved under {args.stats_dir}/")


if __name__ == "__main__":
    main()
