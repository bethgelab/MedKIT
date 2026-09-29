#!/usr/bin/env python3
"""
Pre-compute MEMOIR background features for a given model.

MEMOIR uses a mean-decentered prompt-feature aggregation:
    features - mean(background_features)

Background features should capture the model's hidden representations for
*irrelevant* text (text the model should NOT be edited on), so they must be
collected from a diverse background corpus — NOT from the editing dataset.

Run this ONCE per model before any MEMOIR editing experiment.
Results are saved to the path specified by --output_path (or the
dir_background_features field in the MEMOIR hparams yaml).

Example:
    python generate_memoir_features.py \\
        --model_name Qwen/Qwen3-4B-Instruct-2507 \\
        --layer model.layers[30].mlp.down_proj \\
        --output_path ./data/memoir_features/qwen3-4b.pt \\
        --n_samples 500 \\
        --device 0
"""

import argparse
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm.auto import tqdm


def brackets_to_periods(name: str) -> str:
    return name.replace("[", ".").replace("]", "")


def get_module(model, dotted_path: str):
    parts = dotted_path.split(".")
    m = model
    for i, p in enumerate(parts):
        try:
            if p.isdigit():
                m = m[int(p)]
            else:
                m = getattr(m, p)
        except (AttributeError, IndexError) as e:
            traversed = ".".join(parts[:i])
            available = [n for n, _ in m.named_children()] if hasattr(m, "named_children") else dir(m)
            raise AttributeError(
                f"Failed at '{p}' (step {i}) in path '{dotted_path}'.\n"
                f"  Traversed so far : '{traversed}' → {type(m).__name__}\n"
                f"  Available children: {available[:20]}"
            ) from e
    return m


def collect_features(
    model,
    tokenizer,
    layer_path: str,
    n_samples: int,
    batch_size: int,
    max_length: int,
    device: str,
    agg: str = "mean",
) -> torch.Tensor:
    """
    Run Wikipedia snippets through the model and collect mean-pooled hidden
    states from the *input* to the target layer (down_proj).

    Returns a tensor of shape (n_collected, hidden_dim).
    """
    # Load background corpus (Wikipedia)
    print("Loading Wikipedia background corpus...")
    raw_ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    # Resolve the module to hook
    dotted = brackets_to_periods(layer_path)
    target_module = get_module(model, dotted)

    collected: list[torch.Tensor] = []
    hook_inputs: list[torch.Tensor] = []

    def _hook(module, args, output):
        # args[0] is the input tensor to the Linear layer: (B, S, D)
        x = args[0].detach().float()
        # Mean-pool over sequence dimension (no padding removal for background text)
        feat = x.mean(dim=1)  # (B, D)
        hook_inputs.append(feat.cpu())

    handle = target_module.register_forward_hook(_hook)

    model.eval()
    n_collected = 0
    pbar = tqdm(total=n_samples, desc="Collecting background features")

    texts = []
    for sample in raw_ds:
        text = sample.get("text", "")
        if len(text.split()) < 30:
            continue
        # Truncate to first ~200 words to keep prompts short and diverse
        texts.append(" ".join(text.split()[:200]))
        if len(texts) == batch_size:
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with torch.no_grad():
                model(**enc)
            feats = torch.cat(hook_inputs, dim=0)   # (batch, D)
            hook_inputs.clear()
            collected.append(feats)
            n_collected += feats.shape[0]
            pbar.update(feats.shape[0])
            texts = []
            if n_collected >= n_samples:
                break

    handle.remove()
    pbar.close()

    all_feats = torch.cat(collected, dim=0)[:n_samples]
    print(f"Collected {all_feats.shape[0]} background feature vectors (dim={all_feats.shape[1]})")
    return all_feats


def main():
    parser = argparse.ArgumentParser(description="Pre-compute MEMOIR background features")
    parser.add_argument("--model_name", required=True,
                        help="HuggingFace model ID, e.g. Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--layer", required=True,
                        help="Layer path for the down_proj input hook, "
                             "e.g. model.layers[30].mlp.down_proj")
    parser.add_argument("--output_path", required=True,
                        help="Where to save the .pt file, e.g. ./data/memoir_features/qwen3-4b.pt")
    parser.add_argument("--n_samples", type=int, default=500,
                        help="Number of background feature vectors to collect (default 500; "
                             "MEMOIRAdapter loads the first 100 for the mean)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Forward-pass batch size")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max token length per sample")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index")
    parser.add_argument("--dtype", default="float16",
                        choices=["float16", "bfloat16", "float32"],
                        help="Model precision")
    args = parser.parse_args()

    device = f"cuda:{args.device}"
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Create output directory if needed
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    print(f"Loading model {args.model_name} in {args.dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"No pad token found — set pad_token = eos_token ({tokenizer.eos_token!r})")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()

    feats = collect_features(
        model=model,
        tokenizer=tokenizer,
        layer_path=args.layer,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    torch.save(feats, args.output_path)
    print(f"Saved {feats.shape[0]} background features (shape {tuple(feats.shape)}) to {args.output_path}")


if __name__ == "__main__":
    main()
