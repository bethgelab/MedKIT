import copy
import os
import sys
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast

from ...util.globals import *
from ...util.nethook import Trace, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """
    # Resolve model_name BEFORE extracting language_model, so the stats filename
    # uses the top-level model's HF identifier (e.g. "google_gemma-3-4b-it"),
    # not the empty _name_or_path that sub-configs often have.
    if model_name is None:
        model_name = model.config._name_or_path.replace("/", "_")

    # Vision-language models (e.g. Gemma 3) wrap the text LM under .language_model.
    # Hooks and forward passes must go through the text LM only; the config and
    # module paths (model.layers.X.mlp...) are relative to that submodule.
    if hasattr(model, 'language_model'):
        model = model.language_model

    def get_ds():
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('XXX/XXX/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        if ds_name == "wikipedia":
            # datasets v3+ dropped old dataset scripts; use the Parquet-based mirror
            raw_ds = load_dataset("wikimedia/wikipedia", "20231101.en")
        elif ds_name == "wikitext":
            raw_ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        else:
            raw_ds = load_dataset(ds_name)

        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config, 'seq_length'):
            maxlen = model.config.seq_length
        else:
            maxlen = 4096  # safe default; capped by batch_tokens anyway
                
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    batch_size = 100  # Examine this many dataset texts at once
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config, 'seq_length'):
        npos = model.config.seq_length
    else:
        npos = 4096  # safe default; capped by batch_tokens anyway
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            npos = 4096

    if batch_tokens is None:
        # Cap at 4096 to avoid OOM on long-context models (e.g. Llama-3.1 with 131072 positions)
        batch_tokens = min(npos, 4096)
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = f"_t{batch_tokens}" + size_suffix
    # model_name was already resolved above (before language_model extraction)
    print(f'Batch tokens: {batch_tokens}, precision: {precision}, size suffix: {size_suffix}')

    # Resolve the actual hook path — VLMs expose text layers directly without
    # a leading "model." prefix (e.g. Gemma3TextModel has "layers.X" not "model.layers.X").
    # We keep layer_name unchanged for the filename so MEMIT/AlphaEdit can find the cache.
    _existing_modules = {n for n, _ in model.named_modules()}
    if layer_name in _existing_modules:
        hook_layer_name = layer_name
    elif layer_name.startswith('model.') and layer_name[len('model.'):] in _existing_modules:
        hook_layer_name = layer_name[len('model.'):]
        print(f"Hook path adjusted: '{layer_name}' → '{hook_layer_name}' (VLM text model exposed directly)")
    else:
        hook_layer_name = layer_name  # will raise LookupError downstream with a clear message

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension
    print(f"Stats file: {filename}")

    # Guard against corrupt cache files left by previously killed jobs.
    # A valid file must be loadable and have no all-zero rows or NaNs.
    # CombinedStat.state_dict() uses push_key_prefix so the second-moment
    # matrix is stored under "mom2.mom2", not "mom2".
    #
    # IMPORTANT: tally's load_cached_state calls numpy.load() WITHOUT
    # allow_pickle=True, which raises ValueError for npz files that contain
    # numpy object arrays (e.g. the "constructor" string key saved by
    # CombinedStat.state_dict).  load_cached_state catches that ValueError,
    # returns None, and tally then falls through to make_loader(dataset=None)
    # which crashes / hangs.  To avoid this we do the load ourselves with
    # allow_pickle=True and return early, bypassing tally entirely.
    if filename.exists() and not force_recompute:
        try:
            _cached = np.load(filename, allow_pickle=True)
            # Try prefixed key first (tally cache via CombinedStat), then plain key
            _mom2 = _cached["mom2.mom2"] if "mom2.mom2" in _cached else (
                     _cached["mom2"]       if "mom2"     in _cached else None)
            if _mom2 is None:
                raise ValueError(f"missing mom2 key (keys: {list(_cached.keys())[:6]})")
            _m = _mom2.astype(np.float32)
            # Check for NaN, infinity (fp16 overflow), or all-zero rows
            if np.isnan(_m).any() or np.isinf(_m).any() or np.all(_m.sum(axis=1) == 0):
                raise ValueError("NaN, inf, or all-zero rows")
            print(f"  Cache validation passed (shape={_m.shape}). Loading directly.")
            # Load directly — bypass tally to avoid the allow_pickle issue.
            stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
            stat.load_state_dict(dict(_cached))
            return stat
        except Exception as _e:
            print(f"  Corrupt cache detected ({_e}), deleting and recomputing: {filename}")
            filename.unlink()

    print(f"Computing Cov locally....")
    ds = get_ds()

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,  # 0 = in-process; avoids tokenizer fork deadlock in subprocesses
    )

    batch_count = -(-(sample_size or len(ds)) // batch_size)
    added_stats = 0
    state_dict_last = None
    with (torch.no_grad()):
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch, f"cuda:{hparams.device}")
                with Trace(
                    model, hook_layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)

                stat_before = stat.state_dict() if added_stats > 0 else None
                stat.add(feats)
                test = stat.mom2.moment().float().to("cpu")
                row_sum = torch.sum(test, dim=1)
                if not (torch.isnan(test).any() or torch.isinf(test).any() or torch.any(row_sum == 0)):
                    added_stats += 1 * batch_size
                    state_dict_last = stat.state_dict()  # only save when batch was valid
                else:
                    if stat_before is not None:
                        stat.load_state_dict(stat_before)
                        stat.to_(device=feats.device)
                    else:
                        stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})

    print(f"Added {added_stats} stats")

    test2 = stat.mom2.moment().float().to("cpu")
    row_sum2 = torch.sum(test2, dim=1)
    if torch.any(row_sum2 == 0) or torch.isnan(test2).any() or torch.isinf(test2).any():
        print('zero rows in the stats')
        if state_dict_last is None:
            raise ValueError("Zero rows in the stats")
        stat2 = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
        stat2.load_state_dict(state_dict_last)
        stat2.to_(device=feats.device)
        test3 = stat2.mom2.moment().float().to("cpu")
        row_sum3 = torch.sum(test3, dim=1)
        if torch.any(row_sum3 == 0) or torch.isnan(test3).any() or torch.isinf(test3).any():
            raise ValueError("Zero rows in the stats2")
        else:
            _verify_stat(stat2, filename)
            return stat2
    _verify_stat(stat, filename)
    return stat


def _verify_stat(stat, filename):
    """
    Verify the freshly computed stat and delete the cache file if corrupt,
    so the next run recomputes from scratch rather than loading bad data.
    """
    m = stat.mom2.moment().float().cpu()
    bad = torch.isnan(m).any() or torch.isinf(m).any() or torch.all(m.sum(dim=1) == 0)
    if bad:
        msg = f"Freshly computed stats are corrupt (nan/inf/zero-rows) for {filename}. "
        if filename is not None and Path(filename).exists():
            Path(filename).unlink()
            msg += "Deleted corrupt cache file."
        raise ValueError(msg)
    print(f"  Post-compute validation passed (shape={tuple(m.shape)}).")

def is_valid(feats):
    # Check for NaNs or infinite values
    if torch.isnan(feats).any() or torch.isinf(feats).any():
        return False
    # Check for zero rows
    if (feats.sum(dim=1) == 0).any():
        return False
    return True


def state_dicts_are_identical(state_dict1, state_dict2):
    # Check if keys are identical
    if state_dict1.keys() != state_dict2.keys():
        return False

    # Check if values are identical
    for key in state_dict1:
        if isinstance(state_dict1[key], str):
            if state_dict1[key] != state_dict2[key]:
                print(f"Key {key} is not equal")
                return False
        if isinstance(state_dict1[key], torch.Tensor):
            if not torch.equal(state_dict1[key], state_dict2[key]):
                print(f"Key {key} is not equal")
                return False
        if isinstance(state_dict1[key], int):
            if state_dict1[key] != state_dict2[key]:
                print(f"Key {key} is not equal")
                return False

    return True


if __name__ == "__main__":
    main()
