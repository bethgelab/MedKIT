#!/usr/bin/env python
"""Shared-eval driver for the DPO / GRPO lifelong-LoRA baselines.

The DPO/GRPO training scripts (train_model_{dpo,grpo}_hemonc.py) produce one LoRA
adapter per increment, chained across increments (lifelong CL), at:

    {adapter_root}/{model_folder}/{strategy}/{safe_increment_label}/

Their own eval (constrained label log-prob scoring) is *not* used.  Instead this
driver scores each increment through the SAME harness as every other method in
the sweep — run_medkit.py's `editing_method=EVAL` path — so the numbers are
directly comparable and land in the same `_results.json` schema as the main
methods.

For each increment i we launch one EVAL run of the base model with increment i's
adapter loaded, restricted to increment i's eval window.  Scoring increment i with
the adapter trained *through* increment i reproduces the lifelong-CL semantics
used by SEEKR / O-LoRA / LoRA-Merge.

Adapter injection detail (important):
  * We inject the adapter via `++hparams.adapter_path=<dir>` — read by
    LifelongEditor at load time (easyeditor/editors/editor.py:401) and declared on
    OracleRAGHyperParams so it survives from_hparams()'s field filter.
  * We pass `++adapter_path=null` to null out the TOP-LEVEL adapter_path (the
    generated config sets it to '').  If left as a non-None value the runtime
    injection loop in run_medkit.py copies it onto hparams — clobbering ours —
    and the results filename tag flips from `_base_` to `_finetuned_`, so the
    files no longer line up with the other methods' results.

Increments without training data get no adapter of their own; they are scored
with the latest adapter trained before them (or the plain base model if no
adapter has been trained yet), matching what the training scripts carry forward.

Prerequisites:
  * The per-increment adapters must already exist (run the training script first).
  * The Hydra experiment configs `hemonc/main/{strategy}/{combo}_{config_key}.yaml`
    must exist (shipped; regenerate with
    `python main_experiments/generate_configs.py --method {dpo,grpo}`, one at a time).

Usage (one model × strategy):
    # from experiments/
    python further_baselines/eval_lifelong_adapters.py \
        --combo dpo --config_key medgemma_4b --strategy weekly \
        --adapter_root /path/to/dpo/adapters \
        [--n_increments N] [--skip_existing] [--dry_run]
"""

import argparse
import os
import re
import subprocess
import sys

# Ensure experiments/ (parent dir) is importable for hemonc_batching / model_registry.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../experiments
sys.path.insert(0, _ROOT)

from hemonc_batching import build_increments  # noqa: E402
from further_baselines import model_registry  # noqa: E402
from medkit_data import DEFAULT_DATA_PATH, resolve_data_path  # noqa: E402

# Fixed by the sweep — every method's per-increment results use this filename shape.
RESULT_SUFFIX = "_mix_noevidence_base_gpt-4o-mini-hemonc_results.json"
DS_TAG = "data_-1_42"  # ds_size=-1, ds_seed=42 (matches generated main configs)


def safe_increment_label(label: str) -> str:
    """Match the sanitisation used by the training scripts to name adapter dirs."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label).strip()) or "increment"


def hydra_quote(value: str) -> str:
    """Quote a value for a Hydra command-line override (paths may contain
    characters such as `~` or spaces that the override grammar rejects)."""
    return "'" + str(value).replace("'", "\\'") + "'"


def expected_result_path(metrics_save_dir, strategy, combo, config_key, increment):
    return os.path.join(
        _ROOT, metrics_save_dir, "hemonc", "main", strategy,
        f"{combo}_{config_key}", "EVAL", DS_TAG,
        f"{increment}{RESULT_SUFFIX}",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combo", required=True, choices=["dpo", "grpo"],
                    help="Which baseline's adapters to evaluate (sets the combo dir prefix).")
    ap.add_argument("--config_key", required=True,
                    help="Model config key, e.g. medgemma_4b (see model_registry).")
    ap.add_argument("--strategy", required=True,
                    choices=["daily", "weekly", "monthly"],
                    help="Batching strategy (must match how the adapters were trained).")
    ap.add_argument("--adapter_root", required=True,
                    help="Root under which {model_folder}/{strategy}/{safe_label}/ adapters live.")
    ap.add_argument("--data_path", default=DEFAULT_DATA_PATH,
                    help="hf://<org>/MedKIT or a local CSV (relative to experiments/).")
    ap.add_argument("--crop_year", type=int, default=2025,
                    help="crop_year passed to build_increments (matches the sweep).")
    ap.add_argument("--n_increments", type=int, default=None,
                    help="Cap to the first N increments (for smoke tests).")
    ap.add_argument("--metrics_save_dir", default="metrics/main",
                    help="Where run_medkit.py writes results (relative to experiments/).")
    ap.add_argument("--run_script", default="run_medkit.py")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip increments whose result JSON already exists.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print the commands without launching them.")
    ap.add_argument("--extra_overrides", nargs="*", default=[],
                    help="Extra Hydra overrides appended verbatim (e.g. ds_size=20 for smoke tests).")
    args = ap.parse_args()

    entry = model_registry.get(args.config_key)  # raises with a helpful message if unknown
    data_path_abs = resolve_data_path(
        args.data_path if args.data_path.startswith("hf://") or os.path.isabs(args.data_path)
        else os.path.join(_ROOT, args.data_path))

    increments = build_increments(
        data_path=data_path_abs, strategy=args.strategy, crop_year=args.crop_year,
    )
    if args.n_increments is not None:
        increments = increments[: args.n_increments]

    strategy_adapter_dir = os.path.join(args.adapter_root, entry.model_folder, args.strategy)
    config_name = f"hemonc/main/{args.strategy}/{args.combo}_{args.config_key}"

    print(f"[eval_lifelong_adapters] combo={args.combo} model={args.config_key} "
          f"({entry.model_folder}) strategy={args.strategy} "
          f"increments={len(increments)} hparam=EVAL/{entry.eval_hparam_stem}")

    if not os.path.isdir(strategy_adapter_dir):
        sys.exit(f"[eval_lifelong_adapters] no adapters under {strategy_adapter_dir} "
                 "— run the training script first.")

    def trained_adapter(increment):
        d = os.path.join(strategy_adapter_dir, safe_increment_label(increment))
        return d if os.path.isfile(os.path.join(d, "adapter_config.json")) else None

    # Only score up to the last trained increment: anything after it has simply
    # not been trained yet (e.g. an interrupted run), so don't carry forward there.
    trained = [i for i, inc in enumerate(increments) if trained_adapter(inc)]
    if not trained:
        sys.exit(f"[eval_lifelong_adapters] no trained adapters under {strategy_adapter_dir}.")
    if trained[-1] + 1 < len(increments):
        print(f"  [warning] adapters exist only up to {increments[trained[-1]]}; "
              f"{len(increments) - trained[-1] - 1} later increments are not scored.")
        increments = increments[: trained[-1] + 1]

    n_run, n_skip, n_carried = 0, 0, 0
    adapter_dir = ""  # "" = plain base model (no adapter trained yet)
    for increment in increments:
        candidate = trained_adapter(increment)
        if candidate:
            adapter_dir = candidate
        else:
            # No training data for this increment: carry the previous adapter forward.
            print(f"  [no adapter for {increment}] using "
                  f"{adapter_dir or 'the base model'}")
            n_carried += 1

        if args.skip_existing and os.path.exists(
                expected_result_path(args.metrics_save_dir, args.strategy,
                                     args.combo, args.config_key, increment)):
            print(f"  [skip-existing] increment={increment}")
            n_skip += 1
            continue

        cmd = [
            args.python, args.run_script,
            f"--config-name={config_name}",
            "++adapter_path=null",                       # neutralise top-level (keeps `_base_` tag)
            f"++hparams.adapter_path={hydra_quote(adapter_dir)}",  # adapter to evaluate ('' = base model)
            f"++qa.batching.increment_filter={increment}",
            f"++qa.data_path={hydra_quote(data_path_abs)}",  # score on the same data the adapters saw
            "++checkpoint.save=false",
            "++checkpoint.load=false",
        ] + list(args.extra_overrides)

        print(f"  [increment {increment}] adapter={adapter_dir or '(base model)'}")
        print("    " + " ".join(cmd))
        if args.dry_run:
            n_run += 1
            continue

        # run_medkit.py resolves paths relative to experiments/, so run there.
        res = subprocess.run(cmd, cwd=_ROOT)
        if res.returncode != 0:
            print(f"    !! non-zero exit ({res.returncode}) for increment {increment}")
        n_run += 1

    print(f"[eval_lifelong_adapters] done: launched={n_run} "
          f"skipped_existing={n_skip} carried_forward={n_carried}")


if __name__ == "__main__":
    main()
