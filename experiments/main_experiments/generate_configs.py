#!/usr/bin/env python3
"""
Generate MedKIT main-experiment run configs.

Reads hparam_tuning/figures/best_params.csv and writes one Hydra YAML per
(strategy × method × model) cell into
    hydra/experiments/hemonc/main/{strategy}/{method}_{model}.yaml
Also writes main_experiments/manifest.csv enumerating every config.

The base (architecture-level) hparam YAMLs under hydra/experiments/hparams/
are left untouched — this script only writes EXPERIMENT configs that compose
those base hparams and override the sweep-winner values inline.

Untuned models (medgemma_4b, bio_medical_llama3_8b) inherit the best params
from a donor in the same architecture family (gemma-3-4b / llama31-8b).

Usage (run from experiments/):
    python main_experiments/generate_configs.py                 # write everything
    python main_experiments/generate_configs.py --dry-run       # preview only
    python main_experiments/generate_configs.py --strategy weekly --method memit
    python main_experiments/generate_configs.py --include-adaptllm   # include adaptllm_medicine_chat
"""
from __future__ import annotations

import argparse
import ast
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BEST_PARAMS_CSV = ROOT / "hparam_tuning" / "figures" / "best_params.csv"
CONFIG_ROOT = ROOT / "hydra" / "experiments" / "hemonc" / "main"
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.csv"

STRATEGIES = ["daily", "weekly", "monthly"]

# Past-sample (sentinel) evaluation — strategy-aware to balance thoroughness vs cost.
# `eval_interval` matches the runner's auto-save cadence so past_eval lands on the
# same checkpoints that get persisted; `sentinel_k` is tuned so the pool fills in
# the first third of the run on every strategy; `sentinel_max` is the cap.
PAST_EVAL_BY_STRATEGY = {
    "monthly": {"enabled": True, "eval_interval": 1,  "sentinel_k": 3, "sentinel_max": 100},
    "weekly":  {"enabled": True, "eval_interval": 2,  "sentinel_k": 2, "sentinel_max": 100},
    "daily":   {"enabled": True, "eval_interval": 10, "sentinel_k": 1, "sentinel_max": 100},
}

# (config_key, hparam_stem, family_donor_csv_model, tier)
MODELS_DEFAULT = [
    ("gemma3_4b",              "gemma-3-4b-it",          "gemma-3-4b", "4b"),
    ("qwen3_4b",               "qwen3-4b",               "qwen3-4b",   "4b"),
    ("medgemma_4b",            "medgemma-4b-it",         "gemma-3-4b", "4b"),
    ("llama31_8b",             "llama3.1-8b",            "llama31-8b", "8b"),
    ("bio_medical_llama3_8b",  "bio-medical-llama-3-8b", "llama31-8b", "8b"),
]
MODELS_OPTIONAL = [
    ("adaptllm_medicine_chat", "adaptllm-medicine-chat", "llama31-8b", "8b"),
]

# API-only models — accessed via OpenRouter, no local GPU. Used as
# strong-LM ablations to isolate "does the model fail to use the
# retrieved evidence?" from "does the retriever fail to find it?".
# Only oracle_{abs,gt} configs are generated for these — editing methods
# and gradient-based RAG retrievers can't run against an API endpoint.
#
# (config_key, hparam_stem, donor_csv_model, tier)
MODELS_API = [
    ("claude_opus", "claude-opus", None, "api"),
]
API_METHOD_ALLOWLIST = ("oracle_abs", "oracle_gt")

# Methods. Keys follow submit_initial_edit_jobs.sh / existing config filename
# conventions. `csv_method` is the row label in best_params.csv; None means
# no sweep exists for this method (no overrides written).
#
# `enabled_by_default` controls whether the method is part of the default
# main-matrix manifest.  When False, configs are NOT generated and the method
# does NOT appear in manifest.csv unless the user passes --include-disabled
# (or selects it explicitly via --method <name>).  Use this for variants we've
# decided not to run as part of the standard sweep — see comments per method
# for rationale.
METHODS = {
    "alphaedit": dict(
        enabled_by_default=True,
        skip_pre_edit=True,
        editing_method="AlphaEdit", hparam_dir="AlphaEdit", csv_method="AlphaEdit",
        hparam_prefix="", use_vllm=False, eval_use_vllm=True,
        swept_fields=["nullspace_threshold", "mom2_update_weight"],
        top_level={}, qa_include_evidence=False,
    ),
    "memit": dict(
        enabled_by_default=True,
        skip_pre_edit=True,
        editing_method="MEMIT", hparam_dir="MEMIT", csv_method="MEMIT",
        hparam_prefix="", use_vllm=False, eval_use_vllm=True,
        swept_fields=["kl_factor", "layers", "mom2_update_weight", "v_lr"],
        top_level={}, qa_include_evidence=False,
    ),
    "lora_merge": dict(
        enabled_by_default=True,
        skip_pre_edit=True,
        editing_method="LoRA-Merge", hparam_dir="LoRA-Merge", csv_method="LoRA-Merge",
        hparam_prefix="", use_vllm=False, eval_use_vllm=False,
        swept_fields=["lr", "num_steps"],
        top_level={}, qa_include_evidence=False,
    ),
    "o_lora": dict(
        enabled_by_default=True,
        skip_pre_edit=True,
        editing_method="O-LoRA", hparam_dir="O-LoRA", csv_method="O-LoRA",
        hparam_prefix="", use_vllm=False, eval_use_vllm=False,
        swept_fields=["lr", "num_steps", "orth_lambda"],
        top_level={}, qa_include_evidence=False,
    ),
    "ike": dict(
        enabled_by_default=True,
        skip_pre_edit=True,
        editing_method="IKE", hparam_dir="IKE", csv_method="IKE",
        hparam_prefix="hemonc-", use_vllm=False, eval_use_vllm=True,
        # k pinned (see RAG retrievers below for rationale); not swept.
        swept_fields=[],
        fixed_overrides={"k": 3},
        top_level={}, qa_include_evidence=False,
        # IKE prepends top-k retrieved examples to every prompt, so the
        # concatenated context can exceed the default 2048 vLLM max_model_len.
        vllm_eval_max_model_len_override=8192,
    ),
    "seekr": dict(
        enabled_by_default=True,
        skip_pre_edit=True,
        editing_method="SEEKR", hparam_dir="SEEKR", csv_method="SEEKR",
        hparam_prefix="", use_vllm=False, eval_use_vllm=False,
        swept_fields=["lr", "num_steps", "replay_weight"],
        top_level={}, qa_include_evidence=False,
    ),
    "grace": dict(
        enabled_by_default=True,
        # GRACE's eval-time forward is batch-safe (no scalar .item() calls,
        # torch.cdist + torch.where broadcast over batch). Use default bs=8.
        skip_pre_edit=True,
        editing_method="GRACE", hparam_dir="GRACE", csv_method="GRACE",
        hparam_prefix="", use_vllm=False, eval_use_vllm=False,
        swept_fields=["edit_lr", "eps"],
        top_level={}, qa_include_evidence=False,
    ),
    "memoir": dict(
        enabled_by_default=True,
        eval_gen_batch_size_override=1,
        skip_pre_edit=True,
        editing_method="MEMOIR", hparam_dir="MEMOIR", csv_method="MEMOIR",
        hparam_prefix="", use_vllm=False, eval_use_vllm=False,
        swept_fields=["edit_lr", "irr_threshold"],
        top_level={}, qa_include_evidence=False,
    ),
    "wise": dict(
        enabled_by_default=True,
        # WISE patched to be batch-safe at eval time (per-item routing via
        # torch.where) BUT measured smoke shows bs=8 is ~10% slower than bs=1
        # — WISE's per-forward cost (3 forward passes + memory retrieval
        # loop) dominates, so batching doesn't help. Keep bs=1.
        eval_gen_batch_size_override=1,
        skip_pre_edit=True,
        editing_method="WISE", hparam_dir="WISE", csv_method="WISE",
        hparam_prefix="", use_vllm=False, eval_use_vllm=False,
        swept_fields=["edit_lr", "n_iter"],
        top_level={}, qa_include_evidence=False,
    ),
    # ------------------------------------------------------------------
    # Retrieval baselines: top_k pinned at 3 across all (method × model ×
    # strategy) cells. Rationale (see analysis notes):
    #  - per-method sweep showed Δrewrite_acc is invariant to top_k for both
    #    BM25 and Dense; sweep winners were heterogeneous (1 / 5 / 10) which
    #    contaminates the cross-model retrieval-accuracy comparison;
    #  - top_k=3 is the conventional RAG default, gives Dense recall room
    #    above k=1 (13% → 21%), and bounds prompt-processing cost on 4B
    #    models vs k=5/10.  swept_fields kept empty so we don't read the
    #    sweep CSV for these methods.
    #
    # _gt vs _abs:  _gt retrieves and prepends the ground-truth target label
    # of the top-k corpus entries (label-only context); _abs retrieves and
    # prepends the full abstract text.  Default sweep runs only the _abs
    # variants — abstracts are the ecologically valid retrieval target
    # (a real RAG system retrieves documents, not pre-extracted answer
    # labels), and the _gt variants leak target-shaped context that confounds
    # the headline accuracy story.  Re-enable _gt with --include-disabled
    # if you want the ablation.
    "bm25_gt": dict(
        enabled_by_default=False,  # see "_gt vs _abs" comment above
        editing_method="BM25RAG", hparam_dir="BM25RAG", csv_method="BM25RAG",
        hparam_prefix="", use_vllm=True, eval_use_vllm=False,
        swept_fields=[],
        fixed_overrides={"top_k": 3},
        top_level={"retrieve_target": True}, qa_include_evidence=False,
    ),
    "bm25_abs": dict(
        enabled_by_default=True,
        editing_method="BM25RAG", hparam_dir="BM25RAG", csv_method="BM25RAG",
        hparam_prefix="", use_vllm=True, eval_use_vllm=False,
        swept_fields=[],
        fixed_overrides={"top_k": 3},
        top_level={"retrieve_target": False}, qa_include_evidence=False,
    ),
    "dense_gt": dict(
        enabled_by_default=False,  # see "_gt vs _abs" comment above
        editing_method="DenseRAG", hparam_dir="DenseRAG", csv_method="DenseRAG",
        hparam_prefix="", use_vllm=True, eval_use_vllm=False,
        swept_fields=[],
        fixed_overrides={"top_k": 3},
        top_level={"retrieve_target": True}, qa_include_evidence=False,
    ),
    "dense_abs": dict(
        enabled_by_default=True,
        editing_method="DenseRAG", hparam_dir="DenseRAG", csv_method="DenseRAG",
        hparam_prefix="", use_vllm=True, eval_use_vllm=False,
        swept_fields=[],
        fixed_overrides={"top_k": 3},
        top_level={"retrieve_target": False}, qa_include_evidence=False,
    ),
    # ------------------------------------------------------------------
    # Empty-corpus ablation variants. Identical to *_abs above except
    # `qa.seed_pre_batch_corpus: false`, which causes the RAG wrappers to
    # start with an empty corpus (only adapt()-time evidence ever enters
    # retrieval). Tests "RAG as a streaming learner" vs the default
    # "RAG as a librarian" condition.
    "bm25_abs_empty": dict(
        enabled_by_default=False,  # ablation deferred — see PAPER_EXPERIMENTS.md
        editing_method="BM25RAG", hparam_dir="BM25RAG", csv_method="BM25RAG",
        hparam_prefix="", use_vllm=True, eval_use_vllm=False,
        swept_fields=[],
        fixed_overrides={"top_k": 3},
        top_level={"retrieve_target": False}, qa_include_evidence=False,
        qa_extra_lines=["seed_pre_batch_corpus: false"],
        # Mini-scale ablation: cap at the first 10 increments of the eval
        # window (~70-100 samples) to keep total ablation cost <2 main runs.
        batching_n_increments_override=10,
    ),
    "dense_abs_empty": dict(
        enabled_by_default=False,  # ablation deferred — see PAPER_EXPERIMENTS.md
        editing_method="DenseRAG", hparam_dir="DenseRAG", csv_method="DenseRAG",
        hparam_prefix="", use_vllm=True, eval_use_vllm=False,
        swept_fields=[],
        fixed_overrides={"top_k": 3},
        top_level={"retrieve_target": False}, qa_include_evidence=False,
        qa_extra_lines=["seed_pre_batch_corpus: false"],
        batching_n_increments_override=10,
    ),
    # ------------------------------------------------------------------
    # Oracle baselines (gt-evidence / abs-evidence injected directly into
    # every prompt by the data loader, no retrieval).  Disabled by default
    # because Oracle output is invariant to the batching strategy: the same
    # gold evidence is injected per-prompt regardless of which week/month it
    # belongs to, so running oracle for daily AND weekly AND monthly
    # produces three identical result sets.  Run once (see existing weekly
    # oracle results) and reuse the numbers across all strategies in the
    # analysis.  Re-enable with --include-disabled if you want fresh runs.
    "oracle_gt": dict(
        enabled_by_default=False,
        editing_method="EVAL", hparam_dir="OracleRAG", csv_method=None,
        hparam_prefix="", use_vllm=True, eval_use_vllm=False,
        swept_fields=[],
        top_level={}, qa_include_evidence="ground_truth",
    ),
    "oracle_abs": dict(
        enabled_by_default=False,
        editing_method="EVAL", hparam_dir="OracleRAG", csv_method=None,
        hparam_prefix="", use_vllm=True, eval_use_vllm=False,
        swept_fields=[],
        top_level={}, qa_include_evidence="abstract",
    ),
    # ------------------------------------------------------------------
    # Further-baselines shared-eval configs (DPO / GRPO).
    #
    # These are EVAL-only configs: the LoRA adapter is trained OFFLINE by
    # further_baselines/train_model_{dpo,grpo}_hemonc.py, and then scored through
    # the standard EVAL path by further_baselines/eval_lifelong_adapters.py, which
    # invokes run_medkit.py --config-name=hemonc/main/{strategy}/{dpo,grpo}_{model}
    # per increment with `++hparams.adapter_path=<increment adapter>` and
    # `++qa.batching.increment_filter=<label>`.  Results land under
    # metrics/main/hemonc/main/{strategy}/{dpo,grpo}_{model}/EVAL/...
    #
    # hparam_dir="EVAL" loads the plain base model (no RAG / no baked-in adapter);
    # the adapter is injected at run time.  enabled_by_default=False so they never
    # perturb the main matrix — generate them with `--method dpo` and `--method grpo`
    # (or --include-disabled).  Agentic-RAG is intentionally NOT here: its driver
    # (run_agentic_shared_eval.py) is standalone and consumes no Hydra config.
    "dpo": dict(
        enabled_by_default=False,
        skip_pre_edit=True,
        editing_method="EVAL", hparam_dir="EVAL", csv_method=None,
        hparam_prefix="", use_vllm=False, eval_use_vllm=False,
        swept_fields=[],
        top_level={}, qa_include_evidence=False,
    ),
    "grpo": dict(
        enabled_by_default=False,
        skip_pre_edit=True,
        editing_method="EVAL", hparam_dir="EVAL", csv_method=None,
        hparam_prefix="", use_vllm=False, eval_use_vllm=False,
        swept_fields=[],
        top_level={}, qa_include_evidence=False,
    ),
}


def parse_cell(field: str, raw: str):
    s = (raw or "").strip()
    if s == "":
        return None
    if field == "layers":
        parsed = ast.literal_eval(s)
        return [int(x) for x in parsed]
    if field == "nullspace_threshold":
        return s  # keep string like "rel_max:1e-4"
    if field in {"top_k", "k", "num_steps", "n_iter", "mom2_update_weight"}:
        return int(float(s))
    return float(s)


def load_best_params(csv_path: Path):
    """Return {(csv_method, csv_model, strategy): row_dict}."""
    rows = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[(r["method"], r["model"], r["strategy"])] = r
    return rows


def fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


def render_config(
    *,
    method: str,
    model: str,
    strategy: str,
    hparam_stem: str,
    swept_overrides: dict,
    warnings: list[str],
    flagged: bool,
) -> str:
    spec = METHODS[method]
    hparam_path = f"{spec['hparam_dir']}/{spec['hparam_prefix']}{hparam_stem}"
    experiment = f"hemonc/main/{strategy}/{method}_{model}"
    note = f"MAIN — {spec['editing_method']} / {model} / {strategy}"
    # Distinctive tag for wandb filtering — pins the canonical main-matrix
    # runs on the data_v4 / crop_year=2025 / post-optimizations configs.
    # Filter in wandb UI with: tags:"main_sweep_v4"
    tags = ["main_sweep_v4", "main", strategy, method, model]

    warning_lines = [f"# WARNING: {w}" for w in warnings]
    if flagged:
        warning_lines.append("# NOTE: locality gate flagged in sweep (locality<0.4) — kept per plan decision.")
    warning_block = ("\n".join(warning_lines) + "\n") if warning_lines else ""

    top_level_lines = []
    for k, v in spec["top_level"].items():
        top_level_lines.append(f"{k}: {fmt(v)}")
    top_level_block = ("\n" + "\n".join(top_level_lines) + "\n") if top_level_lines else ""

    swept_lines = []
    for k, v in swept_overrides.items():
        swept_lines.append(f"{k}: {fmt(v)}")
    if swept_lines:
        swept_block = (
            f"\n# sweep-winner overrides for strategy={strategy}\n"
            + "\n".join(swept_lines) + "\n"
        )
    else:
        swept_block = ""

    qa_include_evidence = fmt(spec["qa_include_evidence"])

    pe = PAST_EVAL_BY_STRATEGY[strategy]
    past_eval_block = (
        f"  past_eval:\n"
        f"    enabled: {fmt(pe['enabled'])}\n"
        f"    eval_interval: {pe['eval_interval']}\n"
        f"    sentinel_k: {pe['sentinel_k']}\n"
        f"    sentinel_max: {pe['sentinel_max']}\n"
    )

    # Optional extra lines injected into the `qa:` block (e.g. ablation flags
    # like `seed_pre_batch_corpus: false` for the empty-corpus variant).
    qa_extra_block = ""
    for line in spec.get("qa_extra_lines", []) or []:
        qa_extra_block += f"  {line}\n"

    return f"""# @package _global_
# AUTO-GENERATED by main_experiments/generate_configs.py — do not edit by hand.
{warning_block}defaults:
  - /wandb: wandb
  - /hparams: {hparam_path}
  - _self_

wandb:
  notes: "{note}"
  tags: {fmt(tags)}

experiment: "{experiment}"
debug: false

editing_method: '{spec['editing_method']}'
data_dir: 'data'
ds_size: -1
ds_seed: 42
metrics_save_dir: 'metrics/main'
batch_size: 1
max_minibatch_size: 32

adapter_path: ''
use_vllm: {fmt(spec['use_vllm'])}
# RAG methods load vLLM via the use_vllm=True path, which reads
# vllm_max_model_len (default 8192). BM25/Dense ABS variants prepend full
# abstracts — some HemOnc abstracts + prompt exceed 8192 tokens. Bump to
# 16384 where the model architecture supports it. bio_medical_llama3_8b
# is built on Llama-3 base (max_position_embeddings=8192), so it must
# stay capped at 8192 — vLLM rejects values above the model's max.
vllm_max_model_len: {8192 if model == 'bio_medical_llama3_8b' else 16384}
# Force eval_use_vllm=False for:
#  - Gemma-3 family (multimodal wrapper breaks vLLM state_dict sync)
#  - 8B models on 40GB cards (CuMemAllocator fragments after 2nd wake_up
#    cycle; co-resident doesn't fit LoRA training either). Falls back to
#    HF-batched eval which is ~2x slower but rock-solid on 40GB 8B.
eval_use_vllm: {fmt(False if model in ('gemma3_4b','medgemma_4b','llama31_8b','bio_medical_llama3_8b') else spec['eval_use_vllm'])}
# Sleep-mode: only one model resident on GPU at a time. HF swaps to CPU
# while vLLM generates; vLLM sleeps (KV cache released, weights CPU) during
# the edit step. Required on 40GB cards; safe everywhere.
eval_use_vllm_sleep_mode: {fmt(spec.get('eval_use_vllm_sleep_mode_override', True))}
# vLLM memory budget: ~0.80 when sleep-mode (solo on GPU during eval),
# ~0.40 when co-resident with HF (for methods where sleep mode fragments
# CuMemAllocator — e.g. LoRA-family with PEFT training).
vllm_eval_gpu_memory_utilization: {spec.get('vllm_eval_gpu_memory_utilization_override', 0.80)}
vllm_eval_max_model_len: {spec.get('vllm_eval_max_model_len_override', 2048)}
# HF eval-path knobs (used by GRACE/MEMOIR/WISE and by any eval_use_vllm
# run while vLLM is asleep). Batch-generation replaces the per-prompt
# serial loop in test_prediction_acc. Lower if you hit OOM on 8B × tight
# VRAM; raise for throughput on 80GB cards. Some wrapped-forward methods
# (WISE/MEMOIR/GRACE) override this to 1 because their custom forwards
# do scalar ops like `min_dist.item()` that fail on batched inputs.
eval_gen_batch_size: {spec.get('eval_gen_batch_size_override', 8)}
# Attention backend: 'auto' probes flash-attn at load time and falls back
# to sdpa if not installed. Set to 'sdpa' or 'eager' to force, 'default'
# to let HF decide.
attn_implementation: auto
{top_level_block}{swept_block}
checkpoint:
  save: true
  load: true
  save_dir: 'checkpoints/main'
  # Keep only the latest checkpoint per run.
  # - Completed runs: the 100% snapshot = final edited model state, used
  #   for post-hoc general-capability evaluation.
  # - Partial runs: the latest-saved increment, used for --skip-done resume.
  # Size: ~1 × 10GB/run × 75 runs ≈ 750GB matrix-wide.
  keep_milestones: [1.0]

qa:
  model: 'gpt-4o-mini-hemonc'
  judge_model: 'gpt-4o-hemonc'
  judge_workers: 8
  judge_enabled: true
  fs_examples: false
{past_eval_block}  base_eval: false
  data_format: csv
  data_path: 'hf://bethgelab/MedKIT'  # or a local CSV built with data_construction/
  include_evidence: {qa_include_evidence}
  modes:
    - null
  modes_combine: mix
  increments: []
  filter_edits: false
  filter_conflicting_edits: true
  # Skip per-increment pre-edit evaluation entirely; we use an OOB (base-
  # model) baseline computed once per model (see main_experiments/
  # precompute_oob.sh) and merge at analysis time.  Set to false for runs
  # that need an inline base-model response (e.g. RAG pre-eval).
  skip_pre_edit: {fmt(spec.get('skip_pre_edit', False))}
{qa_extra_block}  batching:
    strategy: {strategy}
    crop_year: 2025
    n_increments: {fmt(spec.get('batching_n_increments_override', None))}
"""


def render_api_config(
    *,
    method: str,
    model: str,
    strategy: str,
    hparam_stem: str,
) -> str:
    """Render an oracle_{abs,gt} config for an API-only model (OpenRouter).

    API models bypass the local-vLLM stack entirely:
      - use_vllm=false, eval_use_vllm=false
      - use_openrouter=true (the entry script reads this and routes
        generation through the OpenRouter client in editor.py)
      - no checkpointing (the LM is stateless)
      - no past_eval (would just re-bill us for old prompts)
    Otherwise mirrors the structure of a regular oracle_{abs,gt} main
    config so analysis tooling can treat it identically.
    """
    spec = METHODS[method]
    hparam_path = f"{spec['hparam_dir']}/{hparam_stem}"
    experiment = f"hemonc/main/{strategy}/{method}_{model}"
    note = f"MAIN — {spec['editing_method']} / {model} (API) / {strategy}"
    tags = ["main_sweep_v4", "main", strategy, method, model, "api"]
    qa_include_evidence = fmt(spec["qa_include_evidence"])

    return f"""# @package _global_
# AUTO-GENERATED by main_experiments/generate_configs.py — do not edit by hand.
# API-only ablation: strong-LM oracle baseline (isolates retrieval-quality
# failure from model-utilization failure). Runs against OpenRouter — no
# local GPU is required, so this is light enough to use as a one-off
# ablation rather than a full-matrix sweep.
defaults:
  - /wandb: wandb
  - /hparams: {hparam_path}
  - _self_

wandb:
  notes: "{note}"
  tags: {fmt(tags)}

experiment: "{experiment}"
debug: false

editing_method: '{spec['editing_method']}'
data_dir: 'data'
ds_size: -1
ds_seed: 42
metrics_save_dir: 'metrics/main'
batch_size: 16
max_minibatch_size: 32

# OpenRouter API path: bypasses local vLLM/HF loading; editor.py reads
# `use_openrouter` and routes generate calls through the OpenRouter client.
adapter_path: ''
use_vllm: false
use_openrouter: true
eval_use_vllm: false

# Stateless API model — no checkpoints to save/load.
checkpoint:
  save: false
  load: false

qa:
  model: 'gpt-4o-mini-hemonc'
  judge_model: 'gpt-4o-hemonc'
  judge_workers: 8
  judge_enabled: true
  fs_examples: false
  past_eval: false
  base_eval: false
  data_format: csv
  data_path: 'hf://bethgelab/MedKIT'  # or a local CSV built with data_construction/
  include_evidence: {qa_include_evidence}
  modes:
    - null
  modes_combine: mix
  increments: []
  filter_edits: false
  filter_conflicting_edits: true
  skip_pre_edit: false
  batching:
    strategy: {strategy}
    crop_year: 2025
    # Mini-scale: only the first 10 increments — keeps OpenRouter API
    # spend bounded.  Override at submit time with
    # `qa.batching.n_increments=null` for a full-scale run.
    n_increments: 10
"""


def resolve_overrides(method: str, donor_csv_model: str, strategy: str, rows: dict):
    """Pull sweep-winner row for (csv_method, donor, strategy) and translate to overrides.

    Returns (overrides_dict, flagged_bool, warnings_list).
    """
    spec = METHODS[method]
    warnings: list[str] = []
    overrides: dict = {}
    flagged = False

    # Sweep-derived overrides (skipped when swept_fields is empty).
    if spec["swept_fields"] and spec["csv_method"] is not None:
        row = rows.get((spec["csv_method"], donor_csv_model, strategy))
        if row is None:
            warnings.append(
                f"no best_params.csv row for ({spec['csv_method']}, {donor_csv_model}, {strategy}); "
                "falling back to base hparams without sweep overrides."
            )
        else:
            for f in spec["swept_fields"]:
                val = parse_cell(f, row.get(f, ""))
                if val is not None:
                    overrides[f] = val
            flagged = str(row.get("flagged", "")).strip().lower() == "true"

    # Fixed overrides win over sweep-derived values (e.g. retrieval baselines
    # pinned at top_k=3 regardless of what the sweep selected).
    fixed = spec.get("fixed_overrides") or {}
    overrides.update(fixed)

    return overrides, flagged, warnings


def build_manifest_rows(
    methods: list[str], models: list[tuple], strategies: list[str]
):
    # Sort: tier (4b before 8b before api), strategy duration (monthly→weekly→daily),
    # method (alphabetical within method group).
    tier_order = {"4b": 0, "8b": 1, "api": 2}
    strat_order = {"monthly": 0, "weekly": 1, "daily": 2}
    method_order = {m: i for i, m in enumerate(METHODS.keys())}

    api_keys = {m[0] for m in MODELS_API}

    out = []
    for model_key, stem, donor_csv, tier in models:
        is_api = model_key in api_keys
        for strategy in strategies:
            # API models always emit the full API_METHOD_ALLOWLIST,
            # regardless of the outer enabled_by_default filter — the
            # allowlist already restricts them to oracle_{abs,gt} (the
            # only methods that work against an API endpoint).  Non-API
            # models follow the caller-supplied `methods` list.
            method_iter = (
                [m for m in API_METHOD_ALLOWLIST if m in METHODS]
                if is_api else methods
            )
            for method in method_iter:
                out.append(
                    dict(
                        tier=tier, model=model_key, stem=stem, donor_csv=donor_csv,
                        strategy=strategy, method=method, is_api=is_api,
                    )
                )
    out.sort(key=lambda r: (
        tier_order.get(r["tier"], 99),
        strat_order[r["strategy"]],
        method_order[r["method"]],
        r["model"],
    ))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=STRATEGIES, default=None,
                    help="Restrict to one batching strategy.")
    ap.add_argument("--method", default=None,
                    help="Restrict to one method key (see METHODS dict). "
                         "Bypasses the enabled_by_default filter, so disabled "
                         "methods can still be generated explicitly.")
    ap.add_argument("--model", default=None,
                    help="Restrict to one model key.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be written but don't touch disk.")
    ap.add_argument("--include-adaptllm", action="store_true",
                    help="Include adaptllm_medicine_chat (excluded by default).")
    ap.add_argument("--include-disabled", action="store_true",
                    help="Include methods with enabled_by_default=False "
                         "(e.g. _gt RAG variants, oracle_*). "
                         "Defaults to off so the standard sweep matrix only "
                         "covers the methods we actually want to run.")
    args = ap.parse_args()

    strategies = [args.strategy] if args.strategy else STRATEGIES
    if args.method:
        # Explicit --method overrides the enabled-by-default filter so users
        # can still regenerate a single disabled method on demand.
        methods = [args.method]
    elif args.include_disabled:
        methods = list(METHODS.keys())
    else:
        methods = [m for m, spec in METHODS.items()
                   if spec.get("enabled_by_default", True)]
    models = list(MODELS_DEFAULT)
    if args.include_adaptllm:
        models += MODELS_OPTIONAL
    # API-only models are always included (cheap to generate; they only emit
    # oracle_{abs,gt} configs — see API_METHOD_ALLOWLIST in the build loop).
    models += MODELS_API
    if args.model:
        models = [m for m in models if m[0] == args.model]
        if not models:
            raise SystemExit(f"Model '{args.model}' not in the allowed list.")

    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")

    disabled_in_scope = [m for m in methods
                         if not METHODS[m].get("enabled_by_default", True)]
    if disabled_in_scope:
        print(f"  Including disabled methods: {disabled_in_scope}")
    skipped = sorted(m for m, spec in METHODS.items()
                     if not spec.get("enabled_by_default", True)
                     and m not in methods)
    if skipped and not args.method:
        print(f"  Skipping (enabled_by_default=False): {skipped}\n"
              f"    pass --include-disabled to include them.")

    rows = load_best_params(BEST_PARAMS_CSV)
    print(f"Loaded {len(rows)} rows from {BEST_PARAMS_CSV.relative_to(ROOT)}")

    manifest_rows = build_manifest_rows(methods, models, strategies)
    print(f"Planning {len(manifest_rows)} configs "
          f"({len(methods)} methods × {len(models)} models × {len(strategies)} strategies).")

    n_written = 0
    n_flagged = 0
    n_missing = 0
    manifest_out = []

    for entry in manifest_rows:
        method = entry["method"]
        model = entry["model"]
        stem = entry["stem"]
        donor_csv = entry["donor_csv"]
        strategy = entry["strategy"]
        is_api = entry.get("is_api", False)

        if is_api:
            # API rows skip the sweep CSV (no per-cell tuning applies) and use
            # the dedicated OpenRouter renderer.
            overrides, flagged, warns = {}, False, []
            yaml_text = render_api_config(
                method=method, model=model, strategy=strategy,
                hparam_stem=stem,
            )
        else:
            overrides, flagged, warns = resolve_overrides(method, donor_csv, strategy, rows)
            if warns:
                n_missing += 1
            if flagged:
                n_flagged += 1
            yaml_text = render_config(
                method=method, model=model, strategy=strategy,
                hparam_stem=stem, swept_overrides=overrides,
                warnings=warns, flagged=flagged,
            )

        rel_path = Path("hydra/experiments/hemonc/main") / strategy / f"{method}_{model}.yaml"
        abs_path = ROOT / rel_path
        config_name = f"hemonc/main/{strategy}/{method}_{model}"

        manifest_out.append({
            "strategy": strategy,
            "method": method,
            "model": model,
            "tier": entry["tier"],
            "config_name": config_name,
            "config_path": str(rel_path),
            "donor_csv_model": donor_csv,
            "flagged": "true" if flagged else "false",
            "has_overrides": "true" if overrides else "false",
            "job_id": "",
            "status": "pending",
            "wandb_run_id": "",
        })

        if args.dry_run:
            continue

        abs_path.parent.mkdir(parents=True, exist_ok=True)
        with open(abs_path, "w") as f:
            f.write(yaml_text)
        n_written += 1

    # Write manifest.
    if not args.dry_run:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(manifest_out[0].keys())
        with open(MANIFEST_PATH, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(manifest_out)
        print(f"\nWrote manifest → {MANIFEST_PATH.relative_to(ROOT)}  ({len(manifest_out)} rows)")
        print(f"Wrote {n_written} config files under {CONFIG_ROOT.relative_to(ROOT)}/")
    else:
        print(f"\n[dry-run] Would write {len(manifest_out)} configs + manifest.")

    if n_missing:
        print(f"  ⚠  {n_missing} configs lacked a best_params.csv row (see WARNING headers).")
    if n_flagged:
        print(f"  ⚠  {n_flagged} configs draw from a locality-flagged sweep winner (kept).")


if __name__ == "__main__":
    main()
