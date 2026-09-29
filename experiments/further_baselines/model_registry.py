"""Model-naming bridge for the Further-Baselines (DPO / GRPO / Agentic-RAG).

The three baselines were written as standalone scripts that key models by their
HuggingFace *folder* name (e.g. ``google_medgemma-4b-it``).  The main sweep and
the analysis pipeline instead key models by a short *config key* (e.g.
``medgemma_4b``), and the Hydra ``hparams/EVAL/<stem>.yaml`` files key them by an
*EVAL hparam stem* (e.g. ``medgemma-4b-it``) whose ``model_name`` is the HF id.

This module is the single source of truth mapping the three naming schemes, so
the launchers (``eval_lifelong_adapters.py``, ``run_agentic_shared_eval.py``, and
the SLURM wrappers) can translate between them.

Sources of truth:
  - config keys / MODELS_DEFAULT  : main_experiments/generate_configs.py
  - script model_folder keys      : further_baselines/*_hemonc.py BASE_MODELS
  - EVAL hparam stems + model_name: hydra/experiments/hparams/EVAL/<stem>.yaml
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelEntry:
    config_key: str      # analysis / sweep key (combo suffix, e.g. dpo_<config_key>)
    model_folder: str    # HF-folder key expected by the training/agentic scripts
    model_name: str      # HuggingFace model id
    eval_hparam_stem: str  # hydra/experiments/hparams/EVAL/<stem>.yaml (no .yaml)
    tier: str            # "4b" / "8b" — matches the scripts' size-based settings


# Order mirrors MODELS_DEFAULT in generate_configs.py; adaptllm is optional.
MODELS = [
    ModelEntry("gemma3_4b",             "google_gemma-3-4b-it",                "google/gemma-3-4b-it",                "gemma-3-4b-it",          "4b"),
    ModelEntry("qwen3_4b",              "Qwen_Qwen3-4B-Instruct-2507",         "Qwen/Qwen3-4B-Instruct-2507",         "qwen3-4b",               "4b"),
    ModelEntry("medgemma_4b",           "google_medgemma-4b-it",               "google/medgemma-4b-it",               "medgemma-4b-it",         "4b"),
    ModelEntry("llama31_8b",            "meta-llama_Llama-3.1-8B-Instruct",    "meta-llama/Llama-3.1-8B-Instruct",    "llama3.1-8b",            "8b"),
    ModelEntry("bio_medical_llama3_8b", "ContactDoctor_Bio-Medical-Llama-3-8B", "ContactDoctor/Bio-Medical-Llama-3-8B", "bio-medical-llama-3-8b", "8b"),
    # Optional 6th model — not part of the 5-model paper sweep.
    ModelEntry("adaptllm_medicine_chat", "AdaptLLM_medicine-chat",             "AdaptLLM/medicine-chat",              "adaptllm-medicine-chat", "8b"),
]

BY_CONFIG_KEY = {m.config_key: m for m in MODELS}
BY_MODEL_FOLDER = {m.model_folder: m for m in MODELS}


def get(config_key: str) -> ModelEntry:
    """Look up a model entry by its config key (e.g. 'medgemma_4b')."""
    try:
        return BY_CONFIG_KEY[config_key]
    except KeyError:
        raise KeyError(
            f"Unknown config_key {config_key!r}. Known: {sorted(BY_CONFIG_KEY)}"
        )


def config_keys(include_optional: bool = False) -> list:
    """Return the default 5 config keys, or all 6 when include_optional=True."""
    keys = [m.config_key for m in MODELS]
    if include_optional:
        return keys
    return [k for k in keys if k != "adaptllm_medicine_chat"]
