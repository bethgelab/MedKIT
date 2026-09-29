"""LoRA adapter setup shared by the DPO / GRPO trainers.

Both baselines start from the same base model as every other method in the
sweep: the first increment attaches a freshly initialised LoRA adapter to the
base model, and each later increment resumes from the adapter trained through
the previous one (lifelong CL).

LoRA hyperparameters per model:
  - rank / alpha: r=64, alpha=128 for the 4B models; r=32, alpha=64 otherwise.
  - target modules: attention projections for the Gemma family; attention + MLP
    projections for all other architectures.
"""

from peft import LoraConfig, PeftModel, get_peft_model

ATTN_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
ATTN_MLP_TARGETS = ATTN_TARGETS + ["gate_proj", "up_proj", "down_proj"]


def lora_config_for_model(model_folder: str) -> LoraConfig:
    lower = model_folder.lower()
    r, alpha = (64, 128) if "4b" in lower else (32, 64)
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=ATTN_TARGETS if "gemma" in lower else ATTN_MLP_TARGETS,
    )


def lora_summary(model_folder: str) -> dict:
    """JSON-serialisable view of lora_config_for_model (for train_config.json)."""
    cfg = lora_config_for_model(model_folder)
    return {"r": cfg.r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
            "target_modules": sorted(cfg.target_modules)}


def attach_adapter(base, adapter_dir, model_folder: str, is_trainable: bool):
    """Wrap `base` with the adapter to train or evaluate.

    adapter_dir=None means no adapter has been trained yet: for training, attach
    a fresh LoRA; for evaluation, return the plain base model.
    """
    if adapter_dir is None:
        if is_trainable:
            return get_peft_model(base, lora_config_for_model(model_folder))
        return base
    return PeftModel.from_pretrained(base, adapter_dir, is_trainable=is_trainable)
