from copy import deepcopy
from typing import Any, Dict, List, Tuple
import math

import torch
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lora_merge_hparams import LoRAMergeHyperParams


def apply_lora_merge_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAMergeHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    LoRA-Merge: train a LoRA adapter on the current batch, then merge it fully
    into the base model via merge_and_unload(). Each increment thus starts from
    a clean (merged) base model — no PEFT wrapper is retained.
    """
    if copy:
        model = deepcopy(model)

    merged_model = execute_lora_merge(model, tok, requests, hparams)
    return merged_model, {}


def execute_lora_merge(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: LoRAMergeHyperParams,
        **kwargs: Any,
) -> AutoModelForCausalLM:
    model.config.use_cache = False
    device = torch.device(f'cuda:{hparams.device}')

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=hparams.rank,
        lora_alpha=hparams.lora_alpha,
        lora_dropout=hparams.lora_dropout,
        layers_to_transform=hparams.layers if len(hparams.layers) > 0 else None,
        target_modules=hparams.target_modules,
    )
    model.to(device)
    peft_model = get_peft_model(model, peft_config)
    peft_model.is_parallelizable = True
    peft_model.model_parallel = True
    peft_model.print_trainable_parameters()

    requests = deepcopy(requests)
    texts = [r["prompt"] for r in requests]
    targets = [r["target_new"] for r in requests]

    lr = hparams.lr * hparams.batch_size / 256
    opt = torch.optim.Adam(
        peft_model.parameters(),
        lr=lr,
        weight_decay=hparams.weight_decay,
    )

    warmup_steps = int(0.1 * hparams.num_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / (hparams.num_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress)) + 1e-6

    scheduler = LambdaLR(opt, lr_lambda)

    if hparams.fp16:
        scaler = GradScaler()

    loss_meter = AverageMeter()
    for it in range(hparams.num_steps):
        loss_meter.reset()
        for txt, tgt in zip(
                chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
        ):
            opt.zero_grad()
            loss = compute_nll_loss(peft_model, tok, txt, tgt, device, hparams)

            loss_meter.update(loss.item(), n=len(txt))
            if hparams.fp16:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

        scheduler.step()
        print(f"[LoRA-Merge] Step {it + 1}/{hparams.num_steps}, loss {loss_meter.avg:.4f}")

    # Merge adapter into base model and return a clean (non-PEFT) model
    merged_model = peft_model.merge_and_unload()
    # Belt-and-suspenders: clear any stale forward hooks on the embedding layer
    # (e.g. make_inputs_require_grads).  torch._dynamo cannot trace in-place
    # requires_grad_() calls during generate(), so these hooks must be gone
    # before evaluation.
    merged_model.get_input_embeddings()._forward_hooks.clear()
    merged_model.config.use_cache = True
    return merged_model


def compute_nll_loss(model, tok, texts, targets, device, hparams):
    """Compute NLL loss on prompt+target sequences, masking the prompt tokens."""
    from easyeditor.util.training_utils import build_tokens_dict
    tokens = build_tokens_dict(tok, texts, targets, hparams.max_length, device)
    return model(**tokens).loss


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def chunks(arr, n):
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
