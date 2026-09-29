"""
O-LoRA: Orthogonal Low-Rank Adaptation for continual knowledge editing.

For each new increment a fresh LoRA adapter is trained on the current batch
with an orthogonality regularisation penalty:

    loss = NLL + orth_lambda * sum_layers sum_i ||A_new @ A_prev_i.T||_F^2

where A matrices have shape (rank x d_in).  After training the adapter is
merged into the base model and the A matrices are stored for future
increments.  The growing list of stored A matrices is maintained on the
executor instance, so it naturally accumulates across calls from
LifelongEditor.
"""
from copy import deepcopy
from typing import Any, Dict, List
import math

import torch
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from .o_lora_hparams import OLoRAHyperParams


class OLoRARewriteExecutor:
    """Stateful executor that stores previous LoRA A matrices for orthogonality."""

    def __init__(self):
        # Maps module name (str) → list of A-weight tensors (CPU, rank × d_in)
        self.prev_A_matrices: Dict[str, List[torch.Tensor]] = {}

    def apply_to_model(
            self,
            model: AutoModelForCausalLM,
            tok: AutoTokenizer,
            requests: List[Dict],
            hparams: OLoRAHyperParams,
            copy=False,
            return_orig_weights=False,
            keep_original_weight=False,
            **kwargs: Any,
    ):
        if copy:
            model = deepcopy(model)

        merged_model = self._execute(model, tok, requests, hparams)
        return merged_model, {}

    def _execute(self, model, tok, requests, hparams):
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

        # Pre-move stored A matrices to device for efficiency during training
        prev_A_on_device = {
            name: [a.to(device) for a in a_list]
            for name, a_list in self.prev_A_matrices.items()
        }

        loss_meter = AverageMeter()
        for it in range(hparams.num_steps):
            loss_meter.reset()
            for txt, tgt in zip(
                    chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
            ):
                opt.zero_grad()

                nll_loss = _compute_nll_loss(peft_model, tok, txt, tgt, device, hparams)
                orth_loss = self._orthogonality_penalty(peft_model, prev_A_on_device, hparams)
                loss = nll_loss + hparams.orth_lambda * orth_loss

                loss_meter.update(nll_loss.item(), n=len(txt))
                loss.backward()
                opt.step()

            scheduler.step()
            print(f"[O-LoRA] Step {it + 1}/{hparams.num_steps}, "
                  f"nll={loss_meter.avg:.4f}, orth={orth_loss.item():.4f}")

        # Store current A matrices (CPU) before merging
        self._collect_A_matrices(peft_model)

        merged_model = peft_model.merge_and_unload()
        merged_model.get_input_embeddings()._forward_hooks.clear()
        merged_model.config.use_cache = True
        return merged_model

    @staticmethod
    def _orthogonality_penalty(
            peft_model,
            prev_A_on_device: Dict[str, List[torch.Tensor]],
            hparams,
    ) -> torch.Tensor:
        """
        Computes sum over all LoRA layers of ||A_current @ A_prev.T||_F^2.
        Returns a scalar tensor (0 if no previous A matrices exist).
        """
        penalty = torch.tensor(0.0, device=torch.device(f'cuda:{hparams.device}'))
        if not prev_A_on_device:
            return penalty

        for name, module in peft_model.named_modules():
            if not hasattr(module, 'lora_A'):
                continue
            # lora_A is an nn.ModuleDict; default adapter key is 'default'
            for adapter_key, lora_a_layer in module.lora_A.items():
                A_current = lora_a_layer.weight  # (rank, d_in)
                if name in prev_A_on_device:
                    for A_prev in prev_A_on_device[name]:
                        # A_prev shape: (rank_prev, d_in) — may differ if rank changed
                        overlap = torch.mm(A_current, A_prev.T)  # (rank, rank_prev)
                        penalty = penalty + torch.sum(overlap ** 2)

        return penalty

    def _collect_A_matrices(self, peft_model) -> None:
        """Extract and store (CPU) A matrices from the current PEFT model."""
        for name, module in peft_model.named_modules():
            if not hasattr(module, 'lora_A'):
                continue
            for adapter_key, lora_a_layer in module.lora_A.items():
                A = lora_a_layer.weight.detach().cpu()
                self.prev_A_matrices.setdefault(name, []).append(A)

        n_layers = len(self.prev_A_matrices)
        n_total = sum(len(v) for v in self.prev_A_matrices.values())
        print(f"[O-LoRA] Stored A matrices: {n_layers} layers, "
              f"{n_total} total (increments × layers)")


def _compute_nll_loss(model, tok, texts, targets, device, hparams):
    mask_token = -100

    if getattr(tok, 'chat_template', None) is not None:
        # Build full sequences using the chat template so training matches the
        # inference format used by instruction-tuned models.
        full_ids_list = []
        prompt_lens = []
        for text, target in zip(texts, targets):
            # Prompt length: user turn + generation prompt (no answer yet)
            prompt_ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            # Full sequence: user turn + assistant answer
            full_ids = tok.apply_chat_template(
                [{"role": "user", "content": text},
                 {"role": "assistant", "content": target}],
                tokenize=True,
                add_generation_prompt=False,
            )
            # Some chat templates BPE-merge the trailing prompt token with the
            # leading target token (e.g. ContactDoctor/Bio-Medical-Llama-3-8B's
            # Human:/Assistant: template merges ' ' + 'superior' -> ' superior'),
            # so len(prompt_ids) overshoots the assistant-content boundary in
            # full_ids and silently masks all label tokens (loss=0). Use the
            # longest common prefix of prompt_ids and full_ids instead.
            lcp = 0
            while (lcp < len(prompt_ids) and lcp < len(full_ids)
                   and prompt_ids[lcp] == full_ids[lcp]):
                lcp += 1
            prompt_lens.append(lcp)
            full_ids_list.append(full_ids)

        # Truncate and left-pad to a uniform length
        max_len = min(max(len(x) for x in full_ids_list), hparams.max_length)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        input_ids = torch.full((len(full_ids_list), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(full_ids_list), max_len), dtype=torch.long)
        labels = torch.full((len(full_ids_list), max_len), mask_token, dtype=torch.long)
        for i, (ids, p_len) in enumerate(zip(full_ids_list, prompt_lens)):
            ids = ids[-max_len:]          # truncate from the left if too long
            p_len = max(0, p_len - (len(full_ids_list[i]) - len(ids)))  # adjust after truncation
            start = max_len - len(ids)    # left-pad position
            input_ids[i, start:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, start:] = 1
            # Only supervise on the assistant's answer tokens
            labels[i, start + p_len:] = torch.tensor(ids[p_len:], dtype=torch.long)
        tokens = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }
    else:
        full_prompt = [f"{p} {l}" for p, l in zip(texts, targets)]
        prompt_ids = tok(list(texts), return_tensors="pt", padding=True, truncation=True,
                         max_length=hparams.max_length)["input_ids"]
        num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
        tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True,
                     max_length=hparams.max_length)
        tokens["labels"] = tokens["input_ids"].clone()
        num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
        for i in range(len(texts)):
            tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i] + num_prompt_toks[i]] = mask_token
        tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
        tokens = {k: v.to(device) for k, v in tokens.items()}

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
