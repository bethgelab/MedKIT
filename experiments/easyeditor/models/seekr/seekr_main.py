"""
SEEKR: Selective Knowledge Exemplar-based Knowledge Retention.

Experience-replay continual learning baseline.  A bounded buffer stores
exemplars from past increments; at each training step a random sample from
the buffer is replayed alongside the current batch to counteract forgetting:

    loss = NLL_current + replay_weight * NLL_replay

After training the adapter is merged into the base model and the buffer is
updated with exemplars from the current increment (randomly subsampled to
buffer_size if the limit is exceeded).

The replay buffer persists on the executor instance and accumulates across
calls from LifelongEditor.
"""
import random
from copy import deepcopy
from typing import Any, Dict, List
import math

import torch
from peft import get_peft_model, LoraConfig, TaskType
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

from .seekr_hparams import SEEKRHyperParams


class SEEKRRewriteExecutor:
    """Stateful executor that maintains an exemplar replay buffer."""

    def __init__(self):
        # Each entry: {'prompt': str, 'target_new': str}
        self.replay_buffer: List[Dict[str, str]] = []

    def apply_to_model(
            self,
            model: AutoModelForCausalLM,
            tok: AutoTokenizer,
            requests: List[Dict],
            hparams: SEEKRHyperParams,
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

        has_replay = len(self.replay_buffer) > 0
        print(f"[SEEKR] Buffer size at start: {len(self.replay_buffer)}")

        loss_meter = AverageMeter()
        for it in range(hparams.num_steps):
            loss_meter.reset()
            for txt, tgt in zip(
                    chunks(texts, hparams.batch_size), chunks(targets, hparams.batch_size)
            ):
                opt.zero_grad()

                # Backward each loss term as soon as it's computed so the
                # forward graph for one batch can be freed before we run the
                # next forward.  The combined-loss form
                #     loss = nll_current + replay_weight * nll_replay
                #     loss.backward()
                # keeps BOTH graphs alive simultaneously until the single
                # backward call — peak activation = current_batch + replay_batch.
                # On Gemma-3 (262K vocab → ~400 MB filtered-logits per seq) that
                # consistently OOMs from increment 2 onward when the replay
                # buffer fills.  Sequential backward gives identical gradients
                # at peak activation = max(current_batch, replay_batch).
                nll_current = _compute_nll_loss(peft_model, tok, txt, tgt, device, hparams)
                nll_current.backward()
                loss_meter.update(nll_current.item(), n=len(txt))
                del nll_current

                if has_replay:
                    replay_size = min(hparams.replay_batch_size, len(self.replay_buffer))
                    replay_samples = random.sample(self.replay_buffer, replay_size)
                    r_texts = [s['prompt'] for s in replay_samples]
                    r_targets = [s['target_new'] for s in replay_samples]
                    nll_replay = _compute_nll_loss(peft_model, tok, r_texts, r_targets,
                                                   device, hparams)
                    (hparams.replay_weight * nll_replay).backward()
                    del nll_replay

                opt.step()

            scheduler.step()
            print(f"[SEEKR] Step {it + 1}/{hparams.num_steps}, "
                  f"nll={loss_meter.avg:.4f}")

        # Update replay buffer with current increment's exemplars
        new_exemplars = [
            {'prompt': r['prompt'], 'target_new': r['target_new']}
            for r in requests
        ]
        self.replay_buffer.extend(new_exemplars)
        if len(self.replay_buffer) > hparams.buffer_size:
            self.replay_buffer = random.sample(self.replay_buffer, hparams.buffer_size)
        print(f"[SEEKR] Buffer size after update: {len(self.replay_buffer)}")

        merged_model = peft_model.merge_and_unload()
        merged_model.get_input_embeddings()._forward_hooks.clear()
        merged_model.config.use_cache = True
        return merged_model


def _compute_nll_loss(model, tok, texts, targets, device, hparams):
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
