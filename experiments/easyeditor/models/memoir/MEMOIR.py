"""
MEMOIR: Model Editing with Minimal Overwrite and Informed Retention

This module implements MEMOIR, a knowledge editing method that uses sparse fine-tuning
and conditional memory activation to efficiently edit large language models.

Key components:
- TopHasher: Implements top-k hashing for sparse feature selection
- MEMOIR: Main wrapper class that injects editing capability into models
- MEMOIRAdapter: Wrapper for the edited layer with sparsely activated residual memory
"""

import gc
import os
import sys
import copy

# Per-forward debug prints (mask overlap ratios, match-sample IDs, irrelevant-
# sample skips) fire ONCE PER GENERATED TOKEN during inference. On a 1000-row
# task with max_tokens=512 that's >500K print() calls — enough to dominate
# wall-clock and balloon SLURM logs to GBs. Off by default; opt in with
# MEMOIR_VERBOSE=1 for one-off debugging.
_VERBOSE = os.environ.get("MEMOIR_VERBOSE", "0") not in ("0", "", "false", "False")
from typing import Optional, List, Tuple

import torch
from torch.nn import functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import numpy as np
import transformers
import wandb

from .utils import parent_module, brackets_to_periods, EarlyStopMeter


class TopHasher:
    """
    Top-k hashing algorithm for sparse feature selection.

    Selects the top-k indices with highest absolute values from input features,
    then applies a permutation to ensure diversity in selection.
    """

    def __init__(self, input_dim: int, top_k: int):
        self.top_k = top_k
        self.input_dim = input_dim
        self.permutation = torch.randperm(input_dim)
        print(f"TopHasher: selecting {top_k} feature indices from {input_dim} dimensions")

    def get_active_indices(self, x: Tensor) -> Tensor:
        assert x.shape[0] == self.input_dim, "Input dimension must match the input dimension of the TopHasher"
        _, indices = torch.topk(x.abs(), self.top_k)
        indices = self.permutation[indices.to(self.permutation.device)]
        return indices.detach().cpu()


class MEMOIR(torch.nn.Module):
    """
    MEMOIR lifelong model editing framework for language models.
    """

    def __init__(self, config, model, device):
        super(MEMOIR, self).__init__()
        self.config = config
        self.model = model
        self.device = device

        layer = config.inner_params[0]

        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)

        if type(adapter_layer) is not MEMOIRAdapter:
            setattr(self.edit_module, self.layer_name,
                    MEMOIRAdapter(config, adapter_layer, transpose=transpose))
            print(f"Successfully inserted MEMOIR adapter into layer: {layer}")

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def __call__(self, **kwargs):
        adapter_layer = self.get_adapter_layer()
        prompt_boundary = kwargs.pop("last_prompt_token_loc_inference", None)
        if prompt_boundary is not None:
            setattr(adapter_layer, "last_prompt_token_loc_inference", prompt_boundary)
        try:
            return self.model(**kwargs)
        finally:
            if prompt_boundary is not None and hasattr(adapter_layer, "last_prompt_token_loc_inference"):
                delattr(adapter_layer, "last_prompt_token_loc_inference")

    def get_adapter_layer(self) -> 'MEMOIRAdapter':
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is MEMOIRAdapter, \
            'Adapter Layer is not added correctly. Expected MEMOIRAdapter.'
        return adapter_layer.to(self.model.device)

    def reset_layer(self):
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    # MEMOIR's adapter has two pieces of state that the default
    # `state_dict()` save MISSES because they aren't registered as
    # Parameters/buffers:
    #   - `hasher.permutation`  (the random feature-index permutation; if not
    #     persisted, a resumed run gets a fresh permutation and ALL stored
    #     edits map to the wrong active indices — silent total corruption.)
    #   - `masks_for_edited_samples`  (per-edit feature masks, queried at
    #     inference to gate the residual memory; without them every input
    #     looks "irrelevant" and no edit fires.)
    # `new_weight` IS a Parameter so it goes through state_dict, but for a
    # generic `load_state_dict` to succeed the model must already be wrapped —
    # the wrapper introduces keys like `…down_proj.original_layer.weight` and
    # `…down_proj.new_weight` that don't exist on a fresh HF model.
    #
    # `save_model` writes a single `<path>.pt` containing the wrapped model
    # state_dict + an extras dict with the hasher and mask state.
    # `load_model` (via `load_memoir_into_model`) restores both onto an
    # already-wrapped fresh model.
    def save_model(self, path: str):
        adapter = self.get_adapter_layer()
        # masks_for_edited_samples may be either an empty list (no edits yet)
        # or a stacked 2-D tensor (after the first edit) — handle both.
        masks = adapter.masks_for_edited_samples
        if isinstance(masks, torch.Tensor):
            masks_to_save = masks.detach().cpu()
        else:
            masks_to_save = list(masks)  # empty list
        state = {
            'format_version': 1,
            'editing_method': 'MEMOIR',
            'layer': self.layer,
            'wrapped_state_dict': self.model.state_dict(),
            'extras': {
                # Hasher permutation MUST be persisted — a fresh randperm() at
                # load time would silently invalidate every stored mask.
                'hasher_permutation': adapter.hasher.permutation.detach().cpu(),
                'hasher_top_k': int(adapter.hasher.top_k),
                'hasher_input_dim': int(adapter.hasher.input_dim),
                'masks_for_edited_samples': masks_to_save,
            },
        }
        torch.save(state, path + '.pt')
        n_masks = (masks_to_save.shape[0]
                   if isinstance(masks_to_save, torch.Tensor)
                   else len(masks_to_save))
        print(f'[MEMOIR] Saved checkpoint to {path}.pt '
              f'(edited samples={n_masks})')

    def load_model(self, path: str):
        try:
            state = torch.load(path + '.pt',
                               map_location=next(self.model.parameters()).device,
                               weights_only=False)
        except TypeError:
            state = torch.load(path + '.pt',
                               map_location=next(self.model.parameters()).device)

        if not isinstance(state, dict) or state.get('editing_method') != 'MEMOIR':
            raise RuntimeError(
                f'MEMOIR.load_model: file at {path}.pt is not a MEMOIR checkpoint '
                f'(got format={type(state).__name__}, '
                f'method={state.get("editing_method") if isinstance(state, dict) else "n/a"}). '
                'If this checkpoint was written by the legacy save path '
                '(before the resume fix), delete it and re-run from scratch.'
            )

        # Restore Parameter/buffer state (incl. new_weight); strict=False so
        # we tolerate stray keys from older HF revisions.
        missing, unexpected = self.model.load_state_dict(
            state['wrapped_state_dict'], strict=False
        )
        if unexpected:
            print(f'[MEMOIR.load_model] Unexpected state_dict keys (ignored): '
                  f'{unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
        if missing:
            print(f'[MEMOIR.load_model] Missing state_dict keys (left at init): '
                  f'{missing[:5]}{"..." if len(missing) > 5 else ""}')

        adapter = self.get_adapter_layer()
        extras = state['extras']
        # Replace the random permutation with the saved one BEFORE any inference.
        saved_perm = extras['hasher_permutation']
        if (saved_perm.numel() != adapter.hasher.permutation.numel()
                or extras.get('hasher_top_k') != adapter.hasher.top_k):
            raise RuntimeError(
                f'[MEMOIR.load_model] Hasher mismatch: saved '
                f'(input_dim={extras.get("hasher_input_dim")}, top_k={extras.get("hasher_top_k")}) '
                f'vs current (input_dim={adapter.hasher.input_dim}, top_k={adapter.hasher.top_k}). '
                'Hparams must match between save and resume.'
            )
        adapter.hasher.permutation = saved_perm.to(adapter.hasher.permutation.device)
        # Restore masks (tensor) or empty-list sentinel.
        masks = extras['masks_for_edited_samples']
        if isinstance(masks, torch.Tensor):
            adapter.masks_for_edited_samples = masks.to(adapter.weight.device)
        else:
            adapter.masks_for_edited_samples = []
        n_masks = (adapter.masks_for_edited_samples.shape[0]
                   if isinstance(adapter.masks_for_edited_samples, torch.Tensor)
                   else 0)
        print(f'[MEMOIR] Loaded checkpoint from {path}.pt '
              f'(edited samples={n_masks})')

    def edit(self, config, tokens, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        self.get_adapter_layer().set_parameter_tunable()

        # Find the first position where labels are NOT -100 (first actual label token)
        # and subtract 1 to get the last prompt token position.  This is robust to
        # both left- and right-padded sequences.  Counting all -100s (old approach)
        # includes trailing padding, pointing past the label tokens entirely.
        has_label = (tokens["labels"] != -100)
        last_prompt_token_loc = has_label.float().argmax(dim=1) - 1
        self.get_adapter_layer().last_prompt_token_loc = last_prompt_token_loc

        loss_meter = EarlyStopMeter()
        optimizer = None

        if config.RUN_SAVE_BACKGROUND_FEATURES:
            _ = self.model(**tokens)
            self.model.eval()
            setattr(eval(f"self.model.{self.layer}"), "training", False)
            delattr(self.get_adapter_layer(), "last_prompt_token_loc")
            return

        # Per-sequence gradient accumulation: run one sequence through the model
        # at a time, call backward() immediately, then free the computation graph
        # before the next sequence.  Peak activation memory is proportional to
        # batch_size=1 instead of the full context-template batch (~12 sequences),
        # reducing activation storage from ~27 GB to ~0.7 GB.  This sidesteps the
        # torch.compile / gradient-checkpointing incompatibility seen on Qwen3 and
        # similar compiled models (where gradient_checkpointing_enable() does not
        # reduce memory because the compiled graph was already traced).
        self.model.train()

        if hasattr(self.model.config, 'batch_size'):
            k = self.config.batch_size
        else:
            k = 1
        bs = tokens["input_ids"].shape[0] - k

        for i in range(config.n_iter):
            if i == 0:
                optimizer = torch.optim.SGD(
                    [self.get_adapter_layer().new_weight],
                    config.edit_lr,
                    weight_decay=1e-5
                )

            if i > 0 and loss_meter.stop():
                break

            optimizer.zero_grad()
            loss_scalar = self._cal_ft_loss_accum(tokens, last_prompt_token_loc, bs, k)

            torch.nn.utils.clip_grad_norm_(
                self.get_adapter_layer().new_weight,
                max_norm=1.0
            )

            if wandb.run:
                wandb.log({'train/loss': loss_scalar}, commit=True)

            print(f"Iteration {i}: loss = {np.round(loss_scalar, 4)}")

            optimizer.step()
            loss_meter.update(loss_scalar)

        self.model.eval()

        setattr(eval(f"self.model.{self.layer}"), "training", False)
        delattr(self.get_adapter_layer(), "last_prompt_token_loc")

    def _cal_ft_loss_accum(self, tokens, last_prompt_token_loc: torch.Tensor,
                           bs: int, k: int) -> float:
        """Run one sequence at a time, backward immediately, accumulate gradients.

        Returns the mean loss as a Python float (detached).  Gradients are
        accumulated into new_weight.grad; the caller should call optimizer.step()
        afterwards (and optimizer.zero_grad() before calling this method).
        """
        loss_fct = CrossEntropyLoss(reduction='none')
        model_dtype = next(self.model.parameters()).dtype
        total_loss = 0.0

        for seq_i in range(bs):
            # Build a single-sequence batch to keep peak activation memory small.
            single_input = {key: val[seq_i:seq_i + 1]
                            for key, val in tokens.items() if key != 'labels'}
            single_input['use_cache'] = False

            single_logits = self.model(**single_input).logits
            if single_logits.dtype != model_dtype:
                single_logits = single_logits.to(model_dtype)

            # single_logits[0, :-1, :] is contiguous (no multi-dim slice copy).
            seq_logits = single_logits[0, :-1, :]          # (seq-1, vocab)
            seq_labels = tokens['labels'][seq_i, 1:]       # (seq-1,)

            seq_loss = loss_fct(seq_logits, seq_labels)    # (seq-1,)

            col_index = last_prompt_token_loc[seq_i]
            label_mask = torch.zeros(seq_loss.shape[0], dtype=torch.bool,
                                     device=seq_loss.device)
            label_mask[col_index - 1:] = True

            denom = label_mask.sum().clamp(min=1)
            seq_nll = (seq_loss * label_mask).sum() / denom

            # Divide by bs so the accumulated gradient equals the mean over the batch.
            (seq_nll / bs).backward()
            total_loss += seq_nll.item()

        return total_loss / bs


class MEMOIRAdapter(torch.nn.Module):
    """
    MEMOIR adapter layer that implements MEMOIR memory module for knowledge editing.
    """

    def __init__(self, config, layer, transpose: bool):
        super(MEMOIRAdapter, self).__init__()

        assert config.prompt_feature_agg in ['last', 'mean', 'mean_decentered'], \
            f"Unknown prompt feature aggregation strategy: {config.prompt_feature_agg}. "
        self.prompt_feature_agg = config.prompt_feature_agg

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config

        self.new_weight = copy.deepcopy(self.weight)
        self.new_weight.data.zero_()

        self.original_layer = copy.deepcopy(self.layer)

        if 'gpt2' in self.config.model_name:
            self.bias = self.layer.bias
        else:
            self.bias = None

        assert not self.weight.requires_grad, \
            'Original layer weights should not be trainable.'

        self.training = False

        self.hasher = TopHasher(self.new_weight.shape[1], top_k=self.config.top_k)

        self.masks_for_edited_samples = []

        # Load pre-computed background features (skip if running save mode or file missing)
        if config.RUN_SAVE_BACKGROUND_FEATURES:
            # In save mode, background features don't exist yet; initialize empty
            print("RUN_SAVE_BACKGROUND_FEATURES=True: skipping background feature load.")
            self.loaded_irrelevant_sample_mean_features = torch.zeros(
                layer.weight.shape[1], device=layer.weight.device
            )
            self.saved_background_features = []
        elif config.dir_background_features and config.dir_background_features != 'null':
            import os
            if os.path.exists(config.dir_background_features):
                print(f"Loading background features for model {config.model_name} from: {config.dir_background_features}")
                self.loaded_irrelevant_sample_features = torch.load(
                    config.dir_background_features
                )[:100, :]
                self.loaded_irrelevant_sample_mean_features = torch.mean(
                    self.loaded_irrelevant_sample_features, dim=0
                ).to(layer.weight.device)
                assert self.loaded_irrelevant_sample_mean_features.shape[0] == layer.weight.shape[1]
            else:
                print(f"Warning: background features file not found at {config.dir_background_features}. "
                      f"Using zero features. Run with RUN_SAVE_BACKGROUND_FEATURES=true first.")
                self.loaded_irrelevant_sample_mean_features = torch.zeros(
                    layer.weight.shape[1], device=layer.weight.device
                )
        else:
            print("No dir_background_features set; using zero background features.")
            self.loaded_irrelevant_sample_mean_features = torch.zeros(
                layer.weight.shape[1], device=layer.weight.device
            )

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def aggregate_prompt_features(self, x: Tensor, prompt_boundary: int, padding_counts: int, verbose: bool = False) -> Tensor:
        if verbose:
            print(f"Prompt features aggregated from token {padding_counts} to token "
                  f"{prompt_boundary} (out of {x.shape[1]} total tokens)")

        if self.prompt_feature_agg == 'last':
            return x[:, prompt_boundary, :]
        elif self.prompt_feature_agg == 'mean':
            return x[:, padding_counts:prompt_boundary + 1, :].mean(1)
        elif self.prompt_feature_agg == 'mean_decentered':
            # Defensive .to(x.device): the construction code uses
            # ``device=layer.weight.device`` so this should already match
            # ``x.device`` in normal use, but a fresh ``model.to(...)`` after
            # adapter install can leave non-buffer attributes stranded
            # (Python attribute, not registered nn.Module buffer). Migrating
            # on first use makes inference robust to that race.
            mean_features = self.loaded_irrelevant_sample_mean_features
            if mean_features.device != x.device:
                mean_features = mean_features.to(x.device)
                self.loaded_irrelevant_sample_mean_features = mean_features
            return (x[:, padding_counts:prompt_boundary + 1, :].mean(1) - mean_features.unsqueeze(0))

    def count_padding_tokens(self, x: Tensor) -> int:
        padding_vec = x[0, :].clone()
        is_padding = (x == padding_vec.unsqueeze(0)).all(dim=1)
        padding_counts = is_padding.sum(dim=0)
        return padding_counts

    def new_weight_forward(self, x: Tensor) -> Tensor:
        B, S, D = x.shape

        if self.training:
            assert not hasattr(self, "last_prompt_token_loc_inference") or getattr(self, "last_prompt_token_loc_inference") is None
            prompt_boundary = self.last_prompt_token_loc[0]
        else:
            assert not hasattr(self, "last_prompt_token_loc") or getattr(self, "last_prompt_token_loc") is None
            if hasattr(self, "last_prompt_token_loc_inference") and self.last_prompt_token_loc_inference is not None:
                prompt_boundary = self.last_prompt_token_loc_inference[0]
            else:
                # Called directly on the underlying model (e.g. evaluation in editor.py bypasses
                # the MEMOIR wrapper, so last_prompt_token_loc_inference is never set).
                # Fall back to the last token in the sequence as the prompt boundary.
                # For RUN_SAVE_BACKGROUND_FEATURES mode this value is unused (save path
                # only cat/saves accumulated training-mode features and calls sys.exit).
                # For normal inference the MEMOIR wrapper should be used; this fallback
                # ensures evaluation does not crash and computes a reasonable feature.
                prompt_boundary = S - 1

        if self.training:
            padding_counts = self.count_padding_tokens(x[0])
        else:
            padding_counts = 0

        prompt_agg_features = self.aggregate_prompt_features(x, prompt_boundary, padding_counts)

        if self.config.RUN_SAVE_BACKGROUND_FEATURES:
            self.func_save_background_features(prompt_agg_features[-1])
            return torch.zeros_like(F.linear(x, self.new_weight))

        active_indices = self.hasher.get_active_indices(prompt_agg_features[0])

        active_parameter_mask = torch.zeros(D, dtype=torch.bool, device=x.device)
        active_parameter_mask[active_indices] = True

        if self.training:
            if all(not torch.equal(active_parameter_mask, prev)
                   for prev in self.masks_for_edited_samples):
                if len(self.masks_for_edited_samples) == 0:
                    self.masks_for_edited_samples = active_parameter_mask.view(1, -1)
                else:
                    self.masks_for_edited_samples = torch.vstack(
                        (self.masks_for_edited_samples, active_parameter_mask)
                    )
        else:
            overlapping_counts = torch.matmul(
                active_parameter_mask.float(),
                self.masks_for_edited_samples.T.float()
            )

            overlap_ratios = overlapping_counts / self.config.top_k

            best_vals, best_idxs = torch.topk(overlap_ratios, k=1)
            best_overlap = best_vals.item()
            best_match_idx = best_idxs.item()

            if _VERBOSE:
                print(f"Mask overlap ratio with closest edited sample: {best_overlap:.4f} "
                      f"(match sample #{best_match_idx})")

            if best_overlap >= self.config.irr_threshold:
                if _VERBOSE:
                    if best_overlap == 1.0:
                        print(f"Identified as a previously edited sample #{best_match_idx}")
                    else:
                        print(f"Identified as rephrased sample (from sample #{best_match_idx})")

                best_match_mask = self.masks_for_edited_samples[best_match_idx].to(
                    active_indices.device
                )
                active_indices = torch.where(best_match_mask > 0.5)[0]
            else:
                if _VERBOSE:
                    print(f"Identified as irrelevant sample; deactivating residual memory")
                down_out = torch.zeros_like(F.linear(x, self.new_weight))
                return down_out

        down_out = F.linear(x[:, :, active_indices], self.new_weight[:, active_indices])
        return down_out

    def forward(self, *args) -> Tensor:
        layer_out = self.original_layer(*args) + self.new_weight_forward(*args)
        return layer_out

    def func_save_background_features(self, irrelevant_sample_features: Tensor):
        if self.training:
            self.saved_background_features.append(irrelevant_sample_features.unsqueeze(0).detach().cpu())
        else:
            self.saved_background_features = torch.cat(self.saved_background_features, dim=0)
            torch.save(self.saved_background_features, self.config.dir_to_save_background_features)
            print(f"Background features saved to {self.config.dir_to_save_background_features}. Exiting program.")
            sys.exit(0)
