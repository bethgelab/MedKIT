import copy
import random

import torch
from torch.nn import functional as F
from .utils import parent_module, brackets_to_periods, EarlyStopMeter, EditingMeanAct
import transformers
import numpy as np
from torch import Tensor
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from .merge import slerp, GTA, linear
import torch.nn as nn
import gc

merge_dict = {
    'slerp': slerp(),
    'ties': GTA('magnitude', 'sum', normalize=True),
    'magnitude_norm': GTA('magnitude', None, normalize=True),
    'magnitude': GTA('magnitude', None, normalize=False),
    'sign': GTA(None, 'sum', normalize=True),
    'dare_ties': GTA('rescaled_random', 'sum'),
    'dare_linear': GTA('random', None),
    'linear': linear()
}

edit_history = []
merge_group_edit_history = []

def euc(query, key, config, act_mask=None, infer=False, per_batch=False):
    """Euclidean distance between two activation tensors.

    Shapes:
      query, key : [B, S, D]
      return     : scalar  (original behaviour, per_batch=False)
                   [B]     (per-item, per_batch=True — safe for bs>1 eval routing)

    When per_batch is True, the aggregate `.mean()` across the batch axis is
    dropped so WISE's routing decisions can be made per-item.  At bs=1 both
    paths return numerically identical values.
    """
    act_fn = ACT2FN[config.hidden_act]
    l2_norm = torch.norm(act_fn(key) - act_fn(query), dim=-1)   # [B, S]
    if infer and l2_norm.size(1) > 100:
        # [B, 1]: largest seq-position per batch item
        topk = torch.topk(l2_norm, k=1, largest=True, dim=-1)
        if per_batch:
            return topk.values.squeeze(-1)   # [B]
        return topk.values.mean()            # scalar (legacy)

    if act_mask is not None:
        return torch.sum(l2_norm * act_mask, dim=1) / torch.sum(act_mask, dim=1)  # [B]
    else:
        return torch.mean(l2_norm, dim=-1)   # [B]


class WISE(torch.nn.Module):
    def __init__(self, config, model, device):
        super(WISE, self).__init__()
        self.config = config
        self.model = model
        self.config = config
        if hasattr(self.model.config, 'hidden_act'):
            self.config.hidden_act = self.model.config.hidden_act
        elif hasattr(self.model.config, 'hidden_activation'):
            # Gemma-3 and some newer models use hidden_activation instead of hidden_act
            self.config.hidden_act = self.model.config.hidden_activation
        elif hasattr(self.model.config, 'activation_function'):
            self.config.hidden_act = self.model.config.activation_function
        else:
            self.config.hidden_act = 'silu'
        # self.tokenizer = model.tokenizer
        layer = config.inner_params[0]
        self.device = device
        self.adapter_layer = None
        self.original_layer = None

        # --- ensure proper formatting (WISE edits weights matrices) ---
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer

        for n, p in self.model.named_parameters():
            p.requires_grad = False

        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # --- Add WISE to chosen layers ---
        self.edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        self.layer_name = self.layer.rsplit(".", 1)[-1]
        adapter_layer = getattr(self.edit_module, self.layer_name)

        if type(adapter_layer) is not WISEAdapter:
            setattr(self.edit_module, self.layer_name, WISEAdapter(config, adapter_layer, transpose=transpose))
            self.original_layer = copy.deepcopy(adapter_layer)
            print(f"New weights successfully inserted into {layer}")
        
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    # Forward
    def __call__(self, **kwargs):
        if not self.config.retrieve:
            if hasattr(self.get_adapter_layer(), 'editing') and not self.get_adapter_layer().editing:
                # final merge
                if not self.get_adapter_layer().original_layer.weight.equal(self.get_adapter_layer().new_weight) and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    self.get_adapter_layer().memory_weight.append(self.get_adapter_layer().new_weight)
                if len(self.get_adapter_layer().memory_weight) > 0 and self.get_adapter_layer().editing_total_cnt >= self.config.save_freq:
                    print('length of memory is ', len(self.get_adapter_layer().memory_weight), '!!!!!!')
                    self.get_adapter_layer().merge_weight()
        return self.model(**kwargs)

    def reset_layer(self):
        layer = getattr(self.edit_module, self.layer_name)
        del layer
        setattr(self.edit_module, self.layer_name, self.get_adapter_layer().original_layer)

    def get_adapter_layer(self):
        adapter_layer = getattr(self.edit_module, self.layer_name)
        assert type(adapter_layer) is WISEAdapter, print('Adapter Layer is not added correctly....')
        return adapter_layer

    # ── Checkpointing ─────────────────────────────────────────────────────────
    # The default `torch.save(model.state_dict())` path is BROKEN for WISE because:
    #   1. At save time `self.model` is the *wrapped* HF model (an MLP layer is
    #      replaced in-place by WISEAdapter via `setattr`), so the state_dict has
    #      keys like `…down_proj.layer.weight, …down_proj.original_layer.weight,
    #      …down_proj.new_weight` — these don't match a fresh, un-wrapped model
    #      at load time and `load_state_dict(strict=True)` raises.
    #   2. WISEAdapter holds critical state OUTSIDE the standard state_dict
    #      mechanism: `memory_weight` (list of accumulated edits — the actual
    #      knowledge) and `memory_mean_act` (per-memory activation thresholds
    #      used by the retrieve-mode router) are plain Python lists, not
    #      registered as parameters/buffers.
    #
    # `save_model` writes a single `<path>.pt` containing the wrapped model's
    # full state_dict + a sidecar dict of the non-Parameter Python state.
    # `load_model` assumes the WISE wrapper has already been re-applied to a
    # fresh HF model (do this via `load_wise_into_model` below, which is the
    # entry point the runner uses on resume).
    def save_model(self, path: str):
        adapter = self.get_adapter_layer()
        state = {
            'format_version': 1,
            'editing_method': 'WISE',
            'layer': self.layer,
            'wrapped_state_dict': self.model.state_dict(),
            'extras': {
                # CRITICAL: the accumulated edits live here. Without persisting
                # this list every resume would discard all prior weight memories.
                'memory_weight': [w.detach().cpu() for w in adapter.memory_weight],
                # Activation stats per stored memory — used by retrieve-mode
                # routing. EditingMeanAct is a plain Python class, picklable.
                'memory_mean_act': adapter.memory_mean_act,
                'editing_mean_act': adapter.editing_mean_act,
                'editing_total_cnt': int(adapter.editing_total_cnt),
                'merge_cnt': int(adapter.merge_cnt),
                # Optional runtime tensors — present only during/after training.
                'weight_mask': (adapter.weight_mask.detach().cpu()
                                if hasattr(adapter, 'weight_mask') else None),
                'used_mask': adapter.used_mask,  # ndarray or None
            },
        }
        torch.save(state, path + '.pt')
        print(f'[WISE] Saved checkpoint to {path}.pt '
              f'(memories={len(state["extras"]["memory_weight"])}, '
              f'edits={state["extras"]["editing_total_cnt"]})')

    def load_model(self, path: str):
        """Load a WISE checkpoint into the (already-wrapped) self.model.

        The runner calls `load_wise_into_model(model, hparams, path)` — that
        helper builds the wrapper first, then dispatches here.
        """
        # `weights_only=False` is required: extras contain custom Python
        # objects (EditingMeanAct, ndarrays). torch>=2.6 changed the default.
        try:
            state = torch.load(path + '.pt',
                               map_location=next(self.model.parameters()).device,
                               weights_only=False)
        except TypeError:
            # Older torch versions don't have weights_only; fall through.
            state = torch.load(path + '.pt',
                               map_location=next(self.model.parameters()).device)

        if not isinstance(state, dict) or state.get('editing_method') != 'WISE':
            raise RuntimeError(
                f'WISE.load_model: file at {path}.pt is not a WISE checkpoint '
                f'(got format={type(state).__name__}, '
                f'method={state.get("editing_method") if isinstance(state, dict) else "n/a"}). '
                'If this checkpoint was written by the legacy save path '
                '(before the resume fix), delete it and re-run from scratch.'
            )

        # Restore Parameter/buffer state — keys now match because the wrapper
        # is in place.
        missing, unexpected = self.model.load_state_dict(
            state['wrapped_state_dict'], strict=False
        )
        if unexpected:
            print(f'[WISE.load_model] Unexpected state_dict keys (ignored): '
                  f'{unexpected[:5]}{"..." if len(unexpected) > 5 else ""}')
        if missing:
            print(f'[WISE.load_model] Missing state_dict keys (left at init): '
                  f'{missing[:5]}{"..." if len(missing) > 5 else ""}')

        # Restore the non-Parameter adapter state.
        adapter = self.get_adapter_layer()
        extras = state['extras']
        adapter.memory_weight = [
            w.to(adapter.weight.device).to(adapter.weight.dtype)
            for w in extras.get('memory_weight', [])
        ]
        adapter.memory_mean_act = extras.get('memory_mean_act', [])
        adapter.editing_mean_act = extras.get('editing_mean_act', adapter.editing_mean_act)
        adapter.editing_total_cnt = int(extras.get('editing_total_cnt', 0))
        adapter.merge_cnt = int(extras.get('merge_cnt', 0))
        wm = extras.get('weight_mask', None)
        if wm is not None:
            adapter.weight_mask = wm.to(adapter.weight.device)
        adapter.used_mask = extras.get('used_mask', None)
        print(f'[WISE] Loaded checkpoint from {path}.pt '
              f'(memories={len(adapter.memory_weight)}, '
              f'edits={adapter.editing_total_cnt})')

    # TODO: generation
    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def edit(self, config, tokens, act_mask=None, deact_mask=None):
        # for retrieve ##
        global edit_history
        global merge_group_edit_history
        edit_history.append([{f"{k1}" : v1.to('cpu') for k1, v1 in tokens.items()}, False])
        # for retrieve ##
        last_prompt_token_loc = (tokens["labels"] == -100).sum(dim=-1) - 1

        try:
            setattr(eval(f"self.model.{self.layer}"), "training", True)
            setattr(eval(f"self.model.{self.layer}"), "editing", True)
            self.get_adapter_layer().set_parameter_tunable()
            if getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
                self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)
        except AttributeError:
            setattr(eval(f"self.model.model.{self.layer}"), "training", True)
            setattr(eval(f"self.model.model.{self.layer}"), "editing", True)
            self.get_adapter_layer().set_parameter_tunable()
            if getattr(eval(f"self.model.model.{self.layer}"), "editing_total_cnt") % self.config.save_freq == 0:
                self.get_adapter_layer().generate_activation_mask(self.config.mask_ratio)

        # --- train Wise value ---
        loss_meter = EarlyStopMeter()
        for i in range(config.n_iter):

            if i == 0:
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.SGD([self.get_adapter_layer().new_weight], config.edit_lr, weight_decay=1e-5)

            ft_loss = self.__cal_ft_loss(tokens, last_prompt_token_loc)

            act_loss = self.__cal_activation_loss(self.get_adapter_layer().original_layer_output, self.get_adapter_layer().new_weight_layer_output,
                                                  config=config, act_mask=act_mask, deact_mask=deact_mask)
            loss = ft_loss + act_loss.to(ft_loss.device)

            if loss_meter.stop():
                self.get_adapter_layer().save_editing_activation()  # add last gradient
                break
            if i == config.n_iter - 1:
                self.get_adapter_layer().save_editing_activation()  # add last gradient

            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                memory_loss = []
                for _ in merge_group_edit_history:
                    idx = 0
                    while True:
                        memo_input, is_used = _[idx]
                        if not is_used:
                            _[idx][1] = True
                            break
                        idx += 1
                        if idx == len(_): ## re Assign
                            for m in range(len(_)):
                                _[m][1] = False
                            idx = 0

                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    memory_act_loss = self.__cal_memory_neg_activation_loss(self.get_adapter_layer().original_layer_output,
                                                    self.get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    memory_loss.append(memory_act_loss.to(ft_loss.device))
                    del memo_input
                neg_memo_loss = torch.stack(memory_loss).mean()
                loss += neg_memo_loss
                if len(edit_history) > 0:
                    memo_input = random.choice(edit_history)[0]
                    memo_input = {f"{k1}" : v1.to(self.config.device) for k1, v1 in memo_input.items()}
                    self.model(**memo_input)

                    pos_memo_loss = self.__cal_memory_pos_activation_loss(self.get_adapter_layer().original_layer_output,
                                                    self.get_adapter_layer().new_weight_layer_output, config=config,
                                                    act_mask=act_mask, deact_mask=deact_mask)
                    del memo_input
                    loss += pos_memo_loss.to(ft_loss.device)
            # for replay Appendix B.3

            optimizer.zero_grad()

            loss.backward()
            self.get_adapter_layer().mask_new_weight_gradient()

            if self.config.retrieve and self.get_adapter_layer().merge_cnt > 0 and self.config.replay:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)} + {np.round(neg_memo_loss.item(), 3)} + {np.round(pos_memo_loss.item(), 3)}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(ft_loss.item(), 3)} + {np.round(act_loss.item(), 3)}"
                )

            optimizer.step()
            loss_meter.update(loss.item())

            if type(self.config.norm_constraint) is float:
                self.__norm_constraint(self.config.norm_constraint)

        # --- pull out info we want to log from the Wise layer ---
        try:
            setattr(eval(f"self.model.{self.layer}"), "editing", False)
            setattr(eval(f"self.model.{self.layer}"), "training", False)

            editing_total_cnt = getattr(eval(f"self.model.{self.layer}"), "editing_total_cnt") + 1
            setattr(eval(f"self.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
        except AttributeError:
            setattr(eval(f"self.model.model.{self.layer}"), "editing", False)
            setattr(eval(f"self.model.model.{self.layer}"), "training", False)

            editing_total_cnt = getattr(eval(f"self.model.model.{self.layer}"), "editing_total_cnt") + 1
            setattr(eval(f"self.model.model.{self.layer}"), "editing_total_cnt", editing_total_cnt)
        #
        if self.config.save_freq is not None and editing_total_cnt % self.config.save_freq == 0:
            self.get_adapter_layer().save_weight()
            print(f'Add New Weight to Memory...')
        if editing_total_cnt % self.config.merge_freq == 0:
            # for retrieve ##
            merge_group_edit_history.append(edit_history)
            edit_history = []
            # for retrieve ##

            self.get_adapter_layer().merge_weight()
            print(f'Merge Weight of (New, Original) Matrix... with {self.config.merge_alg}')

        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()

    def __norm_constraint(self, norm_constraint):
        new_weight = self.get_adapter_layer().new_weight
        original_weight = self.get_adapter_layer().weight
        with torch.no_grad():
            new_weight[...] = torch.clamp(
                new_weight, min=original_weight - norm_constraint, max=original_weight + norm_constraint
            )

    def __cal_ft_loss(self, tokens, last_prompt_token_loc):
        k = 1
        bs = tokens["input_ids"].shape[0] - k
        logits = self.model(**tokens).logits
        shift_logits = logits[:-k, :-1, :].contiguous()
        shift_labels = tokens['labels'][:-k, 1:].contiguous()

        loss_fct = CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(bs, -1)

        label_mask = torch.zeros_like(loss, dtype=torch.bool)

        for i, col_index in enumerate(last_prompt_token_loc[:-k]):
            label_mask[i, col_index - 1:] = True

        ft_loss = ((loss * label_mask).sum(1) / label_mask.sum(1)).mean()
        return ft_loss

    def __cal_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        k = 1
        if act_mask is not None:
            in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config,
                                act_mask=act_mask)
            out_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config,
                                 act_mask=deact_mask)
        else:
            in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
            out_scope_dist = euc(original_layer_output[-k:, ...], new_weight_layer_output[-k:, ...], config)

        loss = out_scope_dist.view(-1,1) - in_scope_dist + config.gamma
        loss2 = out_scope_dist - config.alpha
        loss3 = config.beta - in_scope_dist
        loss3 = torch.mean(loss3[loss3 > 0]) if min(loss3[loss3 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
        loss2 = torch.mean(loss2[loss2 > 0]) if min(loss2[loss2 > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
        loss = torch.mean(loss[loss > 0]) if min(loss[loss > 0].size()) > 0 else torch.tensor(0.).to(original_layer_output.device)
        return loss + loss2 + loss3

    def __cal_memory_pos_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = 20 - in_scope_dist

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

    def __cal_memory_neg_activation_loss(self, original_layer_output, new_weight_layer_output, config=None, act_mask=None,
                              deact_mask=None):
        k = 1
        in_scope_dist = euc(original_layer_output[:-k, ...], new_weight_layer_output[:-k, ...], config)
        loss4 = in_scope_dist - 5

        return torch.mean(loss4[loss4 > 0]) if min(loss4[loss4 > 0].size()) > 0 else torch.tensor(0.)

class WISEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(WISEAdapter, self).__init__()
        # Gemma 3 (and other compiled models) wrap their MLP forward with torch.compile.
        # When they call self.down_proj(...) which is now this adapter, dynamo tries to
        # inline WISEAdapter.forward and crashes on the @torch._dynamo.disable wrapper.
        # suppress_errors=True makes dynamo fall back to eager instead of crashing.
        torch._dynamo.config.suppress_errors = True

        self.layer = layer
        self.weight = self.layer.weight
        self.device = layer.weight.device
        self.config = config
        self.new_weight = copy.deepcopy(self.weight)
        self.original_layer = copy.deepcopy(self.layer)
        self.memory_weight = []
        self.memory_mean_act = []
        self.merge_cnt = 0  # only for retrieve
        assert not self.weight.requires_grad, print('Original Layer can not be tunable....')

        self.used_mask = None 

        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False
        self.editing = False

        self.editing_mean_act = EditingMeanAct()
        self.editing_total_cnt = 0

    def set_parameter_tunable(self):
        self.new_weight.requires_grad = True

    def save_weight(self):
        self.memory_weight.append(copy.deepcopy(self.new_weight))
        self.new_weight = copy.deepcopy(self.original_layer.weight)
        if self.config.retrieve:
            self.memory_mean_act.append(copy.deepcopy(self.editing_mean_act))
            self.editing_mean_act = EditingMeanAct()

    def merge_weight(self):
        if self.config.save_freq is not None:  # for ties dare dare_ties
            if not self.config.retrieve:
                merge_alg = merge_dict[self.config.merge_alg]
                if self.original_layer.weight.equal(self.layer.weight):
                    cur_new_weight = merge_alg.execute([self.config.weights / len(self.memory_weight) for _ in range(len(self.memory_weight))], self.original_layer.weight, self.memory_weight, densities=self.config.densities)
                else:
                    cur_new_weight = merge_alg.execute([0.4 / len(self.memory_weight) for _ in range(len(self.memory_weight))] + [0.6], self.original_layer.weight, self.memory_weight + [self.layer.weight], densities=self.config.densities)
                self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                del self.memory_weight
                self.memory_weight = []
            else:
                merge_alg = merge_dict[self.config.merge_alg]
                merge_num = self.config.merge_freq // self.config.save_freq
                assert len(self.memory_weight) >= merge_num
                new_merge_weight = merge_alg.execute([self.config.weights / merge_num for _ in range(merge_num)], self.original_layer.weight, self.memory_weight[-merge_num:], densities=self.config.densities)
                min_a = 1e9
                for _ in range(merge_num):
                    self.memory_weight.pop()
                    edit_act = self.memory_mean_act.pop()
                    min_a = min(min_a, edit_act.min_act())
                self.new_weight = copy.deepcopy(self.original_layer.weight)
                self.memory_weight.append(new_merge_weight)
                self.memory_mean_act.append(EditingMeanAct(min_a=min_a))
                print(len(self.memory_weight))
                assert len(self.memory_mean_act) == len(self.memory_weight)
                self.merge_cnt += 1
        else:
            merge_alg = merge_dict[self.config.merge_alg]
            cur_new_weight = merge_alg.execute(0.5, self.layer.weight, [self.new_weight],
                                               densities=self.config.densities)
            self.layer.weight = torch.nn.Parameter(cur_new_weight.to(self.layer.weight.device), requires_grad=False)
            self.new_weight = copy.deepcopy(self.original_layer.weight)

    def save_editing_activation(self):
        in_scope_dist = euc(self.original_layer_output[:-1, ...], self.new_weight_layer_output[:-1, ...], self.config)
        self.editing_mean_act.update(in_scope_dist.mean().item())

    def generate_activation_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        p_mask = np.random.choice([1, 0], size=p_grad.size()[0], p=[mask_ratio, 1 - mask_ratio])
        p_mask = torch.from_numpy(p_mask).to(p_grad.device)
        self.weight_mask = p_mask

    def generate_non_overlapping_mask(self, mask_ratio):
        p_grad = self.new_weight.reshape(-1)
        mask_size = int(mask_ratio * p_grad.size()[0])
        if self.used_mask is None:
            self.used_mask = np.zeros(p_grad.size()[0], dtype=bool)
        available_indices = np.where(~self.used_mask)[0]  # 获取未被遮罩的元素索引
        if len(available_indices) < mask_size:
            raise ValueError("Not enough unused elements to generate a new mask.")
        chosen_indices = np.random.choice(available_indices, size=mask_size, replace=False)
        mask_array = np.zeros(p_grad.size()[0], dtype=int)
        mask_array[chosen_indices] = 1
        self.used_mask[chosen_indices] = True  # 更新遮罩状态
        self.weight_mask = torch.from_numpy(mask_array).to(p_grad.device)

    def new_weight_forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.new_weight)

    def mask_new_weight_gradient(self):
        assert self.new_weight.grad is not None, print('Gradient Collection for New Weight error, gradient not found')
        # Add gradient mask after the loss updates
        p_size = self.new_weight.grad.size()
        p_grad = self.new_weight.grad.reshape(-1)

        # mask = torch.from_numpy(np.random.choice([0, 1], size=p_grad.size()[0], p=[.1, .9])).cuda()
        p_grad = p_grad * self.weight_mask
        self.new_weight.grad = p_grad.view(p_size).to(self.new_weight.grad.dtype)

    @torch._dynamo.disable
    def forward(self, *args):
        if self.editing:
            layer_out = self.new_weight_forward(*args)
            self.new_weight_layer_output = layer_out
            self.original_layer_output = self.original_layer(*args)
        else:
            # Eval path — batch-safe: distances are computed per-item and
            # routing is done with torch.where (per-batch masks) instead of
            # Python `if/else` on a scalar. At bs=1 this is numerically
            # identical to the legacy branch; at bs>1 it routes per-item
            # which preserves WISE's intended semantics.
            if not self.config.retrieve:
                original_layer_output = self.original_layer(*args)
                layer_output = self.layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist2 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True, per_batch=True)  # [B]
                dist1 = euc(original_layer_output, layer_output,           self.config, infer=True, per_batch=True)  # [B]
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio

                # Per-item routing decisions:
                #   both close   → use original (base)
                #   dist1 > dist2 → use layer (edited)
                #   else         → use new_weight (candidate)
                both_close   = (dist1 < threshold) & (dist2 < threshold)           # [B]
                prefer_layer = (dist1 > dist2) & ~both_close                        # [B]
                # Broadcast [B] → [B,1,1] so torch.where works on [B,S,D] tensors.
                m_close = both_close.view(-1, 1, 1)
                m_layer = prefer_layer.view(-1, 1, 1)
                layer_out = torch.where(
                    m_close, original_layer_output,
                    torch.where(m_layer, layer_output, new_weight_layer_output),
                )
            else:
                original_layer_output = self.original_layer(*args)
                new_weight_layer_output = self.new_weight_forward(*args)
                dist1 = euc(original_layer_output, new_weight_layer_output, self.config, infer=True, per_batch=True)  # [B]
                threshold = self.editing_mean_act.min_act() * self.config.act_ratio

                # Start with per-item choice between original and new_weight.
                below_th = (dist1 < threshold).view(-1, 1, 1)                        # [B,1,1]
                layer_out = torch.where(below_th, original_layer_output, new_weight_layer_output)
                min_dist = dist1                                                     # [B]

                # For each stored memory, update per-item when that memory's
                # distance exceeds both the current best and the memory's own
                # activation threshold.
                for i in range(len(self.memory_weight)):
                    memory_retrieve_weight = self.memory_weight[i]
                    memory_weight_layer_output = F.linear(*args, memory_retrieve_weight)
                    dist = euc(original_layer_output, memory_weight_layer_output, self.config, infer=True, per_batch=True)  # [B]
                    mem_thr = self.memory_mean_act[i].min_act() * self.config.act_ratio
                    take = (dist > min_dist) & (dist > mem_thr)                      # [B]
                    layer_out = torch.where(take.view(-1, 1, 1), memory_weight_layer_output, layer_out)
                    min_dist = torch.where(take, dist, min_dist)
        return layer_out