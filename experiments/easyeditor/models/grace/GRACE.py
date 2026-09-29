import copy

import torch
from .utils import parent_module, brackets_to_periods
import transformers
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def euc(query, key):
    # Euclidean distance
    if len(key.shape) < 2:
        key = key.view(1, -1)
    return torch.cdist(key, query, p=2)

def perturb_values(chosen_value, num_pert, device):
    # Create a bunch of noised versions of the value, then create batch, then train value
    chosen_value = chosen_value
    noise = torch.normal(0, 1, chosen_value.shape, device=device)
    noise[0] = noise[0]*0
    noise.requires_grad = True
    chosen_value = chosen_value + noise
    return chosen_value

class GRACE(torch.nn.Module):
    def __init__(self, config, model, device):
        super(GRACE, self).__init__()
        self.config = config
        self.log_dict = {}
        self.model = model
        self.config = config
        # self.tokenizer = model.tokenizer
        layer = config.inner_params[0]
        self.device = device
        self.original_layer = None

        # --- ensure proper formatting (GRACE edits ~layers~ not weights matrices) ---        
        suffixes = [".weight", ".bias"]
        self.layer = layer.rsplit(".", 1)[0] if any(layer.endswith(x) for x in suffixes) else layer
        
        for n, p in self.model.named_parameters():
            p.requires_grad = False
        
        if isinstance(self.model, transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel):
            transpose = False
        else:
            transpose = True

        # --- Add GRACE to chosen layers ---
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        original_layer = getattr(edit_module, layer_name)
        if type(original_layer) is not GRACEAdapter:
            setattr(edit_module, layer_name, GRACEAdapter(config, original_layer, transpose=transpose).to(self.device))
            self.original_layer = copy.deepcopy(original_layer)
        
    def __call__(self, **kwargs):
        # if self.config.task == "hallucination":
        #     print(kwargs)
        #     key_id = (kwargs["labels"] == -100).sum() - 1
        #     setattr(eval(f"self.model.{self.layer}"), "key_id", key_id) # Tell GRACE which token to use for its query (default is the last token)
        return self.model(**kwargs)

    def reset_layer(self):
        layer_name = self.layer.rsplit(".", 1)[-1]
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        setattr(edit_module, layer_name, self.original_layer.to(self.device))

    def generate(self, *args, **kwargs):
        setattr(eval(f"self.model.{self.layer}"), "key_id", -1)
        return self.model.generate(*args, **kwargs)

    def rolllback(self,edit_id):
        layer = eval(f"self.model.{self.layer}")
        layer.delete_key(edit_id)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    # GRACE's codebook (keys/epsilons/key_labels/edit_ids) lives as plain Python
    # attributes on the GRACEAdapter, NOT as registered Parameters/buffers — so
    # the default `state_dict()` save loses it entirely.  `values` IS registered
    # (it's created as nn.Parameter inside forward()), but at load time the
    # freshly-wrapped adapter has no `values` attribute yet, so a generic
    # `load_state_dict(strict=True)` would fail too.
    #
    # GRACE never mutates the underlying HF layer's weight (its adapter is
    # purely an output-substitution layer: `layer_out = self.layer(*args)` then
    # optionally swaps in chosen_value), so we don't need to save the base HF
    # model state_dict — only the codebook is required to resume.
    #
    # `save_model` writes a single `<path>.pt` containing the codebook tensors
    # and lists.  `load_model` (via `load_grace_into_model`) restores them onto
    # an already-wrapped fresh model.
    def _get_adapter(self):
        edit_module = parent_module(self.model, brackets_to_periods(self.layer))
        layer_name = self.layer.rsplit(".", 1)[-1]
        adapter = getattr(edit_module, layer_name)
        assert type(adapter) is GRACEAdapter, \
            f'GRACE adapter not present at {self.layer} (got {type(adapter).__name__})'
        return adapter

    def save_model(self, path: str):
        adapter = self._get_adapter()
        # Codebook may be empty if no edits have happened yet (first checkpoint
        # before any add_key call).  Persist a sentinel so load_model knows the
        # file is well-formed but the codebook is uninitialised.
        has_codebook = ('keys' in adapter.__dict__
                        and adapter.keys is not None
                        and adapter.keys.nelement() > 0)
        extras = {
            'has_codebook': has_codebook,
            'key_id': int(getattr(adapter, 'key_id', -1)),
        }
        if has_codebook:
            extras.update({
                'keys': adapter.keys.detach().cpu(),
                'values': adapter.values.detach().cpu(),
                'epsilons': adapter.epsilons.detach().cpu(),
                # Labels are tensors with -100 mask; persist as-is.
                'key_labels': [k.detach().cpu() for k in adapter.key_labels],
                # edit_ids are usually strings (the target_new text); pickled as Python list.
                'edit_ids': list(adapter.edit_ids),
            })
        state = {
            'format_version': 1,
            'editing_method': 'GRACE',
            'layer': self.layer,
            'extras': extras,
        }
        torch.save(state, path + '.pt')
        n_keys = extras['keys'].shape[0] if has_codebook else 0
        print(f'[GRACE] Saved checkpoint to {path}.pt (codebook size={n_keys})')

    def load_model(self, path: str):
        try:
            state = torch.load(path + '.pt',
                               map_location=self.device,
                               weights_only=False)
        except TypeError:
            state = torch.load(path + '.pt', map_location=self.device)

        if not isinstance(state, dict) or state.get('editing_method') != 'GRACE':
            raise RuntimeError(
                f'GRACE.load_model: file at {path}.pt is not a GRACE checkpoint '
                f'(got format={type(state).__name__}, '
                f'method={state.get("editing_method") if isinstance(state, dict) else "n/a"}). '
                'If this checkpoint was written by the legacy save path '
                '(before the resume fix), delete it and re-run from scratch.'
            )

        adapter = self._get_adapter()
        extras = state['extras']
        adapter.key_id = int(extras.get('key_id', -1))
        if not extras.get('has_codebook', False):
            print(f'[GRACE] Loaded {path}.pt — empty codebook (no edits yet)')
            return

        # Restore the codebook directly on the adapter.  Keys/epsilons are
        # plain tensors (mirrors how add_key/init_key_value assign them);
        # values is rebuilt as an nn.Parameter so future edits can extend it.
        device = adapter.device
        keys     = extras['keys'].to(device)
        values   = extras['values'].to(device)
        epsilons = extras['epsilons'].to(device)
        adapter.keys     = keys
        adapter.values   = torch.nn.Parameter(values, requires_grad=True)
        adapter.epsilons = epsilons
        adapter.key_labels = [k.to(device) for k in extras.get('key_labels', [])]
        adapter.edit_ids   = list(extras.get('edit_ids', []))
        print(f'[GRACE] Loaded checkpoint from {path}.pt '
              f'(codebook size={keys.shape[0]})')
          
    def edit(self, config, tokens, edit_id):
        key_id = (tokens["labels"] == -100).sum() - 1
        setattr(eval(f"self.model.{self.layer}"), "key_id", key_id)

        # Diagnostic: log the edit-time input_ids so we can compare against
        # the eval-time input_ids for the same prompt. Same tokens at same
        # positions → same hidden states → queries should match.
        if getattr(config, "grace_debug", False):
            ids = tokens["input_ids"][0].detach().cpu().tolist()
            _k = int(key_id.item()) if hasattr(key_id, "item") else int(key_id)
            print(
                f"[EDIT-IDS] edit_id={edit_id!r} key_id={_k} "
                f"total_len={len(ids)} first5={ids[:5]} "
                f"around_key={ids[max(0,_k-3):_k+2]} last5_prompt={ids[max(0,_k-4):_k+1]}",
                flush=True,
            )

        # --- pass edit label, training mode, and key_id into GRACE ---
        setattr(eval(f"self.model.{self.layer}"), "training", True)
        setattr(eval(f"self.model.{self.layer}"), "edit_label", tokens["labels"])
        setattr(eval(f"self.model.{self.layer}"), "edit_id", edit_id)

        self.losses = []
        # Actions that don't add a new value to self.values. In these cases the
        # iter-0 forward only mutates epsilons (or makes no change at all), so
        # there's no fresh trainable parameter to optimize and the 100-iter
        # training loop is both pointless and unsafe — loss.backward() fails
        # with "element 0 of tensors does not require grad" when the loss path
        # doesn't reach a parameter that was registered between the optimizer
        # being created and backward(). For these cases the existing key+value
        # pair already targets the same label, so the expand-only edit IS the
        # full edit.
        _NO_TRAIN_ACTIONS = {
            "expand_eps_coverage", "expand_eps_moving_avg",
            "label_match_in_eps", "label_match_no_expand",
        }

        # --- train GRACE value ---
        for i in range(config.n_iter):
            # --- insert iteration into each layer (only initiate keys on iteration 1) ---
            setattr(eval(f"self.model.{self.layer}"), "iter", i)

            # --- pass tokens through model (including through the GRACE layer) ---
            outputs = self.model(**tokens)
            if i == 0:
                _action = getattr(eval(f"self.model.{self.layer}"), "_last_action", None)
                if _action in _NO_TRAIN_ACTIONS:
                    # No new value was added — nothing to train this edit.
                    self.losses.append(outputs.loss.detach().cpu().numpy())
                    break
                # --- we only need to create an optimizer for the first iteration (but forward pass instantiates the key, so optimzer is passed after first inference) ---
                optimizer = torch.optim.Adam(self.model.parameters(), config.edit_lr)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            self.losses.append(loss.detach().cpu().numpy())

        self.loss = outputs.loss  # Log final loss (works for both normal and early-break paths)

        # --- pull out info we want to log from the GRACE layer ---
        setattr(eval(f"self.model.{self.layer}"), "training", False)
        adapter = eval(f"self.model.{self.layer}")
        chosen_key = getattr(adapter, "chosen_key")
        nkeys = len(getattr(adapter, "keys"))

        self.log_dict["chosen_key"] =  chosen_key
        self.log_dict["nkeys"] = nkeys

        # --- diagnostic: per-edit codebook stats (cascade probe) ---
        if getattr(config, "grace_debug", False):
            eps = adapter.epsilons.detach().float().flatten().cpu()
            action = getattr(adapter, "_last_action", "unknown")
            sd = getattr(adapter, "_dbg_smallest_distance", None)
            nk = getattr(adapter, "_dbg_nearest_key", None)
            lm = getattr(adapter, "_dbg_label_match", None)
            lm_str = "n/a" if lm is None else ("True" if lm else "False")
            sd_str = f"{float(sd):.4f}" if sd is not None else "n/a"
            # Edit-time query fingerprint — saved by forward() before any
            # add_key / expand_eps decision. Lets us directly compare against
            # the eval-time query for the same prompt (should be ~identical
            # if tokenization/model state match).
            last_q = getattr(adapter, "_dbg_last_query", None)
            if last_q is not None:
                last_q = last_q.float().flatten().cpu()
                qfp = last_q[:5].tolist()
                qnorm = float(last_q.norm().item())
            else:
                qfp, qnorm = [], 0.0

            # Post-training argmax at key position: if training did its job,
            # the model (with the trained value injected) should now predict
            # the target's first token at logits[key_id]. A mismatch means
            # the loss is misleading us or the injection path is broken.
            try:
                with torch.no_grad():
                    out = self.model(**tokens)
                    _k = int(key_id.item()) if hasattr(key_id, "item") else int(key_id)
                    logits_at_key = out.logits[0, _k, :]
                    top5 = torch.topk(logits_at_key, 5)
                    top5_ids = top5.indices.cpu().tolist()
                    top5_vals = top5.values.cpu().tolist()
                    true_next = int(tokens["input_ids"][0, _k + 1].item())
            except Exception as e:
                top5_ids, top5_vals, true_next = [], [], -1

            print(
                f"[GRACE-DEBUG] edit_id={edit_id!r} loss={float(self.loss):.4f} "
                f"action={action} nkeys={nkeys} "
                f"eps[min/mean/max]={eps.min():.4f}/{eps.mean():.4f}/{eps.max():.4f} "
                f"smallest_dist={sd_str} nearest_key={nk} label_match={lm_str} "
                f"init_eps={config.eps} "
                f"qnorm={qnorm:.3f} qfp={['{:.4f}'.format(x) for x in qfp]} "
                f"post_train_top5={top5_ids} true_next={true_next}",
                flush=True,
            )

class GRACEAdapter(torch.nn.Module):
    def __init__(self, config, layer, transpose):
        super(GRACEAdapter, self).__init__()
        # Gemma 3 (and other compiled models) wrap their MLP forward with torch.compile.
        # When they call self.down_proj(...) which is now this adapter, dynamo tries to
        # inline GRACEAdapter.forward and crashes on the @torch._dynamo.disable wrapper.
        # suppress_errors=True makes dynamo fall back to eager instead of crashing.
        torch._dynamo.config.suppress_errors = True

        self.layer = layer
        self.weight = self.layer.weight
        self.init_epsilon = config.eps
        self.dist_fn = config.dist_fn
        self.replacement = config.replacement
        self.device = layer.weight.device
        self.config = config
        self.num_pert = config.num_pert
        self.key_id = -1
        self.ensure_replace_token_loc = False
    
        if transpose:
            self.key_shape = layer.weight.shape[1]
            self.value_shape = layer.weight.shape[0]
        else:
            self.key_shape = layer.weight.shape[0]
            self.value_shape = layer.weight.shape[1]
        self.training = False

    def add_key(self, new_key, new_value, new_edit_id):
        keys = torch.vstack([self.keys, new_key.detach()]) # Add new key to list of keys
        values = torch.nn.Parameter(torch.vstack([self.values, new_value]), requires_grad=True) # Add new value to list of values
        new_epsilon = torch.tensor(self.init_epsilon, device=self.device).view(1)
        if self.epsilons.nelement() == 0:
            epsilons = new_epsilon
        else:
            epsilons = torch.vstack([self.epsilons, new_epsilon]) # Add new epsilon to list of epsilons
        # `keys` is built by appending new_key at the end (vstack), so key_labels
        # and edit_ids must do the same — otherwise key_labels[i] no longer
        # corresponds to keys[i], breaking the label_match check at iter==0.
        # Original code prepended new_label/new_id, which silently misaligned the
        # codebook after the first add and made every subsequent label_match call
        # compare against the wrong stored label.
        key_labels = self.key_labels + [self.edit_label]
        edit_ids = self.edit_ids + [new_edit_id]
        return keys, values, epsilons, key_labels, edit_ids
    
    
    def delete_key(self,edit_id):
        if 'keys' not in self.__dict__ or self.edit_ids==[]:
            print("no keys")
            return
        if edit_id in self.edit_ids:
            index_to_remove = self.edit_ids.index(edit_id)
            self.keys = torch.cat((self.keys[:index_to_remove], self.keys[index_to_remove+1:]), dim=0)
            self.values = torch.nn.Parameter(torch.cat((self.values[:index_to_remove], self.values[index_to_remove+1:]), dim=0), requires_grad=True)
            self.epsilons = torch.cat((self.epsilons[:index_to_remove], self.epsilons[index_to_remove+1:]), dim=0)
            self.key_labels = self.key_labels[:index_to_remove] + self.key_labels[index_to_remove+1:]
            self.edit_ids = self.edit_ids[:index_to_remove] + self.edit_ids[index_to_remove+1:]
            print(self.keys.shape,self.values.shape,self.epsilons.shape,len(self.key_labels),len(self.edit_ids))
        else:
            print("not found")
    
    def init_key_value(self, query, value):
        key = query.detach()
        epsilon = torch.tensor(self.init_epsilon, device=self.device, requires_grad=False).view(1)
        key_label = [self.edit_label]
        edit_ids = [self.edit_id]
        return key, value, epsilon, key_label, edit_ids

    def label_match(self, edit_label, key_label):
        # Compare the meaningful (non-masked) target tokens. The label tensors
        # use -100 to mask the prompt portion; comparing the mean of the full
        # tensor (the original implementation) is dominated by -100 and
        # diverges between two prompts of different lengths even when they
        # encode the same target. Empirically this caused the cascade probe
        # to report label_match=False on every conflict for Llama, including
        # edits that share the exact same target token.
        e = edit_label[edit_label != -100]
        k = key_label[key_label != -100]
        if e.numel() != k.numel():
            return False
        return bool(torch.equal(e, k))

    def split_epsilons_in_half(self, nearest_key, smallest_distance):
        eps_dtype = self.epsilons.dtype
        self.epsilons[nearest_key] = ((smallest_distance / 2) - 1e-5).to(eps_dtype)
        self.epsilons[-1] = (smallest_distance / 2).to(eps_dtype)
    
    @torch._dynamo.disable
    def forward(self, *args):
        # Run layer forward and save what it would have returned for this instance
        layer_out = self.layer(*args)

        ### If training, we need to modify the codebook
        if (not self.training) & ('keys' not in self.__dict__):
            # If it's not training time and we haven't added any keys yet (this is before doing any editing)
            # print(self.__dict__)
            return layer_out
        else:
            if not self.training:
                if self.key_id == -1:
                    token_to_edit = args[0].shape[1] - 1
                    self.key_id = args[0].shape[1] - 1
                else:
                    token_to_edit = min(self.key_id, args[0].shape[1] - 1)
            else:
                token_to_edit = min(self.key_id, args[0].shape[1] - 1)  # args[0].shape[1] - 1 is sequence length
            query = args[0][:, token_to_edit, :] # Just use activation for last token
            # Save the latest query for diagnostics so GRACE.edit() can print
            # the *current* edit's query fingerprint (regardless of whether
            # this edit ended up calling add_key).
            self._dbg_last_query = query.detach()
            if self.config.val_init == "cold":
                new_value = torch.nn.Parameter(torch.rand(1, self.value_shape, requires_grad=True, device=self.device))
            elif self.config.val_init == "warm":
                new_value = torch.nn.Parameter(layer_out[:, token_to_edit, :].detach(), requires_grad=True)

            if self.training and ('keys' not in self.__dict__ or self.keys.nelement() == 0):
                # If no keys exist, initialize keys, values, epsilons, and key labels
                self.keys, self.values, self.epsilons, self.key_labels, self.edit_ids = self.init_key_value(query, new_value)
                self._last_action = "init"
                self._dbg_smallest_distance = None
                self._dbg_nearest_key = None
                self._dbg_label_match = None
            elif self.training and self.iter == 0:
                # Conflict resolution must only run during edit() — gating on
                # `training` prevents eval-time forwards (model.generate) from
                # mutating the codebook. Without this guard, every generated
                # token at eval time re-enters this branch (since edit() sets
                # iter=0 last and may exit early without advancing it),
                # comparing against the stale `self.edit_label` from the
                # previous edit and silently add_key-ing eval queries.
                # Keys exist, so we have decide whether or not to update them (the fact that we've made it to this point means there was an error!)

                # --- search through keys for a match for query ---
                dists = torch.cdist(self.keys.float(), query.float(), p=2).to(self.keys.dtype).view(-1, len(query))
                smallest_distance, nearest_key = dists.min(0)
                self._dbg_smallest_distance = smallest_distance.item()
                self._dbg_nearest_key = int(nearest_key.item())
                self._dbg_label_match = None

                if smallest_distance > (self.init_epsilon + self.epsilons[nearest_key]):
                    # If there's no close key, make a new key
                    self.keys, self.values, self.epsilons, self.key_labels, self.edit_ids = self.add_key(query, new_value,self.edit_id)
                    self._last_action = "add_new_far"
                else:
                    # If there is a close key, we need to handle conflicts
                    _label_match = self.label_match(self.edit_label, self.key_labels[nearest_key])
                    self._dbg_label_match = bool(_label_match)
                    if not _label_match:
                        self.keys, self.values, self.epsilons, self.key_labels, self.edit_ids = self.add_key(query, new_value,self.edit_id)
                        self.split_epsilons_in_half(nearest_key, smallest_distance)
                        self._last_action = "add_new_split_eps"
                    else:
                        # If the current label is the SAME as the nearest label, just make the nearest epsilon bigger
                        if smallest_distance > self.epsilons[nearest_key]:
                            if self.config.eps_expand== "coverage":
                                self.epsilons[nearest_key] = smallest_distance.to(self.epsilons.dtype)
                                self._last_action = "expand_eps_coverage"
                            elif self.config.eps_expand == "moving_average":
                                a = 0.5
                                self.keys[nearest_key] = a*self.keys[nearest_key] + (1-a)*query # Move old key to be halfway between
                                self.epsilons[nearest_key] = smallest_distance.to(self.epsilons.dtype)
                                # self.epsilons[nearest_key] = smallest_distance + self.init_epsilon
                                self._last_action = "expand_eps_moving_avg"
                            else:
                                self._last_action = "label_match_no_expand"
                        else:
                            self._last_action = "label_match_in_eps"
            else:
                # If not iter 0, we don't need to change keys, we just need to learn the value
                pass
        # print(token_to_edit)
        # compute distance from query to all keys and find the closest keys

        dists = torch.cdist(self.keys.float(), query.float(), p=2).to(self.keys.dtype).view(-1, len(query))
        if dists.nelement() == 0:
            return layer_out
        smallest_dist, self.chosen_key = dists.min(0)
        smallest_dist = smallest_dist.view(-1, 1)
        chosen_value = self.values[self.chosen_key]
        eps = self.epsilons[self.chosen_key].view(-1, 1)

        # Diagnostic: log eval-time retrieval stats (gated on grace_debug).
        # Logs only the first forward of each generate() call (token_to_edit ==
        # seq_len - 1), which is the only forward where retrieval actually
        # matters for the prediction. Includes a fingerprint of the query
        # vector (first 5 components) so we can compare against the edit-time
        # query at the same position and see where the divergence comes from.
        if (not self.training) and getattr(self.config, "grace_debug", False):
            _is_first = (token_to_edit == args[0].shape[1] - 1)
            if _is_first:
                _fire = bool((smallest_dist <= eps).all().item())
                _qfp = query[0, :5].detach().float().cpu().tolist()
                _qnorm = float(query[0].detach().float().norm().item())
                print(
                    f"[GRACE-EVAL] seq_len={args[0].shape[1]} token_to_edit={token_to_edit} "
                    f"chosen_key={int(self.chosen_key.item())} "
                    f"smallest_dist={float(smallest_dist.item()):.4f} "
                    f"eps[chosen]={float(eps.item()):.4f} fired={_fire} "
                    f"qnorm={_qnorm:.3f} qfp={['{:.4f}'.format(x) for x in _qfp]}",
                    flush=True,
                )

        if (self.config.val_train == "adv") and (self.training):
            chosen_value = perturb_values(chosen_value, self.num_pert, self.device)

        if self.replacement == "replace_all":
            layer_out = torch.where((smallest_dist <= eps).view(-1, 1, 1), chosen_value.unsqueeze(1).repeat_interleave(layer_out.shape[1], 1), layer_out)
        elif self.replacement == "replace_last":
            layer_out[:, token_to_edit] = torch.where((smallest_dist <= eps), chosen_value, layer_out[:, token_to_edit])
        elif self.replacement == "replace_prompt":
            layer_out[:, :token_to_edit] = torch.where((smallest_dist <= eps), chosen_value, layer_out[:, :token_to_edit])
        else:
            print("token replacement choice not found")
        return layer_out
