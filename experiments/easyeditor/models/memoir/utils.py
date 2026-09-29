"""
Utility functions for the MEMOIR framework. Mostly adapted from the WISE framework.
"""

import transformers
import torch
import os

CONTEXT_TEMPLATES_CACHE = None


def find_sublist_start_index(list1, list2):
    for i in range(len(list1) - len(list2) + 1):
        if all(a == b for a, b in zip(list1[i:i + len(list2)], list2)):
            return i
    return None


def parent_module(model, pname):
    components = pname.split('.')
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent


def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")


def tokenize(batch, tokenizer, device, context_templates=None, hparams=None):
    len_temp = len(context_templates)
    prompts = [item['prompt'] for item in batch]
    labels = [item['target_new'] for item in batch]
    # Fall back to 'prompt' if 'loc_prompt' is not in the request
    loc_prompts = [item.get('loc_prompt', item.get('subject', item['prompt'])) for item in batch]

    mask_token = -100  # ignore_index of CrossEntropyLoss
    if getattr(tokenizer, 'chat_template', None) is not None:
        full_prompt = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": templ.format(p)},
                 {"role": "assistant", "content": l}],
                tokenize=False, add_generation_prompt=False,
            )
            for p, l in zip(prompts, labels) for templ in context_templates
        ]
        prompt_strings = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": templ.format(p)}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts for templ in context_templates
        ]
        prompt_ids = tokenizer(prompt_strings, return_tensors="pt", padding=True, truncation=True)["input_ids"]
    else:
        full_prompt = [f"{templ.format(p + ' ' + l)}" for p, l in zip(prompts, labels) for templ in context_templates]
        prompt_ids = tokenizer([f"{templ.format(p)}" for p in prompts for templ in context_templates],
                               return_tensors="pt", padding=True, truncation=True)["input_ids"]

    full_prompt += loc_prompts  # add for subject activation

    tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()

    num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]

    if hparams.objective_optimization == 'only_label':
        for i in range(len(num_prompt_toks)):
            input_ids_i = tokens["input_ids"][i]
            # Find the first non-padding token to handle left-padded sequences correctly.
            # For left-padded sequences, labels[:num_prompt_toks[i]] would mask the wrong
            # positions (padding tokens + only the first few prompt tokens). We must find
            # the actual content start and mask from there through the prompt end.
            is_not_pad = (input_ids_i != tokenizer.pad_token_id)
            first_real_tok = is_not_pad.nonzero(as_tuple=True)[0][0].item() if is_not_pad.any() else 0
            prompt_end = first_real_tok + num_prompt_toks[i]
            tokens["labels"][i][:prompt_end] = mask_token

    tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    # Left-truncate AFTER labels are fully set so all three tensors stay aligned.
    # Truncate from the left (drop context-template prefix) to preserve the
    # answer tokens at the right end — right-truncation would silently discard
    # the labels and corrupt training.
    seq_len = tokens["input_ids"].shape[1]
    if hasattr(hparams, 'max_length') and seq_len > hparams.max_length:
        print(f"[MEMOIR] Truncating training sequences from {seq_len} → "
              f"{hparams.max_length} tokens (left-truncation to preserve labels).")
        tokens["input_ids"]      = tokens["input_ids"][:, -hparams.max_length:]
        tokens["attention_mask"] = tokens["attention_mask"][:, -hparams.max_length:]
        tokens["labels"]         = tokens["labels"][:, -hparams.max_length:]
    act_masks = []
    deact_masks = []
    for i, loc_prompt in enumerate(loc_prompts):
        if loc_prompt in prompts[i]:  # subject: Factual Editing
            subject_token = tokenizer.encode(' ' + loc_prompt, add_special_tokens=False)
            subject_token1 = tokenizer.encode(loc_prompt, add_special_tokens=False)
            subject_length = len(subject_token)
            act_mask = torch.zeros_like(tokens['input_ids'][int(i * len_temp):int((i + 1) * len_temp)])
            deact_mask = torch.zeros_like(tokens['input_ids'][int(i * len_temp):int((i + 1) * len_temp)])
            for j, token in enumerate(tokens['input_ids'][int(i * len_temp):int((i + 1) * len_temp)]):
                start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token)
                if start_idx is None:
                    start_idx = find_sublist_start_index(token.detach().cpu().numpy().tolist(), subject_token1)
                    subject_length = len(subject_token1)
                act_mask[j][start_idx: start_idx + subject_length] = 1
                deact_mask[j][:start_idx] = 1
                deact_mask[j][start_idx + subject_length:] = 1
        else:  # General Editing
            act_mask = None
            deact_mask = None

        act_masks.append(act_mask)
        deact_masks.append(deact_mask)

    act_masks = [mask.to(device) if mask is not None else None for mask in act_masks]
    deact_masks = [mask.to(device) if mask is not None else None for mask in deact_masks]

    tokens = {key: val.to(device) for key, val in tokens.items()}

    return tokens, act_masks, deact_masks


class EarlyStopMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.pre = 0
        self.val = 1e9
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.pre = self.val
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def stop(self):
        return abs(self.val - self.pre) <= 1e-4 and self.val <= 0.02


def get_context_templates(model, tok, length_params, device):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = []
        prompt_tok = tok(
            ["I", "You", "Because", 'Yes', 'Q: '],
            padding=True,
            return_tensors="pt"
        ).to(device)
        for length, n_gen in length_params:
            gen_token = model.generate(
                input_ids=prompt_tok['input_ids'],
                attention_mask=prompt_tok['attention_mask'],
                max_new_tokens=length,
                num_beams=n_gen // 5,
                num_return_sequences=n_gen // 5,
                pad_token_id=tok.eos_token_id,
            )
            CONTEXT_TEMPLATES_CACHE += tok.batch_decode(gen_token, skip_special_tokens=True)
        CONTEXT_TEMPLATES_CACHE = ['{}'] + [_ + ' {}' for _ in CONTEXT_TEMPLATES_CACHE]

    return CONTEXT_TEMPLATES_CACHE
