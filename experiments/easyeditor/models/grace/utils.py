import transformers
import torch
import os
import numpy as np
import datetime
import struct
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def get_inner_params(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [(n, param_dict[n]) for n in inner_names]

def param_subset(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names]

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

def uuid(digits=4):
    if not hasattr(uuid, "uuid_value"):
        uuid.uuid_value = struct.unpack('I', os.urandom(4))[0] % int(10**digits)

    return uuid.uuid_value

def ckpt_dir():
    """returns the directory in which to store model checkpoints"""
    path = "./ckpts/"
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")
    
def get_params(model):
    return model.state_dict()

def get_shape(p, model): 
    # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
    return p.shape if isinstance(model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

def get_logits(x):
    return x.logits if hasattr(x, "logits") else x

def tokenize(batch, tokenizer, device, test=False):
    prompt, label = batch["prompt"], batch["target_new"]
    if not isinstance(prompt, list):
        prompt=[prompt]
    if not isinstance(label, list):
        label=[label]
    mask_token = -100 # ignore_index of CrossEntropyLoss
    if test or not label:
        tokens = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)
        tokens["labels"] = tokens["input_ids"].clone()
        tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    else:
        if getattr(tokenizer, 'chat_template', None) is not None:
            # Use apply_chat_template(tokenize=True) for BOTH prompt_ids and
            # full tokens so the tokenization exactly matches the eval path
            # (test_prediction_acc uses tokenize=True in one shot). The earlier
            # two-step approach (apply_chat_template(tokenize=False) then
            # tokenizer(...)) produces a different token sequence — specifically
            # it strips the trailing "\n\n" after the assistant header, giving
            # 190 tokens instead of 191 for a typical HemOnc prompt on Llama 3.1.
            # Result: edit-time keys are stored at the <|end_header_id|> token,
            # eval-time queries are computed at the "\n\n" token → different
            # hidden states → smallest_dist ≈ 10 at eval → no firing.
            prompt_ids_list = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True, add_generation_prompt=True,
                    return_tensors="pt", padding=False, truncation=True,
                )[0]
                for p in prompt
            ]
            full_ids_list = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p},
                     {"role": "assistant", "content": l}],
                    tokenize=True, add_generation_prompt=False,
                    return_tensors="pt", padding=False, truncation=True,
                )[0]
                for p, l in zip(prompt, label)
            ]
            num_prompt_toks = [int(pi.shape[0]) for pi in prompt_ids_list]

            # Right-pad to a common length so we can stack into a batch tensor.
            max_len = max(int(fi.shape[0]) for fi in full_ids_list)
            pad_id = tokenizer.pad_token_id
            if pad_id is None:
                pad_id = tokenizer.eos_token_id
            input_ids = torch.full((len(full_ids_list), max_len), pad_id, dtype=full_ids_list[0].dtype)
            attention_mask = torch.zeros((len(full_ids_list), max_len), dtype=torch.long)
            for i, fi in enumerate(full_ids_list):
                n = int(fi.shape[0])
                input_ids[i, :n] = fi
                attention_mask[i, :n] = 1
            tokens = {"input_ids": input_ids, "attention_mask": attention_mask}
        else:
            full_prompt = [f"{p} {l}" for p, l in zip(prompt, label)]
            prompt_ids = tokenizer(list(prompt), return_tensors="pt", padding=True, truncation=True)["input_ids"]
            num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in prompt_ids]
            tokens = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)

        tokens["labels"] = tokens["input_ids"].clone()
        for i in range(len(prompt)):
            tokens["labels"][i][:num_prompt_toks[i]] = mask_token

        # Mask pad positions via attention_mask (1 for valid, 0 for pad).
        # The old `input_ids == pad_token_id` check is unsafe when pad_token_id
        # equals a legitimate special token appearing in the response — e.g.
        # Llama 3.1 Instruct sets pad_token = eos_token = <|eot_id|>, and the
        # assistant message ends with <|eot_id|>. That legitimate token was
        # being masked, inflating (labels==-100).sum() by 1 and shifting
        # GRACE.edit()'s key_id off the true last-prompt position.
        if "attention_mask" in tokens:
            tokens["labels"][tokens["attention_mask"] == 0] = mask_token
        else:
            tokens["labels"][tokens["input_ids"] == tokenizer.pad_token_id] = mask_token

    tokens = {f"{k1}" : v1.to(device) for k1, v1 in tokens.items()}
    return tokens

