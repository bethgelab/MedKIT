"""
Shared training utilities for knowledge-editing methods.
"""
import torch


def build_tokens_dict(tok, texts, targets, max_length, device):
    """Tokenize (prompt, target) pairs for causal-LM training.

    When the tokenizer has a chat template (instruction-tuned models) the
    prompt is formatted as a user turn and the target as the assistant turn so
    that training matches the inference-time format.  Loss is masked on the
    prompt tokens; only the assistant answer tokens contribute to the NLL.

    For base models without a chat template the original concatenation
    approach is preserved exactly.

    Returns a dict with keys ``input_ids``, ``attention_mask``, ``labels``,
    all already moved to *device*.
    """
    mask_token = -100

    if getattr(tok, 'chat_template', None) is not None:
        full_ids_list, prompt_lens = [], []
        for text, target in zip(texts, targets):
            p_ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = tok.apply_chat_template(
                [{"role": "user", "content": text},
                 {"role": "assistant", "content": target}],
                tokenize=True,
                add_generation_prompt=False,
            )
            # Some chat templates BPE-merge the trailing prompt token with the
            # leading target token (e.g. ContactDoctor/Bio-Medical-Llama-3-8B's
            # Human:/Assistant: template merges ' ' + 'superior' -> ' superior'),
            # so len(p_ids) overshoots the assistant-content boundary in
            # full_ids and silently masks all label tokens (loss=0). Use the
            # longest common prefix of p_ids and full_ids instead.
            lcp = 0
            while (lcp < len(p_ids) and lcp < len(full_ids)
                   and p_ids[lcp] == full_ids[lcp]):
                lcp += 1
            prompt_lens.append(lcp)
            full_ids_list.append(full_ids)

        max_len = min(max(len(x) for x in full_ids_list), max_length)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        B = len(full_ids_list)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), mask_token, dtype=torch.long)
        for i, (ids, p_len) in enumerate(zip(full_ids_list, prompt_lens)):
            # Truncate from the left to preserve the answer tokens
            ids = ids[-max_len:]
            p_len = max(0, p_len - (len(full_ids_list[i]) - len(ids)))
            start = max_len - len(ids)  # left-pad
            input_ids[i, start:] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, start:] = 1
            labels[i, start + p_len:] = torch.tensor(ids[p_len:], dtype=torch.long)

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "labels": labels.to(device),
        }

    else:
        full_prompt = [f"{p} {l}" for p, l in zip(texts, targets)]
        prompt_ids = tok(
            list(texts), return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )["input_ids"]
        num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
        tokens = tok(
            full_prompt, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        tokens["labels"] = tokens["input_ids"].clone()
        num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
        for i in range(len(texts)):
            tokens["labels"][i][num_pad_toks[i]: num_pad_toks[i] + num_prompt_toks[i]] = mask_token
        tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = mask_token
        return {k: v.to(device) for k, v in tokens.items()}
