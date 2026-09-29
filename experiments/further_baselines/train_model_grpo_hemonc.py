# --- import bootstrap ------------------------------------------------------
# `hemonc_batching` / `hemonc_eval_protocol` live one directory up (experiments/).
# Add that dir to sys.path so the script runs no matter how it is launched.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ---------------------------------------------------------------------------
import gc
import re
import json
import math
import argparse
from typing import Dict, Tuple, List, Optional

import pandas as pd
import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from further_baselines.lora_init import attach_adapter, lora_summary
from medkit_data import DEFAULT_DATA_PATH, resolve_data_path
from hemonc_batching import build_increments, filter_by_increment
from hemonc_eval_protocol import judge_open_rows, load_increment_df

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception as e:
    raise ImportError(
        "This script requires TRL with GRPOTrainer support. Install/update `trl`."
    ) from e

VALID_LABELS = {"superior", "inferior", "no difference"}
ORDERED_LABELS = ["superior", "inferior", "no difference"]
BASE_MODELS: Dict[str, str] = {
    "google_medgemma-4b-it": "google/medgemma-4b-it",
    "google_medgemma-27b-text-it": "google/medgemma-27b-text-it",
    "BioMistral_BioMistral-7B": "BioMistral/BioMistral-7B",
    "ContactDoctor_Bio-Medical-Llama-3-8B": "ContactDoctor/Bio-Medical-Llama-3-8B",
    "Intelligent-Internet_II-Medical-8B": "Intelligent-Internet/II-Medical-8B",
    "AdaptLLM_medicine-LLM": "AdaptLLM/medicine-LLM",
    "AdaptLLM_medicine-chat": "AdaptLLM/medicine-chat",
    "medalpaca_medalpaca-7b": "medalpaca/medalpaca-7b",
    "meta-llama_Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "google_gemma-3-4b-it": "google/gemma-3-4b-it",
    "Qwen_Qwen3-4B-Instruct-2507": "Qwen/Qwen3-4B-Instruct-2507",
}

MAIN_Q_COLS = ["closed question 1", "closed question 2", "closed question 3"]
MIRROR_Q_COL = "closed question m"
MIRROR_ANS_COL = "closed question m answer"
LOCALITY_Q_COL = "locality_question"
LOCALITY_GT_COL = "locality_ground_truth"
REQUIRED_DATA_COLUMNS = [
    "date", "evidence", "regimen", "comparator", "condition", "context",
    "endpoint", "endpoint_type", "answer", "closed question 1",
    "closed question 2", "closed question 3", MIRROR_Q_COL, MIRROR_ANS_COL,
    "year", LOCALITY_Q_COL, LOCALITY_GT_COL,
]
SOURCE_OUTPUT_COLUMNS = [
    "date", "year", "evidence", "regimen", "comparator", "condition",
    "context", "endpoint", "endpoint_type", "answer", "ground truth",
    "closed question 1", "closed question 2", "closed question 3",
    MIRROR_Q_COL, MIRROR_ANS_COL, "open question 1", "open generation 1",
    "option 1", "option 2", "option 3", LOCALITY_Q_COL, LOCALITY_GT_COL,
]


def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_increment_label(label: str) -> str:
    label = str(label).strip()
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label) or "increment"


def validate_data_layout(data_path: str):
    header = pd.read_csv(data_path, nrows=0)
    columns = header.columns.tolist()
    missing = [c for c in REQUIRED_DATA_COLUMNS if c not in columns]
    if missing:
        raise ValueError(
            "Input CSV is missing required HemOnc columns: "
            + ", ".join(missing)
            + f". Found columns: {columns}"
        )
    extras = [c for c in columns if c not in REQUIRED_DATA_COLUMNS]
    print(
        f"[data] layout ok: {len(columns)} columns, "
        f"{len(REQUIRED_DATA_COLUMNS)} required present, {len(extras)} extra",
        flush=True,
    )
    if extras:
        print(f"[data] extra columns ignored by this pipeline: {extras}", flush=True)


def add_source_output_fields(record: dict, row: pd.Series) -> dict:
    for col in SOURCE_OUTPUT_COLUMNS:
        if col in row.index:
            record[f"source_{col}"] = row.get(col, "")
    for col in row.index:
        if re.match(r"^(open question|open generation|option) \d+$", str(col)):
            record[f"source_{col}"] = row.get(col, "")
    return record


def normalize_label(x) -> Optional[str]:
    x = str(x).strip().lower()
    return x if x in VALID_LABELS else None


def format_prompt_for_model(tokenizer, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return prompt


def truncate_text(text: str, max_chars: int = 1200) -> str:
    text = str(text).strip().replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def train_settings_for_model(folder: str) -> Tuple[int, int, int, int]:
    lower = folder.lower()
    if "27b" in lower:
        return 1024, 1, 8, 1
    if "4b" in lower:
        return 768, 1, 2, 2
    return 768, 1, 4, 1


def eval_settings_for_model(folder: str) -> Tuple[int, int]:
    lower = folder.lower()
    if "27b" in lower:
        return 1, 1024
    if "4b" in lower:
        return 2, 1024
    return 1, 1024


def build_eval_prompt(evidence: str, question: str, include_evidence=True) -> str:
    if str(include_evidence).lower() in {"false", "0", "none", "no"}:
        return f"""{question}

Final answer must be exactly one of: superior, inferior, no difference.

Response:
"""
    return f"""You are answering a medical comparison question using the provided evidence.
Choose exactly one label from:
- superior
- inferior
- no difference

Use only the evidence. If the evidence is mixed or does not clearly favor one side, answer no difference.

Question:
{question}

Evidence:
{truncate_text(evidence, 1800)}

Final answer must be exactly one of: superior, inferior, no difference.

Response:
"""


def build_open_generation_prompt(evidence: str, question: str, include_evidence=True) -> str:
    if str(include_evidence).lower() in {"false", "0", "none", "no"}:
        return f"""{question}

Answer:
"""
    return f"""You are a knowledgeable medical assistant supporting oncologists and hematologists.
Answer the clinical question using only the provided evidence. Be concise and clinically precise.

Question:
{question}

Evidence:
{truncate_text(evidence, 1800)}

Answer:
"""


def build_grpo_prompt(evidence: str, question: str) -> str:
    return (
        f"Evidence:\n{evidence}\n\n{question}\n\n"
        "Respond with exactly one label from: superior, inferior, no difference.\nResponse:"
    )


def split_by_cutoff(df: pd.DataFrame, cutoff: str):
    dfx = df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    cutoff_ts = pd.Timestamp(cutoff)
    pre_df = dfx[dfx["date"].notna() & (dfx["date"] <= cutoff_ts)].copy()
    post_df = dfx[dfx["date"].notna() & (dfx["date"] > cutoff_ts)].copy()
    return pre_df, post_df


def split_stage2_train_val(pre_df: pd.DataFrame, val_start: str):
    dfx = pre_df.copy()
    dfx["date"] = pd.to_datetime(dfx["date"], errors="coerce")
    val_start_ts = pd.Timestamp(val_start)
    train_df = dfx[dfx["date"] < val_start_ts].copy()
    val_df = dfx[dfx["date"] >= val_start_ts].copy()
    return train_df, val_df


def prepare_df(df: pd.DataFrame):
    df = df.copy()
    df = df[df["evidence"].notna()].copy()
    df["evidence"] = df["evidence"].astype(str).str.strip()
    df = df[df["evidence"].str.len() >= 20].copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


def build_grpo_examples(df: pd.DataFrame, include_main: bool = True, include_mirror: bool = True):
    rows = []
    for idx, row in df.iterrows():
        evidence = str(row.get("evidence", "")).strip()
        if not evidence:
            continue
        if include_main:
            gold_main = normalize_label(row.get("answer", ""))
            if gold_main is not None:
                for col in MAIN_Q_COLS:
                    q = str(row.get(col, "")).strip()
                    if q and q.lower() != "nan":
                        rows.append({
                            "row_index": idx,
                            "task": "main",
                            "question_col": col,
                            "prompt": build_grpo_prompt(evidence, q),
                            "gold_label": gold_main,
                            "date": str(row.get("date", "")),
                            "year": row.get("year", None),
                        })
        if include_mirror:
            gold_mirror = normalize_label(row.get(MIRROR_ANS_COL, ""))
            q_m = str(row.get(MIRROR_Q_COL, "")).strip()
            if gold_mirror is not None and q_m and q_m.lower() != "nan":
                rows.append({
                    "row_index": idx,
                    "task": "mirror",
                    "question_col": MIRROR_Q_COL,
                    "prompt": build_grpo_prompt(evidence, q_m),
                    "gold_label": gold_mirror,
                    "date": str(row.get("date", "")),
                    "year": row.get("year", None),
                })
    return rows


def build_eval_records_main(df: pd.DataFrame, include_evidence=True):
    records = []
    for idx, row in df.iterrows():
        gold = normalize_label(row.get("answer", ""))
        if gold is None:
            continue
        evidence = str(row.get("evidence", "")).strip()
        for c in MAIN_Q_COLS:
            q = str(row.get(c, "")).strip()
            if not q or q.lower() == "nan":
                continue
            rec = {
                "row_index": idx,
                "eval_type": "main",
                "question_col": c,
                "question": q,
                "date": row.get("date", None),
                "year": row.get("year", None),
                "condition": row.get("condition", ""),
                "context": row.get("context", ""),
                "endpoint": row.get("endpoint", ""),
                "endpoint_type": row.get("endpoint_type", ""),
                "regimen": row.get("regimen", ""),
                "comparator": row.get("comparator", ""),
                    "prompt": build_eval_prompt(evidence, q, include_evidence=include_evidence),
                "gold_label": gold,
            }
            records.append(add_source_output_fields(rec, row))
    return records


def build_eval_records_mirror(df: pd.DataFrame, include_evidence=True):
    records = []
    for idx, row in df.iterrows():
        gold = normalize_label(row.get(MIRROR_ANS_COL, ""))
        q = str(row.get(MIRROR_Q_COL, "")).strip()
        evidence = str(row.get("evidence", "")).strip()
        if gold is None or not q or q.lower() == "nan":
            continue
        rec = {
            "row_index": idx,
            "eval_type": "mirror",
            "question_col": MIRROR_Q_COL,
            "question": q,
            "date": row.get("date", None),
            "year": row.get("year", None),
            "condition": row.get("condition", ""),
            "context": row.get("context", ""),
            "endpoint": row.get("endpoint", ""),
            "endpoint_type": row.get("endpoint_type", ""),
            "regimen": row.get("regimen", ""),
            "comparator": row.get("comparator", ""),
            "prompt": build_eval_prompt(evidence, q, include_evidence=include_evidence),
            "gold_label": gold,
        }
        records.append(add_source_output_fields(rec, row))
    return records


def build_eval_records_locality(df: pd.DataFrame, include_evidence=True):
    records = []
    for idx, row in df.iterrows():
        gold = normalize_label(row.get(LOCALITY_GT_COL, ""))
        q = str(row.get(LOCALITY_Q_COL, "")).strip()
        evidence = str(row.get("evidence", "")).strip()
        if gold is None or not q or q.lower() == "nan":
            continue
        rec = {
            "row_index": idx,
            "eval_type": "locality",
            "question_col": LOCALITY_Q_COL,
            "question": q,
            "date": row.get("date", None),
            "year": row.get("year", None),
            "condition": row.get("condition", ""),
            "context": row.get("context", ""),
            "endpoint": row.get("endpoint", ""),
            "endpoint_type": row.get("endpoint_type", ""),
            "regimen": row.get("regimen", ""),
            "comparator": row.get("comparator", ""),
            "prompt": build_eval_prompt(evidence, q, include_evidence=include_evidence),
            "gold_label": gold,
        }
        records.append(add_source_output_fields(rec, row))
    return records


def find_open_generation_pairs(df: pd.DataFrame) -> List[Tuple[str, str]]:
    pairs = []
    for col in df.columns:
        match = re.fullmatch(r"open question (\d+)", str(col))
        if not match:
            continue
        generation_col = f"open generation {match.group(1)}"
        if generation_col in df.columns:
            pairs.append((col, generation_col))
    return sorted(pairs, key=lambda x: int(x[0].rsplit(" ", 1)[1]))


def build_eval_records_open_generation(df: pd.DataFrame, include_evidence=True):
    records = []
    pairs = find_open_generation_pairs(df)
    for idx, row in df.iterrows():
        judge_gt = {
            "condition": str(row.get("condition", "")),
            "context": str(row.get("context", "")),
            "target": str(row.get("ground truth", "")),
        }
        evidence = str(row.get("evidence", "")).strip()
        for question_col, generation_col in pairs:
            for eval_type, prompt_col in [("open_question", question_col), ("open_generation", generation_col)]:
                q = str(row.get(prompt_col, "")).strip()
                if not q or q.lower() == "nan":
                    continue
                gold = str(row.get("ground truth", "")).strip()
                rec = {
                    "row_index": idx,
                    "eval_type": eval_type,
                    "question_col": prompt_col,
                    "question": q,
                    "date": row.get("date", None),
                    "year": row.get("year", None),
                    "condition": row.get("condition", ""),
                    "context": row.get("context", ""),
                    "endpoint": row.get("endpoint", ""),
                    "endpoint_type": row.get("endpoint_type", ""),
                    "regimen": row.get("regimen", ""),
                    "comparator": row.get("comparator", ""),
                    "prompt": build_open_generation_prompt(evidence, q, include_evidence=include_evidence),
                    "gold_label": gold,
                    "reference_generation": gold,
                    "judge_ground_truth": judge_gt,
                }
                records.append(add_source_output_fields(rec, row))
    return records


def _completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(completion)


def reward_exact_label(completions, gold_label=None, **kwargs):
    rewards = []
    for comp, gold in zip(completions, gold_label):
        text = _completion_to_text(comp).strip().lower()
        pred = "invalid"
        if "no difference" in text:
            pred = "no difference"
        elif "superior" in text:
            pred = "superior"
        elif "inferior" in text:
            pred = "inferior"
        rewards.append(1.0 if pred == gold else 0.0)
    return rewards


def reward_valid_label(completions, **kwargs):
    rewards = []
    for comp in completions:
        text = _completion_to_text(comp).strip().lower()
        is_valid = ("no difference" in text) or ("superior" in text) or ("inferior" in text)
        rewards.append(0.2 if is_valid else -0.2)
    return rewards


@torch.no_grad()
def predict_batch_generate(model, tokenizer, prompts, batch_size: int, max_length: int):
    preds = []
    raws = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batches"):
        batch = prompts[i:i + batch_size]
        toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        toks = {k: v.to(model.device) for k, v in toks.items()}
        gen = model.generate(**toks, max_new_tokens=6, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        new_tokens = gen[:, toks["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for t in texts:
            raws.append(t)
            tl = t.strip().lower()
            if "no difference" in tl:
                preds.append("no difference")
            elif "superior" in tl:
                preds.append("superior")
            elif "inferior" in tl:
                preds.append("inferior")
            else:
                preds.append("invalid")
        del toks, gen, new_tokens, texts
        torch.cuda.empty_cache()
    return preds, raws


@torch.no_grad()
def generate_texts(model, tokenizer, prompts: List[str], batch_size: int, max_length: int, max_new_tokens: int):
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generate"):
        batch = [format_prompt_for_model(tokenizer, p) for p in prompts[i:i + batch_size]]
        old_truncation_side = getattr(tokenizer, "truncation_side", "right")
        try:
            tokenizer.truncation_side = "left"
            toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        finally:
            tokenizer.truncation_side = old_truncation_side
        toks = {k: v.to(model.device) for k, v in toks.items()}
        gen = model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen[:, toks["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend([t.strip() for t in texts])
        del toks, gen, new_tokens, texts
        torch.cuda.empty_cache()
    return outputs


@torch.no_grad()
def score_labels(model, tokenizer, prompt: str, max_length: int) -> Tuple[str, Dict[str, float]]:
    formatted_prompt = format_prompt_for_model(tokenizer, prompt)
    prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    if tokenizer.bos_token_id is not None and (not prompt_ids or prompt_ids[0] != tokenizer.bos_token_id):
        prompt_ids = [tokenizer.bos_token_id] + prompt_ids
    scores = {}
    for label in ORDERED_LABELS:
        label_ids = tokenizer.encode(label, add_special_tokens=False)
        if not label_ids:
            scores[label] = float("-inf")
            continue
        input_ids = prompt_ids + label_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[-max_length:]
        label_start = len(input_ids) - len(label_ids)
        if label_start <= 0:
            scores[label] = float("-inf")
            continue
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_tensor)
        logits = model(input_ids=input_tensor, attention_mask=attention_mask).logits[0]
        token_log_probs = torch.log_softmax(logits[label_start - 1:-1, :], dim=-1)
        target_ids = input_tensor[0, label_start:]
        label_score = token_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).mean()
        scores[label] = float(label_score.detach().cpu())
        del input_tensor, attention_mask, logits, token_log_probs, target_ids, label_score
    pred = max(ORDERED_LABELS, key=lambda x: scores.get(x, float("-inf")))
    torch.cuda.empty_cache()
    return pred, scores


def checkpoint_predictions(rows: List[dict], out_csv: str, total_records: int):
    if not rows:
        return
    partial_csv = out_csv.replace(".csv", ".partial.csv")
    progress_json = out_csv.replace(".csv", ".progress.json")
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(partial_csv, index=False)
    progress = {
        "completed": int(len(pred_df)),
        "total": int(total_records),
        "accuracy_so_far": float(pred_df["correct"].mean()) if len(pred_df) else None,
        "valid_prediction_rate_so_far": float(pred_df["is_valid_pred"].mean()) if len(pred_df) else None,
        "label_counts_so_far": {str(k): int(v) for k, v in pred_df["pred_label"].value_counts().to_dict().items()},
    }
    save_json(progress, progress_json)
    print(f"[progress] {os.path.basename(out_csv)} {progress}", flush=True)


def checkpoint_open_generations(rows: List[dict], out_csv: str, total_records: int):
    if not rows:
        return
    partial_csv = out_csv.replace(".csv", ".partial.csv")
    progress_json = out_csv.replace(".csv", ".progress.json")
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(partial_csv, index=False)
    nonempty = pred_df["raw_generation"].astype(str).str.strip().ne("").sum()
    progress = {
        "completed": int(len(pred_df)),
        "total": int(total_records),
        "nonempty_generation_rate_so_far": float(nonempty / len(pred_df)) if len(pred_df) else None,
    }
    save_json(progress, progress_json)
    print(f"[progress] {os.path.basename(out_csv)} {progress}", flush=True)


def evaluate_and_save_predictions(model, tokenizer, eval_records, batch_size, max_length, out_csv, checkpoint_every: int = 0):
    if len(eval_records) == 0:
        empty_df = pd.DataFrame()
        empty_df.to_csv(out_csv, index=False)
        return {"total": 0, "correct": 0, "accuracy": None, "valid_prediction_rate": None}, empty_df
    rows = []
    for rec in tqdm(eval_records, desc=f"Eval {os.path.basename(out_csv)}"):
        pred, label_scores = score_labels(model, tokenizer, rec["prompt"], max_length)
        row = dict(rec)
        row["pred_label"] = pred
        row["raw_generation"] = pred
        row["is_valid_pred"] = int(pred in VALID_LABELS)
        row["correct"] = int(pred == rec["gold_label"])
        row["label_scores"] = json.dumps(label_scores)
        rows.append(row)
        if checkpoint_every and checkpoint_every > 0 and len(rows) % checkpoint_every == 0:
            checkpoint_predictions(rows, out_csv, len(eval_records))
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_csv, index=False)
    checkpoint_predictions(rows, out_csv, len(eval_records))
    metrics = {
        "total": int(len(pred_df)),
        "correct": int(pred_df["correct"].sum()),
        "accuracy": float(pred_df["correct"].mean()) if len(pred_df) else None,
        "valid_prediction_rate": float(pred_df["is_valid_pred"].mean()) if len(pred_df) else None,
    }
    return metrics, pred_df


def evaluate_and_save_open_generations(
    model,
    tokenizer,
    eval_records,
    batch_size,
    max_length,
    out_csv,
    checkpoint_every: int = 0,
    max_new_tokens: int = 256,
    judge_model: Optional[str] = None,
    judge_workers: int = 4,
    judge_enabled: bool = True,
):
    if len(eval_records) == 0:
        empty_df = pd.DataFrame()
        empty_df.to_csv(out_csv, index=False)
        return {"total": 0, "nonempty_generations": 0, "nonempty_generation_rate": None}, empty_df
    rows = []
    for i in tqdm(range(0, len(eval_records), batch_size), desc=f"Eval {os.path.basename(out_csv)}"):
        batch_records = eval_records[i:i + batch_size]
        generations = generate_texts(
            model,
            tokenizer,
            [r["prompt"] for r in batch_records],
            batch_size=batch_size,
            max_length=max_length,
            max_new_tokens=max_new_tokens,
        )
        for rec, generation in zip(batch_records, generations):
            row = dict(rec)
            row["pred_label"] = ""
            row["raw_generation"] = generation
            row["is_valid_pred"] = ""
            row["correct"] = ""
            row["label_scores"] = ""
            row["judge_ground_truth"] = rec.get("judge_ground_truth", rec.get("gold_label", {}))
            row["judge_score"] = None
            row["judge_explanation"] = None
            row["judge_flags"] = None
            rows.append(row)
        if checkpoint_every and checkpoint_every > 0 and len(rows) % checkpoint_every == 0:
            checkpoint_open_generations(rows, out_csv, len(eval_records))
    rows = judge_open_rows(rows, judge_model=judge_model, judge_workers=judge_workers, judge_enabled=judge_enabled and bool(judge_model))
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(out_csv, index=False)
    checkpoint_open_generations(rows, out_csv, len(eval_records))
    nonempty = pred_df["raw_generation"].astype(str).str.strip().ne("").sum()
    metrics = {
        "total": int(len(pred_df)),
        "nonempty_generations": int(nonempty),
        "nonempty_generation_rate": float(nonempty / len(pred_df)) if len(pred_df) else None,
    }
    return metrics, pred_df


def compute_yearwise_accuracy(pred_df: pd.DataFrame, cutoff_year: int):
    if pred_df is None or len(pred_df) == 0 or "year" not in pred_df.columns:
        return {"all_years": {"n": 0, "accuracy": None, "valid_prediction_rate": None}, "pre_or_on_cutoff": {"n": 0, "accuracy": None, "valid_prediction_rate": None}, "post_cutoff": {"n": 0, "accuracy": None, "valid_prediction_rate": None}, "by_year": []}
    df = pred_df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    if len(df) == 0:
        return {"all_years": {"n": 0, "accuracy": None, "valid_prediction_rate": None}, "pre_or_on_cutoff": {"n": 0, "accuracy": None, "valid_prediction_rate": None}, "post_cutoff": {"n": 0, "accuracy": None, "valid_prediction_rate": None}, "by_year": []}
    df["year"] = df["year"].astype(int)
    yearwise = []
    for year, sub in df.groupby("year"):
        yearwise.append({"year": int(year), "n": int(len(sub)), "correct": int(sub["correct"].sum()), "accuracy": float(sub["correct"].mean()), "valid_prediction_rate": float(sub["is_valid_pred"].mean()), "period": "pre_or_on_cutoff" if year <= cutoff_year else "post_cutoff"})
    yearwise = sorted(yearwise, key=lambda x: x["year"])
    pre_mask = df["year"] <= cutoff_year
    post_mask = df["year"] > cutoff_year
    return {
        "all_years": {"n": int(len(df)), "accuracy": float(df["correct"].mean()) if len(df) else None, "valid_prediction_rate": float(df["is_valid_pred"].mean()) if len(df) else None},
        "pre_or_on_cutoff": {"n": int(pre_mask.sum()), "accuracy": float(df.loc[pre_mask, "correct"].mean()) if pre_mask.any() else None, "valid_prediction_rate": float(df.loc[pre_mask, "is_valid_pred"].mean()) if pre_mask.any() else None},
        "post_cutoff": {"n": int(post_mask.sum()), "accuracy": float(df.loc[post_mask, "correct"].mean()) if post_mask.any() else None, "valid_prediction_rate": float(df.loc[post_mask, "is_valid_pred"].mean()) if post_mask.any() else None},
        "by_year": yearwise,
    }


def compute_row_level_majority_accuracy(pred_df: pd.DataFrame):
    if pred_df is None or len(pred_df) == 0 or "row_index" not in pred_df.columns:
        return {"num_rows": 0, "majority_accuracy": None}
    row_results = []
    for _, sub in pred_df.groupby("row_index"):
        golds = sub["gold_label"].dropna().unique().tolist()
        if len(golds) != 1:
            continue
        gold = golds[0]
        pred_counts = sub["pred_label"].value_counts().to_dict()
        pred = max(pred_counts.items(), key=lambda x: (x[1], x[0]))[0]
        row_results.append(int(pred == gold))
    if len(row_results) == 0:
        return {"num_rows": 0, "majority_accuracy": None}
    return {"num_rows": int(len(row_results)), "majority_accuracy": float(sum(row_results) / len(row_results))}


def plot_yearwise_accuracy(yearwise_json: dict, title: str, out_path: str):
    rows = yearwise_json.get("by_year", [])
    if not rows:
        return
    years = [r["year"] for r in rows]
    accs = [r["accuracy"] for r in rows]
    counts = [r["n"] for r in rows]
    plt.figure(figsize=(10, 5))
    plt.plot(years, accs, marker="o")
    for x, y, n in zip(years, accs, counts):
        plt.annotate(str(n), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
    plt.ylim(0, 1)
    plt.xlabel("Year")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def limit_records(records: List[dict], max_records: int) -> List[dict]:
    if max_records and max_records > 0:
        return records[:max_records]
    return records


def evaluate_split(model, tokenizer, split_name, split_df, batch_size, max_length, model_result_dir, cutoff_year):
    eval_records_main = build_eval_records_main(split_df)
    eval_records_mirror = build_eval_records_mirror(split_df)
    eval_records_locality = build_eval_records_locality(split_df)
    metrics_main, pred_df_main = evaluate_and_save_predictions(model, tokenizer, eval_records_main, batch_size, max_length, os.path.join(model_result_dir, f"predictions_main_{split_name}.csv"))
    metrics_mirror, pred_df_mirror = evaluate_and_save_predictions(model, tokenizer, eval_records_mirror, batch_size, max_length, os.path.join(model_result_dir, f"predictions_mirror_{split_name}.csv"))
    metrics_locality, pred_df_locality = evaluate_and_save_predictions(model, tokenizer, eval_records_locality, batch_size, max_length, os.path.join(model_result_dir, f"predictions_locality_{split_name}.csv"))
    yearwise_main = compute_yearwise_accuracy(pred_df_main, cutoff_year)
    yearwise_mirror = compute_yearwise_accuracy(pred_df_mirror, cutoff_year)
    yearwise_locality = compute_yearwise_accuracy(pred_df_locality, cutoff_year)
    save_json(yearwise_main, os.path.join(model_result_dir, f"yearwise_main_{split_name}.json"))
    save_json(yearwise_mirror, os.path.join(model_result_dir, f"yearwise_mirror_{split_name}.json"))
    save_json(yearwise_locality, os.path.join(model_result_dir, f"yearwise_locality_{split_name}.json"))
    plot_yearwise_accuracy(yearwise_main, f"{split_name} | Main accuracy by year", os.path.join(model_result_dir, f"yearwise_main_{split_name}.png"))
    plot_yearwise_accuracy(yearwise_mirror, f"{split_name} | Mirror accuracy by year", os.path.join(model_result_dir, f"yearwise_mirror_{split_name}.png"))
    plot_yearwise_accuracy(yearwise_locality, f"{split_name} | Locality accuracy by year", os.path.join(model_result_dir, f"yearwise_locality_{split_name}.png"))
    split_metrics = {
        "main": metrics_main,
        "mirror": metrics_mirror,
        "locality": metrics_locality,
        "row_level_main": compute_row_level_majority_accuracy(pred_df_main),
    }
    save_json(split_metrics, os.path.join(model_result_dir, f"metrics_{split_name}.json"))
    return split_metrics


def evaluate_increment(
    model,
    tokenizer,
    increment_label: str,
    strategy: str,
    batch_df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    strategy_result_dir: str,
    cutoff_year: int,
    max_eval_records_per_type: int = 0,
    checkpoint_every: int = 0,
    open_generation_max_new_tokens: int = 256,
    judge_model: Optional[str] = None,
    judge_workers: int = 4,
    judge_enabled: bool = True,
    include_evidence=True,
):
    safe_label = safe_increment_label(increment_label)
    print(f"\n================ EVALUATING INCREMENT: {increment_label} ================\n", flush=True)

    eval_records_main = limit_records(build_eval_records_main(batch_df, include_evidence=include_evidence), max_eval_records_per_type)
    eval_records_mirror = limit_records(build_eval_records_mirror(batch_df, include_evidence=include_evidence), max_eval_records_per_type)
    eval_records_locality = limit_records(build_eval_records_locality(batch_df, include_evidence=include_evidence), max_eval_records_per_type)
    eval_records_open = limit_records(build_eval_records_open_generation(batch_df, include_evidence=include_evidence), max_eval_records_per_type)
    print(
        f"[info] eval_records increment={increment_label} "
        f"main={len(eval_records_main)} mirror={len(eval_records_mirror)} "
        f"locality={len(eval_records_locality)} open_generation={len(eval_records_open)}",
        flush=True,
    )

    main_csv = os.path.join(strategy_result_dir, f"{safe_label}_main_predictions.csv")
    mirror_csv = os.path.join(strategy_result_dir, f"{safe_label}_mirror_predictions.csv")
    locality_csv = os.path.join(strategy_result_dir, f"{safe_label}_locality_predictions.csv")
    open_csv = os.path.join(strategy_result_dir, f"{safe_label}_open_generation_predictions.csv")
    combined_csv = os.path.join(strategy_result_dir, f"{safe_label}_predictions.csv")
    metrics_json = os.path.join(strategy_result_dir, f"{safe_label}_metrics.json")

    metrics_main, pred_df_main = evaluate_and_save_predictions(model, tokenizer, eval_records_main, batch_size, max_length, main_csv, checkpoint_every)
    metrics_mirror, pred_df_mirror = evaluate_and_save_predictions(model, tokenizer, eval_records_mirror, batch_size, max_length, mirror_csv, checkpoint_every)
    metrics_locality, pred_df_locality = evaluate_and_save_predictions(model, tokenizer, eval_records_locality, batch_size, max_length, locality_csv, checkpoint_every)
    metrics_open, pred_df_open = evaluate_and_save_open_generations(
        model,
        tokenizer,
        eval_records_open,
        batch_size,
        max_length,
        open_csv,
        checkpoint_every=checkpoint_every,
        max_new_tokens=open_generation_max_new_tokens,
        judge_model=judge_model,
        judge_workers=judge_workers,
        judge_enabled=judge_enabled,
    )

    pred_dfs = [df for df in [pred_df_main, pred_df_mirror, pred_df_locality, pred_df_open] if df is not None and len(df) > 0]
    combined_pred_df = pd.concat(pred_dfs, ignore_index=True) if pred_dfs else pd.DataFrame()
    combined_pred_df.to_csv(combined_csv, index=False)

    metrics = {
        "increment": str(increment_label),
        "batching_strategy": strategy,
        "batch_rows": int(len(batch_df)),
        "eval_record_counts": {
            "main": int(len(eval_records_main)),
            "mirror": int(len(eval_records_mirror)),
            "locality": int(len(eval_records_locality)),
            "open_generation": int(len(eval_records_open)),
            "combined": int(len(combined_pred_df)),
        },
        "main": metrics_main,
        "mirror": metrics_mirror,
        "locality": metrics_locality,
        "open_generation": metrics_open,
        "row_level_main": compute_row_level_majority_accuracy(pred_df_main),
        "yearwise": {
            "main": compute_yearwise_accuracy(pred_df_main, cutoff_year),
            "mirror": compute_yearwise_accuracy(pred_df_mirror, cutoff_year),
            "locality": compute_yearwise_accuracy(pred_df_locality, cutoff_year),
        },
        "files": {
            "metrics": metrics_json,
            "predictions": combined_csv,
            "main_predictions": main_csv,
            "mirror_predictions": mirror_csv,
            "locality_predictions": locality_csv,
            "open_generation_predictions": open_csv,
        },
    }
    save_json(metrics, metrics_json)
    print(f"[evaluation] saved metrics to {metrics_json}", flush=True)
    print(f"[evaluation] saved predictions to {combined_csv}", flush=True)
    return metrics


def load_peft_causal_lm(base_model: str, adapter_dir: Optional[str], tokenizer, dtype, cache_dir: str, offload_dir: str, trust_remote_code: bool, model_folder: str, is_trainable: bool):
    if "27b" in model_folder.lower():
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=cache_dir,
            offload_folder=offload_dir,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map={"": 0},
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )
    if len(tokenizer) > base.get_input_embeddings().weight.shape[0]:
        base.resize_token_embeddings(len(tokenizer))
    if is_trainable and hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
        base.config.use_cache = False
    model = attach_adapter(base, adapter_dir, model_folder, is_trainable)
    return model, base


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=DEFAULT_DATA_PATH, help="hf://<org>/MedKIT or a local CSV")
    ap.add_argument("--data", default=None, help="Backward-compatible alias for --data_path")
    ap.add_argument("--model_folder", required=True)
    ap.add_argument("--cutoff", default="2022-12-31")
    ap.add_argument("--val_start", default="2022-01-01")
    ap.add_argument("--adapter_root", required=True)
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--cache_dir", default=None, help="HF model cache (default: HF_HOME / ~/.cache/huggingface)")
    ap.add_argument("--offload_dir", required=True)
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--sample_limit", type=int, default=0)
    ap.add_argument("--batching_strategy", choices=["none", "daily", "weekly", "monthly", "quarterly", "publication"], default="none")
    ap.add_argument("--crop_year", type=int, default=None)
    ap.add_argument("--n_increments", type=int, default=None)
    ap.add_argument("--max_eval_records_per_type", type=int, default=0)
    ap.add_argument("--checkpoint_every", type=int, default=25)
    ap.add_argument("--open_generation_max_new_tokens", type=int, default=256)
    ap.add_argument("--judge_model", default="google/gemini-2.0-flash-001")
    ap.add_argument("--judge_workers", type=int, default=4)
    ap.add_argument("--judge_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--filter_conflicting_edits", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--include_evidence", default="true")
    args = ap.parse_args()

    if args.data is not None:  # legacy alias
        args.data_path = args.data
    args.data_path = resolve_data_path(args.data_path)
    validate_data_layout(args.data_path)

    if args.model_folder not in BASE_MODELS:
        raise ValueError(f"Unknown model folder: {args.model_folder}")
    base_model = BASE_MODELS[args.model_folder]
    train_max_length, train_epochs, train_grad_accum, train_bs = train_settings_for_model(args.model_folder)
    eval_batch_size, eval_max_length = eval_settings_for_model(args.model_folder)
    cutoff_year = pd.Timestamp(args.cutoff).year

    grpo_adapter_dir = os.path.join(args.adapter_root, args.model_folder)
    model_result_dir = os.path.join(args.result_root, args.model_folder)
    strategy_adapter_dir = os.path.join(grpo_adapter_dir, args.batching_strategy)
    strategy_result_dir = os.path.join(model_result_dir, args.batching_strategy)
    os.makedirs(grpo_adapter_dir, exist_ok=True)
    os.makedirs(model_result_dir, exist_ok=True)
    os.makedirs(strategy_adapter_dir, exist_ok=True)
    os.makedirs(strategy_result_dir, exist_ok=True)
    os.makedirs(args.offload_dir, exist_ok=True)

    print(
        f"[batching] strategy={args.batching_strategy} "
        f"crop_year={args.crop_year} n_increments={args.n_increments}",
        flush=True,
    )
    increments = build_increments(
        data_path=args.data_path,
        strategy=args.batching_strategy,
        crop_year=args.crop_year,
    )
    if args.n_increments is not None:
        increments = increments[: args.n_increments]
    if not increments:
        raise ValueError(
            f"No increments were built for strategy={args.batching_strategy}, "
            f"crop_year={args.crop_year}, n_increments={args.n_increments}."
        )
    print(
        f"[batching] built {len(increments)} increments: first={increments[0]} last={increments[-1]}",
        flush=True,
    )

    trust_remote_code = any(x in base_model.lower() for x in ["ii-medical", "adaptllm"])
    tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=args.cache_dir, trust_remote_code=trust_remote_code, use_fast=not trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    save_json({
        "base_model": base_model,
        "model_folder": args.model_folder,
        "init": "fresh LoRA on the base model",
        "lora": lora_summary(args.model_folder),
        "train_max_length": train_max_length,
        "train_epochs": train_epochs,
        "train_grad_accum": train_grad_accum,
        "train_batch_size": train_bs,
        "learning_rate": args.learning_rate,
        "num_generations": args.num_generations,
        "method": "GRPO",
        "answer_selection": "constrained_label_logprob_scoring",
        "reward_components": ["exact_label", "valid_label_bonus"],
        "train_tasks": ["main", "mirror"],
        "batching_strategy": args.batching_strategy,
        "crop_year": args.crop_year,
        "n_increments": args.n_increments,
        "increments": [str(x) for x in increments],
        "data_path": args.data_path,
    }, os.path.join(strategy_result_dir, "train_config.json"))

    all_results = {}
    current_adapter_dir = None  # no adapter yet: increment 1 trains a fresh LoRA on the base model
    for increment in increments:
        batch_df = load_increment_df(
            args.data_path,
            strategy=args.batching_strategy,
            increment=increment,
            crop_year=args.crop_year,
            filter_conflicting_edits=args.filter_conflicting_edits,
        )
        batch_df = prepare_df(batch_df)
        print(f"[batching] increment {increment}: {len(batch_df)} rows", flush=True)

        safe_label = safe_increment_label(increment)
        increment_adapter_dir = os.path.join(strategy_adapter_dir, safe_label)
        os.makedirs(increment_adapter_dir, exist_ok=True)

        train_examples = build_grpo_examples(batch_df, include_main=True, include_mirror=True)
        if args.sample_limit and args.sample_limit > 0:
            train_examples = train_examples[: args.sample_limit]
        print(f"[training] increment {increment}: {len(train_examples)} GRPO examples", flush=True)

        if len(train_examples) > 0:
            train_ds = Dataset.from_list(train_examples)
            model, base = load_peft_causal_lm(
                base_model,
                current_adapter_dir,
                tokenizer,
                dtype,
                args.cache_dir,
                args.offload_dir,
                trust_remote_code,
                args.model_folder,
                is_trainable=True,
            )
            model.print_trainable_parameters()
            num_update_steps_per_epoch = max(1, math.ceil(len(train_ds) / (train_bs * train_grad_accum)))
            derived_steps = train_epochs * num_update_steps_per_epoch
            warmup_steps = int(0.03 * derived_steps)
            grpo_args = GRPOConfig(
                output_dir=increment_adapter_dir,
                max_steps=derived_steps,
                per_device_train_batch_size=train_bs,
                gradient_accumulation_steps=train_grad_accum,
                learning_rate=args.learning_rate,
                lr_scheduler_type="linear",
                warmup_steps=warmup_steps,
                logging_steps=10,
                save_steps=derived_steps,
                save_total_limit=1,
                bf16=use_bf16,
                fp16=not use_bf16,
                optim="paged_adamw_8bit",
                weight_decay=0.01,
                max_grad_norm=1.0,
                report_to="none",
                remove_unused_columns=False,
                dataloader_num_workers=0,
                dataloader_pin_memory=True,
                seed=42,
                max_prompt_length=max(256, train_max_length - 32),
                max_completion_length=8,
                num_generations=args.num_generations,
            )
            trainer = GRPOTrainer(
                model=model,
                reward_funcs=[reward_exact_label, reward_valid_label],
                args=grpo_args,
                train_dataset=train_ds,
                processing_class=tokenizer,
            )
            save_json({
                "increment": str(increment),
                "batching_strategy": args.batching_strategy,
                "batch_rows": int(len(batch_df)),
                "num_train_examples": int(len(train_ds)),
                "source_adapter_dir": current_adapter_dir,
                "adapter_dir": increment_adapter_dir,
                "num_update_steps_per_epoch": int(num_update_steps_per_epoch),
                "train_steps": int(derived_steps),
                "warmup_steps": int(warmup_steps),
            }, os.path.join(strategy_result_dir, f"{safe_label}_train_config.json"))
            trainer.train()
            trainer.save_model(increment_adapter_dir)
            tokenizer.save_pretrained(increment_adapter_dir)
            del trainer, model, base, train_ds
            gc.collect()
            torch.cuda.empty_cache()
            current_adapter_dir = increment_adapter_dir
        else:
            print(f"[training] increment {increment}: no train examples, carrying adapter forward", flush=True)

        model, base = load_peft_causal_lm(
            base_model,
            current_adapter_dir,
            tokenizer,
            dtype,
            args.cache_dir,
            args.offload_dir,
            trust_remote_code,
            args.model_folder,
            is_trainable=False,
        )
        model.eval()
        increment_metrics = evaluate_increment(
            model,
            tokenizer,
            increment_label=str(increment),
            strategy=args.batching_strategy,
            batch_df=batch_df,
            batch_size=eval_batch_size,
            max_length=eval_max_length,
            strategy_result_dir=strategy_result_dir,
            cutoff_year=cutoff_year,
            max_eval_records_per_type=args.max_eval_records_per_type,
            checkpoint_every=args.checkpoint_every,
            open_generation_max_new_tokens=args.open_generation_max_new_tokens,
            judge_model=args.judge_model,
            judge_workers=args.judge_workers,
            judge_enabled=args.judge_enabled,
            include_evidence=args.include_evidence,
        )
        increment_metrics["adapter_dir"] = current_adapter_dir
        all_results[str(increment)] = increment_metrics
        del model, base
        gc.collect()
        torch.cuda.empty_cache()

    summary_path = os.path.join(strategy_result_dir, "metrics.json")
    save_json({
        "model_folder": args.model_folder,
        "base_model": base_model,
        "cutoff": args.cutoff,
        "val_start": args.val_start,
        "eval_batch_size": eval_batch_size,
        "eval_max_length": eval_max_length,
        "method": "GRPO",
        "answer_selection": "constrained_label_logprob_scoring",
        "batching_strategy": args.batching_strategy,
        "crop_year": args.crop_year,
        "n_increments": args.n_increments,
        "num_increments": int(len(increments)),
        "increments": all_results,
    }, summary_path)
    print(f"[evaluation] saved metrics to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
