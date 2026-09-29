# --- import bootstrap ------------------------------------------------------
# Standalone Agentic-RAG script; also imported as a helper library (TfidfRetriever
# + prompt builders) by run_agentic_shared_eval.py.  `hemonc_batching` /
# `hemonc_eval_protocol` live one directory up (experiments/); add it to sys.path.
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ---------------------------------------------------------------------------
import gc
import re
import json
import argparse
from typing import Dict, Tuple, List, Optional

import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from transformers import AutoTokenizer, AutoModelForCausalLM
from medkit_data import DEFAULT_DATA_PATH, resolve_data_path
from hemonc_batching import build_increments, filter_by_increment
from hemonc_eval_protocol import judge_open_rows, load_increment_df


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
    "date",
    "evidence",
    "regimen",
    "comparator",
    "condition",
    "context",
    "endpoint",
    "endpoint_type",
    "answer",
    "closed question 1",
    "closed question 2",
    "closed question 3",
    MIRROR_Q_COL,
    MIRROR_ANS_COL,
    "year",
    LOCALITY_Q_COL,
    LOCALITY_GT_COL,
]
SOURCE_OUTPUT_COLUMNS = [
    "date",
    "year",
    "evidence",
    "regimen",
    "comparator",
    "condition",
    "context",
    "endpoint",
    "endpoint_type",
    "answer",
    "ground truth",
    "closed question 1",
    "closed question 2",
    "closed question 3",
    MIRROR_Q_COL,
    MIRROR_ANS_COL,
    "open question 1",
    "open generation 1",
    "option 1",
    "option 2",
    "option 3",
    LOCALITY_Q_COL,
    LOCALITY_GT_COL,
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


def clean_text_field(x) -> str:
    x = str(x).strip()
    if not x or x.lower() in {"nan", "none"}:
        return ""
    return re.sub(r"\s+", " ", x)


def format_prompt_for_model(tokenizer, prompt: str) -> str:
    """Use an instruction/chat template when the tokenizer provides one."""
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


def eval_settings_for_model(folder: str) -> Tuple[int, int]:
    lower = folder.lower()
    if "27b" in lower:
        return 1, 1024
    if "4b" in lower:
        return 2, 1024
    return 1, 1024


def prepare_df(df: pd.DataFrame):
    df = df.copy()
    df = df[df["evidence"].notna()].copy()
    df["evidence"] = df["evidence"].astype(str).str.strip()
    df = df[df["evidence"].str.len() >= 20].copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df


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


def build_eval_records_main(df: pd.DataFrame):
    records = []
    for idx, row in df.iterrows():
        gold = normalize_label(row.get("answer", ""))
        if gold is None:
            continue
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
                    "gold_label": gold,
                }
            records.append(add_source_output_fields(rec, row))
    return records


def build_eval_records_mirror(df: pd.DataFrame):
    records = []
    for idx, row in df.iterrows():
        gold = normalize_label(row.get(MIRROR_ANS_COL, ""))
        q = str(row.get(MIRROR_Q_COL, "")).strip()
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
                "gold_label": gold,
            }
        records.append(add_source_output_fields(rec, row))
    return records


def build_eval_records_locality(df: pd.DataFrame):
    records = []
    for idx, row in df.iterrows():
        gold = normalize_label(row.get(LOCALITY_GT_COL, ""))
        q = str(row.get(LOCALITY_Q_COL, "")).strip()
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


def build_eval_records_open_generation(df: pd.DataFrame):
    records = []
    pairs = find_open_generation_pairs(df)
    for idx, row in df.iterrows():
        judge_gt = {
            "condition": str(row.get("condition", "")),
            "context": str(row.get("context", "")),
            "target": str(row.get("ground truth", "")),
        }
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
                    "gold_label": gold,
                    "reference_generation": gold,
                    "judge_ground_truth": judge_gt,
                }
                records.append(add_source_output_fields(rec, row))
    return records


class TfidfRetriever:
    def __init__(self, texts: List[str], row_indices: List[int], max_features: int = 50000):
        self.texts = texts
        self.row_indices = row_indices
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
        )
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5):
        if not str(query).strip():
            query = " ".join(re.findall(r"[A-Za-z0-9]+", query)) or "medical treatment comparison"
        qvec = self.vectorizer.transform([query])
        scores = linear_kernel(qvec, self.doc_matrix).ravel()
        if len(scores) == 0:
            return []
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for i in top_idx:
            results.append(
                {
                    "corpus_pos": int(i),
                    "row_index": int(self.row_indices[i]),
                    "score": float(scores[i]),
                    "evidence": self.texts[i],
                }
            )
        return results


def truncate_text(text: str, max_chars: int = 900) -> str:
    text = str(text).strip().replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def sanitize_generated_query(text: str) -> str:
    text = clean_text_field(text)
    if not text:
        return ""
    text = text.splitlines()[0].strip()
    text = re.sub(r"^(search query|improved search query)\s*:\s*", "", text, flags=re.I)
    text = text.strip(" \t\"'")
    return text


def build_structured_query(record: dict) -> str:
    parts = [
        clean_text_field(record.get("condition", "")),
        clean_text_field(record.get("context", "")),
        clean_text_field(record.get("endpoint", "")),
        clean_text_field(record.get("endpoint_type", "")),
        clean_text_field(record.get("regimen", "")),
        clean_text_field(record.get("comparator", "")),
    ]
    seen = set()
    kept = []
    for part in parts:
        key = part.lower()
        if part and key not in seen:
            seen.add(key)
            kept.append(part)
    return " ".join(kept)


def combine_queries(*queries: str) -> str:
    terms = []
    seen = set()
    for query in queries:
        for term in re.split(r"\s+(?:AND\s+)?", clean_text_field(query)):
            term = term.strip(" \t\"'(),;")
            key = term.lower()
            if len(term) >= 2 and key not in seen:
                seen.add(key)
                terms.append(term)
    return " ".join(terms)


@torch.no_grad()
def generate_texts(model, tokenizer, prompts: List[str], batch_size: int, max_input_length: int, max_new_tokens: int):
    outputs = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generate"):
        batch = [format_prompt_for_model(tokenizer, p) for p in prompts[i : i + batch_size]]
        old_truncation_side = getattr(tokenizer, "truncation_side", "right")
        try:
            tokenizer.truncation_side = "left"
            toks = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            )
        finally:
            tokenizer.truncation_side = old_truncation_side
        toks = {k: v.to(model.device) for k, v in toks.items()}
        gen = model.generate(
            **toks,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen[:, toks["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend([t.strip() for t in texts])
        del toks, gen, new_tokens, texts
        torch.cuda.empty_cache()
    return outputs


def build_query_prompt(question: str) -> str:
    return f"""You are helping retrieve medical evidence for a question.
Write a short search query using the key disease, treatment, comparator, and endpoint terms.
Do not answer the question.
Return only the search query.

Question: {question}

Search query:"""


def build_refine_query_prompt(question: str, snippets: List[str], original_query: str) -> str:
    joined = "\n".join([f"Snippet {i+1}: {truncate_text(s, 220)}" for i, s in enumerate(snippets[:3])])
    return f"""You are improving a search query for medical evidence retrieval.
Given the question, the initial query, and a few retrieved snippets, write one improved search query.
Do not answer the question.
Return only the improved search query.

Question: {question}
Initial query: {original_query}
{joined}

Improved search query:"""


def build_answer_prompt(question: str, retrieved_docs: List[dict]) -> str:
    context_blocks = []
    for i, d in enumerate(retrieved_docs, start=1):
        context_blocks.append(f"Evidence {i}:\n{truncate_text(d['evidence'], 700)}")
    context = "\n\n".join(context_blocks)
    return f"""You are answering a medical comparison question using retrieved evidence.
Choose exactly one label from:
- superior
- inferior
- no difference

Use only the retrieved evidence. If the evidence is mixed or does not clearly favor one side, answer no difference.
Return only the label.

Question:
{question}

Retrieved evidence:
{context}

Final answer must be exactly one of: superior, inferior, no difference.

Response:
"""


def build_open_generation_prompt(question: str, retrieved_docs: List[dict]) -> str:
    context_blocks = []
    for i, d in enumerate(retrieved_docs, start=1):
        context_blocks.append(f"Evidence {i}:\n{truncate_text(d['evidence'], 900)}")
    context = "\n\n".join(context_blocks)
    return f"""You are a knowledgeable medical assistant supporting oncologists and hematologists.
Answer the clinical question using only the retrieved evidence. Be concise and clinically precise.

Question:
{question}

Retrieved evidence:
{context}

Answer:
"""


def parse_label(text: str) -> str:
    tl = str(text).strip().lower()
    if "no difference" in tl:
        return "no difference"
    if "superior" in tl:
        return "superior"
    if "inferior" in tl:
        return "inferior"
    return "invalid"


@torch.no_grad()
def score_labels(model, tokenizer, prompt: str, max_input_length: int) -> Tuple[str, Dict[str, float]]:
    """Choose among the valid labels by conditional log probability.

    This avoids the common failure mode where a model continues a retrieved
    abstract instead of emitting one of the allowed labels.
    """
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
        if len(input_ids) > max_input_length:
            input_ids = input_ids[-max_input_length:]

        label_start = len(input_ids) - len(label_ids)
        if label_start <= 0:
            scores[label] = float("-inf")
            continue

        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_tensor)
        logits = model(input_ids=input_tensor, attention_mask=attention_mask).logits[0]
        token_log_probs = torch.log_softmax(logits[label_start - 1 : -1, :], dim=-1)
        target_ids = input_tensor[0, label_start:]
        label_score = token_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1).mean()
        scores[label] = float(label_score.detach().cpu())

        del input_tensor, attention_mask, logits, token_log_probs, target_ids, label_score

    pred = max(ORDERED_LABELS, key=lambda x: scores.get(x, float("-inf")))
    torch.cuda.empty_cache()
    return pred, scores


def compute_yearwise_accuracy(pred_df: pd.DataFrame, cutoff_year: int):
    if pred_df is None or len(pred_df) == 0 or "year" not in pred_df.columns:
        return {
            "all_years": {"n": 0, "accuracy": None, "valid_prediction_rate": None},
            "pre_or_on_cutoff": {"n": 0, "accuracy": None, "valid_prediction_rate": None},
            "post_cutoff": {"n": 0, "accuracy": None, "valid_prediction_rate": None},
            "by_year": [],
        }

    df = pred_df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].notna()].copy()
    if len(df) == 0:
        return {
            "all_years": {"n": 0, "accuracy": None, "valid_prediction_rate": None},
            "pre_or_on_cutoff": {"n": 0, "accuracy": None, "valid_prediction_rate": None},
            "post_cutoff": {"n": 0, "accuracy": None, "valid_prediction_rate": None},
            "by_year": [],
        }

    df["year"] = df["year"].astype(int)
    yearwise = []
    for year, sub in df.groupby("year"):
        yearwise.append(
            {
                "year": int(year),
                "n": int(len(sub)),
                "correct": int(sub["correct"].sum()),
                "accuracy": float(sub["correct"].mean()),
                "valid_prediction_rate": float(sub["is_valid_pred"].mean()),
                "period": "pre_or_on_cutoff" if year <= cutoff_year else "post_cutoff",
            }
        )

    yearwise = sorted(yearwise, key=lambda x: x["year"])
    pre_mask = df["year"] <= cutoff_year
    post_mask = df["year"] > cutoff_year

    return {
        "all_years": {
            "n": int(len(df)),
            "accuracy": float(df["correct"].mean()) if len(df) else None,
            "valid_prediction_rate": float(df["is_valid_pred"].mean()) if len(df) else None,
        },
        "pre_or_on_cutoff": {
            "n": int(pre_mask.sum()),
            "accuracy": float(df.loc[pre_mask, "correct"].mean()) if pre_mask.any() else None,
            "valid_prediction_rate": float(df.loc[pre_mask, "is_valid_pred"].mean()) if pre_mask.any() else None,
        },
        "post_cutoff": {
            "n": int(post_mask.sum()),
            "accuracy": float(df.loc[post_mask, "correct"].mean()) if post_mask.any() else None,
            "valid_prediction_rate": float(df.loc[post_mask, "is_valid_pred"].mean()) if post_mask.any() else None,
        },
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

    return {
        "num_rows": int(len(row_results)),
        "majority_accuracy": float(sum(row_results) / len(row_results)),
    }


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


def run_agentic_prediction(
    model,
    tokenizer,
    retriever: TfidfRetriever,
    record: dict,
    batch_size: int,
    max_input_length: int,
    top_k: int,
    rounds: int,
):
    question = record["question"]
    structured_query = build_structured_query(record)
    generated_q1 = generate_texts(
        model,
        tokenizer,
        [build_query_prompt(question)],
        batch_size=1,
        max_input_length=max_input_length,
        max_new_tokens=32,
    )[0]
    generated_q1 = sanitize_generated_query(generated_q1)
    q1 = combine_queries(structured_query, generated_q1) or generated_q1 or structured_query or question

    first_docs = retriever.search(q1, top_k=top_k)

    refined_query = q1
    final_docs = first_docs
    if rounds >= 2 and len(first_docs) > 0:
        generated_refined_query = generate_texts(
            model,
            tokenizer,
            [build_refine_query_prompt(question, [d["evidence"] for d in first_docs], q1)],
            batch_size=1,
            max_input_length=max_input_length,
            max_new_tokens=32,
        )[0]
        generated_refined_query = sanitize_generated_query(generated_refined_query)
        refined_query = combine_queries(structured_query, generated_refined_query) or generated_refined_query or q1
        final_docs = retriever.search(refined_query, top_k=top_k)

    answer_prompt = build_answer_prompt(question, final_docs)
    pred, label_scores = score_labels(
        model=model,
        tokenizer=tokenizer,
        prompt=answer_prompt,
        max_input_length=max_input_length,
    )

    trace = {
        "initial_query": q1,
        "refined_query": refined_query,
        "retrieved_row_indices": [d["row_index"] for d in final_docs],
        "retrieved_scores": [round(float(d["score"]), 6) for d in final_docs],
        "retrieved_snippets": [truncate_text(d["evidence"], 220) for d in final_docs],
        "raw_generation": pred,
        "label_scores": label_scores,
    }
    return pred, trace


def run_agentic_open_generation(
    model,
    tokenizer,
    retriever: TfidfRetriever,
    record: dict,
    batch_size: int,
    max_input_length: int,
    top_k: int,
    rounds: int,
    max_new_tokens: int,
):
    question = record["question"]
    structured_query = build_structured_query(record)
    generated_q1 = generate_texts(
        model,
        tokenizer,
        [build_query_prompt(question)],
        batch_size=1,
        max_input_length=max_input_length,
        max_new_tokens=32,
    )[0]
    generated_q1 = sanitize_generated_query(generated_q1)
    q1 = combine_queries(structured_query, generated_q1) or generated_q1 or structured_query or question

    first_docs = retriever.search(q1, top_k=top_k)
    refined_query = q1
    final_docs = first_docs
    if rounds >= 2 and len(first_docs) > 0:
        generated_refined_query = generate_texts(
            model,
            tokenizer,
            [build_refine_query_prompt(question, [d["evidence"] for d in first_docs], q1)],
            batch_size=1,
            max_input_length=max_input_length,
            max_new_tokens=32,
        )[0]
        generated_refined_query = sanitize_generated_query(generated_refined_query)
        refined_query = combine_queries(structured_query, generated_refined_query) or generated_refined_query or q1
        final_docs = retriever.search(refined_query, top_k=top_k)

    answer_text = generate_texts(
        model,
        tokenizer,
        [build_open_generation_prompt(question, final_docs)],
        batch_size=batch_size,
        max_input_length=max_input_length,
        max_new_tokens=max_new_tokens,
    )[0]
    trace = {
        "initial_query": q1,
        "refined_query": refined_query,
        "retrieved_row_indices": [d["row_index"] for d in final_docs],
        "retrieved_scores": [round(float(d["score"]), 6) for d in final_docs],
        "retrieved_snippets": [truncate_text(d["evidence"], 220) for d in final_docs],
        "raw_generation": answer_text,
    }
    return answer_text, trace


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


def evaluate_and_save_predictions(
    model,
    tokenizer,
    retriever,
    eval_records,
    batch_size,
    max_length,
    top_k,
    rounds,
    out_csv,
    checkpoint_every: int = 0,
):
    if len(eval_records) == 0:
        empty_df = pd.DataFrame()
        empty_df.to_csv(out_csv, index=False)
        return {"total": 0, "correct": 0, "accuracy": None, "valid_prediction_rate": None}, empty_df

    rows = []
    for rec in tqdm(eval_records, desc=f"Eval {os.path.basename(out_csv)}"):
        pred, trace = run_agentic_prediction(
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            record=rec,
            batch_size=batch_size,
            max_input_length=max_length,
            top_k=top_k,
            rounds=rounds,
        )
        row = dict(rec)
        row["prompt"] = rec["question"]
        row["pred_label"] = pred
        row["raw_generation"] = trace["raw_generation"]
        row["is_valid_pred"] = int(pred in VALID_LABELS)
        row["correct"] = int(pred == rec["gold_label"])
        row["initial_query"] = trace["initial_query"]
        row["refined_query"] = trace["refined_query"]
        row["retrieved_row_indices"] = json.dumps(trace["retrieved_row_indices"])
        row["retrieved_scores"] = json.dumps(trace["retrieved_scores"])
        row["retrieved_snippets"] = json.dumps(trace["retrieved_snippets"])
        row["label_scores"] = json.dumps(trace["label_scores"])
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
    retriever,
    eval_records,
    batch_size,
    max_length,
    top_k,
    rounds,
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
    for rec in tqdm(eval_records, desc=f"Eval {os.path.basename(out_csv)}"):
        answer_text, trace = run_agentic_open_generation(
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            record=rec,
            batch_size=batch_size,
            max_input_length=max_length,
            top_k=top_k,
            rounds=rounds,
            max_new_tokens=max_new_tokens,
        )
        row = dict(rec)
        row["prompt"] = rec["question"]
        row["pred_label"] = ""
        row["raw_generation"] = trace["raw_generation"]
        row["is_valid_pred"] = ""
        row["correct"] = ""
        row["judge_ground_truth"] = rec.get("judge_ground_truth", rec.get("gold_label", {}))
        row["judge_score"] = None
        row["judge_explanation"] = None
        row["judge_flags"] = None
        row["initial_query"] = trace["initial_query"]
        row["refined_query"] = trace["refined_query"]
        row["retrieved_row_indices"] = json.dumps(trace["retrieved_row_indices"])
        row["retrieved_scores"] = json.dumps(trace["retrieved_scores"])
        row["retrieved_snippets"] = json.dumps(trace["retrieved_snippets"])
        row["label_scores"] = ""
        rows.append(row)
        if checkpoint_every and checkpoint_every > 0 and len(rows) % checkpoint_every == 0:
            checkpoint_open_generations(rows, out_csv, len(eval_records))

    pred_df = pd.DataFrame(rows)
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


def limit_records(records: List[dict], max_records: int) -> List[dict]:
    if max_records and max_records > 0:
        return records[:max_records]
    return records


def evaluate_split(
    model,
    tokenizer,
    retriever,
    split_name,
    split_df,
    batch_size,
    max_length,
    top_k,
    rounds,
    model_result_dir,
    cutoff_year,
    max_eval_records_per_type: int = 0,
    checkpoint_every: int = 0,
):
    print(f"\n================ EVALUATING SPLIT: {split_name} ================\n")

    eval_records_main = limit_records(build_eval_records_main(split_df), max_eval_records_per_type)
    eval_records_mirror = limit_records(build_eval_records_mirror(split_df), max_eval_records_per_type)
    eval_records_locality = limit_records(build_eval_records_locality(split_df), max_eval_records_per_type)
    print(
        f"[info] eval_records split={split_name} "
        f"main={len(eval_records_main)} mirror={len(eval_records_mirror)} locality={len(eval_records_locality)}",
        flush=True,
    )

    metrics_main, pred_df_main = evaluate_and_save_predictions(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        eval_records=eval_records_main,
        batch_size=batch_size,
        max_length=max_length,
        top_k=top_k,
        rounds=rounds,
        out_csv=os.path.join(model_result_dir, f"predictions_main_{split_name}.csv"),
        checkpoint_every=checkpoint_every,
    )
    metrics_mirror, pred_df_mirror = evaluate_and_save_predictions(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        eval_records=eval_records_mirror,
        batch_size=batch_size,
        max_length=max_length,
        top_k=top_k,
        rounds=rounds,
        out_csv=os.path.join(model_result_dir, f"predictions_mirror_{split_name}.csv"),
        checkpoint_every=checkpoint_every,
    )
    metrics_locality, pred_df_locality = evaluate_and_save_predictions(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        eval_records=eval_records_locality,
        batch_size=batch_size,
        max_length=max_length,
        top_k=top_k,
        rounds=rounds,
        out_csv=os.path.join(model_result_dir, f"predictions_locality_{split_name}.csv"),
        checkpoint_every=checkpoint_every,
    )

    yearwise_main = compute_yearwise_accuracy(pred_df_main, cutoff_year)
    yearwise_mirror = compute_yearwise_accuracy(pred_df_mirror, cutoff_year)
    yearwise_locality = compute_yearwise_accuracy(pred_df_locality, cutoff_year)

    save_json(yearwise_main, os.path.join(model_result_dir, f"yearwise_main_{split_name}.json"))
    save_json(yearwise_mirror, os.path.join(model_result_dir, f"yearwise_mirror_{split_name}.json"))
    save_json(yearwise_locality, os.path.join(model_result_dir, f"yearwise_locality_{split_name}.json"))

    plot_yearwise_accuracy(yearwise_main, f"{split_name} | Main accuracy by year", os.path.join(model_result_dir, f"yearwise_main_{split_name}.png"))
    plot_yearwise_accuracy(yearwise_mirror, f"{split_name} | Mirror accuracy by year", os.path.join(model_result_dir, f"yearwise_mirror_{split_name}.png"))
    plot_yearwise_accuracy(yearwise_locality, f"{split_name} | Locality accuracy by year", os.path.join(model_result_dir, f"yearwise_locality_{split_name}.png"))

    row_majority_main = compute_row_level_majority_accuracy(pred_df_main)
    split_metrics = {
        "main": metrics_main,
        "mirror": metrics_mirror,
        "locality": metrics_locality,
        "row_level_main": row_majority_main,
    }
    save_json(split_metrics, os.path.join(model_result_dir, f"metrics_{split_name}.json"))
    return split_metrics


def evaluate_increment(
    model,
    tokenizer,
    retriever,
    increment_label: str,
    strategy: str,
    batch_df: pd.DataFrame,
    batch_size: int,
    max_length: int,
    top_k: int,
    rounds: int,
    strategy_result_dir: str,
    cutoff_year: int,
    max_eval_records_per_type: int = 0,
    checkpoint_every: int = 0,
    open_generation_max_new_tokens: int = 256,
    judge_model: Optional[str] = None,
    judge_workers: int = 4,
    judge_enabled: bool = True,
):
    safe_label = safe_increment_label(increment_label)
    print(f"\n================ EVALUATING INCREMENT: {increment_label} ================\n", flush=True)

    eval_records_main = limit_records(build_eval_records_main(batch_df), max_eval_records_per_type)
    eval_records_mirror = limit_records(build_eval_records_mirror(batch_df), max_eval_records_per_type)
    eval_records_locality = limit_records(build_eval_records_locality(batch_df), max_eval_records_per_type)
    eval_records_open = limit_records(build_eval_records_open_generation(batch_df), max_eval_records_per_type)
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

    metrics_main, pred_df_main = evaluate_and_save_predictions(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        eval_records=eval_records_main,
        batch_size=batch_size,
        max_length=max_length,
        top_k=top_k,
        rounds=rounds,
        out_csv=main_csv,
        checkpoint_every=checkpoint_every,
    )
    metrics_mirror, pred_df_mirror = evaluate_and_save_predictions(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        eval_records=eval_records_mirror,
        batch_size=batch_size,
        max_length=max_length,
        top_k=top_k,
        rounds=rounds,
        out_csv=mirror_csv,
        checkpoint_every=checkpoint_every,
    )
    metrics_locality, pred_df_locality = evaluate_and_save_predictions(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        eval_records=eval_records_locality,
        batch_size=batch_size,
        max_length=max_length,
        top_k=top_k,
        rounds=rounds,
        out_csv=locality_csv,
        checkpoint_every=checkpoint_every,
    )
    metrics_open, pred_df_open = evaluate_and_save_open_generations(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        eval_records=eval_records_open,
        batch_size=batch_size,
        max_length=max_length,
        top_k=top_k,
        rounds=rounds,
        out_csv=open_csv,
        checkpoint_every=checkpoint_every,
        max_new_tokens=open_generation_max_new_tokens,
        judge_model=judge_model,
        judge_workers=judge_workers,
        judge_enabled=judge_enabled,
    )

    pred_dfs = [df for df in [pred_df_main, pred_df_mirror, pred_df_locality, pred_df_open] if df is not None and len(df) > 0]
    if pred_dfs:
        combined_pred_df = pd.concat(pred_dfs, ignore_index=True)
    else:
        combined_pred_df = pd.DataFrame()
    combined_pred_df.to_csv(combined_csv, index=False)

    yearwise_main = compute_yearwise_accuracy(pred_df_main, cutoff_year)
    yearwise_mirror = compute_yearwise_accuracy(pred_df_mirror, cutoff_year)
    yearwise_locality = compute_yearwise_accuracy(pred_df_locality, cutoff_year)
    row_majority_main = compute_row_level_majority_accuracy(pred_df_main)

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
        "row_level_main": row_majority_main,
        "yearwise": {
            "main": yearwise_main,
            "mirror": yearwise_mirror,
            "locality": yearwise_locality,
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


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=DEFAULT_DATA_PATH, help="hf://<org>/MedKIT or a local CSV")
    ap.add_argument("--data", default=None, help="Backward-compatible alias for --data_path")
    ap.add_argument("--model_folder", required=True)
    ap.add_argument("--cutoff", default="2022-12-31")
    ap.add_argument("--val_start", default="2022-01-01")
    ap.add_argument("--result_root", required=True)
    ap.add_argument("--cache_dir", default=None, help="HF model cache (default: HF_HOME / ~/.cache/huggingface)")
    ap.add_argument("--offload_dir", default="offload_agentic", help="Scratch dir for accelerate weight offloading")
    ap.add_argument("--retrieval_top_k", type=int, default=5)
    ap.add_argument("--retrieval_rounds", type=int, default=2)
    ap.add_argument("--retrieval_corpus", choices=["train_only", "pre_cutoff_all"], default="pre_cutoff_all")
    ap.add_argument("--max_corpus_docs", type=int, default=0)
    ap.add_argument("--eval_splits", default="all,pre_cutoff,post_cutoff")
    ap.add_argument("--max_eval_records_per_type", type=int, default=0)
    ap.add_argument("--checkpoint_every", type=int, default=25)
    ap.add_argument("--open_generation_max_new_tokens", type=int, default=256)
    ap.add_argument("--judge_model", default="google/gemini-2.0-flash-001")
    ap.add_argument("--judge_workers", type=int, default=4)
    ap.add_argument("--judge_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--filter_conflicting_edits", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--batching_strategy", choices=["none", "daily", "weekly", "monthly", "quarterly", "publication"], default="none")
    ap.add_argument("--crop_year", type=int, default=None)
    ap.add_argument("--n_increments", type=int, default=None)
    args = ap.parse_args()

    if args.data is not None:  # legacy alias
        args.data_path = args.data
    args.data_path = resolve_data_path(args.data_path)
    validate_data_layout(args.data_path)

    if args.model_folder not in BASE_MODELS:
        raise ValueError(f"Unknown model folder: {args.model_folder}")

    base_model = BASE_MODELS[args.model_folder]
    eval_batch_size, eval_max_length = eval_settings_for_model(args.model_folder)
    cutoff_year = pd.Timestamp(args.cutoff).year
    model_result_dir = os.path.join(args.result_root, args.model_folder)
    strategy_result_dir = os.path.join(model_result_dir, args.batching_strategy)
    os.makedirs(model_result_dir, exist_ok=True)
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
        f"[batching] built {len(increments)} increments: "
        f"first={increments[0]} last={increments[-1]}",
        flush=True,
    )

    print(f"[info] model_folder={args.model_folder}")
    print(f"[info] base_model={base_model}")
    print(f"[info] cutoff={args.cutoff}")
    print(f"[info] val_start={args.val_start}")
    print(f"[info] result_dir={model_result_dir}")
    print(f"[info] strategy_result_dir={strategy_result_dir}")
    print(f"[info] retrieval_top_k={args.retrieval_top_k}")
    print(f"[info] retrieval_rounds={args.retrieval_rounds}")
    print(f"[info] retrieval_corpus={args.retrieval_corpus}")
    print(f"[info] eval_splits={args.eval_splits}")
    print(f"[info] max_eval_records_per_type={args.max_eval_records_per_type}")
    print(f"[info] checkpoint_every={args.checkpoint_every}")
    print(f"[info] open_generation_max_new_tokens={args.open_generation_max_new_tokens}")
    print(f"[info] judge_model={args.judge_model} judge_workers={args.judge_workers} judge_enabled={args.judge_enabled}")

    lower = base_model.lower()
    trust_remote_code = any(x in lower for x in ["ii-medical", "adaptllm"])

    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        cache_dir=args.cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=not trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"

    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    if "27b" in args.model_folder.lower():
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=args.cache_dir,
            offload_folder=args.offload_dir,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
            device_map={"": 0},
            cache_dir=args.cache_dir,
            trust_remote_code=trust_remote_code,
            attn_implementation="sdpa",
        )

    embed_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embed_size:
        model.resize_token_embeddings(len(tokenizer))

    model.eval()

    run_config = {
        "method": "agentic_search",
        "answer_selection": "constrained_label_logprob_scoring",
        "base_model": base_model,
        "model_folder": args.model_folder,
        "eval_batch_size": eval_batch_size,
        "eval_max_length": eval_max_length,
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_rounds": args.retrieval_rounds,
        "retrieval_corpus": args.retrieval_corpus,
        "eval_splits": args.eval_splits,
        "max_eval_records_per_type": args.max_eval_records_per_type,
        "checkpoint_every": args.checkpoint_every,
        "open_generation_max_new_tokens": args.open_generation_max_new_tokens,
        "batching_strategy": args.batching_strategy,
        "crop_year": args.crop_year,
        "n_increments": args.n_increments,
        "increments": [str(x) for x in increments],
        "data_path": args.data_path,
        "cutoff": args.cutoff,
        "val_start": args.val_start,
    }
    save_json(run_config, os.path.join(strategy_result_dir, "run_config.json"))

    all_results = {}
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

        if len(batch_df) == 0:
            metrics_path = os.path.join(strategy_result_dir, f"{safe_increment_label(increment)}_metrics.json")
            empty_metrics = {
                "increment": str(increment),
                "batching_strategy": args.batching_strategy,
                "batch_rows": 0,
                "skipped": True,
                "reason": "No rows remained after prepare_df filtering.",
            }
            save_json(empty_metrics, metrics_path)
            all_results[str(increment)] = empty_metrics
            print(f"[evaluation] saved metrics to {metrics_path}", flush=True)
            continue

        corpus_df = batch_df[["evidence"]].copy().join(batch_df.drop(columns=["evidence"], errors="ignore"))
        corpus_df = corpus_df.drop_duplicates(subset=["evidence"]).copy()
        if args.max_corpus_docs and args.max_corpus_docs > 0:
            corpus_df = corpus_df.head(args.max_corpus_docs).copy()
        retriever = TfidfRetriever(
            texts=corpus_df["evidence"].astype(str).tolist(),
            row_indices=corpus_df.index.astype(int).tolist(),
        )

        increment_metrics = evaluate_increment(
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            increment_label=str(increment),
            strategy=args.batching_strategy,
            batch_df=batch_df,
            batch_size=eval_batch_size,
            max_length=eval_max_length,
            top_k=args.retrieval_top_k,
            rounds=args.retrieval_rounds,
            strategy_result_dir=strategy_result_dir,
            cutoff_year=cutoff_year,
            max_eval_records_per_type=args.max_eval_records_per_type,
            checkpoint_every=args.checkpoint_every,
            open_generation_max_new_tokens=args.open_generation_max_new_tokens,
            judge_model=args.judge_model,
            judge_workers=args.judge_workers,
            judge_enabled=args.judge_enabled,
        )
        increment_metrics["retrieval_corpus_docs"] = int(len(corpus_df))
        all_results[str(increment)] = increment_metrics

    summary_metrics = {
        "model_folder": args.model_folder,
        "base_model": base_model,
        "method": "agentic_search",
        "cutoff": args.cutoff,
        "val_start": args.val_start,
        "eval_batch_size": eval_batch_size,
        "eval_max_length": eval_max_length,
        "retrieval_top_k": args.retrieval_top_k,
        "retrieval_rounds": args.retrieval_rounds,
        "retrieval_corpus": args.retrieval_corpus,
        "eval_splits": args.eval_splits,
        "max_eval_records_per_type": args.max_eval_records_per_type,
        "checkpoint_every": args.checkpoint_every,
        "open_generation_max_new_tokens": args.open_generation_max_new_tokens,
        "batching_strategy": args.batching_strategy,
        "crop_year": args.crop_year,
        "n_increments": args.n_increments,
        "num_increments": int(len(increments)),
        "increments": all_results,
    }
    summary_path = os.path.join(strategy_result_dir, "metrics.json")
    save_json(summary_metrics, summary_path)
    print(f"[evaluation] saved metrics to {summary_path}", flush=True)
    print(json.dumps(summary_metrics, indent=2))

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
