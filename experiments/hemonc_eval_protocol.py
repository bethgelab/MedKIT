import concurrent.futures
import json
import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


VALID_LABELS = {"superior", "inferior", "no difference"}


def detect_refusal(text):
    if not text or not isinstance(text, str):
        return False
    stripped = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return bool(
        re.search(
            r"(i(?:'m| am)\s+(?:sorry|afraid|not\s+able|unable)|"
            r"i\s+(?:cannot|can'?t|am\s+not\s+able|am\s+unable|won'?t|refuse)\b|"
            r"as\s+an\s+ai(?:\s+language)?\s+model|"
            r"(?:consult|speak\s+(?:to|with)|seek|talk\s+to|ask)\s+(?:a|your|an?)\s+"
            r"(?:doctor|healthcare\s+(?:provider|professional)|physician|medical\s+professional|oncologist|clinician))",
            stripped,
            flags=re.IGNORECASE,
        )
    )


def normalize_label(x) -> Optional[str]:
    x = str(x).strip().lower()
    return x if x in VALID_LABELS else None


def filter_conflicting_edits_df(df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["condition", "context", "endpoint", "regimen", "comparator"]
    if not all(c in df.columns for c in key_cols + ["answer", "date"]):
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "conflicting_edit" in df.columns:
        is_conflicting = df["conflicting_edit"].astype(bool)
    else:
        is_conflicting = df.groupby(key_cols)["answer"].transform("nunique") > 1
    latest = df[is_conflicting].sort_values("date").groupby(key_cols, sort=False).tail(1)
    return pd.concat([df[~is_conflicting], latest]).sort_index().reset_index(drop=True)


def load_increment_df(
    data_path: str,
    strategy: str = "none",
    increment: Optional[str] = None,
    crop_year: Optional[int] = None,
    filter_conflicting_edits: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    if strategy != "none" and increment is not None and increment != "hemonc_full":
        from hemonc_batching import filter_by_increment

        df = filter_by_increment(df, strategy=strategy, increment=increment, crop_year=crop_year)
    elif crop_year is not None:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df[df["date"].dt.year >= int(crop_year)].reset_index(drop=True)
    if filter_conflicting_edits:
        before = len(df)
        df = filter_conflicting_edits_df(df)
        print(f"  [filter_conflicting_edits] {before - len(df)} rows removed, {len(df)} rows remain")
    if len(df) == 0:
        raise ValueError(f"No rows loaded for increment={increment!r}, strategy={strategy!r}, crop_year={crop_year!r}")
    return df


def load_csv_data(
    data_path,
    include_evidence=True,
    increment=None,
    batching_cfg=None,
    _df_override=None,
    filter_conflicting_edits=True,
):
    if _df_override is not None:
        df = _df_override.copy()
    else:
        strategy = getattr(batching_cfg, "strategy", "none") if batching_cfg is not None else "none"
        crop_year = getattr(batching_cfg, "crop_year", None) if batching_cfg is not None else None
        df = load_increment_df(data_path, strategy, increment, crop_year, filter_conflicting_edits)

    has_locality = "locality_question" in df.columns
    has_cq45 = "closed question 4" in df.columns and "closed question 5" in df.columns
    has_cq_m = "closed question m" in df.columns
    records = []
    for _, row in df.iterrows():
        evidence = str(row["evidence"])
        ground_truth_statement = str(row["ground truth"])

        def make_prompt(q, _ev=evidence, _gts=ground_truth_statement):
            if include_evidence in (True, "abstract"):
                return f"Evidence: {_ev}\n\n{str(q)}"
            if include_evidence == "ground_truth":
                return f"Evidence: {_gts}\n\n{str(q)}"
            return str(q)

        answer = str(row["answer"])
        gt_dict = {
            "condition": str(row["condition"]),
            "context": str(row["context"]),
            "target": ground_truth_statement,
        }
        record = {
            "src": make_prompt(row["closed question 1"]),
            "alt": answer,
            "subject": f"{row['regimen']} compared to {row['comparator']}",
            "evidence": evidence,
            "ground_truth_statement": ground_truth_statement,
            "tag": f"{row['date']}|{row['regimen']}|{row['comparator']}|{row['condition']}|{row['context']}|{row['endpoint']}",
            "date": str(row["date"]),
            "rephrase": make_prompt(row["closed question 2"]),
            "genv2_cq_2": {"prompt": make_prompt(row["closed question 2"]), "ground_truth": answer},
            "genv2_cq_3": {"prompt": make_prompt(row["closed question 3"]), "ground_truth": answer},
            "genv2_oq_1": {"prompt": make_prompt(row["open question 1"]), "ground_truth": gt_dict},
            "genv2_og_1": {"prompt": make_prompt(row["open generation 1"]), "ground_truth": gt_dict},
        }
        if has_cq45:
            record["genv2_cq_4"] = {"prompt": make_prompt(row["closed question 4"]), "ground_truth": answer}
            record["genv2_cq_5"] = {"prompt": make_prompt(row["closed question 5"]), "ground_truth": answer}
        if has_cq_m:
            record["genv2_cq_m"] = {"prompt": make_prompt(row["closed question m"]), "ground_truth": str(row["closed question m answer"])}
        if has_locality:
            loc_q = row["locality_question"]
            loc_a = row["locality_ground_truth"]
            record["loc"] = make_prompt(str(loc_q)) if not pd.isna(loc_q) else None
            record["loc_ans"] = str(loc_a) if not pd.isna(loc_a) else None
        records.append(record)
    return records


def extract_data(test_data):
    prompts = [r["src"] for r in test_data]
    rephrase_prompts = [r["rephrase"] for r in test_data] if "rephrase" in test_data[0] else None
    target_new = [r["alt"] for r in test_data]
    tags = [r.get("tag", "") for r in test_data]
    subject = [r["subject"] for r in test_data]
    evidence = [r["evidence"] for r in test_data]
    locality_inputs = {"neighborhood": {"prompt": [], "ground_truth": []}}
    if "loc" in test_data[0]:
        locality_inputs["neighborhood"]["prompt"] = [r["loc"] for r in test_data]
        locality_inputs["neighborhood"]["ground_truth"] = [r["loc_ans"] for r in test_data]
    else:
        locality_inputs = None
    portability_inputs = {}
    for key in test_data[0].keys():
        if key.startswith("genv2_"):
            portability_inputs[key] = {
                "prompt": [r[key]["prompt"] for r in test_data],
                "ground_truth": [r[key]["ground_truth"] for r in test_data],
            }
    return prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject, evidence


def llm_as_judge(mode: str, answers: List[str], ground_truth: List[dict], judge_model: str, max_workers: int = 4):
    instructions_open_qa = """
You are a medical expert tasked with evaluating how well a model's response, comparing two treatment options, aligns with a ground truth.
Rate alignment on a 1-5 Likert scale.

Format your answer as:
Score: <1-5>
Explanation: <brief justification>
"""
    instructions_open_gen = """
You are a medical expert evaluating whether a model's free-text treatment recommendation is consistent with a known pairwise clinical comparison.
Score 1-5. Also report:
Flags:
- mentions_A: <YES/NO>
- mentions_B: <YES/NO>
- preference: <A preferred / B preferred / No clear preference / Neither mentioned>

Output format:
Score: <1-5>
Flags:
- mentions_A: <YES/NO>
- mentions_B: <YES/NO>
- preference: <A preferred / B preferred / No clear preference / Neither mentioned>
Explanation: <brief justification>
"""
    instructions_closed = "Answer with exactly one word: YES if the response conveys the correct choice, NO otherwise."
    instructions = instructions_open_qa if mode == "open_qa" else instructions_open_gen if mode == "open_gen" else instructions_closed
    aliases = {
        "gpt-4o-hemonc": "openai/gpt-4o",
        "gpt-4o-mini-hemonc": "openai/gpt-4o-mini",
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
    }
    openrouter_model = aliases.get(judge_model, judge_model)
    from openai import OpenAI
    import openai as _openai

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ.get("OPENROUTER_API_KEY"), max_retries=0)
    if openrouter_model.endswith(":free"):
        max_workers = 1

    def _judge_one(args):
        idx, resp_text, gt = args
        for n_try in range(10):
            try:
                if mode == "closed_qa_fallback":
                    task = f"Question: {gt['question']}\nCorrect Answer: Option {gt['target']}\nModel Response: {resp_text}"
                else:
                    task = f"Condition: {gt['condition']}\nContext: {gt['context']}\nGround Truth: {gt['target']}\nResponse: {resp_text}"
                api_resp = client.chat.completions.create(
                    model=openrouter_model,
                    messages=[{"role": "user", "content": f"Instructions:\n{instructions}\n\nTask:\n{task}\n\n"}],
                )
                raw = api_resp.choices[0].message.content.strip()
                if mode == "closed_qa_fallback":
                    return idx, 1.0 if "YES" in raw.upper() else 0.0, raw, None
                score_match = re.search(r"Score\s*:\s*(\d+)", raw, flags=re.IGNORECASE)
                if not score_match:
                    raise ValueError(f"Could not parse judge score from: {raw}")
                score = int(score_match.group(1))
                if score > 5 and score > 10 and score <= 15:
                    score -= 10
                if score < 1 or score > 5:
                    raise ValueError(f"Score out of range: {score}")
                explanation = raw.split("Explanation", 1)[-1].lstrip(": \n") if "Explanation" in raw else raw
                flags = None
                if mode == "open_gen":
                    flags = {"mentions_A": None, "mentions_B": None, "preference": None}
                    ma = re.search(r"mentions_A\s*:\s*(YES|NO)", raw, re.IGNORECASE)
                    mb = re.search(r"mentions_B\s*:\s*(YES|NO)", raw, re.IGNORECASE)
                    pref = re.search(r"preference\s*:\s*(A preferred|B preferred|No clear preference|Neither mentioned)", raw, re.IGNORECASE)
                    if ma:
                        flags["mentions_A"] = ma.group(1).upper()
                    if mb:
                        flags["mentions_B"] = mb.group(1).upper()
                    if pref:
                        flags["preference"] = pref.group(1)
                return idx, score, explanation, flags
            except _openai.RateLimitError as e:
                wait = min(300.0, 5.0 * (2 ** n_try))
                logging.getLogger(__name__).warning("llm_as_judge rate limit: sleeping %.0fs — %s", wait, e)
                time.sleep(wait)
            except Exception as e:
                wait = min(120.0, 5.0 * (2 ** n_try))
                logging.getLogger(__name__).warning("llm_as_judge error: sleeping %.0fs — %s", wait, e)
                time.sleep(wait)
        return idx, None, None, None

    args_list = [(i, resp, gt) for i, (resp, gt) in enumerate(zip(answers, ground_truth))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_judge_one, args_list))
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results], [r[2] for r in results], [r[3] for r in results]


def judge_open_rows(rows: List[dict], judge_model: str, judge_workers: int = 4, judge_enabled: bool = True) -> List[dict]:
    if not judge_enabled or not rows:
        for row in rows:
            row["judge_score"] = None
            row["judge_explanation"] = None
            row["judge_flags"] = None
            row["correct"] = ""
        return rows
    for eval_type, mode in [("open_question", "open_qa"), ("open_generation", "open_gen")]:
        idxs = [i for i, r in enumerate(rows) if r.get("eval_type") == eval_type]
        if not idxs:
            continue
        answers = [rows[i].get("raw_generation", "") for i in idxs]
        gts = [rows[i].get("judge_ground_truth", {}) for i in idxs]
        scores, explanations, flags = llm_as_judge(mode, answers, gts, judge_model, judge_workers)
        for i, score, exp, flg in zip(idxs, scores, explanations, flags):
            rows[i]["judge_score"] = score
            rows[i]["judge_explanation"] = exp
            rows[i]["judge_flags"] = json.dumps(flg) if flg is not None else None
            rows[i]["correct"] = int(score is not None and score >= 4)
    return rows


def _extract_sample_metrics(pm):
    portability = pm.get("portability", {})
    sample_cq = {}
    for m_key, p_key, q_num in [
        ("rewrite_acc", None, 1),
        ("rephrase_acc", None, 2),
        (None, "genv2_cq_3", 3),
        (None, "genv2_cq_4", 4),
        (None, "genv2_cq_5", 5),
        (None, "genv2_cq_m", "m"),
    ]:
        if m_key is not None and m_key in pm:
            val = pm[m_key]
            sample_cq[q_num] = float(val[0] if isinstance(val, list) else val)
        elif p_key in portability:
            perf = portability[p_key].get("performance", {})
            acc = perf.get("acc") if isinstance(perf, dict) else None
            if acc is not None:
                sample_cq[q_num] = float(acc[0] if isinstance(acc, list) else acc)
    oq = portability.get("genv2_oq_1", {}).get("performance", {})
    og = portability.get("genv2_og_1", {}).get("performance", {})
    loc = pm.get("locality", {}).get("neighborhood", {}).get("performance", {})
    return {
        "closed": sample_cq,
        "open_question_score": float(oq["score"]) / 5.0 if isinstance(oq, dict) and oq.get("score") is not None else None,
        "open_generation_score": float(og["score"]) / 5.0 if isinstance(og, dict) and og.get("score") is not None else None,
        "locality_acc": float(loc.get("acc")) if isinstance(loc, dict) and loc.get("acc") is not None else None,
    }


def _phase_summary_stats(phase_pms):
    closed_by_q = {1: [], 2: [], 3: [], 4: [], 5: [], "m": []}
    oq, og, loc = [], [], []
    for pm in phase_pms:
        s = _extract_sample_metrics(pm)
        for q, v in s["closed"].items():
            closed_by_q[q].append(v)
        if s["open_question_score"] is not None:
            oq.append(s["open_question_score"])
        if s["open_generation_score"] is not None:
            og.append(s["open_generation_score"])
        if s["locality_acc"] is not None:
            loc.append(s["locality_acc"])
    stats = {}
    labels = {1: "q1", 2: "q2", 3: "q3", 4: "q4", 5: "q5", "m": "qm"}
    means = []
    for q, vals in closed_by_q.items():
        if vals:
            stats[f"Closed Questions/{labels[q]}"] = float(np.mean(vals))
            means.append(float(np.mean(vals)))
    if means:
        stats["Closed Questions/mean"] = float(np.mean(means))
    if oq:
        stats["Open Questions/score"] = float(np.mean(oq))
    if og:
        stats["Open Generation/score"] = float(np.mean(og))
    if loc:
        stats["Locality/acc"] = float(np.mean(loc))
    return stats

