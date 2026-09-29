#!/usr/bin/env python3
"""
MedKIT dataset construction pipeline.

Reads the HemOnc Knowledge Base tables + cached PubMed abstracts and emits
the flat benchmark CSV used by the experiments. Self-contained — no other
preprocessing scripts are required (run `add_conflict_flag.py` afterwards
to attach the `conflicting_edit` boolean column).

Pipeline steps:
  1. Load HemOnc tables (`ref.table.csv`, `study_results.csv`,
     `indications.csv`, `efficacy.xlsx`, `stage.xlsx`); drop NaNs on required
     keys; deduplicate.
  2. Normalize free-text efficacy into the 3-class label
     {superior, inferior, no difference} via a frozen vocabulary lookup
     plus a regex rule map.
  3. Endpoint-aware canonicalization: rank by endpoint type
     (Primary > Co-primary > Secondary > Undesignated) and clinical priority
     (OS > PFS > DFS > EFS > RFS > FFS > TTP > rPFS > other); keep one
     canonical row per (study, condition, regimen, comparator, context).
  4. Build both directions of every comparison `(R, C, label)` and
     `(C, R, flipped_label)` so the relational generalization probe is
     materialized.
  5. Attach disease stage information from the indications table.
  6. Fetch (or load from cache) the PubMed abstracts of every cited PMID.
  7. Render the seven probes per update from a fixed library of prompt
     templates (anchor + 2 lexical paraphrases + relational mirror +
     compositional + operational + locality).
  8. Drop exact duplicates and any contradictory rows.
  9. Generate locality probes via a deterministic four-level cascade over
     oncology groups.
 10. Collapse mirror direction-pairs into one primary row plus a stored
     mirrored question.
 11. Locality fallback for any rows still missing a probe after collapse.

Random choices use a fixed seed so the construction is fully reproducible.

Usage:
    python process_medkit.py --raw_dir raw_hemonc --output medkit_v4.csv
"""
from __future__ import annotations

import argparse
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import pandas as pd
from Bio import Entrez
from tqdm import tqdm

# NCBI requires a real contact email for Entrez calls; only needed when the
# bundled abstract cache (raw_hemonc/docs.csv) is incomplete.
Entrez.email = os.environ.get("ENTREZ_EMAIL")

# ---------------------------------------------------------------------------
# Endpoint priority tables
# ---------------------------------------------------------------------------

ENDPOINT_TYPE_RANK: dict[str, int] = {
    "Primary": 0,
    "Co-primary": 1,
    "Secondary": 2,
    "Undesignated": 3,
}  # NaN → 4

ENDPOINT_RANK: dict[str, int] = {
    "OS": 0,
    "PFS": 1,
    "DFS": 2,
    "EFS": 3,
    "RFS": 4,
    "FFS": 5,
    "TTP": 6,
    "rPFS": 7,
}  # unlisted → 99


def _endpoint_sort_key(row: pd.Series) -> tuple[int, int]:
    type_rank = ENDPOINT_TYPE_RANK.get(row["endpoint_type"], 4)
    ep_rank = ENDPOINT_RANK.get(str(row["endpoint"]), 99)
    return (type_rank, ep_rank)


# ---------------------------------------------------------------------------
# Rule-based efficacy fallback
# ---------------------------------------------------------------------------
# The new HemOnc study_results.csv (v2+) uses a richer, free-text vocabulary
# for the `efficacy` field (e.g. "Longer OS", "Did not meet primary endpoint of
# PFS") that is not covered by the legacy efficacy.xlsx mapping.  These rules
# provide a pattern-based fallback that correctly labels ~96 % of the otherwise
# unmapped rows.
# ---------------------------------------------------------------------------

_SUPERIOR_PATS: list[str] = [
    r"^longer\b", r"^higher\b", r"^better\b", r"^improved\b",
    r"^more\b", r"^greater\b", r"^increased\b",
    r"^seems to have longer\b", r"^seems to have higher\b",
    r"^seems to have better\b", r"^seems to have more\b",
    r"^seems to have greater\b",
    r"^might have longer\b", r"^might have higher\b",
    r"^might have better\b", r"^might have more\b",
    r"^most likely has longer\b", r"^most likely has higher\b",
    r"^most likely has better\b", r"^most likely has more\b",
    r"^most likely has greater\b", r"^most likely has increased\b",
    r"^most likely has improved\b",
]
_INFERIOR_PATS: list[str] = [
    r"^shorter\b", r"^lower\b", r"^worse\b", r"^fewer\b",
    r"^less\b", r"^decreased\b",
    r"^seems to have shorter\b", r"^seems to have lower\b",
    r"^seems to have worse\b", r"^seems to have fewer\b",
    r"^seems to have less\b",
    r"^might have shorter\b", r"^might have lower\b",
    r"^might have worse\b", r"^might have fewer\b",
    r"^most likely has shorter\b", r"^most likely has lower\b",
    r"^most likely has worse\b", r"^most likely has fewer\b",
    r"^most likely has less\b", r"^most likely has decreased\b",
]
_NODIFF_PATS: list[str] = [
    r"^did not meet\b", r"^does not meet\b", r"^failed to meet\b",
    r"^no (?:significant |statistically )?difference\b",
    r"^similar\b", r"^comparable\b", r"^equivalent\b",
    r"^non-inferior\b", r"^not significant\b",
    r"^met.*non-inferiority",
]

_SUPERIOR_RE = re.compile("|".join(_SUPERIOR_PATS), re.IGNORECASE)
_INFERIOR_RE = re.compile("|".join(_INFERIOR_PATS), re.IGNORECASE)
_NODIFF_RE   = re.compile("|".join(_NODIFF_PATS),   re.IGNORECASE)


def _rule_map_efficacy(value: str) -> str | None:
    """Return 'Superior' | 'Inferior' | 'No Difference' | None via regex rules.

    Called only when the efficacy.xlsx lookup returns 'Other' or no match,
    to handle the richer free-text vocabulary introduced in HemOnc v2+.
    Returns None for values that are genuinely ambiguous or uninformative
    (e.g. 'Not reported', 'TBD …', 'Inconclusive').
    """
    v = str(value).strip()
    if _NODIFF_RE.search(v):
        return "No Difference"
    if _SUPERIOR_RE.search(v):
        return "Superior"
    if _INFERIOR_RE.search(v):
        return "Inferior"
    return None


# ---------------------------------------------------------------------------
# Oncology group mapping (for locality questions)
# ---------------------------------------------------------------------------

ONCOLOGY_GROUPS: dict[str, list[str]] = {
    "Hematologic": [
        # classic hematologic malignancies
        "leukemia", "lymphoma", "myeloma", "myeloid", "myelodysplastic",
        "hodgkin", "mds", "cll", "cml", "aml", "all", "myeloproliferative",
        "waldenstrom", "macroglobulinemia", "plasmacytoma",
        # myeloid / bone-marrow disorders
        "myelofibrosis", "polycythemia", "thrombocythemia", "mastocytosis",
        "eosinophilic", "hypereosinophilic", "aplastic anemia",
        # red-cell / haemoglobin disorders
        "sickle cell", "thalassemia", "hemolytic anemia", "hemolytic anaemia",
        "cold agglutinin", "warm autoimmune hemolytic", "paroxysmal nocturnal",
        "hemoglobinuria", "pyruvate kinase",
        # plasma-cell & related
        "amyloidosis", "castleman",
        # platelet / coagulation disorders
        "thrombocytopenia", "thrombocytopenic purpura", "antiphospholipid",
        "coagulopathy", "hemophilia", "haemophilia", "von willebrand",
        "venous thromboembolism", "thromboembolism", "pulmonary embolism",
        # transplant / graft complications
        "graft versus host", "graft-versus-host", "gvhd",
        "stem cell mobilization", "hsct",
        # vascular hematologic
        "hereditary hemorrhagic telangiectasia", "vasculitis",
    ],
    "Lung": ["lung", "nsclc", "sclc", "mesothelioma"],
    "GI": [
        "colorectal", "gastric", "esophageal", "pancreatic",
        "hepatocellular", "cholangiocarcinoma", "biliary", "colon",
        "rectal", "gastrointestinal", "duodenal", "appendix", "anal",
        "hepatoblastoma", "hepatic veno-occlusive",
    ],
    "Breast": ["breast"],
    "GU": [
        "prostate", "bladder", "renal", "urothelial", "kidney", "testicular",
        "wilms",
    ],
    "Gynecologic": [
        "cervical", "ovarian", "endometrial", "uterine", "vulvar", "fallopian",
        "gestational trophoblastic", "trophoblastic",
    ],
    "HeadNeck": [
        "head and neck", "nasopharyngeal", "oropharyngeal", "laryngeal",
        "thyroid", "salivary", "oral cavity", "hypopharyngeal",
    ],
    "CNS": [
        "brain", "glioma", "glioblastoma", "meningioma", "cns",
        "medulloblastoma", "neuroblastoma", "subependymal giant cell",
    ],
    "Melanoma": ["melanoma", "skin", "merkel", "basal cell"],
    "Sarcoma": [
        "sarcoma", "gastrointestinal stromal",
        "desmoid", "tenosynovial giant cell",
    ],
    "NET": [
        "neuroendocrine", "pheochromocytoma", "paraganglioma", "carcinoid",
        "adrenocortical",
    ],
}


def _keyword_matches(keyword: str, cond_lower: str) -> bool:
    # Short keywords are acronyms (ALL, AML, MDS, ...) and must match as whole
    # words: as substrings, "all" would match "small cell lung cancer" or
    # "locally advanced".  Longer keywords match as substrings so that e.g.
    # "sarcoma" still covers "osteosarcoma".
    if len(keyword) <= 4:
        return re.search(rf"\b{re.escape(keyword)}\b", cond_lower) is not None
    return keyword in cond_lower


def condition_to_group(condition: str) -> str | None:
    cond_lower = condition.lower()
    for group, keywords in ONCOLOGY_GROUPS.items():
        if any(_keyword_matches(kw, cond_lower) for kw in keywords):
            return group
    return None


# ---------------------------------------------------------------------------
# PubMed fetch
# ---------------------------------------------------------------------------

def get_text(pmid: str, retries: int = 50) -> tuple[str, str]:
    for i in range(retries):
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="text")
            record = Entrez.read(handle)
            article = record["PubmedArticle"][0]["MedlineCitation"]["Article"]
            doc = article["ArticleTitle"]
            if "Abstract" in article:
                doc = " ".join([doc] + article["Abstract"]["AbstractText"])
            return pmid, doc
        except HTTPError as e:
            if e.code == 429:
                time.sleep(i)
            else:
                raise Exception(
                    f"Failed to fetch data for PMID {pmid} after {retries} retries."
                )
    raise Exception(f"Failed to fetch data for PMID {pmid} after {retries} retries.")


# ---------------------------------------------------------------------------
# Mirror-pair helpers  (ported from build_hemonc_v3.py)
# ---------------------------------------------------------------------------

def flip_answer(answer: int) -> int:
    """1↔2, 3→3."""
    if answer == 1:
        return 2
    if answer == 2:
        return 1
    return answer


def flip_answer_str(answer: str) -> str:
    """superior↔inferior, no difference unchanged."""
    a = str(answer).strip().lower()
    if a == "superior":
        return "inferior"
    if a == "inferior":
        return "superior"
    return "no difference"


def find_mirror_pairs(group_df: pd.DataFrame) -> list[tuple[int, int]]:
    rows = list(group_df.itertuples())
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i, row_a in enumerate(rows):
        if row_a.Index in used:
            continue
        for j, row_b in enumerate(rows):
            if i == j or row_b.Index in used:
                continue
            if row_a.regimen == row_b.comparator and row_a.comparator == row_b.regimen:
                pairs.append((row_a.Index, row_b.Index))
                used.add(row_a.Index)
                used.add(row_b.Index)
                break
    return pairs


# ---------------------------------------------------------------------------
# Q&A templates
# ---------------------------------------------------------------------------

# Endpoint-specific variants (used when endpoint is known)
CLOSED_SYSTEM_PROMPT = """You are a knowledgeable medical assistant supporting oncologists and hematologists in evaluating treatment options.

Given a clinical question, compare two treatment options with respect to a specific outcome endpoint and determine their relative efficacy based on current clinical evidence.

Answer with exactly one of the following three options:
- superior      (Treatment 1 outperforms Treatment 2 on the specified endpoint)
- inferior      (Treatment 1 underperforms Treatment 2 on the specified endpoint)
- no difference (no meaningful difference between the two treatments on the specified endpoint)

Do not include any explanation or additional text.

Task:
{}
Response:"""
OPEN_SYSTEM_PROMPT = """You are a knowledgeable medical assistant supporting oncologists and hematologists in evaluating treatment options.

Given a condition, context, and endpoint, compare two treatment options with respect to the specific outcome endpoint and summarize their relative efficacy based on current clinical evidence.

Task:
{}
Response:"""
CLOSED_QN_EP = [
    "Choose an option that best describes the {ENDPOINT} outcome of {REGIMEN} compared to {COMPARATOR} when used to treat {CONDITION} ({CONTEXT}).",
    "Select the option that most accurately reflects the {ENDPOINT} outcome of {REGIMEN} versus {COMPARATOR} in treating {CONDITION} ({CONTEXT}).",
    "Which option best summarizes the {ENDPOINT} results of {REGIMEN} compared to {COMPARATOR} for {CONDITION} ({CONTEXT})?",
]
OPEN_QN_EP = (
    "Condition: {CONDITION}, Context: {CONTEXT}, Endpoint: {ENDPOINT}, "
    "Treatment 1: {REGIMEN}, Treatment 2: {COMPARATOR}"
)

# Endpoint-agnostic fallback (used when endpoint is missing)
CLOSED_QN_NOEP = [
    "Choose an option that best describes the efficacy of {REGIMEN} compared to {COMPARATOR} when used to treat {CONDITION} ({CONTEXT}).",
    "Select the option that most accurately reflects the effectiveness of {REGIMEN} versus {COMPARATOR} in treating {CONDITION} ({CONTEXT}).",
    "Which option best summarizes the comparative efficacy of {REGIMEN} and {COMPARATOR} for managing {CONDITION} ({CONTEXT})?",
]
OPEN_QN_NOEP = (
    "Condition: {CONDITION}, Context: {CONTEXT}, "
    "Treatment 1: {REGIMEN}, Treatment 2: {COMPARATOR}"
)

# Open generation template (unchanged)
OPEN_GEN = """
    You are a knowledgeable medical assistant supporting oncologists and hematologists in evaluating treatment options.\n
    Given a condition and clinical context, your task is to provide a concise overview over the relevant components of a treatment that is consistent with current clinical guidelines.\n
    Be as specific as possible, including: (1) Drug components, (2) Timing and sequencing, (3) Dosage and duration, (4) Route of administration.\n\n
    Condition: {CONDITION}, Context: {CONTEXT}\n
    Treatment: 
    """

# Template used to synthesise mirror questions for orphaned rows
MIRROR_Q_TEMPLATE_EP = (
    "Choose an option that best describes the {ENDPOINT} outcome of {REGIMEN} "
    "compared to {COMPARATOR} when used to treat {CONDITION} ({CONTEXT})."
)
MIRROR_Q_TEMPLATE_NOEP = (
    "Choose an option that best describes the efficacy of {REGIMEN} "
    "compared to {COMPARATOR} when used to treat {CONDITION} ({CONTEXT})."
)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_medkit(
    raw_dir: str = "raw_hemonc",
    output_path: str = "medkit_v4.csv",
    seed: int = 42,
) -> None:
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1. Load source tables
    # ------------------------------------------------------------------
    columns_ref = ["study", "condition", "pmid", "pub.date"]
    columns_result = [
        "study", "condition", "regimen", "comparator",
        "efficacy", "context", "endpoint", "endpoint_type",
    ]
    columns_indication = ["study", "condition", "stage_or_status"]

    preprocess = lambda df, col: (
        df[col].dropna().drop_duplicates().sort_values(col).reset_index(drop=True)
    )

    ref = preprocess(pd.read_csv(f"{raw_dir}/ref.table.csv"), columns_ref)
    indication = preprocess(pd.read_csv(f"{raw_dir}/indications.csv"), columns_indication)

    # For study_results: only require the key columns; endpoint/endpoint_type may be NaN.
    _result_raw = pd.read_csv(f"{raw_dir}/study_results.csv")[columns_result]
    _required = ["study", "condition", "regimen", "comparator", "efficacy", "context"]
    result = (
        _result_raw.dropna(subset=_required)
        .drop_duplicates()
        .sort_values(_required)
        .reset_index(drop=True)
    )

    efficacy_df = pd.read_excel(f"{raw_dir}/efficacy.xlsx")
    efficacy2label: dict[str, str] = dict(
        zip(efficacy_df["efficacy_raw"], efficacy_df["efficacy_std"])
    )
    efficacy2label = {
        k: (v if "Might Be " not in v else v[len("Might Be "):])
        for k, v in efficacy2label.items()
    }

    def _normalize_efficacy(value: str) -> str | None:
        """Resolve an efficacy string to 'Superior' | 'Inferior' | 'No Difference' | None.

        Priority:
          1. efficacy.xlsx lookup (covers the legacy HemOnc vocabulary).
          2. Regex rule-map (covers the richer free-text vocabulary in HemOnc v2+).
          3. None  → row is discarded (uninformative / ambiguous values).
        """
        std = efficacy2label.get(value)
        if std is not None and std != "Other":
            return std
        return _rule_map_efficacy(value)

    stage_df = pd.read_excel(f"{raw_dir}/stage.xlsx")
    stage2label: dict[str, str] = dict(zip(stage_df["stage_raw"], stage_df["stage_std"]))

    # ------------------------------------------------------------------
    # 2. Build study → PMIDs / dates maps
    # ------------------------------------------------------------------
    study2pmids: dict[tuple, set] = defaultdict(set)
    study2dates: dict[tuple, set] = defaultdict(set)
    for _, row in ref.iterrows():
        key = (row["study"], row["condition"])
        study2pmids[key].add(row["pmid"])
        study2dates[key].add(row["pub.date"])

    # ------------------------------------------------------------------
    # 3. Resolve canonical endpoint per comparison
    #    Group by (study, condition, regimen, comparator, context) and keep
    #    the single row with the highest-priority endpoint.
    # ------------------------------------------------------------------
    group_cols = ["study", "condition", "regimen", "comparator", "context"]
    result = result.dropna(subset=["efficacy"])

    canonical_rows: list[pd.Series] = []
    for _, grp in result.groupby(group_cols, sort=False):
        # Filter to rows with a valid (non-None) efficacy label
        grp = grp[grp["efficacy"].map(lambda x: _normalize_efficacy(x) is not None)]
        if grp.empty:
            continue
        # Sort by endpoint priority and pick the first (best) row
        grp = grp.copy()
        grp["_sort_key"] = grp.apply(_endpoint_sort_key, axis=1)
        best = grp.sort_values("_sort_key").iloc[0]
        canonical_rows.append(best)

    result_canonical = pd.DataFrame(canonical_rows).reset_index(drop=True)
    print(
        f"  study_results: {len(result):,} rows → {len(result_canonical):,} canonical rows "
        f"after endpoint dedup."
    )

    # ------------------------------------------------------------------
    # 4. Build study2result
    # ------------------------------------------------------------------
    study2result: dict[tuple, set] = defaultdict(set)
    for _, row in result_canonical.iterrows():
        key = (row["study"], row["condition"])
        if key not in study2pmids:
            continue
        if row["regimen"] == row["comparator"]:
            continue

        label = _normalize_efficacy(row["efficacy"])
        if label is None:
            continue  # should not happen after Step 3 filter, but guard anyway
        switched = (
            label.replace("Inferior", "Superior")
            if "Inferior" in label
            else label.replace("Superior", "Inferior")
        )
        date = min(
            study2dates[key],
            key=lambda d: datetime.strptime(d, "%Y-%m-%d"),
        )
        endpoint = row["endpoint"] if pd.notna(row.get("endpoint")) else ""
        ep_type = row["endpoint_type"] if pd.notna(row.get("endpoint_type")) else ""

        study2result[key].add(
            (row["regimen"], row["comparator"], label, date, row["context"], endpoint, ep_type)
        )
        study2result[key].add(
            (row["comparator"], row["regimen"], switched, date, row["context"], endpoint, ep_type)
        )

    # Remove studies from study2pmids that have no results
    for study in set(study2pmids.keys()) - set(study2result.keys()):
        del study2pmids[study]

    # ------------------------------------------------------------------
    # 5. Build study2stage
    # ------------------------------------------------------------------
    study2stage: dict[tuple, set] = defaultdict(set)
    for _, row in indication.iterrows():
        key = (row["study"], row["condition"])
        if key not in study2result:
            continue
        study2stage[key].add(stage2label.get(row["stage_or_status"], row["stage_or_status"]))

    # ------------------------------------------------------------------
    # 6. Load / fetch PubMed documents
    # ------------------------------------------------------------------
    pmid2doc: dict = {}
    path_doc = f"{raw_dir}/docs.csv"
    if os.path.exists(path_doc):
        docs_df = pd.read_csv(path_doc)
        pmid2doc = dict(zip(docs_df["pmid"], docs_df["doc"]))

    all_pmids = {pmid for pmids in study2pmids.values() for pmid in pmids}
    pmids_missing = all_pmids - set(pmid2doc.keys())
    pmids_missing |= {
        pmid for pmid, doc in pmid2doc.items()
        if not isinstance(doc, str) and isinstance(doc, float) and np.isnan(doc)
    }
    # HemOnc uses 999999999x placeholders for references without a PubMed
    # record (e.g. pre-MEDLINE papers); there is nothing to fetch for them.
    pmids_missing = {pmid for pmid in pmids_missing if int(pmid) < 9_999_999_990}

    if pmids_missing:
        if not Entrez.email:
            raise RuntimeError(
                f"{len(pmids_missing)} abstracts are missing from {path_doc} and must be "
                "fetched from NCBI Entrez. Set the ENTREZ_EMAIL environment variable "
                "to your email address and re-run."
            )
        pmid2doc_missing: dict = {}
        with ThreadPoolExecutor(max_workers=10) as exe:
            futures = [exe.submit(get_text, pmid) for pmid in pmids_missing]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching PMIDs"):
                try:
                    pmid, doc = future.result()
                    pmid2doc_missing[pmid] = doc
                except Exception as e:
                    print(f"  Fetch error: {e}")
        pmid2doc.update(pmid2doc_missing)
        pmid2doc_df = pd.DataFrame(list(pmid2doc.items()), columns=["pmid", "doc"])
        pmid2doc_df.to_csv(path_doc, index=False)

    # ------------------------------------------------------------------
    # 7. Build flat dataset
    # ------------------------------------------------------------------
    option2idx = {"superior": 1, "inferior": 2, "no difference": 3}
    dataset_rows: list[dict] = []

    def _sort_pmids(pmids):
        """Sort PMIDs numerically when possible, lexicographically as a
        fallback. Returns a deterministic ordering across runs (independent
        of Python's hash randomization)."""
        return sorted(
            pmids,
            key=lambda p: (0, int(p)) if str(p).isdigit() else (1, str(p)),
        )

    for key, values in study2result.items():
        # Build evidence and pmids in lock-step over a deterministically
        # sorted PMID list. Skip PMIDs whose document is missing/empty so
        # that `evidence` chunks correspond 1:1 to `pmids` entries.
        sorted_pmids = _sort_pmids(study2pmids[key])
        ev_chunks: list[str] = []
        kept_pmids: list[str] = []
        for pmid in sorted_pmids:
            doc = pmid2doc.get(pmid, "")
            if not isinstance(doc, str) or not doc:
                continue
            ev_chunks.append(doc)
            kept_pmids.append(str(pmid))
        if not ev_chunks:
            continue
        evidence = "\n\n".join(ev_chunks)
        pmids_str = ";".join(kept_pmids)
        stage = "" if key not in study2stage else " ({})".format(
            ", ".join(sorted(study2stage[key]))
        )
        condition = key[1] + stage

        # `values` is a set: sort it so row order (and hence the downstream
        # locality picks and seeded mirror-pair draws) is independent of
        # Python's hash randomization.
        for regimen, comparator, efficacy, date, context, endpoint, ep_type in sorted(
            values, key=lambda t: tuple(map(str, t))
        ):
            if regimen == comparator:
                continue

            fmt = {
                "REGIMEN": regimen,
                "COMPARATOR": comparator,
                "CONDITION": condition,
                "CONTEXT": context,
                "ENDPOINT": endpoint,
            }

            # Choose template set based on endpoint availability
            has_endpoint = bool(endpoint)
            if has_endpoint and endpoint != "Could not be determined":
                closed_qns = [t.format(**fmt) for t in CLOSED_QN_EP]
                open_qn = OPEN_QN_EP.format(**fmt)
            else:
                closed_qns = [t.format(**fmt) for t in CLOSED_QN_NOEP]
                open_qn = OPEN_QN_NOEP.format(**fmt)

            open_gen = OPEN_GEN.format(CONDITION=condition, CONTEXT=context)

            closed_answer = option2idx[efficacy.lower()]

            # Ground truth includes endpoint when available
            if has_endpoint:
                ground_truth = (
                    f"{regimen} {efficacy.lower()} to {comparator} "
                    f"for {condition} ({context}) [endpoint: {endpoint}]"
                )
            else:
                ground_truth = (
                    f"{regimen} {efficacy.lower()} to {comparator} "
                    f"for {condition} ({context})"
                )

            dataset_rows.append({
                "date": date,
                "evidence": evidence,
                "pmids": pmids_str,
                "regimen": regimen,
                "comparator": comparator,
                "condition": condition,
                "context": context,
                "endpoint": endpoint,
                "endpoint_type": ep_type,
                "answer": efficacy.lower(),
                "ground truth": ground_truth,
                "closed question 1": CLOSED_SYSTEM_PROMPT.format(closed_qns[0]),
                "closed question 2": CLOSED_SYSTEM_PROMPT.format(closed_qns[1]),
                "closed question 3": CLOSED_SYSTEM_PROMPT.format(closed_qns[2]),
                "open question 1": OPEN_SYSTEM_PROMPT.format(open_qn),
                "open generation 1": open_gen,
                "option 1": "superior",
                "option 2": "inferior",
                "option 3": "no difference",
            })

    df = pd.DataFrame(dataset_rows)

    # Drop exact duplicates (same question + answer); keep=False removes
    # contradictory pairs where same question yields different answers.
    subset_dedup = [
        "date", "evidence", "regimen", "comparator", "condition",
        "context", "closed question 1", "answer",
    ]
    before = len(df)
    df = df.dropna().drop_duplicates(subset=subset_dedup[:-1], keep=False).reset_index(drop=True)
    print(
        f"  Dataset: {before:,} rows → {len(df):,} after dedup "
        f"(removed {before - len(df):,} conflicting/duplicate rows)."
    )

    # Extract year
    df["year"] = pd.to_datetime(df["date"]).dt.year

    # ------------------------------------------------------------------
    # 8. Locality question generation
    # ------------------------------------------------------------------
    print("Generating locality questions ...")
    df = _add_locality_questions(df)

    # ------------------------------------------------------------------
    # 9. Collapse mirror pairs
    # ------------------------------------------------------------------
    print("Collapsing mirror pairs ...")
    df = _collapse_mirror_pairs(df, rng)

    # ------------------------------------------------------------------
    # 10. Locality fallback for rows still missing a locality question
    # ------------------------------------------------------------------
    print("Applying locality fallback ...")
    df = _apply_locality_fallback(df)

    # ------------------------------------------------------------------
    # 11. Write output
    # ------------------------------------------------------------------
    col_order = [
        "date", "evidence", "pmids", "regimen", "comparator", "condition", "context",
        "endpoint", "endpoint_type",
        "answer", "ground truth",
        "closed question 1", "closed question 2", "closed question 3",
        "closed question m", "closed question m answer",
        "open question 1", "open generation 1",
        "option 1", "option 2", "option 3", "year",
        "locality_question", "locality_ground_truth",
    ]
    df = df[[c for c in col_order if c in df.columns]]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    # ------------------------------------------------------------------
    # 12. Summary
    # ------------------------------------------------------------------
    total = len(df)
    ep_filled = df["endpoint"].astype(str).str.strip().ne("").sum()
    cq_m_filled = df["closed question m"].astype(str).str.strip().ne("").sum()
    loc_filled = df["locality_question"].astype(str).str.strip().ne("").sum()

    print(f"\nSaved {total:,} rows → {out}")
    print(f"  endpoint populated   : {ep_filled:,} / {total:,} ({100*ep_filled/total:.1f}%)")
    print(f"  closed question m    : {cq_m_filled:,} / {total:,} ({100*cq_m_filled/total:.1f}%)")
    print(f"  locality_question    : {loc_filled:,} / {total:,} ({100*loc_filled/total:.1f}%)")
    print(f"  answer distribution  : {df['answer'].value_counts().to_dict()}")


# ---------------------------------------------------------------------------
# Locality question helpers
# ---------------------------------------------------------------------------

def _add_locality_questions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a locality question to every row using a cascade of criteria:

    1. Same oncology group, different condition, different context, different answer
    2. Same oncology group, different condition  (relax context/answer)
    3. Cross-group: any condition, different context, different answer
    4. Cross-group: any other condition  (most permissive)
    """
    df = df.copy()
    df["locality_question"] = ""
    df["locality_ground_truth"] = ""

    df["_group"] = df["condition"].apply(condition_to_group)
    by_group: dict[str, pd.DataFrame] = {
        g: sub for g, sub in df.groupby("_group", sort=False) if g is not None
    }

    filled = 0
    for idx in df.index:
        row = df.loc[idx]
        group = row["_group"]
        best = None

        # Levels 1–2: same oncology group
        if group is not None and group in by_group:
            pool = by_group[group]
            cands = pool[
                (pool["condition"] != row["condition"])
                & (pool["context"] != row["context"])
                & (pool["answer"] != row["answer"])
            ]
            if not cands.empty:
                best = cands.sort_values("date", kind="stable").iloc[0]
            else:
                cands = pool[pool["condition"] != row["condition"]]
                if not cands.empty:
                    best = cands.sort_values("date", kind="stable").iloc[0]

        # Levels 3–4: cross-group fallback
        if best is None:
            pool = df[df["condition"] != row["condition"]]
            cands = pool[
                (pool["context"] != row["context"])
                & (pool["answer"] != row["answer"])
            ]
            if not cands.empty:
                best = cands.sort_values("date", kind="stable").iloc[0]
            elif not pool.empty:
                best = pool.sort_values("date", kind="stable").iloc[0]

        if best is not None:
            df.at[idx, "locality_question"] = best["open question 1"]
            df.at[idx, "locality_ground_truth"] = best["ground truth"]
            filled += 1

    df = df.drop(columns=["_group"])
    print(f"  Locality questions generated: {filled:,} / {len(df):,} rows.")
    return df


# ---------------------------------------------------------------------------
# Mirror pair collapsing
# ---------------------------------------------------------------------------

def _make_output_row(primary: pd.Series, mirror_q: str, mirror_ans: int) -> dict:
    row: dict = primary.to_dict()
    row["closed question m"] = mirror_q
    row["closed question m answer"] = mirror_ans
    return row


def _collapse_mirror_pairs(df: pd.DataFrame, rng: random.Random) -> pd.DataFrame:
    """Collapse (R vs C) and (C vs R) siblings into one row per comparison.

    Grouping key includes endpoint + endpoint_type + frozenset{regimen, comparator}
    so that rows differing only in endpoint stay in separate groups — preventing
    the cross-endpoint mirror contamination that the naïve (date, evidence,
    condition, context) grouping caused.

    Every output row gets a freshly synthesized `closed question m`: the
    closed-question template with regimen/comparator swapped, wrapped in the
    same system prompt as the primary cq1. `closed question m answer` is the
    mechanical flip of the primary's answer.
    """
    df = df.reset_index(drop=True).copy()
    df["_cmp_key"] = [frozenset([r, c]) for r, c in zip(df["regimen"], df["comparator"])]
    group_keys = [
        "date", "evidence", "condition", "context",
        "endpoint", "endpoint_type", "_cmp_key",
    ]

    output_rows: list[dict] = []
    n_pairs = 0
    n_orphans = 0
    for _, grp in df.groupby(group_keys, sort=False, dropna=False):
        if len(grp) == 1:
            primary = grp.iloc[0]
            n_orphans += 1
        else:
            # 2+ rows: the two directions of the same comparison on the same
            # endpoint. Pick one direction uniformly at random as primary.
            primary = grp.iloc[0] if rng.random() < 0.5 else grp.iloc[1]
            n_pairs += 1

        endpoint = str(primary.get("endpoint", "")).strip()
        fmt = {
            "REGIMEN": primary["comparator"],  # swapped for mirror
            "COMPARATOR": primary["regimen"],  # swapped for mirror
            "CONDITION": primary["condition"],
            "CONTEXT": primary["context"],
            "ENDPOINT": endpoint,
        }
        if endpoint and endpoint != "Could not be determined":
            mirror_raw = MIRROR_Q_TEMPLATE_EP.format(**fmt)
        else:
            mirror_raw = MIRROR_Q_TEMPLATE_NOEP.format(
                **{k: v for k, v in fmt.items() if k != "ENDPOINT"}
            )
        mirror_q = CLOSED_SYSTEM_PROMPT.format(mirror_raw)
        mirror_ans = flip_answer_str(primary["answer"])

        primary = primary.drop("_cmp_key")
        output_rows.append(_make_output_row(primary, mirror_q, mirror_ans))

    result = pd.DataFrame(output_rows)
    print(
        f"  Mirror collapse: {len(df):,} → {len(result):,} rows "
        f"({n_pairs} paired, {n_orphans} orphans)."
    )
    return result


# ---------------------------------------------------------------------------
# Locality fallback (for rows that still have no locality question)
# ---------------------------------------------------------------------------

def _apply_locality_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing locality questions using a cascade of increasingly relaxed criteria:

    1. Same oncology group, different condition, different context, different answer
    2. Same oncology group, different condition  (relax context/answer constraint)
    3. Any condition (cross-group), different condition, different context, different answer
    4. Any condition (cross-group), different condition  (most permissive)

    Levels 3–4 apply to rows whose condition has no oncology group match, or whose
    group has no valid same-group candidates.
    """
    df = df.copy()
    df["locality_question"] = df["locality_question"].fillna("").astype(str)
    df["locality_ground_truth"] = df["locality_ground_truth"].fillna("").astype(str)

    missing_mask = df["locality_question"].str.strip() == ""
    if not missing_mask.any():
        print("  No rows with missing locality questions — fallback not needed.")
        return df

    n_before = missing_mask.sum()
    df["_group"] = df["condition"].apply(condition_to_group)
    by_group: dict[str, pd.DataFrame] = {
        g: sub for g, sub in df.groupby("_group", sort=False) if g is not None
    }

    filled = 0
    for idx in df.index[missing_mask]:
        row = df.loc[idx]
        group = row["_group"]

        best = None

        # Levels 1–2: same oncology group
        if group is not None and group in by_group:
            pool = by_group[group]
            # Level 1: strict
            cands = pool[
                (pool["condition"] != row["condition"])
                & (pool["context"] != row["context"])
                & (pool["answer"] != row["answer"])
            ]
            if not cands.empty:
                best = cands.sort_values("date", kind="stable").iloc[0]
            else:
                # Level 2: relax context/answer
                cands = pool[pool["condition"] != row["condition"]]
                if not cands.empty:
                    best = cands.sort_values("date", kind="stable").iloc[0]

        # Levels 3–4: cross-group (all other conditions in the dataset)
        if best is None:
            pool = df[df["condition"] != row["condition"]]
            # Level 3: strict cross-group
            cands = pool[
                (pool["context"] != row["context"])
                & (pool["answer"] != row["answer"])
            ]
            if not cands.empty:
                best = cands.sort_values("date", kind="stable").iloc[0]
            else:
                # Level 4: any other condition
                if not pool.empty:
                    best = pool.sort_values("date", kind="stable").iloc[0]

        if best is not None:
            df.at[idx, "locality_question"] = best["open question 1"]
            df.at[idx, "locality_ground_truth"] = best["ground truth"]
            filled += 1

    df = df.drop(columns=["_group"])
    n_after = (df["locality_question"].str.strip() == "").sum()
    print(
        f"  Locality fallback: filled {filled:,} rows "
        f"({n_before:,} missing → {n_after:,} still missing)."
    )
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified HemOnc preprocessing pipeline (v2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--raw_dir", "--raw-dir",
        dest="raw_dir",
        default="raw_hemonc",
        help="Folder containing the HemOnc CSV/XLSX tables and docs.csv (default: raw_hemonc)",
    )
    parser.add_argument(
        "--output", "--out-csv",
        dest="output",
        default="medkit_v4.csv",
        help="Output CSV path (default: medkit_v4.csv)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    process_medkit(
        raw_dir=args.raw_dir,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
