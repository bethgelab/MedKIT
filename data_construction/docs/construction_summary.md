# MedKIT Benchmark Construction — Detailed Summary

## 1. Source data

**HemOnc Knowledge Base**, snapshot **2026-03-12**, distributed via the
Harvard Dataverse under CC BY-NC-SA 4.0. The frozen snapshot used by
this release lives under [`../raw_hemonc/`](../raw_hemonc/).

**Tables consumed by the pipeline:**

| Table | Role |
|-------|------|
| `study_results.csv` | Central results table: one row per `(study, condition, regimen, comparator, context, endpoint)` with a free-text `efficacy` field and `endpoint` / `endpoint_type` (Primary/Co-primary/Secondary/Undesignated). |
| `ref.table.csv` | Bibliographic metadata: `study`, `pmid`, `pub.date`, `condition`. |
| `indications.csv` | Per-study `stage_or_status` (e.g. advanced, metastatic). |
| `efficacy.xlsx` | Lookup mapping raw efficacy strings → standardized labels (`Superior`, `Inferior`, `No Difference`, `Other`, `Might Be X`). |
| `stage.xlsx` | Stage-name normalization. |
| `docs.csv` | Cached PubMed abstracts keyed by PMID. Pre-populated in this snapshot; the pipeline back-fills missing PMIDs via NCBI Entrez on first run. |

Other HemOnc tables (variants, drugs, conditions, authors, sigs, etc.)
are present in `raw_hemonc/` for completeness but are not consumed by
the construction pipeline.

**Semantics**: each `study_results` row is "in `study` S, on `condition`
C (`context` X), `regimen` R compared to `comparator` C′ showed `efficacy`
E on `endpoint` EP".

## 2. Pipeline

Entry point: [`../process_medkit.py`](../process_medkit.py). Run-once
post-step: [`../add_conflict_flag.py`](../add_conflict_flag.py).

**Step 1 — Load & filter source tables.** Read `ref`, `indications`,
`study_results`; drop NaNs on required keys; deduplicate.

**Step 2 — Efficacy normalization (free-text → 3-class label).**
- Priority 1: `efficacy.xlsx` lookup (covers legacy HemOnc vocabulary;
  strip "Might Be " prefix to collapse hedged labels into their base
  class).
- Priority 2: regex rule-map fallback for free-text values not in the
  lookup. Patterns:
  - **Superior**: "longer", "higher", "better", "improved", "more",
    "greater", "increased" (+ hedged variants).
  - **Inferior**: "shorter", "lower", "worse", "fewer", "less",
    "decreased".
  - **No Difference**: "did not meet", "no significant difference",
    "similar", "non-inferior", "met non-inferiority", etc.
- Uninformative values ("Not reported", "Inconclusive", "TBD") are
  discarded.

**Step 3 — Endpoint-aware canonicalization.**
- Group rows by `(study, condition, regimen, comparator, context)`.
- When multiple rows report different endpoints for the same comparison,
  keep one canonical row via priority ranking:
  - `endpoint_type`: Primary > Co-primary > Secondary > Undesignated
    (NaN last).
  - `endpoint`: OS > PFS > DFS > EFS > RFS > FFS > TTP > rPFS > other.
- This removes endpoint redundancy while preserving the clinically most
  relevant outcome per comparison.

**Step 4 — Build comparison pairs in both directions.**
For each canonical row, add both `(R vs C, label)` and
`(C vs R, flipped_label)` to a `study → set(results)` map.
`Superior ↔ Inferior`; `No Difference` is unchanged. Each comparison
is dated by the earliest `pub.date` across its associated PMIDs.

**Step 5 — Attach disease stage.** Look up `indications.stage_or_status`
via `stage.xlsx` and append to the condition string as `" (stageA, stageB)"`.

**Step 6 — Fetch PubMed evidence.** For every PMID linked to a kept
study, ensure the abstract is in `docs.csv` (download via NCBI Entrez
if missing; 10-thread pool, retries). Per-edit `evidence` is the
deterministically-ordered concatenation of the abstracts of all PMIDs
tied to that `(study, condition)`. PMIDs are sorted numerically; missing
abstracts are skipped so each `evidence` chunk corresponds 1:1 to a PMID
in the row's `pmids` column.

**Step 7 — Generate question variants for each comparison.**
- Endpoint-aware (`endpoint` known and ≠ "Could not be determined") vs.
  endpoint-agnostic templates.
- **Closed questions (3 paraphrases)** wrapped in a shared system prompt
  that instructs the model to answer one of `{superior, inferior,
  no difference}`:
  - Endpoint-aware: "Choose an option that best describes the
    {ENDPOINT} outcome of {REGIMEN} compared to {COMPARATOR} when used
    to treat {CONDITION} ({CONTEXT})." + two rephrasings.
  - Endpoint-agnostic: analogous templates referring to "efficacy" /
    "effectiveness".
- **Open question** — a structured prompt
  (condition / context / endpoint / treatment 1 / treatment 2) under the
  same system prompt family.
- **Open generation** — treatment-description task asking for the
  guideline-concordant regimen for the (condition, context), including
  components, timing, dosage, route.
- **Options**: `option 1 = superior`, `option 2 = inferior`,
  `option 3 = no difference`.
- **Ground-truth string** includes the endpoint when available:
  `"{R} {label} to {C} for {condition} ({context}) [endpoint: {EP}]"`.

**Step 8 — Deduplication.**
- Drop exact duplicates on `(date, evidence, regimen, comparator,
  condition, context, closed question 1)`.
- Use `keep=False` on the same subset so any *contradictory* rows
  (same setup, inconsistent label across study reports) are dropped
  entirely, preventing ambiguous edits from entering the benchmark.

**Step 9 — Locality probe generation.**
For each row, find a candidate edit that is "close but unrelated" via
a deterministic four-level cascade:
1. Same oncology group, different condition, different context,
   different answer.
2. Same oncology group, different condition (relax context / answer).
3. Cross-group, different condition, different context, different
   answer.
4. Cross-group, any different condition.

Candidate selection is deterministic (sort by date, stable). The
oncology-group mapping covers 12 groups (Hematologic, Lung, GI, Breast,
GU, Gynecologic, HeadNeck, CNS, Melanoma, Sarcoma, NET, Other) defined
by curated keyword lists in `process_medkit.py`. The locality question
is the *open question* of the candidate; `locality_ground_truth` is the
candidate's ground-truth string.

**Step 10 — Mirror pair collapsing.**
- Group rows by `(date, evidence, condition, context, endpoint,
  endpoint_type, frozenset({regimen, comparator}))`.
- Within each group, find the `(R vs C)` and `(C vs R)` siblings created
  in Step 4.
- With a fixed RNG seed (`42`), pick one direction as *primary*; store
  the other's closed-question text and answer as `closed question m` /
  `closed question m answer`. This gives each edit a built-in
  rephrase / reversal probe while halving the row count.
- Orphans (single-direction rows with no mirror partner) get a
  *synthesized* mirror question via the template, with the answer
  flipped.

**Step 11 — Locality fallback.** For any row still missing a locality
probe after the mirror collapse, re-run the four-level cascade.

**Step 12 — Conflict flag** ([`add_conflict_flag.py`](../add_conflict_flag.py)).
Mark rows where the same `(condition, context, endpoint, regimen,
comparator)` appears multiple times across different studies / dates
with disagreeing answers. Stored as `conflicting_edit` (boolean) — used
downstream as a clean / contested split.

## 3. Final benchmark

After running `process_medkit.py` followed by `add_conflict_flag.py`,
the output CSV (default `medkit_v4_conflict_flagged.csv`) carries
**6,196 rows × 25 columns**.

**Columns per row (one factual update):**

- **Provenance** — `date`, `year`, `evidence` (concatenated PubMed
  abstracts in PMID-sorted order), `pmids` (semicolon-separated PMID
  list, same order as `evidence` chunks).
- **Clinical fact** — `regimen`, `comparator`, `condition` (with stage
  suffix), `context`, `endpoint`, `endpoint_type`.
- **Label** — `answer` ∈ {superior, inferior, no difference},
  `ground truth` (verbalized statement with endpoint when known).
- **Closed-QA probes** — `closed question 1/2/3` (three paraphrases,
  endpoint-aware or -agnostic), `option 1/2/3`, plus `closed question m`
  (mirrored direction) and `closed question m answer`.
- **Open probes** — `open question 1` (compositional), `open generation 1`
  (operational treatment description).
- **Locality** — `locality_question`, `locality_ground_truth`.
- **Conflict flag** — `conflicting_edit` (added by `add_conflict_flag.py`).

**Answer encoding**: `option 1 = superior`, `option 2 = inferior`,
`option 3 = no difference`.

The construction is byte-deterministic given the fixed inputs in
`raw_hemonc/` and the seed (`42`); re-runs produce identical output.

## 4. Temporal batching for sequential editing

The pipeline does not produce batch-assignment artifacts. Sequential
batching is constructed at experiment time by
[`../../experiments/hemonc_batching.py`](../../experiments/hemonc_batching.py),
called from `experiments/run_medkit.py` based on the active Hydra
config's `qa.batching.strategy` field.

**Supported strategies** (selected via Hydra config):
- `none` — single batch, all rows.
- `publication` — one batch per unique publication.
- `daily` — one batch per calendar day (`YYYY-MM-DD`).
- `weekly` — one batch per ISO week (`YYYY-Www`).
- `monthly` — one batch per calendar month (`YYYY-MM`).
- `quarterly` — one batch per quarter (`YYYY-Qq`).

**Optional cropping** via `qa.batching.crop_year`. The paper experiments
use `crop_year: 2025` (post-2025 records only) to minimize overlap with
model pre-training corpora when evaluating knowledge integration on
truly novel evidence.
