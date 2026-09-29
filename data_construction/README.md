# MedKIT — Dataset construction

This subproject reconstructs the MedKIT benchmark from the upstream
HemOnc Knowledge Base snapshot and PubMed abstracts. The output is the
same parquet that is published on the Hugging Face Hub as
`bethgelab/MedKIT`; this code lets you verify the construction
end-to-end.

## What the pipeline does

1. Loads the HemOnc tables (`ref.table.csv`, `study_results.csv`,
   `indications.csv`, `efficacy.xlsx`, `stage.xlsx`) from
   `raw_hemonc/`.
2. Normalizes free-text efficacy into a 3-class label
   (`{superior, inferior, no difference}`) via a frozen vocabulary
   lookup plus a regex rule map.
3. Selects one canonical row per `(study, condition, regimen,
   comparator, context)` group using endpoint priority (Primary >
   Co-primary > Secondary > Undesignated; OS > PFS > DFS > ...).
4. Builds both directions of every comparison (`R vs C`, `C vs R`) so
   the relational generalization probe is materialized.
5. Fetches the supporting PubMed abstracts for every cited PMID via
   the NCBI Entrez API. Abstracts are cached in `raw_hemonc/docs.csv`,
   which is shipped pre-populated; the fetch only re-triggers if the
   cache is incomplete.
6. Renders the seven probes per update from a fixed prompt-template
   library (anchor + 2 lexical paraphrases + relational mirror +
   compositional open QA + operational open generation + locality).
7. Deduplicates, drops contradictory rows, attaches stage strings,
   collapses mirror pairs, and pairs each update with a same-group
   locality probe.
8. Flags rows whose
   `(condition, context, endpoint, regimen, comparator)` appears with
   disagreeing labels across studies as `conflicting_edit=True`.

A 12-step description with rationale is in `docs/construction_summary.md`.

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended.

## Run

The pipeline is a single command:

```bash
# from data_construction/
python process_medkit.py \
    --raw-dir raw_hemonc \
    --out-csv ./medkit_v4.csv

python add_conflict_flag.py \
    --in-csv ./medkit_v4.csv \
    --out-csv ./medkit_v4_conflict_flagged.csv
```

If any abstracts are missing from the bundled `raw_hemonc/docs.csv`
cache, the pipeline fetches them from NCBI Entrez, which requires a
real contact email. Set `ENTREZ_EMAIL` in that case (the script exits
with an error if it needs to fetch and the variable is unset):

```bash
export ENTREZ_EMAIL="you@example.org"   # use your real email when running
```

The resulting `medkit_v4_conflict_flagged.csv` contains the same
comparisons as the parquet on the Hugging Face Hub; see
[Reproducibility](#reproducibility) for the exact relationship.

Expected runtime on a typical workstation: ~5 minutes if `docs.csv` is
already populated (the default with the snapshot bundled here);
~30–60 minutes the first time if the abstract cache needs to be
rebuilt from Entrez.

## Sequential batching is built at experiment time

There is no precomputed batch-assignment file in this subproject.
Sequential editing experiments construct chronological batches
on-the-fly inside the experiment harness (see
`experiments/hemonc_batching.py`, called from `experiments/run_medkit.py`)
based on the `qa.batching.strategy` field of the active Hydra config
(`none` / `publication` / `daily` / `weekly` / `monthly` / `quarterly`).
Re-running an experiment with a different strategy is therefore as
simple as flipping that config field — no preprocessing step is
required between dataset construction and experiment.

## Outputs

After running both scripts you get:

- `medkit_v4.csv` — the 6,196-row benchmark before the conflict flag.
- `medkit_v4_conflict_flagged.csv` — same with the boolean
  `conflicting_edit` column added.

### Reproducibility

The pipeline is deterministic: repeated runs produce byte-identical
output, independent of Python's hash seed. The published parquet was
built with an earlier revision that differs in two ways:

1. **Hash-dependent row order.** Its row order depended on Python's
   hash randomization, which in turn affected the seeded random choices
   downstream.
2. **Oncology-group assignment.** It matched group keywords as
   substrings, so the hematologic acronym `all` also matched conditions
   such as "non-*small* cell lung cancer" or "*locally* advanced". This
   put 1,105 updates (17.8%), including every lung-cancer update, in
   the hematologic group, so their locality probes were drawn from
   hematologic conditions. Short keywords (acronyms) now match as whole
   words.

A fresh run therefore matches the published parquet on the benchmark
content (the same 6,196 rows, 5,861 comparisons and gold labels) but
not on every sampled field:

- the direction kept for a mirror pair (`R vs C` vs. `C vs R`) differs
  for about 43% of rows,
- the locality probe differs for most rows; in a fresh run 99.8% of
  locality probes come from the update's own oncology group, versus
  82.2% in the published parquet, and
- a handful of `conflicting_edit` flags differ (159 vs. 154), since
  conflicts are detected per direction.

The Hugging Face parquet is the canonical version of MedKIT and is
what all reported experiments use; load it directly to reproduce the
paper's numbers.

## License and attribution

- Code: Apache 2.0 (see top-level `LICENSE`).
- Bundled HemOnc snapshot: CC BY-NC-SA 4.0; see `raw_hemonc/NOTICE.md`.
- PubMed abstracts: courtesy of the U.S. National Library of Medicine;
  see NLM terms of use.
