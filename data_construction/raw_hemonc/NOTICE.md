# HemOnc Knowledge Base — Frozen snapshot

The CSV / XLSX / TAB files in this directory are a frozen copy of the
**HemOnc Knowledge Base, snapshot 2026-03-12**, redistributed here so
that you can run `process_medkit.py` without separately
downloading from the Harvard Dataverse.

## Source

- Project: [HemOnc.org](https://hemonc.org/)
- Distribution: Harvard Dataverse (HemOncKB)

## License

CC BY-NC-SA 4.0 (Creative Commons Attribution-NonCommercial-ShareAlike
4.0 International). For academic and non-commercial users only.

Full terms: https://creativecommons.org/licenses/by-nc-sa/4.0/

The MedKIT release inherits this license through the ShareAlike clause.

## Required citation

> Warner JL, Dymshyts D, Reich CG, Gurley MJ, Hochheiser H, Moldwin ZH,
> Belenkaya R, Williams AE, Yang PC. **HemOnc: A new standard
> vocabulary for chemotherapy regimen representation in the OMOP
> common data model.** *Journal of Biomedical Informatics* 2019;
> 96:103239. doi:10.1016/j.jbi.2019.103239.

For the most up-to-date HemOnc snapshot and the canonical citation
form, refer to the HemOncKB Dataverse record.

## Contents

The files in this directory follow HemOncKB's standard naming
convention. Only the tables consumed by `process_medkit.py` are
included:

- `ref.table.csv` — bibliographic metadata (study ↔ PMID ↔ pub.date).
- `study_results.csv` — central trial results table.
- `indications.csv` — per-study stage / status.
- `efficacy.xlsx` — legacy lookup mapping free-text efficacy strings
  to standardized labels.
- `stage.xlsx` — stage-name normalization.
- `docs.csv` — pre-fetched PubMed abstracts cache (PMID ↔ abstract).

The remaining HemOncKB tables (authors, persons, drugs, variants,
etc.) are not used by the pipeline and are not redistributed here;
obtain them from the HemOncKB Dataverse record if needed.
