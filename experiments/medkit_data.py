"""Locate the MedKIT benchmark file.

Every entry point passes its configured data path through `resolve_data_path`,
which returns a local CSV path that the rest of the code reads with pandas:

  * ``hf://<org>/<name>`` — the benchmark on the Hugging Face Hub (the default,
    see DEFAULT_DATA_PATH).  Downloaded once and cached as CSV under
    ``$MEDKIT_DATA_DIR`` (default: ``experiments/data/medkit/``).
  * anything else — a local CSV, e.g. the output of
    ``data_construction/process_medkit.py`` + ``add_conflict_flag.py``.
    Relative paths are resolved against the working directory, as before.
"""

import os
from pathlib import Path

HF_PREFIX = "hf://"
DEFAULT_DATA_PATH = "hf://bethgelab/MedKIT"
CACHE_ROOT = Path(os.environ.get(
    "MEDKIT_DATA_DIR", Path(__file__).resolve().parent / "data" / "medkit"))


def resolve_data_path(path) -> str:
    path = str(path)
    if not path.startswith(HF_PREFIX):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"MedKIT data file not found: {path!r}. Use the Hugging Face copy "
                f"({DEFAULT_DATA_PATH}) or build it with data_construction/.")
        return path

    repo_id = path[len(HF_PREFIX):].strip("/")
    local = CACHE_ROOT / repo_id.replace("/", "__") / "medkit.csv"
    if not local.exists():
        from datasets import load_dataset
        print(f"[medkit_data] downloading {repo_id} from the Hugging Face Hub -> {local}")
        df = load_dataset(repo_id, split="train").to_pandas()
        local.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically so concurrent jobs never read a half-written file.
        tmp = local.with_name(f"{local.name}.{os.getpid()}.tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, local)
    return str(local)
