"""
hemonc_batching.py — Batching strategies for MedKIT sequential editing experiments.

Splits the MedKIT dataset into chronologically ordered batches that are fed to the
lifelong editing loop one at a time, simulating how new trial evidence accumulates.

Supported strategies
--------------------
none         Single batch containing all data (backward-compatible default)
publication  One batch per unique publication (unique evidence text), ~3 960 batches
daily        One batch per calendar date (YYYY-MM-DD), ~2 274 batches post-2000
weekly       One batch per ISO week  (YYYY-Www),      ~1 110 batches post-2000
monthly      One batch per calendar month (YYYY-MM),  ~  305 batches post-2000  ← recommended
quarterly    One batch per calendar quarter (YYYY-Qq), ~  100 batches post-2000

Batch labels are designed to be:
  - Chronologically sortable via plain string sort
  - Safe as filesystem path components (no special chars)
  - Human-readable at a glance

Usage
-----
    from hemonc_batching import build_increments, filter_by_increment

    # 1. Derive the ordered list of batches once at experiment start
    increments = build_increments(
        data_path="path/to/Hemonc_Edit_v4.csv",
        strategy="monthly",
        crop_year=2000,
    )
    # → ['2000-01', '2000-02', ..., '2025-05']

    # 2. Inside the increment loop, filter the full DataFrame to this batch
    batch_df = filter_by_increment(full_df, strategy="monthly", increment="2023-06")
"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

# Internal column used during label assignment (dropped before returning)
_BATCH_COL = "_batch_label"

_VALID_STRATEGIES = ("none", "publication", "daily", "weekly", "monthly", "quarterly")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with 'date' parsed as datetime if it isn't already."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _assign_labels(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Add a '_batch_label' column to df.

    Assumes df already has a datetime 'date' column and an 'evidence' column
    (only needed for the 'publication' strategy).
    """
    df = df.copy()

    if strategy == "none":
        df[_BATCH_COL] = "hemonc_full"

    elif strategy == "publication":
        # Each unique evidence text = one publication.
        # Label: pub_{YYYY-MM-DD}_{per_date_counter:04d}
        # Sort by (date, evidence) for a fully deterministic order.
        pubs = (
            df[["evidence", "date"]]
            .drop_duplicates(subset="evidence")
            .sort_values(["date", "evidence"])
            .reset_index(drop=True)
        )
        pubs["_date_str"] = pubs["date"].dt.strftime("%Y-%m-%d")
        pubs["_counter"] = pubs.groupby("_date_str").cumcount()
        pubs[_BATCH_COL] = (
            "pub_"
            + pubs["_date_str"]
            + "_"
            + pubs["_counter"].apply(lambda x: f"{x:04d}")
        )
        label_map = dict(zip(pubs["evidence"], pubs[_BATCH_COL]))
        df[_BATCH_COL] = df["evidence"].map(label_map)

    elif strategy == "daily":
        df[_BATCH_COL] = df["date"].dt.strftime("%Y-%m-%d")

    elif strategy == "weekly":
        # ISO week label: YYYY-Www  (zero-padded week number)
        # pd.Period("W") gives long interval strings; use isocalendar() instead.
        iso = df["date"].dt.isocalendar()
        df[_BATCH_COL] = (
            iso["year"].astype(str)
            + "-W"
            + iso["week"].astype(str).str.zfill(2)
        )

    elif strategy == "monthly":
        # YYYY-MM  (Period string, already sorts correctly as plain string)
        df[_BATCH_COL] = df["date"].dt.to_period("M").astype(str)

    elif strategy == "quarterly":
        # YYYY-Qq  (e.g. '2023Q3')
        df[_BATCH_COL] = df["date"].dt.to_period("Q").astype(str)

    else:
        raise ValueError(
            f"Unknown batching strategy '{strategy}'. "
            f"Valid options: {', '.join(_VALID_STRATEGIES)}"
        )

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_increments(
    data_path: str | Path,
    strategy: str = "none",
    crop_year: Optional[int] = None,
) -> list[str]:
    """
    Derive the chronologically ordered list of batch labels from the HemOnc CSV.

    Only the 'date' and 'evidence' columns are read; the full CSV is not loaded
    into memory here.

    Parameters
    ----------
    data_path : path to Hemonc_Edit_vX.csv
    strategy  : batching granularity (see module docstring)
    crop_year : if set, exclude rows with year < crop_year (e.g. 2000)

    Returns
    -------
    Sorted list of unique batch label strings in chronological order.
    """
    if strategy == "none":
        return ["hemonc_full"]

    if strategy not in _VALID_STRATEGIES:
        raise ValueError(
            f"Unknown batching strategy '{strategy}'. "
            f"Valid options: {', '.join(_VALID_STRATEGIES)}"
        )

    usecols = ["date", "evidence"] if strategy == "publication" else ["date"]
    df = pd.read_csv(data_path, usecols=usecols)
    df = _ensure_datetime(df)

    if crop_year is not None:
        df = df[df["date"].dt.year >= int(crop_year)]
        if df.empty:
            raise ValueError(
                f"No rows remain in '{data_path}' after cropping to year >= {crop_year}."
            )

    df = _assign_labels(df, strategy)
    labels = sorted(df[_BATCH_COL].dropna().unique().tolist())

    print(
        f"[hemonc_batching] strategy='{strategy}', crop_year={crop_year}: "
        f"{len(labels)} batches  "
        f"(first: '{labels[0]}'  last: '{labels[-1]}')"
    )
    return labels


def build_pre_batch_corpus(
    data_path: str | Path,
    strategy: str,
    increments: list[str],
    crop_year: Optional[int] = None,
) -> "pd.DataFrame":
    """Return all rows whose batch label is strictly before the first experiment increment.

    These records form the fixed background corpus for a RAG experiment.
    Only records chronologically prior to the first experiment increment are
    included — future batches are intentionally excluded to prevent data leakage.

    Parameters
    ----------
    data_path  : path to Hemonc_Edit_vX.csv
    strategy   : same strategy used with build_increments()
    increments : the ordered list of increment labels that will be processed in
                 the experiment (as returned by build_increments() + slicing)
    crop_year  : same crop_year used with build_increments()

    Returns
    -------
    DataFrame of pre-batch rows (original columns, no _batch_label column).
    Empty DataFrame if no rows precede the first experiment increment.
    """
    if strategy == "none" or not increments:
        return pd.DataFrame()

    # Load WITHOUT crop_year so we can see all historical records
    df = pd.read_csv(data_path)
    df = _ensure_datetime(df)
    df = _assign_labels(df, strategy)

    first_increment = increments[0]  # increments are chronologically ordered
    pre_batch_df = (
        df[df[_BATCH_COL] < first_increment]
        .drop(columns=[_BATCH_COL])
        .reset_index(drop=True)
    )
    print(
        f"[hemonc_batching] pre-batch corpus: {len(pre_batch_df)} rows "
        f"(total={len(df)}, strictly before first increment '{first_increment}')"
    )
    return pre_batch_df


def filter_by_increment(
    df: pd.DataFrame,
    strategy: str,
    increment: str,
    crop_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Filter a pre-loaded DataFrame to the rows belonging to one increment label.

    This is called inside the per-increment loading loop, so the full DataFrame
    is only read from disk once and then sliced cheaply here.

    Parameters
    ----------
    df        : full DataFrame from pd.read_csv (must contain 'date' and
                'evidence' columns)
    strategy  : same strategy used with build_increments()
    increment : one label from the list returned by build_increments()
    crop_year : same crop_year used with build_increments(); applied before
                the label filter so the mapping is identical

    Returns
    -------
    Filtered DataFrame (original columns preserved, no extra _batch_label column).
    Row order follows the original CSV order within the batch.
    """
    if strategy == "none":
        if crop_year is not None:
            df = _ensure_datetime(df)
            df = df[df["date"].dt.year >= int(crop_year)].copy()
        return df

    df = _ensure_datetime(df)

    if crop_year is not None:
        df = df[df["date"].dt.year >= int(crop_year)].copy()

    df = _assign_labels(df, strategy)
    result = df[df[_BATCH_COL] == increment].drop(columns=[_BATCH_COL]).reset_index(drop=True)

    if result.empty:
        raise ValueError(
            f"Increment '{increment}' not found in the dataset "
            f"(strategy='{strategy}', crop_year={crop_year}). "
            f"Re-run build_increments() with the same parameters to get valid labels."
        )

    return result
