#!/usr/bin/env python3
"""
main_experiments/status.py — Progress monitor for the main-experiment matrix.

Joins four sources of truth to give a single view of what's going on:
  1. manifest.csv            — the planned 266 (strategy × method × model) runs
  2. metrics/main/…          — actual result files written by each run
  3. squeue                  — live SLURM state (RUNNING / PENDING)
  4. LOGS/main/…             — most recent log file per run, used for ETA and
                               failure classification

Each run is classified into exactly one status bucket:

  complete        all planned increments have result files on disk
  running         SLURM job for this run is currently RUNNING
  pending         SLURM job is queued but not yet started (PD / CF)
  stalled         partial progress on disk, no SLURM job, no recent error
                  (e.g. walltime-killed or segfault-on-exit)
  failed          most-recent log shows a Traceback/OOM/etc., not in queue
  not_submitted   no results, no log, no queue entry

Usage (from experiments/):
  python main_experiments/status.py                     # grouped summary + detail
  python main_experiments/status.py --summary           # just the counts
  python main_experiments/status.py --strategy weekly   # filter
  python main_experiments/status.py --status running    # show only RUNNING runs
  python main_experiments/status.py --json              # machine-readable
  python main_experiments/status.py --failed            # shortcut: show failures + last error
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.csv"
METRICS_ROOT = ROOT / "metrics" / "main" / "hemonc" / "main"
LOGS_ROOT = ROOT / "LOGS" / "main"
# Benchmark location used to count increments: hf://<org>/MedKIT (default) or a local CSV.
DATA_PATH = os.environ.get("MEDKIT_DATA_PATH", "hf://bethgelab/MedKIT")

CROP_YEAR_DEFAULT = 2025

STATUS_ORDER = ["complete", "running", "pending", "stalled", "failed", "stale", "not_submitted"]

# ANSI colours — suppressed when stdout isn't a TTY.
def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s
def green(s): return _c("32", s)
def yellow(s): return _c("33", s)
def red(s): return _c("31", s)
def cyan(s): return _c("36", s)
def dim(s): return _c("2", s)

STATUS_COLOR = {
    "complete":      green,
    "running":       cyan,
    "pending":       yellow,
    "stalled":       yellow,
    "failed":        red,
    "stale":         yellow,
    "not_submitted": dim,
}

# --------------------------------------------------------------------- helpers


@dataclass
class RunState:
    strategy: str
    method: str
    model: str
    tier: str
    editing_method: str
    config_name: str
    # Progress
    done: int = 0
    total: int = 0
    pct: float = 0.0
    # SLURM
    job_id: Optional[str] = None
    queue_state: Optional[str] = None
    elapsed_s: Optional[int] = None
    # Timing
    per_inc_s: Optional[float] = None
    eta_s: Optional[int] = None
    # Failure details
    status: str = "not_submitted"
    last_log: Optional[str] = None
    error_hint: Optional[str] = None


def squeue_jobs() -> dict[str, dict]:
    """Return {job_name: {id, state, elapsed_s}} for the current user."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h",
             "-o", "%A|%j|%T|%M"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}
    jobs = {}
    for line in out.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 4:
            continue
        jid, name, state, elapsed = parts
        # Parse elapsed time "HH:MM:SS" or "D-HH:MM:SS" or "MM:SS"
        jobs[name] = {
            "id": jid,
            "state": state,
            "elapsed_s": _parse_elapsed(elapsed),
        }
    return jobs


def _parse_elapsed(s: str) -> int:
    try:
        if "-" in s:
            d, rest = s.split("-", 1)
            h, m, sec = rest.split(":")
            return int(d)*86400 + int(h)*3600 + int(m)*60 + int(sec)
        parts = s.split(":")
        if len(parts) == 3:
            h, m, sec = parts
            return int(h)*3600 + int(m)*60 + int(sec)
        if len(parts) == 2:
            m, sec = parts
            return int(m)*60 + int(sec)
    except Exception:
        pass
    return 0


_BATCHES_CACHE: dict = {}
def total_increments(strategy: str, crop_year: int = CROP_YEAR_DEFAULT) -> int:
    """Compute the total batches for a strategy+crop_year (cached)."""
    key = (strategy, crop_year)
    if key in _BATCHES_CACHE:
        return _BATCHES_CACHE[key]
    try:
        sys.path.insert(0, str(ROOT))
        from hemonc_batching import build_increments
        from medkit_data import resolve_data_path
        # build_increments prints diagnostics to stdout; silence them so that
        # `status.py --json` produces parseable JSON (callers pipe its stdout
        # straight into `python -c 'json.load(sys.stdin)'`).
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            inc = build_increments(resolve_data_path(DATA_PATH), strategy=strategy, crop_year=crop_year)
        n = len(inc)
    except Exception:
        n = -1
    _BATCHES_CACHE[key] = n
    return n


def count_done(strategy: str, method: str, model: str, editing_method: str) -> int:
    """Count result JSON files (excludes sentinel, predictions)."""
    d = METRICS_ROOT / strategy / f"{method}_{model}" / editing_method / "data_-1_42"
    if not d.is_dir():
        return 0
    return sum(
        1 for p in d.glob("*_results.json")
        if "sentinel" not in p.name and "predictions" not in p.name
    )


_LOG_GLOB = "main_edit_main-{strategy}-{method}-{model}_*.out"
def latest_log(strategy: str, method: str, model: str) -> Optional[Path]:
    pat = str(LOGS_ROOT / _LOG_GLOB.format(strategy=strategy, method=method, model=model))
    files = sorted(glob(pat))
    return Path(files[-1]) if files else None


_ERROR_PATTERNS = [
    (re.compile(r"OutOfMemoryError|out of memory|OOM", re.I), "OOM"),
    (re.compile(r"CUDA Error: out of memory"), "CUDA-OOM"),
    (re.compile(r"illegal memory access"), "CUDA-illegal"),
    (re.compile(r"RecompileLimitExceeded"), "dynamo-recompile"),
    (re.compile(r"InferenceMode"), "inference-tensor"),
    (re.compile(r"no module or parameter named"), "vLLM-sync"),
    (re.compile(r"decoder prompt.*longer than"), "prompt-too-long"),
    (re.compile(r"Tensor with \d+ elements cannot be converted to Scalar"), "batch-scalar"),
    (re.compile(r"has no attribute 'generate'"), "no-generate"),
    (re.compile(r"TIME LIMIT|walltime"), "walltime"),
    (re.compile(r"Segmentation fault|Aborted \(core dumped\)"), "segfault"),
    (re.compile(r"Traceback \(most recent call last\)"), "traceback"),
]

def scan_log_for_error(log_path: Path) -> Optional[str]:
    """Return a short error-class hint, or None if the log looks clean."""
    try:
        # Read only the last 8 KB — errors cluster at the end.
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size > 8192:
                f.seek(-8192, 2)
            tail = f.read().decode(errors="ignore")
    except Exception:
        return None
    for pat, label in _ERROR_PATTERNS:
        if pat.search(tail):
            return label
    return None


def inc_times_from_log(log_path: Path) -> list[float]:
    """Extract per-increment wall-time deltas in seconds from log timestamps."""
    TS = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?)\]")
    try:
        lines = log_path.read_text(errors="ignore").splitlines()
    except Exception:
        return []
    inc_ts: list[datetime] = []
    last_ts: Optional[datetime] = None
    for ln in lines:
        m = TS.search(ln)
        if m:
            last_ts = datetime.fromisoformat(m.group(1).replace(",", "."))
        if "++++ Increment:" in ln:
            # Grab the timestamp from nearest following log line
            # (the banner itself has no timestamp)
            if last_ts is not None:
                inc_ts.append(last_ts)
    if last_ts is not None:
        inc_ts.append(last_ts)  # cap with last line so final increment's duration is estimable
    deltas = []
    for i in range(len(inc_ts) - 1):
        dt = (inc_ts[i+1] - inc_ts[i]).total_seconds()
        if dt > 0:
            deltas.append(dt)
    # First increment often has startup overhead — drop it if we have enough.
    if len(deltas) > 2:
        deltas = deltas[1:]
    return deltas


def format_hms(seconds: Optional[int]) -> str:
    if seconds is None:
        return "—"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def bar(pct: float, width: int = 20) -> str:
    filled = int(pct / 100 * width)
    return "▕" + "█" * filled + "░" * (width - filled) + "▏"


# ------------------------------------------------------------------- main loop


def build_state(filter_strategy=None, filter_method=None, filter_model=None) -> list[RunState]:
    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f))

    if filter_strategy:
        rows = [r for r in rows if r["strategy"] == filter_strategy]
    if filter_method:
        rows = [r for r in rows if r["method"] == filter_method]
    if filter_model:
        rows = [r for r in rows if r["model"] == filter_model]

    jobs = squeue_jobs()   # one squeue call

    states: list[RunState] = []
    for r in rows:
        strat, method, model, tier = r["strategy"], r["method"], r["model"], r["tier"]
        em = _read_editing_method(r["config_path"])
        rs = RunState(
            strategy=strat, method=method, model=model, tier=tier,
            editing_method=em, config_name=r["config_name"],
        )
        rs.total = total_increments(strat)
        rs.done = count_done(strat, method, model, em)
        rs.pct = 100.0 * rs.done / rs.total if rs.total > 0 else 0.0

        # SLURM match by job name convention from submit.sh
        jname = f"main-{strat}-{method}-{model}"
        if jname in jobs:
            j = jobs[jname]
            rs.job_id = j["id"]
            rs.queue_state = j["state"]
            rs.elapsed_s = j["elapsed_s"]

        # Timing / ETA from log (if any)
        lp = latest_log(strat, method, model)
        rs.last_log = lp.name if lp else None
        if lp:
            deltas = inc_times_from_log(lp)
            if deltas:
                # Use median of observed per-increment deltas
                rs.per_inc_s = sorted(deltas)[len(deltas) // 2]
        if rs.per_inc_s and rs.total > rs.done:
            rs.eta_s = int(rs.per_inc_s * (rs.total - rs.done))

        # Classification
        rs.status = _classify(rs, lp)
        if rs.status == "failed" and lp:
            rs.error_hint = scan_log_for_error(lp)

        states.append(rs)
    return states


def _read_editing_method(config_path: str) -> str:
    p = ROOT / config_path
    try:
        with open(p) as f:
            for line in f:
                if line.startswith("editing_method:"):
                    return line.split(":", 1)[1].strip().strip("'\"")
    except FileNotFoundError:
        pass
    return ""


def _classify(rs: RunState, log_path: Optional[Path]) -> str:
    # Stale: result files exist but far exceed the expected count for this
    # strategy → they're leftovers from a previous config (e.g. crop_year
    # change).  Don't trust them as "complete".
    if rs.total > 0 and rs.done > rs.total * 1.2:
        return "stale"
    if rs.total > 0 and rs.done >= rs.total:
        return "complete"
    if rs.queue_state in ("RUNNING", "CG"):
        return "running"
    if rs.queue_state in ("PENDING", "CF", "RD"):
        return "pending"
    if log_path is None and rs.done == 0:
        return "not_submitted"
    # Has log or partial progress, not in queue
    err = scan_log_for_error(log_path) if log_path else None
    if err and err not in ("segfault",) and rs.done < rs.total:
        # Segfault-on-exit with complete disk state is not a real failure
        return "failed"
    if rs.done > 0:
        return "stalled"
    return "failed" if err else "not_submitted"


# --------------------------------------------------------------------- display


def print_summary(states: list[RunState]):
    counts = Counter(s.status for s in states)
    total = len(states)
    done = counts.get("complete", 0)
    running = counts.get("running", 0)
    pending = counts.get("pending", 0)
    stalled = counts.get("stalled", 0)
    failed = counts.get("failed", 0)
    stale = counts.get("stale", 0)
    not_sub = counts.get("not_submitted", 0)

    # Progress fraction of CASES done across all planned runs.
    # Clamp per-run done at total to avoid stale-data inflating the overall %.
    total_cases = sum(s.total for s in states)
    done_cases = sum(min(s.done, s.total) if s.total > 0 else 0 for s in states)
    pct_cases = 100.0 * done_cases / total_cases if total_cases > 0 else 0.0

    print()
    print(f"  {'Runs':<18}{total:>4}")
    print(f"    {green('✓ complete'):<26}{done:>4}   ({100*done/total:.0f}%)" if total else "")
    print(f"    {cyan('▶ running'):<26}{running:>4}")
    print(f"    {yellow('… pending'):<26}{pending:>4}")
    print(f"    {yellow('○ stalled'):<26}{stalled:>4}")
    print(f"    {red('✗ failed'):<26}{failed:>4}")
    if stale:
        print(f"    {yellow('! stale (old config)'):<26}{stale:>4}")
    print(f"    {dim('· not submitted'):<26}{not_sub:>4}")
    print()
    print(f"  {'Increments'}      {done_cases:>5} / {total_cases} "
          f"{bar(pct_cases)}  {pct_cases:.1f}%")

    # Total ETA = max ETA across running jobs (parallel), plus serial sum of stalled+pending+not_submitted?
    # More useful: show per-status total remaining time if run serially
    running_etas = [s.eta_s for s in states if s.status == "running" and s.eta_s]
    remaining_etas = [s.eta_s for s in states if s.status in ("stalled", "pending", "not_submitted") and s.eta_s]
    if running_etas:
        print(f"  Longest running ETA :  {format_hms(max(running_etas))}")
    if remaining_etas:
        print(f"  Remaining (serial)  :  {format_hms(sum(remaining_etas))}"
              f"  ({len(remaining_etas)} runs)")


def print_detail_table(states: list[RunState], limit_status: Optional[str] = None):
    if limit_status:
        states = [s for s in states if s.status == limit_status]
    if not states:
        return

    # Sort: incomplete stuff first, grouped by status then method+model
    order = {k: i for i, k in enumerate(STATUS_ORDER)}
    states = sorted(states, key=lambda s: (order[s.status], s.strategy, s.method, s.model))

    # Column widths
    hdr = f"  {'status':<12} {'strategy':<8} {'method':<12} {'model':<24} {'progress':<24} {'pct':>5}  {'elapsed':>8}  {'eta':>8}  hint"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in states:
        progress_str = f"{s.done}/{s.total} " + bar(s.pct, width=14)
        elapsed = format_hms(s.elapsed_s) if s.elapsed_s else "—"
        eta = format_hms(s.eta_s) if s.eta_s else "—"
        hint = s.error_hint or ("" if s.status != "failed" else "?")
        color = STATUS_COLOR.get(s.status, str)
        print(f"  {color(f'{s.status:<12}')} "
              f"{s.strategy:<8} {s.method:<12} {s.model:<24} "
              f"{progress_str:<24} {s.pct:>4.0f}%  {elapsed:>8}  {eta:>8}  {dim(hint)}")


def print_json(states: list[RunState]):
    print(json.dumps([asdict(s) for s in states], indent=2, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["monthly", "weekly", "daily"])
    ap.add_argument("--method")
    ap.add_argument("--model")
    ap.add_argument("--status", choices=STATUS_ORDER,
                    help="Show only runs in this status")
    ap.add_argument("--summary", action="store_true",
                    help="Counts only — no per-run detail table")
    ap.add_argument("--failed", action="store_true",
                    help="Shortcut: show failed runs with error hints")
    ap.add_argument("--json", action="store_true", help="Machine-readable output")
    args = ap.parse_args()

    states = build_state(args.strategy, args.method, args.model)

    if args.json:
        print_json(states)
        return

    limit = args.status or ("failed" if args.failed else None)

    if not args.summary or not limit:
        print_summary(states)
    if not args.summary:
        print_detail_table(states, limit_status=limit)
    print()


if __name__ == "__main__":
    main()
