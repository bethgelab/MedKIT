#!/usr/bin/env bash
# main_experiments/submit.sh — Submit the full main-experiment matrix to SLURM.
#
# Reads main_experiments/manifest.csv (produced by generate_configs.py) and
# submits one sbatch per row, in the manifest's ordering (4B models first, then
# 8B; monthly → weekly → daily; methods in METHODS-dict order).
#
# Filters:
#   --strategy <daily|weekly|monthly>   restrict to one batching strategy
#   --method <key>                      restrict to one method (e.g. memit, bm25_gt)
#   --model  <key>                      restrict to one model
#   --skip-done                         skip configs whose final checkpoint exists
#   --order manifest                    disable in-script re-sort (use manifest order as-is)
#
# Environment:
#   DRY_RUN=1  print sbatch commands without submitting
#
# Usage examples:
#   ./main_experiments/submit.sh                           # submit everything
#   ./main_experiments/submit.sh --strategy monthly        # just monthly
#   ./main_experiments/submit.sh --method memit --skip-done
#   DRY_RUN=1 ./main_experiments/submit.sh --strategy weekly --model gemma3_4b
#
# Monitor:
#   squeue -u $USER
#   python main_experiments/status.py

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MANIFEST="$SCRIPT_DIR/manifest.csv"
SLURM_SCRIPT="${SLURM_SCRIPT:-$SCRIPT_DIR/slurm/main_edit.slurm}"
DRY_RUN="${DRY_RUN:-0}"

STRATEGY_FILTER=""
METHOD_FILTER=""
MODEL_FILTER=""
SKIP_DONE=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --strategy) STRATEGY_FILTER="$2"; shift 2 ;;
        --method)   METHOD_FILTER="$2";   shift 2 ;;
        --model)    MODEL_FILTER="$2";    shift 2 ;;
        --skip-done) SKIP_DONE=1;         shift ;;
        --order)    shift 2 ;;  # reserved, manifest order is already authoritative
        -h|--help)
            sed -n '1,25p' "$0"; exit 0 ;;
        *)
            echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: $MANIFEST not found. Run: python main_experiments/generate_configs.py"
    exit 1
fi
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: $SLURM_SCRIPT not found."
    exit 1
fi

mkdir -p "$REPO_ROOT/LOGS/main"

echo "Submitting MAIN edit jobs"
[ -n "$STRATEGY_FILTER" ] && echo "  strategy filter : $STRATEGY_FILTER"
[ -n "$METHOD_FILTER" ]   && echo "  method filter   : $METHOD_FILTER"
[ -n "$MODEL_FILTER" ]    && echo "  model filter    : $MODEL_FILTER"
[ "$SKIP_DONE" = "1" ]    && echo "  skip-done       : on"
[ "$DRY_RUN"   = "1" ]    && echo "  [DRY RUN — no jobs will be submitted]"
echo ""

# Quick helper: checkpoint dir for a given experiment, mirrors the entry script.
#   {save_dir}/{experiment}/{editing_method}/data_{ds_size}_{ds_seed}/
# ds_size=-1, ds_seed=42 are fixed in every generated config.
is_done() {
    local experiment="$1" editing_method="$2"
    local ckpt_dir="$REPO_ROOT/checkpoints/main/${experiment}/${editing_method}/data_-1_42"
    [ -d "$ckpt_dir" ] || return 1
    # Find any args.json with a 100% milestone marker — we treat presence of
    # a '_final' suffix or a file tagged as the last increment as done.  Since
    # the entry script always writes a checkpoint for the FINAL batch, we
    # check for a sentinel '<experiment>/.done' file written by the runner;
    # fall back to "at least one args.json exists" with a warning so the user
    # can manually inspect.
    [ -f "$ckpt_dir/.done" ]
}

n_submitted=0
n_skipped=0
n_total=0

# Parse manifest, skip header.
{
    read -r _header
    while IFS=, read -r strategy method model tier config_name config_path donor flagged has_ov job_id status wandb_run_id; do
        n_total=$((n_total + 1))

        [ -n "$STRATEGY_FILTER" ] && [ "$strategy" != "$STRATEGY_FILTER" ] && continue
        [ -n "$METHOD_FILTER" ]   && [ "$method"   != "$METHOD_FILTER" ]   && continue
        [ -n "$MODEL_FILTER" ]    && [ "$model"    != "$MODEL_FILTER" ]    && continue

        # Derive editing_method from the config (grep the YAML once; cheaper than
        # maintaining a parallel map here).
        editing_method=$(grep -m1 "^editing_method:" "$REPO_ROOT/$config_path" | awk -F"'" '{print $2}')
        experiment="hemonc/main/${strategy}/${method}_${model}"

        if [ "$SKIP_DONE" = "1" ] && is_done "$experiment" "$editing_method"; then
            echo "  [skip-done] $config_name"
            n_skipped=$((n_skipped + 1))
            continue
        fi

        JOB_NAME="main-${strategy}-${method}-${model}"
        CMD="sbatch --job-name=${JOB_NAME} --export=ALL,CONFIG_NAME=${config_name} ${SLURM_SCRIPT}"

        if [ "$DRY_RUN" = "1" ]; then
            echo "  [dry] $CMD"
            n_submitted=$((n_submitted + 1))
        else
            OUT=$($CMD)
            JOB_ID=$(echo "$OUT" | awk '{print $NF}')
            echo "  submitted ${config_name} → job ${JOB_ID}"
            n_submitted=$((n_submitted + 1))
        fi
    done
} < "$MANIFEST"

echo ""
echo "Total in manifest    : $n_total"
echo "Submitted this run   : $n_submitted"
[ "$SKIP_DONE" = "1" ] && echo "Skipped (already done): $n_skipped"
echo ""
echo "Monitor with : python main_experiments/status.py"
echo "             : squeue -u \$USER"
echo "Logs at      : $REPO_ROOT/LOGS/main/"
echo "Results at   : $REPO_ROOT/metrics/main/"
