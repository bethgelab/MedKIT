#!/usr/bin/env bash
# submit_smoke.sh — small runtime-estimation smokes
#
# Submits a fixed, short-running variant of each selected config through the
# production main_edit.slurm template, using HYDRA_OVERRIDES to cap runtime
# (2 increments, no checkpointing). Designed to produce enough data to
# extrapolate full-run costs for the main matrix.
#
# Usage (run from experiments/):
#   ./main_experiments/submit_smoke.sh                     # defaults
#   ./main_experiments/submit_smoke.sh --strategy monthly  # pick a strategy
#   ./main_experiments/submit_smoke.sh --model gemma3_4b
#
# Monitor:
#   squeue -u $USER | grep smoke-
#   python main_experiments/status.py --method <m> --model <model> --strategy <s>

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SLURM_SCRIPT="${SLURM_SCRIPT:-$SCRIPT_DIR/slurm/main_edit.slurm}"

# Representative methods — one per complexity class.
METHODS_DEFAULT="memit lora_merge ike grace bm25_gt"
MODEL_DEFAULT="qwen3_4b"
STRATEGY_DEFAULT="weekly"
N_INCREMENTS_DEFAULT="2"
# Short walltime override — smokes should finish well under this.
WALLTIME_DEFAULT="3:00:00"

STRATEGY="$STRATEGY_DEFAULT"
MODEL="$MODEL_DEFAULT"
METHODS="$METHODS_DEFAULT"
N_INCREMENTS="$N_INCREMENTS_DEFAULT"
WALLTIME="$WALLTIME_DEFAULT"
DRY_RUN="${DRY_RUN:-0}"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --strategy)     STRATEGY="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --methods)      METHODS="$2"; shift 2 ;;
        --n-increments) N_INCREMENTS="$2"; shift 2 ;;
        --time)         WALLTIME="$2"; shift 2 ;;
        -h|--help)      sed -n '1,18p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "ERROR: $SLURM_SCRIPT not found." >&2
    exit 1
fi

mkdir -p "$REPO_ROOT/LOGS/main"

echo "Smoke test"
echo "  strategy       : $STRATEGY"
echo "  model          : $MODEL"
echo "  methods        : $METHODS"
echo "  n_increments   : $N_INCREMENTS"
echo "  walltime       : $WALLTIME"
echo "  slurm template : $SLURM_SCRIPT"
[ "$DRY_RUN" = "1" ] && echo "  [DRY RUN]"
echo ""

# Build the HYDRA_OVERRIDES string once. NO COMMAS inside the value — sbatch
# --export uses commas as its own separator, and our last attempts got
# silently truncated at the first comma inside a list literal.  We omit
# wandb.tags for the same reason; set tags post-hoc if needed.
OVERRIDES="qa.batching.n_increments=${N_INCREMENTS} checkpoint.save=false checkpoint.load=false"

n_submitted=0
for M in $METHODS; do
    CONFIG_NAME="hemonc/main/${STRATEGY}/${M}_${MODEL}"
    JOB_NAME="smoke-${M}-${MODEL}"
    LOG_PATH="$REPO_ROOT/LOGS/main/smoke_${STRATEGY}_${M}_${MODEL}_%j.out"

    # Using explicit --export so the value-passing is unambiguous.
    # Quoting OVERRIDES as a single string keeps SLURM's comma-parser off
    # the equals-signs inside our override pairs.
    CMD=(sbatch
        --job-name="$JOB_NAME"
        --time="$WALLTIME"
        -o "$LOG_PATH"
        --export="ALL,CONFIG_NAME=$CONFIG_NAME,HYDRA_OVERRIDES=$OVERRIDES"
        "$SLURM_SCRIPT"
    )

    if [ "$DRY_RUN" = "1" ]; then
        printf '  [dry] '
        printf '%q ' "${CMD[@]}"
        echo
    else
        OUT=$("${CMD[@]}")
        JOB_ID=$(echo "$OUT" | awk '{print $NF}')
        echo "  submitted $CONFIG_NAME → job $JOB_ID → $LOG_PATH"
        n_submitted=$((n_submitted + 1))
    fi
done

echo ""
[ "$DRY_RUN" != "1" ] && echo "Submitted $n_submitted job(s)."
echo "Monitor : squeue -u \$USER | grep smoke-"
echo "Logs at : $REPO_ROOT/LOGS/main/smoke_${STRATEGY}_*.out"
