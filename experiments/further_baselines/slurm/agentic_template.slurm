#!/bin/bash
# Generic SLURM template for the Agentic-RAG baseline (one array task per model).
#
# Copy this file to agentic_<your-cluster>.slurm, fill in the FIXME placeholders,
# and submit from experiments/:
#
#   sbatch --export=ALL,BATCHING_STRATEGY=weekly further_baselines/slurm/agentic_<your-cluster>.slurm
#
# Runs run_agentic_shared_eval.py: query -> retrieve -> refine -> retrieve -> answer,
# scored through the shared eval harness (same metrics/output schema as the main methods).
#
#SBATCH --job-name=fb-agentic
#SBATCH --partition=FIXME-partition          # e.g. h100, a100, gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --array=0-4                          # the 5 paper models
#SBATCH -o FIXME-absolute-path-to-LOGS/agentic_%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=FIXME@example.com
set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=FIXME-models-cache-dir
export HF_DATASETS_CACHE=FIXME-datasets-cache-dir
# OPENROUTER_API_KEY must be exported for the LLM judge (and for BACKEND=openrouter).

SCRIPT_DIR=FIXME-absolute-path-to-experiments
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"

source FIXME-path-to-miniconda/etc/profile.d/conda.sh
conda activate medkit   # the main experiments env (requirements.txt) is sufficient

# ── Config (override via --export=ALL,VAR=... at submit time) ────────────────
DATA="${DATA:-hf://bethgelab/MedKIT}"   # or a local CSV built with data_construction/
BATCHING_STRATEGY="${BATCHING_STRATEGY:-weekly}"
CROP_YEAR="${CROP_YEAR:-2025}"
N_INCREMENTS="${N_INCREMENTS:-}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-hemonc}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
RETRIEVAL_TOP_K="${RETRIEVAL_TOP_K:-5}"
RETRIEVAL_ROUNDS="${RETRIEVAL_ROUNDS:-2}"
# ── Variant axes (defaults reproduce the paper setting) ──────────────────────
#   RETRIEVER=tfidf|dense   CORPUS=self|accumulating   BACKEND=hf|openrouter
# Dense + accumulating corpus:  RETRIEVER=dense CORPUS=accumulating
# Frontier-model generator:     BACKEND=openrouter OPENROUTER_STEM=claude-sonnet-4-5
# Use COMBO_NAME to write variants to distinct dirs (e.g. agentic_rag_dense_accum -> <name>_<model>).
RETRIEVER="${RETRIEVER:-tfidf}"
CORPUS="${CORPUS:-self}"
BACKEND="${BACKEND:-hf}"
OPENROUTER_STEM="${OPENROUTER_STEM:-claude-sonnet-4-5}"
OPENROUTER_THINKING="${OPENROUTER_THINKING:-0}"
COMBO_NAME="${COMBO_NAME:-}"

CONFIG_KEYS=(gemma3_4b qwen3_4b medgemma_4b llama31_8b bio_medical_llama3_8b)
CONFIG_KEY="${CONFIG_KEYS[$SLURM_ARRAY_TASK_ID]}"

echo "=== Agentic-RAG | model=$CONFIG_KEY strategy=$BATCHING_STRATEGY retriever=$RETRIEVER corpus=$CORPUS backend=$BACKEND ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

VARG=(--retriever "$RETRIEVER" --corpus "$CORPUS" --backend "$BACKEND")
[[ "$BACKEND" == "openrouter" ]] && VARG+=(--openrouter_stem "$OPENROUTER_STEM" --openrouter_thinking "$OPENROUTER_THINKING")
[[ -n "$COMBO_NAME" ]] && VARG+=(--combo_name "${COMBO_NAME}_${CONFIG_KEY}")
[[ -n "$N_INCREMENTS" ]] && VARG+=(--n_increments "$N_INCREMENTS")

python further_baselines/run_agentic_shared_eval.py \
  --config_key "$CONFIG_KEY" --strategy "$BATCHING_STRATEGY" \
  --data_path "$DATA" --crop_year "$CROP_YEAR" \
  --retrieval_top_k "$RETRIEVAL_TOP_K" --retrieval_rounds "$RETRIEVAL_ROUNDS" \
  --judge_model "$JUDGE_MODEL" --judge_workers "$JUDGE_WORKERS" \
  "${VARG[@]}"

echo "=== Agentic-RAG done: $CONFIG_KEY / $BATCHING_STRATEGY ($RETRIEVER/$CORPUS/$BACKEND) ==="
