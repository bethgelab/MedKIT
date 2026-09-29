#!/bin/bash
# Generic SLURM template for the GRPO baseline (one array task per model).
#
# Copy this file to grpo_<your-cluster>.slurm, fill in the FIXME placeholders,
# and submit from experiments/:
#
#   sbatch --export=ALL,BATCHING_STRATEGY=weekly further_baselines/slurm/grpo_<your-cluster>.slurm
#
# Per array task (one model):
#   1. Train per-increment lifelong LoRA adapters with train_model_grpo_hemonc.py,
#      starting from a fresh LoRA on the base model (like every other method).
#   2. Score each increment through the standard EVAL harness with
#      eval_lifelong_adapters.py (same metrics/output schema as the main methods).
#
#SBATCH --job-name=fb-grpo
#SBATCH --partition=FIXME-partition          # e.g. h100, a100, gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --array=0-4                          # the 5 paper models
#SBATCH -o FIXME-absolute-path-to-LOGS/grpo_%x_%A_%a.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=FIXME@example.com
set -euo pipefail

# ── Environment ───────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=FIXME-models-cache-dir
export HF_DATASETS_CACHE=FIXME-datasets-cache-dir
# OPENROUTER_API_KEY must be exported for the LLM judge (submit with --export=ALL).

SCRIPT_DIR=FIXME-absolute-path-to-experiments
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"

source FIXME-path-to-miniconda/etc/profile.d/conda.sh
conda activate medkit_baselines   # env with further_baselines/requirements.txt installed

# ── Config (override via --export=ALL,VAR=... at submit time) ────────────────
DATA="${DATA:-hf://bethgelab/MedKIT}"   # or a local CSV built with data_construction/
ADAPTER_ROOT="${ADAPTER_ROOT:-$SCRIPT_DIR/further_baselines/artifacts/grpo/adapters}"
RESULT_ROOT="${RESULT_ROOT:-$SCRIPT_DIR/further_baselines/artifacts/grpo/results}"   # standalone self-eval output
OFFLOAD_DIR="${OFFLOAD_DIR:-$SCRIPT_DIR/further_baselines/artifacts/grpo/offload_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
BATCHING_STRATEGY="${BATCHING_STRATEGY:-weekly}"
CROP_YEAR="${CROP_YEAR:-2025}"
N_INCREMENTS="${N_INCREMENTS:-}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-hemonc}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
mkdir -p "$ADAPTER_ROOT" "$RESULT_ROOT" "$OFFLOAD_DIR"
trap 'rm -rf "$OFFLOAD_DIR"' EXIT

# Parallel arrays: config_key (sweep) ↔ model_folder (HF folder key).
CONFIG_KEYS=(gemma3_4b qwen3_4b medgemma_4b llama31_8b bio_medical_llama3_8b)
MODEL_FOLDERS=(google_gemma-3-4b-it Qwen_Qwen3-4B-Instruct-2507 google_medgemma-4b-it meta-llama_Llama-3.1-8B-Instruct ContactDoctor_Bio-Medical-Llama-3-8B)
CONFIG_KEY="${CONFIG_KEYS[$SLURM_ARRAY_TASK_ID]}"
MODEL_FOLDER="${MODEL_FOLDERS[$SLURM_ARRAY_TASK_ID]}"

echo "=== GRPO baseline | model=$CONFIG_KEY ($MODEL_FOLDER) strategy=$BATCHING_STRATEGY ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

NARG=(); [[ -n "$N_INCREMENTS" ]] && NARG=(--n_increments "$N_INCREMENTS")

# ── Step 1: train per-increment adapters ────────────────────────────────────
python further_baselines/train_model_grpo_hemonc.py \
  --model_folder "$MODEL_FOLDER" \
  --data_path "$DATA" \
  --adapter_root "$ADAPTER_ROOT" \
  --result_root "$RESULT_ROOT" \
  --offload_dir "$OFFLOAD_DIR" \
  --batching_strategy "$BATCHING_STRATEGY" \
  --crop_year "$CROP_YEAR" \
  --learning_rate "$LEARNING_RATE" \
  --num_generations "$NUM_GENERATIONS" \
  --judge_model "$JUDGE_MODEL" --judge_workers "$JUDGE_WORKERS" \
  --filter_conflicting_edits --include_evidence false \
  "${NARG[@]}"

# ── Step 2: score each increment through the shared EVAL harness ─────────────
python further_baselines/eval_lifelong_adapters.py \
  --combo grpo --config_key "$CONFIG_KEY" --strategy "$BATCHING_STRATEGY" \
  --adapter_root "$ADAPTER_ROOT" --data_path "$DATA" --crop_year "$CROP_YEAR" \
  --skip_existing "${NARG[@]}"

echo "=== GRPO baseline done: $CONFIG_KEY / $BATCHING_STRATEGY ==="
