#!/bin/bash
# Generic SLURM template for MedKIT main experiments.
#
# Copy this file to main_edit_<your-cluster>.slurm and fill in the FIXME
# placeholders below. Then submit via:
#
#   export SLURM_SCRIPT="$PWD/main_experiments/slurm/main_edit_<your-cluster>.slurm"
#   bash main_experiments/submit.sh --strategy monthly
#
#SBATCH --job-name=main-edit-${CONFIG_NAME:-unknown}
#SBATCH --partition=FIXME-partition          # e.g. h100, a100, gpu
#SBATCH --gres=gpu:1                         # adjust if your scheduler needs gpu:H100:1 etc.
#SBATCH --cpus-per-task=9
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH -o FIXME-absolute-path-to-LOGS-main/main_edit_%x_%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=FIXME@example.com

# ── Environment ───────────────────────────────────────────────────────────────
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_CACHE=FIXME-models-cache-dir
export HF_DATASETS_CACHE=FIXME-datasets-cache-dir
# OPENROUTER_API_KEY must be exported in your ~/.bashrc on the login node
# (NOT OPENAI_API_KEY, NOT OPEN_ROUTER_KEY). Submit propagates env via --export=ALL.

# ── Diagnostics ───────────────────────────────────────────────────────────────
scontrol show job "$SLURM_JOB_ID"
echo "NODE: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo "OPENROUTER_API_KEY set: ${OPENROUTER_API_KEY:+yes}"

# ── Run ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR=FIXME-absolute-path-to-experiments
cd "$SCRIPT_DIR"
mkdir -p LOGS/main

# Source conda from wherever it lives on your cluster.
source FIXME-path-to-miniconda/etc/profile.d/conda.sh
conda activate medkit_vllm

echo "Running MAIN edit experiment: CONFIG_NAME=${CONFIG_NAME:-NOT SET}"
echo "HYDRA_OVERRIDES=${HYDRA_OVERRIDES:-(none)}"
echo ""

python run_medkit.py --config-name="${CONFIG_NAME}" ${HYDRA_OVERRIDES:-}
