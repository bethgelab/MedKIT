# MedKIT — Main Experiments

Orchestration layer over Hydra + SLURM that runs the paper's full
sweep: **5 models × 3 batching strategies × ~15 method variants = 261
main configs**, plus **5 out-of-the-box (OOB) eval baselines**, for
**266 total runs**.

This directory contains the config generator, the submission /
monitoring scripts, the SLURM template, and the run manifest. The
generated Hydra configs themselves live under
`../hydra/experiments/hemonc/main/{daily,weekly,monthly,oob_eval}/`.

---

## Matrix

| Dimension          | Values                                                                                                                                  |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| **Models** (5)     | `gemma3_4b`, `qwen3_4b`, `medgemma_4b`, `llama31_8b`, `bio_medical_llama3_8b`                                                            |
| **Batching** (3)   | `daily` (~98 batches), `weekly` (~48 batches), `monthly` (~14 batches), all with `crop_year: 2025` (post-2025 records only)              |
| **Methods** (11)   | Editing: `alphaedit`, `memit`, `wise`, `grace`, `memoir`, `ike`. Continual: `lora_merge`, `o_lora`, `seekr`. Retrieval: `bm25`, `dense`. |
| **Method configs** (15) | Editing methods (1× each = 9) + RAG variants `bm25_gt`/`bm25_abs`/`dense_gt`/`dense_abs` (4) + oracle baselines `oracle_gt`/`oracle_abs` (2) |
| **OOB baselines**  | One per model (`oob_eval/`) — out-of-the-box performance with no edits.                                                                  |
| **Judge**          | `gpt-4o-hemonc` alias (resolves to `openai/gpt-4o` via OpenRouter; can be overridden — see "Change the judge" below).                    |
| **Total runs**     | 5 × 3 × ~17.4 ≈ 261 main + 5 OOB = **266**.                                                                                              |

The `crop_year: 2025` cap restricts the update stream to records
published from 2025 onward. This is the chronological-novelty
protocol used in the paper: it minimizes the chance that "successful
integration" merely surfaces pre-existing parametric knowledge from
the model's training corpus.

`adaptllm_medicine_chat` is excluded by default; it can be re-enabled
with `--include-adaptllm` when regenerating configs.

## Layout

```
main_experiments/
├── README.md              # this file
├── generate_configs.py    # writes run YAMLs + manifest.csv from best_params.csv
├── submit.sh              # submits the matrix to SLURM (filters, dry-run, skip-done)
├── submit_smoke.sh        # 2-increment smoke test for a single (method, model)
├── status.py              # progress monitor (checkpoints + squeue)
├── manifest.csv           # AUTO-GENERATED list of all 266 runs
└── slurm/
    └── main_edit_template.slurm   # generic SLURM template (FIXME placeholders)
```

Run configs live under
`../hydra/experiments/hemonc/main/{strategy}/{method}_{model}.yaml`
and are all auto-generated — do not edit by hand. Base
architecture-level hparams live under
`../hydra/experiments/hparams/{METHOD}/{model}.yaml` and are untouched
by the generator. Sweep-winner overrides (from
`../hparam_tuning/figures/best_params.csv`) are written inline in each
run config, so daily / weekly / monthly configs coexist cleanly.

---

## Quick start

### 1. Install dependencies (from `experiments/`)

```bash
conda create -n medkit_vllm python=3.10 -y
conda activate medkit_vllm
pip install -r pip-requirements.txt
```

The `pip-requirements.txt` is the full pinned snapshot used for the
paper experiments (Torch 2.6.0, Transformers 4.52.4, vLLM 0.8.5). If
pip resolution fails on your cluster, fall back to `pip install
-r requirements.txt` for an unpinned install.

Verify:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability())"
```

### 2. Pre-compute method-specific statistics (one-shot, per model)

Three of the methods need precomputed artifacts that are not shipped
with the release. **Skip any step you don't need** — the artifacts are
specific to a single method family.

**MEMIT and AlphaEdit** — second-moment statistics over a Wikipedia
corpus (~1.5–3 GB per model, ~6–12 h on one H100):

```bash
python ../precompute_stats.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --layers 4,5,6,7,8 \
    --stats_dir ./data/stats \
    --n_samples 100 \
    --batch_tokens 4096
```

Repeat for each model in the matrix. The output stats serve both MEMIT
and AlphaEdit.

**MEMOIR** — background features over a diverse corpus, used for
mean-decentering at edit time:

```bash
python ../generate_memoir_features.py \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --layer model.layers[30].mlp.down_proj \
    --output_path ./data/memoir_features/qwen3-4b.pt \
    --n_samples 500 \
    --device 0
```

The output path must match `dir_background_features` in the
corresponding `../hydra/experiments/hparams/MEMOIR/<model>.yaml`.

### 3. Set the LLM-judge API key

Open-form probes (compositional, operational, locality) are scored by
an LLM judge accessed via OpenRouter:

```bash
echo 'export OPENROUTER_API_KEY="sk-or-v1-..."' >> ~/.bashrc
source ~/.bashrc
echo "${OPENROUTER_API_KEY:0:10}..."   # confirm non-empty
```

The variable name must be exactly `OPENROUTER_API_KEY`. To route
through Azure OpenAI directly instead, see the comments in
`../easyeditor/evaluate/evaluate.py`.

### 4. Create your cluster's SLURM template

Copy the template and fill in the `FIXME` placeholders for your
cluster:

```bash
cp slurm/main_edit_template.slurm slurm/main_edit_<your-cluster>.slurm
$EDITOR slurm/main_edit_<your-cluster>.slurm
```

Placeholders: GPU partition, GRES string, log path, optional
`--mail-user`, `TRANSFORMERS_CACHE` / `HF_DATASETS_CACHE` scratch
directories, `SCRIPT_DIR` (absolute path to your `experiments/`
directory), and the `conda.sh` init path.

### 5. Smoke-test (~10 min)

```bash
export SLURM_SCRIPT="$PWD/slurm/main_edit_<your-cluster>.slurm"
bash submit_smoke.sh --strategy weekly --methods "memit"
```

When `squeue -u $USER` shows the job RUNNING and the log shows the
model loading + first increment progressing without errors, you're
clear for the full sweep.

### 6. Submit the sweep

```bash
export SLURM_SCRIPT="$PWD/slurm/main_edit_<your-cluster>.slurm"

# Dry-run — print what would be submitted, submit nothing
DRY_RUN=1 bash submit.sh --strategy weekly | head -20

# Real submission of all 87 weekly configs
bash submit.sh --strategy weekly

# Filter by method and/or model
bash submit.sh --strategy weekly --method memit --model qwen3_4b

# Skip configs whose final-increment sentinel `.done` exists
bash submit.sh --strategy weekly --skip-done
```

Each accepted `sbatch` records its job id to stdout. Resubmitting an
already-running config picks up from the last saved increment because
every generated config has `checkpoint.save: true` and
`checkpoint.load: true`.

### 7. Monitor

```bash
# Live SLURM state plus checkpoint progress and parsed log errors
python status.py --strategy weekly

# Just failures
python status.py --strategy weekly --status failed

# Filter by method, write status column back to manifest, also fetch wandb run id
python status.py --method memit --update-manifest --wandb
```

Statuses: `complete`, `running`, `pending`, `stalled` (partial
progress, no job in queue), `failed` (latest log shows an error),
`not_submitted`. `stalled` runs are usually the right target for
`bash submit.sh --skip-done`.

### 8. Where results land

- **Metrics**: `../metrics/main/hemonc/main/<strategy>/<method>_<model>/<editing_method>/data_-1_42/*.json`
- **Checkpoints**: `../checkpoints/main/hemonc/main/<strategy>/<method>_<model>/<editing_method>/data_-1_42/`
- **Logs**: `LOGS/main/main_edit_main-<strategy>-<method>-<model>_<jobid>.out`
- **Weights & Biases**: when `WANDB_ENTITY` is set, runs log under
  the project name in `../hydra/experiments/wandb/wandb.yaml`.

---

## Reference

### How configs are composed

- Every paper config has `defaults: [/wandb: wandb, /hparams:
  <METHOD>/<model>, _self_]`. The wandb config is the clean entity-`null`
  default; the hparams file carries architecture-level settings; sweep
  winners are inlined in the run config itself.
- Untuned models inherit best params from a same-architecture donor:
  `medgemma_4b ← gemma-3-4b`, `bio_medical_llama3_8b ← llama31-8b`.
- Oracle methods (`oracle_gt`, `oracle_abs`) have nothing to sweep:
  `editing_method: EVAL` plus `qa.include_evidence: ground_truth |
  abstract`. No retrieval, no sweep, just the base hparam file.
- RAG variants (`bm25_gt`, `bm25_abs`, `dense_gt`, `dense_abs`) set
  `retrieve_target: true|false` at the top level. `_gt` retrieves
  using the ground-truth comparison statement; `_abs` retrieves using
  abstracts. Both share the tuned `top_k`.
- Submit ordering is baked into `manifest.csv`: 4B models before 8B,
  monthly → weekly → daily, methods in the fixed `generate_configs.py`
  order. Quick wins land first; long 8B × daily runs are at the tail
  so failure patterns surface early.

### Regenerating configs

```bash
# Run from experiments/
python main_experiments/generate_configs.py                  # all 266
python main_experiments/generate_configs.py --dry-run        # preview only
python main_experiments/generate_configs.py --strategy weekly --method memit
python main_experiments/generate_configs.py --include-adaptllm
```

Regenerate whenever `../hparam_tuning/figures/best_params.csv` changes
or you want to edit the model / method / strategy lists in
`generate_configs.py`. The script overwrites configs and rewrites
`manifest.csv`.

### Changing the judge

Edit the `qa.judge_model` default in `generate_configs.py` (single
string constant inside `render_config`) and regenerate. Default value:
`gpt-4o-hemonc`. Any alias registered in
`../easyeditor/evaluate/evaluate_utils.py::MODEL_ALIASES` works;
unknown strings pass through as a raw OpenRouter model id.

### Eval-time vLLM (`eval_use_vllm`)

Editing methods default to HuggingFace `model.generate()` for eval,
which is the dominant cost (~95% of wall time). Setting
`eval_use_vllm: true` dual-loads a companion vLLM engine for eval-only
generation; HF still runs the edit step. After each edit, the HF
`state_dict` is pushed into vLLM via `load_weights(...)` (~1–5 s per
batch).

**Enabled by default** in `generate_configs.py` for: `alphaedit`,
`memit`, `lora_merge`, `o_lora`, `ike`, `seekr` (post-edit forward is
a plain state_dict with no inference-time wrapping).

**Disabled** (and must stay that way): `grace`, `memoir`, `wise` —
these wrap the target layer with `GRACEAdapter` / `MEMOIRAdapter` /
`WISEAdapter` whose `forward()` does codebook lookup, feature-based
sparse masking, or activation-distance routing at inference time;
vLLM cannot replicate those semantics.

Tuning knobs (top-level in each generated config when present):

- `vllm_eval_gpu_memory_utilization` (default `0.40`) — vLLM's share
  of VRAM. Tune down to `0.3` if HF + vLLM OOMs on 8B × A100 80GB.
- `vllm_eval_max_model_len` (default `2048`) — caps vLLM KV cache.

---

## Notes and gotchas

- **Locality gate.** 6 AlphaEdit × Llama-family × {daily, weekly,
  monthly} cells hit the `locality < 0.4` gate during hparam tuning.
  We keep the sweep-winner anyway (max rewrite_acc); generated configs
  carry a `# NOTE: locality gate flagged …` comment so you can find
  them with `grep`.
- **Bio-Medical-LLaMA-3 max context = 8192.** Configs cap
  `vllm_max_model_len: 8192` for this model only. Some Dense-Abstract
  / BM25-Abstract prompts may still exceed this; the prompt builder
  left-truncates and prints a `[generate_fast_vllm] left-truncated
  N/M prompts to ...` warning. Expected behavior, not an error.
- **vLLM v1 strict prompt-length validation.** A "decoder prompt
  longer than maximum model length" error from any non
  bio-medical-llama config indicates a real bug.
- **AlphaEdit `nullspace_threshold`.** A string like
  `'rel_max:1e-4'`. The generator keeps it as a quoted string in the
  YAML so Hydra passes it through untouched.
- **`OPENAI_API_KEY` is not required.** The OpenAI client in
  `../easyeditor/editors/utils.py:GPTWrapper` is only instantiated
  when editing a model whose name contains `gpt`, which is not the
  case for any of the five paper-evaluated open-weights models. Leave
  the env var unset unless you extend the codebase to edit an
  OpenAI-API-served model.
