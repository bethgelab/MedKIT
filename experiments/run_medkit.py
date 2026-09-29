import importlib
import logging
import os.path
import shutil
import sys

import numpy as np
import pandas as pd

import torch
import gc

sys.path.append('')
import json
import random
import hydra
from omegaconf import DictConfig, open_dict
from omegaconf import OmegaConf as oc
import wandb
from easyeditor import (
    FTHyperParams,
    IKEHyperParams,
    KNHyperParams,
    MEMITHyperParams,
    ROMEHyperParams,
    LoRAHyperParams,
    MENDHyperParams,
    SERACHparams,
    GraceHyperParams,
    R_ROMEHyperParams,
    WISEHyperParams,
    RAGHyperParams,
    PTuningHyperParams,
    AlphaEditHyperParams,
    MEMOIRHyperParams,
    LoRAMergeHyperParams,
    OLoRAHyperParams,
    SEEKRHyperParams,
    )
from easyeditor import BaseEditor, LifelongEditor
from easyeditor.models.ike import encode_ike_facts
from easyeditor.models.ike import encode_hemonc_ike_facts, apply_hemonc_ike_to_model
from easyeditor.models.oracle_rag import OracleRAGHyperParams
from easyeditor.models.bm25_rag import BM25RAGHyperParams
from easyeditor.models.dense_rag import DenseRAGHyperParams
from sentence_transformers import SentenceTransformer
from easyeditor import ZsreDataset
from medkit_data import resolve_data_path


# ── Pre-edit cache helpers ─────────────────────────────────────────────────────

def _pre_edit_cache_path(args, hparams, increment, mode, setting):
    """Return the path of the pre-edit cache file for this run.

    The cache key is per-(editing_method, model, dataset, strategy, increment,
    mode, setting). Sequential edits produce different model states at inc≥2,
    so sharing across methods would leak state between them — include
    `editing_method` in both the directory and filename.

    Within a single method, the cache is still reused across resumes and
    across other runs with identical config (e.g. rerunning a seed).
    """
    model_slug = hparams.model_name.replace('/', '_').replace('-', '_')
    method_slug = str(args.editing_method).replace('/', '_')
    data_slug = os.path.splitext(os.path.basename(args.qa.data_path))[0]
    batching_cfg = getattr(args.qa, 'batching', None)
    strategy = getattr(batching_cfg, 'strategy', 'none') if batching_cfg else 'none'
    crop_year = getattr(batching_cfg, 'crop_year', '') if batching_cfg else ''
    n_increments = getattr(batching_cfg, 'n_increments', '') if batching_cfg else ''
    batching_slug = f"{strategy}_cy{crop_year}_n{n_increments}" if crop_year else strategy
    cache_dir = os.path.join(args.metrics_save_dir, 'pre_edit_cache', model_slug, method_slug)
    fname = f"{increment}_{mode}_{setting}_{data_slug}_ds{args.ds_size}_{batching_slug}_pre_edit.json"
    return os.path.join(cache_dir, fname)


def _load_pre_edit_cache(path):
    """Load cached pre-edit metrics from disk, or return None if not found."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _save_pre_edit_cache(path, pre_metrics):
    """Save pre-edit metrics list to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(pre_metrics, f, indent=4)


def _resolve_save_mode(args):
    """Return the mode label used in checkpoint filenames.

    MUST stay in sync with the in-loop computation in main() that decides
    `mode` right before save_checkpoint is called.  If the two diverge,
    resume reads the wrong filename and silently restarts from increment 1.

    Logic (mirrors main loop):
      - len(modes) > 1 OR modes_combine == 'mix'  → 'mix'
      - len(modes) > 1 AND modes_combine == 'concat' → 'concat'
      - else → modes[0]   (single explicit mode, or None)
    """
    modes = list(args.qa.modes)
    combine = args.qa.modes_combine
    if len(modes) > 1 or combine == 'mix':
        return 'mix'
    if len(modes) > 1 and combine == 'concat':
        return 'concat'
    return modes[0]


@hydra.main(version_base=None, config_path="hydra/experiments")
def main(args: DictConfig) -> None:
    # Merge base defaults under the experiment config so the experiment config always wins.
    # Hydra's subdirectory composition has version-dependent ordering issues with base config
    # inheritance, so we apply it explicitly here: base provides defaults, args overrides.
    _base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "hydra/experiments/base_config.yaml")
    _base = oc.load(_base_path)
    with open_dict(_base):
        _base.pop('defaults', None)   # strip Hydra-only key before merging
    args = oc.merge(_base, args)      # args wins over base defaults
    # Resolve the benchmark location (hf://<org>/MedKIT or a local CSV) once, up front.
    if getattr(args.qa, 'data_format', 'json') == 'csv':
        with open_dict(args):
            args.qa.data_path = resolve_data_path(args.qa.data_path)
    print(f'ARGS: {args}')

    if args.editing_method == 'FT':
        editing_hparams = FTHyperParams
    elif args.editing_method == 'IKE':
        editing_hparams = IKEHyperParams
    elif args.editing_method == 'KN':
        editing_hparams = KNHyperParams
    elif args.editing_method == 'MEMIT':
        editing_hparams = MEMITHyperParams
    elif args.editing_method == 'ROME':
        editing_hparams = ROMEHyperParams
    elif args.editing_method == 'LoRA':
        editing_hparams = LoRAHyperParams
    elif args.editing_method == 'MEND':
        editing_hparams = MENDHyperParams
    elif args.editing_method == 'SERAC':
        editing_hparams = SERACHparams
    elif args.editing_method == 'GRACE':
        editing_hparams = GraceHyperParams
    elif args.editing_method == 'R-ROME':
        editing_hparams = R_ROMEHyperParams
    elif args.editing_method == 'WISE':
        editing_hparams = WISEHyperParams
    elif args.editing_method == 'EVAL':
        editing_hparams = OracleRAGHyperParams
    elif args.editing_method == 'RAG':
        editing_hparams = RAGHyperParams
    elif args.editing_method == 'PTuning':
        editing_hparams = PTuningHyperParams
    elif args.editing_method == 'AlphaEdit':
        editing_hparams = AlphaEditHyperParams
    elif args.editing_method == 'MEMOIR':
        editing_hparams = MEMOIRHyperParams
    elif args.editing_method == 'LoRA-Merge':
        editing_hparams = LoRAMergeHyperParams
    elif args.editing_method == 'O-LoRA':
        editing_hparams = OLoRAHyperParams
    elif args.editing_method == 'SEEKR':
        editing_hparams = SEEKRHyperParams
    elif args.editing_method == 'OracleRAG':
        editing_hparams = OracleRAGHyperParams
    elif args.editing_method == 'BM25RAG':
        editing_hparams = BM25RAGHyperParams
    elif args.editing_method == 'DenseRAG':
        editing_hparams = DenseRAGHyperParams
    else:
        raise NotImplementedError(f"Unknown editing_method: {args.editing_method!r}")

    # ── Auto-build increments from batching strategy ─────────────────────────
    # If qa.batching.strategy is set (and not 'none'), derive the ordered list
    # of batch labels from the dataset and replace qa.increments.  This runs
    # before the checkpoint logic so the full increment list is always available
    # for checkpoint resume filtering.
    _batching_cfg = getattr(args.qa, 'batching', None)
    _strategy     = getattr(_batching_cfg, 'strategy', 'none') if _batching_cfg is not None else 'none'
    if _strategy != 'none':
        from hemonc_batching import build_increments as _build_increments
        _crop_year = getattr(_batching_cfg, 'crop_year', None)
        _auto_increments = _build_increments(
            data_path=args.qa.data_path,
            strategy=_strategy,
            crop_year=_crop_year,
        )
        _n_increments = getattr(_batching_cfg, 'n_increments', None)
        if _n_increments:
            _auto_increments = _auto_increments[:_n_increments]
        with open_dict(args):
            args.qa.increments = _auto_increments
        print(f'[batching] Auto-built {len(_auto_increments)} increments '
              f"for strategy='{_strategy}' crop_year={_crop_year}"
              + (f" (capped at n_increments={_n_increments})" if _n_increments else ""))

    # ── Optional single-increment filter ─────────────────────────────────────
    # When qa.batching.increment_filter is set (str or list of labels), restrict
    # the increment list to just those labels.  Used by the DPO/GRPO shared-eval
    # driver (further_baselines/eval_lifelong_adapters.py) to score one increment
    # per invocation, so increment i is evaluated with the LoRA adapter trained
    # through increment i (lifelong-CL semantics, as for SEEKR / O-LoRA / LoRA-Merge).
    _inc_filter = getattr(_batching_cfg, 'increment_filter', None) if _batching_cfg is not None else None
    if _inc_filter:
        _keep = {str(_inc_filter)} if isinstance(_inc_filter, str) else {str(x) for x in _inc_filter}
        with open_dict(args):
            args.qa.increments = [x for x in args.qa.increments if str(x) in _keep]
        print(f'[batching] increment_filter={_inc_filter!r} → {list(args.qa.increments)}')

    print(f'Args: {args}')

    random.seed(args.ds_seed)
    np.random.seed(args.ds_seed)

    hparams = editing_hparams.from_hparams(config=args.hparams)

    # Inject experiment-level fields into hparams at runtime so that experiment
    # configs can override hparam defaults without editing the hparam YAML files.
    for field in ('adapter_path', 'use_vllm', 'vllm_max_model_len',
                  'lora_max_rank', 'use_openrouter',
                  'retrieve_target', 'eval_use_vllm', 'eval_use_vllm_sleep_mode',
                  'vllm_eval_gpu_memory_utilization', 'vllm_eval_max_model_len',
                  'eval_gen_batch_size', 'attn_implementation'):
        val = getattr(args, field, None)
        if val is not None:
            setattr(hparams, field, val)

    # ── Pre-batch corpus for RAG methods ─────────────────────────────────────
    # When batching is enabled, build a static background corpus from all records
    # that precede the experiment's first increment.  RAG wrappers are seeded with
    # this corpus at construction time; adapt() then adds each new batch on top.
    # OracleRAG is excluded: it uses exact-match dict lookup only and has no use
    # for a background corpus (see easyeditor/models/oracle_rag/oracle_rag_main.py).
    _rag_methods = ('IKE', 'BM25RAG', 'DenseRAG', 'RAG')
    pre_batch_records = []
    # When `qa.seed_pre_batch_corpus` is False, RAG wrappers start with an
    # empty corpus and only see evidence from adapt() calls — the "online" /
    # streaming variant used in ablations against the default "library"
    # condition (pre-2025 abstracts pre-loaded). Defaults to True to match
    # the historical behaviour.
    _seed_pre_batch = getattr(args.qa, 'seed_pre_batch_corpus', True)
    if _strategy != 'none' and args.editing_method in _rag_methods and _seed_pre_batch:
        from hemonc_batching import build_pre_batch_corpus as _build_pre_batch
        _pre_batch_df = _build_pre_batch(
            data_path=args.qa.data_path,
            strategy=_strategy,
            increments=list(args.qa.increments),
            crop_year=_crop_year,
        )
        if len(_pre_batch_df) > 0:
            pre_batch_records = load_csv_data(
                args.qa.data_path,
                include_evidence=False,
                _df_override=_pre_batch_df,
                filter_conflicting_edits=getattr(args.qa, 'filter_conflicting_edits', False),
            )
            print(f'[RAG corpus] Pre-batch corpus: {len(pre_batch_records)} records '
                  f'(before first increment "{args.qa.increments[0]}")')
    elif _strategy != 'none' and args.editing_method in _rag_methods and not _seed_pre_batch:
        print('[RAG corpus] seed_pre_batch_corpus=False — starting with empty corpus '
              '(adapt() calls grow it from this run\'s edits only).')

    if args.editing_method == 'IKE':
        if getattr(hparams, 'hemonc', False):
            # HemOnc IKE: pre-build demonstration index from the pre-batch corpus
            # (or all records when no batching is configured).
            ike_corpus_records = pre_batch_records if pre_batch_records else load_csv_data(
                args.qa.data_path, include_evidence=False,
                filter_conflicting_edits=getattr(args.qa, 'filter_conflicting_edits', False),
            )
            sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
            train_ds = encode_hemonc_ike_facts(sentence_model, ike_corpus_records, hparams)
        else:
            # Original IKE (ZSRE): encode training data and route through editor.edit()
            train_data_path = os.path.join(args.data_dir, 'zsre_mend_train_10000.json')
            train_ds = ZsreDataset(train_data_path)
            sentence_model = SentenceTransformer(hparams.sentence_model_name).to(f'cuda:{hparams.device}')
            encode_ike_facts(sentence_model, train_ds, hparams)
    else:
        train_ds = None

    editor = LifelongEditor.from_hparams(hparams)
    edited_model = None

    if args.checkpoint.load:
        # IMPORTANT: this MUST match the mode computation done inside the
        # main loop (around the save_checkpoint call site), otherwise resume
        # silently fails: save writes `{inc}_mix_…_args.json` while load
        # would look for `{inc}_None_…_args.json`. Use the shared helper.
        mode = _resolve_save_mode(args)
        outstanding_increments, wandb_run_id = load_checkpoint(
            args,
            mode,
            None if args.editing_method == 'EVAL' else editor.model,
            hparams=hparams,
        )
        if wandb_run_id is None:
            print('Cannot proceed wandb run - no checkpoint found')
            wandb_run_id = wandb.util.generate_id()
    else:
        wandb_run_id = wandb.util.generate_id()
        outstanding_increments = args.qa.increments
    print(f'wandb_run_id: {wandb_run_id}')

    # Build setting tag upfront (needed for run name before wandb.init).
    # Two-part tag: {evidence|noevidence}_{finetuned|base}
    _is_evidence  = getattr(args.qa, 'include_evidence', True)
    _adapter_path = getattr(args, 'adapter_path', None)
    _is_finetuned = bool(_adapter_path)
    setting = f"{'evidence' if _is_evidence else 'noevidence'}_{'finetuned' if _is_finetuned else 'base'}"

    # Friendly run name: eval-<model>-<setting_abbrev>
    _SETTING_ABBREV = {
        'evidence_finetuned':   'ft-ev',
        'noevidence_finetuned': 'ft-noev',
        'evidence_base':        'base-ev',
        'noevidence_base':      'base-noev',
    }
    _model_tag = args.experiment.replace('eval_cutoff_', '').replace('eval_', '')
    _run_name  = f"eval-{_model_tag}-{_SETTING_ABBREV.get(setting, setting)}"

    if not args.debug:
        os.makedirs(os.path.join(args.metrics_save_dir, args.experiment, args.editing_method), exist_ok=True)
        wandb.init(project=args.wandb.project,
                   entity=args.wandb.entity,
                   name=_run_name,
                   id=wandb_run_id,
                   resume='allow',
                   notes=args.wandb.notes,
                   # Log to W&B only when an entity is configured (WANDB_ENTITY).
                   mode=None if args.wandb.entity else 'disabled',
                   config=oc.to_container(args, resolve=True, throw_on_missing=True),
                   settings=wandb.Settings(_service_wait=300))

    print(f' Running {args.editing_method} on increments: {args.qa.increments} with modes: {args.qa.modes}')
    sentinel_pool  = []   # list of {'increment': str, 'record': dict}
    _batch_counter = 0    # counts processed (non-skipped) batches
    _ckpt_count    = 0    # counts increments that triggered a checkpoint call
    total_batches  = len(args.qa.increments)
    if args.checkpoint.save:
        _eff_interval = _resolve_save_interval(args)
        print(f'[checkpoint] save_interval={_eff_interval} '
              f'keep_milestones={list(getattr(args.checkpoint, "keep_milestones", []) or [])} '
              f'total_batches={total_batches}')
    all_data       = {}   # increment → list of records (collected for downstream use)
    for increment in args.qa.increments:
        print(f'++++++++++++ Increment: {increment} ++++++++++++')
        for mode in args.qa.modes:
            if mode is None:
                mode_prefix = ''
            else:
                mode_prefix = f'{mode}_'
            if getattr(args.qa, 'data_format', 'json') == 'csv':
                include_evidence = getattr(args.qa, 'include_evidence', True)
                test_data = load_csv_data(
                    args.qa.data_path,
                    include_evidence=include_evidence,
                    increment=increment,
                    batching_cfg=getattr(args.qa, 'batching', None),
                    filter_conflicting_edits=getattr(args.qa, 'filter_conflicting_edits', False),
                )
            else:
                test_data = json.load(open(os.path.join(args.data_dir, f'{increment}/{mode_prefix}qa_{args.qa.model}.json'), 'r', encoding='utf-8'))

            if args.ds_size > 0 and len(test_data) > args.ds_size:
                random.seed(args.ds_seed)
                test_data = random.sample(test_data, args.ds_size)
            all_data[increment] = test_data
            prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject, evidence = extract_data(test_data)
            # Build prompt → ground_truth_statement mapping before any shuffle so we can
            # resolve the (possibly reordered) prompts list to the correct statement.
            _gts_by_prompt = {r['src']: r.get('ground_truth_statement', '') for r in test_data}

        # Single source of truth for the mode label that ends up in checkpoint
        # filenames — also used by load_checkpoint above.  Keep the shuffle in
        # the 'mix' branch (only for compatibility with the historic semantics
        # where mode=='mix' implied the post-load shuffle of the inputs).
        mode = _resolve_save_mode(args)
        if mode == 'mix':
            random.seed(args.ds_seed)
            prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject, evidence = shuffle_inputs(
                prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject, evidence
            )

        # Resolve ground_truth_statements in the (possibly shuffled) prompt order.
        ground_truth_statements = [_gts_by_prompt.get(p, '') for p in prompts]

        # ── Respect eval_portability / eval_locality flags ────────────────────
        # When set to false in the config, skip those evaluations entirely by
        # passing empty dicts.  Useful for fast debug runs focused on rewrite_acc.
        if not getattr(args.qa, 'eval_portability', True):
            portability_inputs = {}
        if not getattr(args.qa, 'eval_locality', True):
            locality_inputs = {}

        if args.editing_method in ['WISE', 'GRACE']:
            # Use HemOnc locality questions as out-of-scope reference prompts
            # instead of the ZSRE dataset (which is domain-mismatched and often absent).
            _neighborhood = (locality_inputs or {}).get('neighborhood', {})
            _loc_qs = _neighborhood.get('prompt', [])
            if _loc_qs:
                loc_prompts = _loc_qs[:len(prompts)]
            else:
                print('Warning: no locality questions available — running WISE/GRACE without locality prompts.')
                loc_prompts = None
        else:
            loc_prompts = None

        if not increment in outstanding_increments:
            print(f'Increment {increment} already processed')
            continue

        _ckpt_count += 1

        # ── Resolve max_minibatch_size ────────────────────────────────────────
        # If max_minibatch_size is set, cap args.batch_size to min(max_minibatch_size,
        # len(prompts)) for this increment.  This lets small weekly batches go through
        # in a single apply_algo call while preventing OOM on large yearly batches.
        _max_mb = getattr(args, 'max_minibatch_size', None)
        if _max_mb is not None:
            with open_dict(args):
                args.batch_size = min(int(_max_mb), len(prompts))
            print(f'[batch_size] max_minibatch_size={_max_mb} → {args.batch_size} '
                  f'(increment {increment} has {len(prompts)} edits)')

        if args.editing_method == 'EVAL':
            summary_metrics, metrics = editor.eval_samples(
                prompts=prompts,
                rephrase_prompts=rephrase_prompts,
                target_new=target_new,
                tags=tags,
                ground_truth=evidence,
                few_shot_examples=args.qa.fs_examples,
                portability_inputs=portability_inputs,
                locality_inputs=locality_inputs,
                batch_size=args.batch_size,
                judge_model=getattr(args.qa, 'judge_model', None),
                judge_workers=getattr(args.qa, 'judge_workers', 4),
                judge_enabled=getattr(args.qa, 'judge_enabled', True),
            )
            if not args.debug:
                if args.checkpoint.save:
                    save_checkpoint(args, increment, mode, wandb_run_id,
                                    batch_idx=_ckpt_count,
                                    total_batches=total_batches,
                                    is_final=(increment == args.qa.increments[-1]))
                eval_metrics_to_wandb(summary_metrics, metrics, tags, setting)
            else:
                print(summary_metrics)
        elif args.editing_method in ('RAG', 'PTuning', 'OracleRAG', 'BM25RAG', 'DenseRAG') or \
                (args.editing_method == 'IKE' and getattr(hparams, 'hemonc', False)):
            metrics, edited_model = editor.rag(
                args=args,
                prompts=prompts,
                rephrase_prompts=rephrase_prompts,
                target_new=target_new,
                ground_truth=evidence,
                ground_truth_statements=ground_truth_statements,
                tags=tags,
                edited_model=edited_model,
                loc_prompts=loc_prompts,
                subject=subject,
                train_ds=train_ds,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                keep_original_weight=True,
                few_shot_examples=args.qa.fs_examples,
                past_eval_interval=args.qa.past_eval_interval,
                verbose=False,
                initial_corpus_records=pre_batch_records,
            )

        else:
            # ── Pre-edit handling ─────────────────────────────────────────────
            # Paper framing: we compare post-edit performance to the UN-edited
            # base model ("OOB") rather than to the sequentially-edited state
            # at increment k−1.  Benefits:
            #   - one pre-eval per (model, case), not per (method, increment)
            #   - no cross-method contamination (cache was unsafe before)
            #   - cleaner delta interpretation
            #
            # When `qa.skip_pre_edit` is true, we inject an empty `pre` field
            # per case and skip the pre-eval loop entirely.  The OOB baseline
            # is produced by a separate eval run (see main_experiments/
            # precompute_oob.sh) whose outputs live under metrics/main/oob_eval/
            # and are merged at analysis time.
            #
            # When false (e.g. RAG methods that need to see the base model's
            # first-increment response), the legacy cached pre-eval path runs.
            _NO_PRE_EDIT_METHODS = {'EVAL', 'RAG', 'OracleRAG', 'BM25RAG', 'DenseRAG', 'IKE'}
            _skip_pre_edit = bool(getattr(args.qa, 'skip_pre_edit', False))
            _use_cache = (not args.debug
                          and not _skip_pre_edit
                          and args.editing_method not in _NO_PRE_EDIT_METHODS)
            _pre_edit_loaded = None
            _cache_path = None
            if _skip_pre_edit:
                # Inject empty pre-entries so per-case indexing in editor
                # stays aligned with requests; downstream analysis reads the
                # OOB file for the actual baseline.
                _n_cases = len(prompts)
                _pre_edit_loaded = [{'pre': {}} for _ in range(_n_cases)]
                print(f'[pre_edit] qa.skip_pre_edit=True — injecting {_n_cases} empty pre entries')
            elif _use_cache:
                _cache_path = _pre_edit_cache_path(args, hparams, increment, mode, setting)
                _pre_edit_loaded = _load_pre_edit_cache(_cache_path)
                if _pre_edit_loaded is not None:
                    print(f'[pre_edit_cache] Loaded {len(_pre_edit_loaded)} entries from {_cache_path}')
                else:
                    print(f'[pre_edit_cache] No cache found — will compute pre-edit eval')

            metrics, edited_model, _ = editor.edit(
                args=args,
                prompts=prompts,
                rephrase_prompts=rephrase_prompts,
                target_new=target_new,
                tags=tags,
                loc_prompts=loc_prompts,
                subject=subject,
                train_ds=train_ds,
                locality_inputs=locality_inputs,
                portability_inputs=portability_inputs,
                keep_original_weight=False,
                few_shot_examples=args.qa.fs_examples,
                past_eval_interval=args.qa.past_eval_interval,
                verbose=False,
                pre_edit=_pre_edit_loaded,
            )

            # Save pre-edit results to cache for future experiments
            if _use_cache and _pre_edit_loaded is None and _cache_path is not None:
                pre_only = [{'pre': m['pre']} for m in metrics if 'pre' in m]
                if pre_only:
                    _save_pre_edit_cache(_cache_path, pre_only)
                    print(f'[pre_edit_cache] Saved {len(pre_only)} entries to {_cache_path}')

            if args.checkpoint.save and not args.debug:
                save_checkpoint(args, increment, mode, wandb_run_id, model=edited_model,
                                batch_idx=_ckpt_count,
                                total_batches=total_batches,
                                is_final=(increment == args.qa.increments[-1]))

            if not args.debug and metrics:
                eval_metrics_to_wandb({}, metrics, tags, setting)

        # ── Sentinel pool: bounded-cost past retention eval ───────────────────
        _past_cfg = getattr(args.qa, 'past_eval', None)
        if _past_cfg and getattr(_past_cfg, 'enabled', False) and args.editing_method != 'EVAL':
            sentinel_k    = getattr(_past_cfg, 'sentinel_k',    3)
            sentinel_max  = getattr(_past_cfg, 'sentinel_max',  150)
            eval_interval = getattr(_past_cfg, 'eval_interval', 1)

            # Always add current batch samples to pool
            k = min(sentinel_k, len(test_data))
            for rec in random.sample(test_data, k):
                sentinel_pool.append({'increment': increment, 'record': rec})
            if len(sentinel_pool) > sentinel_max:
                sentinel_pool = random.sample(sentinel_pool, sentinel_max)

            _batch_counter += 1

            # Evaluate only every eval_interval batches, and only if past batches exist in pool
            _past_in_pool = [e for e in sentinel_pool if e['increment'] != increment]
            if _batch_counter % eval_interval == 0 and len(_past_in_pool) > 0:
                _sent_metrics = _eval_sentinel(editor, sentinel_pool, increment, args, edited_model)
                if not args.debug:
                    _sentinel_metrics_to_wandb(_sent_metrics, increment)
                    _save_sentinel_metrics(_sent_metrics, args, increment, mode, setting)
                else:
                    print(f'[sentinel] cq_avg={_sent_metrics["sentinel_cq_avg"]:.3f} '
                          f'pool={_sent_metrics["pool_size"]} '
                          f'batches={_sent_metrics["n_batches_covered"]}')

            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

        # ── Metrics saving ────────────────────────────────────────────────────
        # Normal runs (debug: false) → metrics_save_dir/experiment/...
        # Debug runs with save_debug_metrics: true → metrics/debug/experiment/...
        # Debug runs without save_debug_metrics → skip saving (original behaviour)
        _save_debug_metrics = args.debug and getattr(args.qa, 'save_debug_metrics', False)
        if not args.debug or _save_debug_metrics:
            if _save_debug_metrics:
                save_dir = os.path.join('metrics', 'debug', args.experiment,
                                        args.editing_method, f'data_{args.ds_size}_{args.ds_seed}')
            else:
                save_dir = os.path.join(args.metrics_save_dir, args.experiment, args.editing_method,
                                        f'data_{args.ds_size}_{args.ds_seed}')
            os.makedirs(save_dir, exist_ok=True)
            # `setting` was computed before wandb.init; reuse it here.
            json.dump(metrics, open(os.path.join(save_dir,
                                                 f'{increment}_{mode}_{setting}_{args.qa.model}_results.json'), 'w'), indent=4)
            if args.editing_method == 'EVAL':
                save_predictions_csv(metrics, tags, setting,
                                     os.path.join(save_dir, f'{increment}_{mode}_predictions.csv'))
        if hasattr(edited_model, 'out') and len(edited_model.out) > 0:
            data = pd.DataFrame(edited_model.out, columns=['context', 'question', 'answers'])
            data.to_csv(os.path.join(args.metrics_save_dir, args.experiment, args.editing_method,
                                     f'data_{args.ds_size}_{args.ds_seed}',
                                     f'{increment}_{mode}_{args.qa.model}_results.csv'), index=False)


def load_csv_data(data_path, include_evidence=True, increment=None, batching_cfg=None,
                  _df_override=None, filter_conflicting_edits=True):
    """Load HemOnc CSV and convert rows to the JSON record format expected by extract_data().

    Supports both v2 datasets (with 'closed question 4' and 'closed question 5') and
    v3 datasets (with 'closed question m' / 'closed question m answer' instead).

    Parameters
    ----------
    data_path      : path to Hemonc_Edit_vX.csv
    include_evidence : prepend "Evidence: ..." to every prompt when True
    increment      : batch label string from build_increments(); if None, all rows
                     are returned (backward-compatible behaviour)
    batching_cfg   : OmegaConf node with fields ``strategy`` and ``crop_year``
                     (qa.batching from the experiment YAML).  When None or
                     strategy=='none', no filtering is applied.
    _df_override   : if provided, use this pre-filtered DataFrame directly and skip
                     CSV loading and batching filter steps.
    filter_conflicting_edits : when True, rows that belong to a conflicting group
                     (same condition/context/endpoint/regimen/comparator tuple with
                     multiple distinct answers) are deduplicated by keeping only the
                     latest update (highest date) per group.
    """
    if _df_override is not None:
        df = _df_override.copy()
    else:
        df = pd.read_csv(data_path)

        # ── Apply batching filter ────────────────────────────────────────────────
        strategy  = getattr(batching_cfg, 'strategy',  'none') if batching_cfg is not None else 'none'
        crop_year = getattr(batching_cfg, 'crop_year', None)   if batching_cfg is not None else None

        if strategy != 'none' and increment is not None and increment != 'hemonc_full':
            from hemonc_batching import filter_by_increment
            df = filter_by_increment(df, strategy=strategy, increment=increment,
                                     crop_year=crop_year)
            print(f"  [batching:{strategy}] increment '{increment}': {len(df)} rows")
        elif crop_year is not None:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df[df['date'].dt.year >= int(crop_year)].reset_index(drop=True)
            print(f"  [batching:none] crop_year={crop_year}: {len(df)} rows")

    # ── Filter conflicting edits ─────────────────────────────────────────────
    if filter_conflicting_edits:
        _key_cols = ['condition', 'context', 'endpoint', 'regimen', 'comparator']
        _before = len(df)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Detect conflicting groups (≥2 distinct answers for the same tuple).
        # Use the pre-computed column when present, otherwise compute on the fly.
        if 'conflicting_edit' in df.columns:
            _is_conflicting = df['conflicting_edit'].astype(bool)
        else:
            _is_conflicting = df.groupby(_key_cols)['answer'].transform('nunique') > 1
        # Keep only the latest row per conflicting group; retain all clean rows.
        _latest = (
            df[_is_conflicting]
            .sort_values('date')
            .groupby(_key_cols, sort=False)
            .tail(1)
        )
        df = pd.concat([df[~_is_conflicting], _latest]).sort_index().reset_index(drop=True)
        print(f"  [filter_conflicting_edits] {_before - len(df)} rows removed, {len(df)} rows remain")

    if len(df) == 0:
        _strategy = getattr(batching_cfg, 'strategy', 'none')
        _crop_year = getattr(batching_cfg, 'crop_year', None)
        raise ValueError(
            f"No rows loaded for increment='{increment}' "
            f"(strategy='{_strategy}', crop_year={_crop_year}). "
            "Check that the increment label is valid and the data path is correct."
        )

    has_locality = 'locality_question' in df.columns
    has_cq45 = 'closed question 4' in df.columns and 'closed question 5' in df.columns
    has_cq_m = 'closed question m' in df.columns
    if has_locality:
        n_nan = df['locality_question'].isna().sum()
        if n_nan:
            print(f"Locality columns detected: {n_nan}/{len(df)} rows have no locality question (locality eval skipped for those).")
        else:
            print(f"Locality columns detected: all {len(df)} rows have valid locality questions.")
    if has_cq_m:
        print(f"v3 dataset detected: loading 'closed question m' as mirrored portability probe.")
    records = []
    for _, row in df.iterrows():
        evidence = str(row['evidence'])
        ground_truth_statement = str(row['ground truth'])

        def make_prompt(q, _ev=evidence, _gts=ground_truth_statement):
            # include_evidence controls what (if anything) is prepended:
            #   False / 'none'         → plain question, no prepended context
            #   True  / 'abstract'     → prepend the trial abstract
            #   'ground_truth'         → prepend the full ground-truth statement
            #                            (e.g. "Regimen A inferior to B for C [PFS]")
            if include_evidence in (True, 'abstract'):
                return f"Evidence: {_ev}\n\n{str(q)}"
            if include_evidence == 'ground_truth':
                return f"Evidence: {_gts}\n\n{str(q)}"
            return str(q)

        answer = str(row['answer'])
        gt_dict_oq = {
            'condition': str(row['condition']),
            'context': str(row['context']),
            'target': str(row['ground truth']),
        }
        gt_dict_og = {
            'condition': str(row['condition']),
            'context': str(row['context']),
            'target': str(row['ground truth']),
        }
        record = {
            'src':      make_prompt(row['closed question 1']),
            'alt':      answer,
            'subject':  f"{row['regimen']} compared to {row['comparator']}",
            'evidence': evidence,
            'ground_truth_statement': str(row['ground truth']),
            'tag':      f"{row['date']}|{row['regimen']}|{row['comparator']}|{row['condition']}|{row['context']}|{row['endpoint']}",
            'date':     str(row['date']),
            'rephrase': make_prompt(row['closed question 2']),
            'genv2_cq_2': {'prompt': make_prompt(row['closed question 2']), 'ground_truth': answer},
            'genv2_cq_3': {'prompt': make_prompt(row['closed question 3']), 'ground_truth': answer},
            'genv2_oq_1': {'prompt': make_prompt(row['open question 1']),   'ground_truth': gt_dict_oq},
            'genv2_og_1': {'prompt': make_prompt(row['open generation 1']), 'ground_truth': gt_dict_og},
        }
        # v2 datasets: closed questions 4 and 5
        if has_cq45:
            record['genv2_cq_4'] = {'prompt': make_prompt(row['closed question 4']), 'ground_truth': answer}
            record['genv2_cq_5'] = {'prompt': make_prompt(row['closed question 5']), 'ground_truth': answer}
        # v3 datasets: mirrored closed question (regimen/comparator swapped, answer flipped)
        if has_cq_m:
            mirror_answer = str(row['closed question m answer'])
            record['genv2_cq_m'] = {'prompt': make_prompt(row['closed question m']), 'ground_truth': mirror_answer}
        if has_locality:
            loc_q = row['locality_question']
            loc_a = row['locality_ground_truth']
            record['loc']     = make_prompt(str(loc_q)) if not pd.isna(loc_q) else None
            record['loc_ans'] = str(loc_a) if not pd.isna(loc_a) else None
        records.append(record)
    return records


def save_predictions_csv(metrics, tags, setting, save_path):
    """Flatten all_metrics into a CSV compatible with the existing predictions schema.

    Appends to any existing file at save_path, replacing only rows that share the
    same `setting` value (so each of the four eval modes can be written independently
    and re-runs safely overwrite their own rows without touching other modes).
    """
    Q_KEYS = [
        ('rewrite_acc',   'closed question 1', 'cq'),
        ('rephrase_acc',  'closed question 2', 'cq'),
    ]
    PORT_KEYS = [
        ('genv2_cq_3', 'closed question 3', 'cq'),
        ('genv2_cq_4', 'closed question 4', 'cq'),   # v2 datasets only
        ('genv2_cq_5', 'closed question 5', 'cq'),   # v2 datasets only
        ('genv2_cq_m', 'closed question m', 'cq'),   # v3 datasets: mirrored question
        ('genv2_oq_1', 'open question 1',   'oq'),
        ('genv2_og_1', 'open generation 1', 'og'),
    ]
    rows = []
    for m, tag in zip(metrics, tags):
        parts = tag.split('|')
        if len(parts) < 5:
            continue
        date, regimen, comparator, condition, context = parts[0], parts[1], parts[2], parts[3], parts[4]
        pm = m.get('past', m)

        base = {'date': date, 'regimen': regimen, 'comparator': comparator,
                'condition': condition, 'context': context, 'setting': setting}

        # Main rewrite / rephrase
        for metric_key, qcol, qtype in Q_KEYS:
            val = pm.get(metric_key)
            if val is None:
                continue
            acc = val[0] if isinstance(val, list) else val
            ref_key = 'rewrite_refused' if metric_key == 'rewrite_acc' else 'rephrase_refused'
            trc_key = 'rewrite_truncated' if metric_key == 'rewrite_acc' else 'rephrase_truncated'
            rows.append({**base, 'question_col': qcol, 'correct': int(round(float(acc))),
                         'judge_score': None, 'judge_explanation': None,
                         'refused': bool(pm.get(ref_key, False)),
                         'truncated': bool(pm.get(trc_key, False))})

        # Portability inputs
        portability = pm.get('portability', {})
        for p_key, qcol, qtype in PORT_KEYS:
            perf_dict = portability.get(p_key, {}).get('performance', None)
            if perf_dict is None:
                continue
            refused = bool(perf_dict.get('refused', False)) if isinstance(perf_dict, dict) else False
            truncated = bool(perf_dict.get('truncated', False)) if isinstance(perf_dict, dict) else False
            if qtype == 'cq':
                acc = perf_dict.get('acc', [None])
                acc = acc[0] if isinstance(acc, list) else acc
                if acc is not None:
                    rows.append({**base, 'question_col': qcol,
                                 'correct': int(round(float(acc))),
                                 'judge_score': None, 'judge_explanation': None,
                                 'refused': refused, 'truncated': truncated})
            else:
                score = perf_dict.get('score') if isinstance(perf_dict, dict) else None
                if score is not None:
                    rows.append({**base, 'question_col': qcol,
                                 'correct': int(score >= 4),
                                 'judge_score': score,
                                 'judge_explanation': perf_dict.get('explanation', ''),
                                 'refused': refused, 'truncated': truncated})

    if rows:
        new_df = pd.DataFrame(rows)
        if os.path.exists(save_path):
            existing_df = pd.read_csv(save_path)
            # Drop any rows belonging to this setting so a re-run overwrites cleanly
            existing_df = existing_df[existing_df['setting'] != setting]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        combined_df.to_csv(save_path, index=False)
        print(f"Saved predictions CSV to {save_path} "
              f"({len(new_df)} new rows for setting='{setting}', "
              f"{len(combined_df)} total rows)")


def extract_data(test_data):
    prompts, rephrase_prompts, target_new, subject, tags, evidence = [], [], [], [], [], []
    locality_inputs = {
        'neighborhood': {
            'prompt': [],
            'ground_truth': [],
        },
    }
    portability_inputs = {}
    prompts += [test_data_['src'] for test_data_ in test_data]
    target_new += [edit_data_['alt'] for edit_data_ in test_data]
    subject += [edit_data_['subject'] for edit_data_ in test_data]
    evidence += [edit_data_['evidence'] for edit_data_ in test_data]

    if 'rephrase' in test_data[0].keys():
        rephrase_prompts += [edit_data_['rephrase'] for edit_data_ in test_data]
    else:
        rephrase_prompts = None
    if 'loc' in test_data[0].keys():
        locality_prompts = [edit_data_['loc'] for edit_data_ in test_data]
        locality_ans = [edit_data_['loc_ans'] for edit_data_ in test_data]
        locality_inputs['neighborhood']['prompt'] += locality_prompts
        locality_inputs['neighborhood']['ground_truth'] += locality_ans
    else:
        locality_inputs = None
    if 'tag' in test_data[0].keys():
        tags += [edit_data_['tag'] for edit_data_ in test_data]
    else:
        tags = None
    if 'mhop' in test_data[0].keys():
        portability_inputs['mhop'] = {
            'prompt': [],
            'ground_truth': [],
        }
        mhop_questions = [edit_data_['mhop'] for edit_data_ in test_data]
        mhop_answers = [edit_data_['mhop_ans'] for edit_data_ in test_data]
        portability_inputs['mhop']['prompt'] += mhop_questions
        portability_inputs['mhop']['ground_truth'] += mhop_answers

    genv2_questions = {}
    for key in test_data[0].keys():
        if key.startswith('genv2_'):
            genv2_prompts = [edit_data_[key]['prompt'] for edit_data_ in test_data]
            genv2_ground_truth = [edit_data_[key]['ground_truth'] for edit_data_ in test_data]
            genv2_questions[key] = {'prompt': genv2_prompts, 'ground_truth': genv2_ground_truth}
            if key not in portability_inputs:
                portability_inputs[key] = {
                    'prompt': [],
                    'ground_truth': [],
                }
    for key in genv2_questions.keys():
        print(f'Adding {key} to portability inputs '
              f'({len(genv2_questions[key]["prompt"])} samples)')
        portability_inputs[key]['prompt'] += genv2_questions[key]['prompt']
        portability_inputs[key]['ground_truth'] += genv2_questions[key]['ground_truth']

    return prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject, evidence


def shuffle_inputs(prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject, evidence):
    shuffle_idx = list(range(len(prompts)))
    random.shuffle(shuffle_idx)
    prompts = [prompts[i] for i in shuffle_idx]
    rephrase_prompts = [rephrase_prompts[i] for i in shuffle_idx]
    target_new = [target_new[i] for i in shuffle_idx]
    if locality_inputs is not None:
        locality_inputs['neighborhood']['prompt'] = [locality_inputs['neighborhood']['prompt'][i] for i in shuffle_idx]
        locality_inputs['neighborhood']['ground_truth'] = [locality_inputs['neighborhood']['ground_truth'][i] for i in
                                                           shuffle_idx]
    subject = [subject[i] for i in shuffle_idx]
    evidence = [evidence[i] for i in shuffle_idx]
    tags = [tags[i] for i in shuffle_idx]
    for key in portability_inputs.keys():
        portability_inputs[key]['prompt'] = [portability_inputs[key]['prompt'][i] for i in shuffle_idx]
        portability_inputs[key]['ground_truth'] = [portability_inputs[key]['ground_truth'][i] for i in shuffle_idx]
    return prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject, evidence


_AUTO_SAVE_INTERVAL = {
    'none': 1, 'monthly': 1, 'quarterly': 1,
    'weekly': 2, 'daily': 10, 'publication': 20,
}


def _resolve_save_interval(args):
    """Return the effective save_interval (int >= 1).

    `checkpoint.save_interval` may be null in config; in that case we pick a
    reasonable default from `qa.batching.strategy`.
    """
    explicit = getattr(args.checkpoint, 'save_interval', None)
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except (TypeError, ValueError):
            pass
    batching = getattr(args.qa, 'batching', None)
    strategy = getattr(batching, 'strategy', 'none') if batching is not None else 'none'
    return _AUTO_SAVE_INTERVAL.get(strategy, 1)


def _delete_checkpoint_files(checkpoint_dir, increment, mode, model_name):
    """Remove all on-disk artifacts for a single checkpoint (state_dict, RAG
    save_model directory, and args.json)."""
    base = os.path.join(checkpoint_dir, f'{increment}_{mode}_{model_name}')
    # state_dict .pt
    pt_path = base + '.pt'
    if os.path.exists(pt_path):
        try:
            os.remove(pt_path)
        except OSError as e:
            print(f'[checkpoint] Could not remove {pt_path}: {e}')
    # RAG save_model path — could be a directory OR a file at `base`
    if os.path.isdir(base):
        shutil.rmtree(base, ignore_errors=True)
    elif os.path.isfile(base):
        try:
            os.remove(base)
        except OSError as e:
            print(f'[checkpoint] Could not remove {base}: {e}')
    # args.json done-marker
    args_path = base + '_args.json'
    if os.path.exists(args_path):
        try:
            os.remove(args_path)
        except OSError as e:
            print(f'[checkpoint] Could not remove {args_path}: {e}')


def _prune_checkpoints(checkpoint_dir, args, mode, milestones, total_batches):
    """Keep only the checkpoints closest to each fractional milestone (plus the
    most-recent save for resume).  Delete the rest.

    Milestones are in [0, 1]; target index = round(m * total_batches) - 1 over
    `args.qa.increments`.  If `milestones` is empty we keep only the latest save.
    """
    if not os.path.isdir(checkpoint_dir):
        return
    suffix = f'_{mode}_{args.qa.model}_args.json'
    increments_list = list(args.qa.increments)
    inc_to_idx = {inc: i for i, inc in enumerate(increments_list)}

    saves = []  # list of (idx, increment)
    for fname in os.listdir(checkpoint_dir):
        if not fname.endswith(suffix):
            continue
        inc = fname[:-len(suffix)]
        if inc in inc_to_idx:
            saves.append((inc_to_idx[inc], inc))
    if not saves:
        return
    saves.sort()
    saved_idxs = [s[0] for s in saves]

    keep = set()
    # Latest-for-resume: always keep the highest saved index
    keep.add(saves[-1][1])
    # Milestone targets
    if milestones and total_batches and total_batches > 0:
        for m in milestones:
            try:
                mf = float(m)
            except (TypeError, ValueError):
                continue
            target = max(0, int(round(mf * total_batches)) - 1)
            # pick saved index closest to target (ties → lower index)
            best_i = min(range(len(saved_idxs)),
                         key=lambda i: (abs(saved_idxs[i] - target), saved_idxs[i]))
            keep.add(saves[best_i][1])

    pruned = []
    for idx, inc in saves:
        if inc in keep:
            continue
        _delete_checkpoint_files(checkpoint_dir, inc, mode, args.qa.model)
        pruned.append(inc)
    if pruned:
        print(f'[checkpoint] Pruned {len(pruned)} old checkpoint(s); '
              f'kept {sorted(keep)}')


def save_checkpoint(args, increment, mode, wandb_run_id, model=None,
                    batch_idx=None, total_batches=None, is_final=False):
    """Write a checkpoint pair (args.json [+ model]) and prune old ones.

    Parameters
    ----------
    batch_idx : int or None
        1-based count of processed increments in this run.  Used with
        `save_interval` to decide whether to write.  `None` → always write
        (legacy callers / EVAL path).
    total_batches : int or None
        Total number of increments in this run; used to resolve milestone
        targets during pruning.
    is_final : bool
        When True, force a write regardless of interval (end of run).
    """
    checkpoint_dir = f'{args.checkpoint.save_dir}/{args.experiment}/{args.editing_method}/data_{args.ds_size}_{args.ds_seed}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not args.checkpoint.save:
        return

    # EVAL path (model is None and caller doesn't specify batch_idx) always
    # writes the tiny args.json done-marker — resume relies on it to skip
    # already-evaluated batches precisely.  When model is given we gate by
    # interval.
    if model is not None and batch_idx is not None:
        interval = _resolve_save_interval(args)
        if (batch_idx % interval != 0) and not is_final:
            return

    args_dict = oc.to_container(args, resolve=True, throw_on_missing=True)
    args_dict.update({'wandb_run_id': wandb_run_id})

    args_path = os.path.join(checkpoint_dir, f'{increment}_{mode}_{args.qa.model}_args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    print(f'Saved args for increment {increment} to {args_path}')
    if model is not None:
        model_path = os.path.join(checkpoint_dir, f'{increment}_{mode}_{args.qa.model}')
        if hasattr(model, 'save_model'):
            model.save_model(model_path)
        else:
            # Standard HuggingFace model (e.g. after MEMIT/WISE/AlphaEdit editing)
            torch.save(model.state_dict(), model_path + '.pt')
            print(f'Saved model state_dict to {model_path}.pt')

    # Prune old checkpoints (milestones + latest-for-resume)
    milestones = getattr(args.checkpoint, 'keep_milestones', None) or []
    try:
        milestones = list(milestones)
    except TypeError:
        milestones = []
    _prune_checkpoints(checkpoint_dir, args, mode, milestones, total_batches)


# Adapter-wrapping methods whose checkpoints can't be loaded as a plain HF
# state_dict.  For these, we wrap a fresh HF model first (the wrap is in-place
# via setattr on the target MLP layer) and then call the wrapper's load_model
# to restore Parameter/buffer state AND the Python-only state (memory_weight
# list for WISE, codebook for GRACE, hasher.permutation+masks for MEMOIR).
_ADAPTER_METHOD_LOADERS = {
    'WISE':   ('easyeditor.models.wise',   'load_wise_into_model'),
    'GRACE':  ('easyeditor.models.grace',  'load_grace_into_model'),
    'MEMOIR': ('easyeditor.models.memoir', 'load_memoir_into_model'),
}


def _resume_adapter_method(method, model, hparams, model_path):
    """Dispatch to the per-method `load_*_into_model` helper.

    Imported lazily so the runner doesn't pull in WISE/GRACE/MEMOIR (and their
    transitive deps) when running other editing methods.
    """
    mod_path, fn_name = _ADAPTER_METHOD_LOADERS[method]
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)(model, hparams, model_path)


def load_checkpoint(args, mode, model=None, hparams=None):
    """Resume from the most recent checkpoint <= the final increment.

    Parameters
    ----------
    model : nn.Module or None
        The fresh, un-edited HF model (typically `editor.model`).  None for
        EVAL runs (no model state to restore).
    hparams : object or None
        Editor hparams; required for adapter-wrapping methods (WISE/GRACE/
        MEMOIR) so the wrapper can be re-constructed at load time before the
        saved adapter state is restored.  Optional for other methods.
    """
    checkpoint_dir = f'{args.checkpoint.save_dir}/{args.experiment}/{args.editing_method}/data_{args.ds_size}_{args.ds_seed}'
    if args.checkpoint.load:
        wandb_run_id = None
        outstanding_increments = []
        for increment in args.qa.increments[::-1]:
            try:
                with open(os.path.join(checkpoint_dir, f'{increment}_{mode}_{args.qa.model}_args.json'), 'r') as f:
                    ckp_args = json.load(f)
                wandb_run_id = ckp_args['wandb_run_id']
                if model is not None:
                    model_path = os.path.join(checkpoint_dir, f'{increment}_{mode}_{args.qa.model}')
                    if args.editing_method in _ADAPTER_METHOD_LOADERS:
                        # WISE/GRACE/MEMOIR — wrap-then-load.  Need hparams.
                        if hparams is None:
                            raise RuntimeError(
                                f'load_checkpoint: hparams required for '
                                f'{args.editing_method} resume but none provided.'
                            )
                        _resume_adapter_method(args.editing_method, model, hparams, model_path)
                    elif hasattr(model, 'load_model'):
                        model.load_model(model_path)
                    else:
                        # Standard HuggingFace model saved as state_dict
                        state_dict = torch.load(model_path + '.pt',
                                                map_location=next(model.parameters()).device)
                        model.load_state_dict(state_dict)
                        del state_dict
                print(f'Loaded checkpoint for increment {increment}')
                break
            except FileNotFoundError:
                outstanding_increments.append(increment)
                continue
        outstanding_increments = outstanding_increments[::-1]
        return outstanding_increments, wandb_run_id
    else:
        return args.qa.increments, None


def _eval_sentinel(editor, pool, current_increment, args, edited_model):
    """Evaluate the sentinel pool with closed-QA only (no LLM judge).

    Runs a single editor.eval_samples() call over the full pool with
    judge_enabled=False, then groups results by batch of origin.

    Returns a dict suitable for wandb logging and JSON disk saving.
    """
    # Build parallel lists from pool records
    records = [entry['record'] for entry in pool]
    increments = [entry['increment'] for entry in pool]

    prompts, rephrase_prompts, target_new, tags, locality_inputs, portability_inputs, subject, evidence = extract_data(records)

    _, all_metrics = editor.eval_samples(
        prompts=prompts,
        rephrase_prompts=rephrase_prompts,
        target_new=target_new,
        tags=tags,
        ground_truth=evidence,
        few_shot_examples=False,
        portability_inputs=portability_inputs,
        locality_inputs=locality_inputs,
        edited_model=edited_model,
        batch_size=args.batch_size,
        judge_enabled=False,
    )

    # Closed-QA keys to aggregate. The first element is the (top-level key,
    # portability sub-key) tuple — exactly one of the two is non-None.
    CQ_KEYS = [
        ('rewrite_acc',  None),
        ('rephrase_acc', None),
        (None, 'genv2_cq_3'),
        (None, 'genv2_cq_4'),
        (None, 'genv2_cq_5'),
        (None, 'genv2_cq_m'),
    ]

    def _task_label(m_key, p_key):
        return m_key if m_key is not None else p_key

    def _task_value(pm, m_key, p_key):
        if m_key is not None:
            v = pm.get(m_key)
            if v is None:
                return None
            return float(v[0] if isinstance(v, list) else v)
        perf = pm.get('portability', {}).get(p_key, {}).get('performance', {})
        acc = perf.get('acc') if isinstance(perf, dict) else None
        if acc is None:
            return None
        return float(acc[0] if isinstance(acc, list) else acc)

    def _cq_avg(m):
        pm = m.get('past', m)
        vals = []
        for m_key, p_key in CQ_KEYS:
            v = _task_value(pm, m_key, p_key)
            if v is not None:
                vals.append(v)
        return float(np.mean(vals)) if vals else None

    # ── Aggregate by origin increment (existing behaviour) ──────────────
    from collections import defaultdict
    by_origin = defaultdict(list)
    # Per-task buckets: task_label → origin → list of per-case values
    per_task_buckets: dict[str, dict[str, list[float]]] = {
        _task_label(mk, pk): defaultdict(list) for mk, pk in CQ_KEYS
    }

    for inc, m in zip(increments, all_metrics):
        pm = m.get('past', m)
        cq = _cq_avg(m)
        if cq is not None:
            by_origin[inc].append(cq)
        for mk, pk in CQ_KEYS:
            v = _task_value(pm, mk, pk)
            if v is not None:
                per_task_buckets[_task_label(mk, pk)][inc].append(v)

    by_origin_list = [
        {'increment': inc, 'n_samples': len(vals), 'cq_avg': float(np.mean(vals))}
        for inc, vals in sorted(by_origin.items())
    ]

    all_cq = [v for vals in by_origin.values() for v in vals]
    sentinel_cq_avg = float(np.mean(all_cq)) if all_cq else None

    # ── Per-task aggregates (additive — sentinel_cq_avg / by_origin
    # remain unchanged so older analysis code keeps working). ─────────
    sentinel_per_task_avg: dict[str, float | None] = {}
    by_origin_per_task: dict[str, list[dict]] = {}
    for task_label, origin_map in per_task_buckets.items():
        flat = [v for vals in origin_map.values() for v in vals]
        sentinel_per_task_avg[task_label] = (
            float(np.mean(flat)) if flat else None
        )
        by_origin_per_task[task_label] = [
            {'increment': inc, 'n_samples': len(vals),
             'avg': float(np.mean(vals))}
            for inc, vals in sorted(origin_map.items())
        ]

    return {
        'evaluated_after': current_increment,
        'pool_size': len(pool),
        'n_batches_covered': len(by_origin_list),
        'sentinel_cq_avg': sentinel_cq_avg,
        'by_origin': by_origin_list,
        # New (additive) per-task fields. Old readers that look up
        # sentinel_cq_avg / by_origin will simply ignore them.
        'sentinel_per_task_avg': sentinel_per_task_avg,
        'by_origin_per_task': by_origin_per_task,
    }


def _sentinel_metrics_to_wandb(sentinel_metrics, current_increment):
    """Log sentinel pool results to wandb as a retention curve table + scalars."""
    table = wandb.Table(columns=['origin_batch', 'n_samples', 'cq_avg'])
    for row in sentinel_metrics['by_origin']:
        table.add_data(row['increment'], row['n_samples'], row['cq_avg'])

    log_dict = {f'retention_curve/{current_increment}': table}
    if sentinel_metrics['sentinel_cq_avg'] is not None:
        log_dict['Retention/sentinel_cq_avg']   = sentinel_metrics['sentinel_cq_avg']
    log_dict['Retention/n_batches_covered'] = sentinel_metrics['n_batches_covered']
    log_dict['Retention/pool_size']         = sentinel_metrics['pool_size']
    wandb.log(log_dict)


def _save_sentinel_metrics(sentinel_metrics, args, increment, mode, setting):
    """Save sentinel evaluation results to a JSON file alongside batch results."""
    save_dir = os.path.join(args.metrics_save_dir, args.experiment, args.editing_method,
                            f'data_{args.ds_size}_{args.ds_seed}')
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'{increment}_{mode}_{setting}_sentinel.json')
    with open(path, 'w') as f:
        json.dump(sentinel_metrics, f, indent=4)
    print(f'Saved sentinel metrics to {path}')


# ── Metric-extraction helpers (shared by wandb logger and any future callers) ─

_CQ_MAP = [
    ('rewrite_acc',  None,         1),   # (top-level key, portability key, q_num)
    ('rephrase_acc', None,         2),
    (None, 'genv2_cq_3',           3),
    (None, 'genv2_cq_4',           4),   # v2 datasets only
    (None, 'genv2_cq_5',           5),   # v2 datasets only
    (None, 'genv2_cq_m',          'm'),  # v3 datasets: mirrored question
]
_Q_LABEL = {1: 'q1', 2: 'q2', 3: 'q3', 4: 'q4', 5: 'q5', 'm': 'qm'}


def _extract_sample_metrics(pm):
    """Return (sample_cq, oq_score, og_score, loc_acc, sample_refused, sample_truncated) from one phase-metrics dict.

    sample_cq         : dict {q_num → float acc}
    oq_score          : float 0–1 (judge score / 5), or None
    og_score          : float 0–1 (judge score / 5), or None
    loc_acc           : float 0–1, or None
    sample_refused    : dict {label → bool}  — labels: q1,q2,q3,q4,q5,qm,oq,og,loc
    sample_truncated  : dict {label → bool}  — same labels
    """
    portability = pm.get('portability', {})

    sample_cq = {}
    sample_refused = {}
    sample_truncated = {}

    # --- Closed Questions ---
    for m_key, p_key, q_num in _CQ_MAP:
        if m_key is not None:
            val = pm.get(m_key)
            if val is None:
                continue
            acc = val[0] if isinstance(val, list) else val
            # rewrite/rephrase flags live at the top level
            ref_key = 'rewrite_refused' if m_key == 'rewrite_acc' else 'rephrase_refused'
            trc_key = 'rewrite_truncated' if m_key == 'rewrite_acc' else 'rephrase_truncated'
            ref = pm.get(ref_key)
            trc = pm.get(trc_key)
        else:
            perf = portability.get(p_key, {}).get('performance', {})
            if perf is None or perf == {}:
                continue
            if isinstance(perf, dict):
                acc = perf.get('acc')
                if isinstance(acc, list):
                    acc = acc[0]
                ref = perf.get('refused')
                trc = perf.get('truncated')
            elif isinstance(perf, (int, float, np.floating)):
                acc = float(perf)  # GRACE returns a raw scalar
                ref = None
                trc = None
            else:
                continue
        if acc is not None:
            sample_cq[q_num] = float(acc)
        label = _Q_LABEL.get(q_num)
        if label is not None:
            if ref is not None:
                sample_refused[label] = bool(ref)
            if trc is not None:
                sample_truncated[label] = bool(trc)

    # --- Open Questions ---
    oq_score = None
    oq_perf = portability.get('genv2_oq_1', {}).get('performance', {})
    if isinstance(oq_perf, dict):
        if oq_perf.get('score') is not None:
            oq_score = float(oq_perf['score']) / 5.0
        if oq_perf.get('refused') is not None:
            sample_refused['oq'] = bool(oq_perf['refused'])
        if oq_perf.get('truncated') is not None:
            sample_truncated['oq'] = bool(oq_perf['truncated'])

    # --- Open Generation ---
    og_score = None
    og_perf = portability.get('genv2_og_1', {}).get('performance', {})
    if isinstance(og_perf, dict):
        if og_perf.get('score') is not None:
            og_score = float(og_perf['score']) / 5.0
        if og_perf.get('refused') is not None:
            sample_refused['og'] = bool(og_perf['refused'])
        if og_perf.get('truncated') is not None:
            sample_truncated['og'] = bool(og_perf['truncated'])

    # --- Locality ---
    loc_acc = None
    loc_data = pm.get('locality', {})
    for _loc_key in ('neighborhood_acc', 'neighborhood'):
        _loc = loc_data.get(_loc_key)
        if _loc is None:
            continue
        if isinstance(_loc, dict):
            _perf = _loc.get('performance', {})
            if isinstance(_perf, dict):
                _acc = _perf.get('acc')
                if _perf.get('refused') is not None:
                    sample_refused['loc'] = bool(_perf['refused'])
                if _perf.get('truncated') is not None:
                    sample_truncated['loc'] = bool(_perf['truncated'])
            else:
                _acc = None
        else:
            _acc = _loc
        if _acc is not None:
            loc_acc = float(_acc[0] if isinstance(_acc, list) else _acc)
        break

    return sample_cq, oq_score, og_score, loc_acc, sample_refused, sample_truncated


def _phase_summary_stats(phase_pms):
    """Compute batch-level summary stats for a list of per-sample phase-metrics dicts.

    Returns a flat dict ready to be prefixed and logged to wandb:
      Closed Questions/{q1,q2,q3,q4,q5,qm} — per-variant mean acc across batch
      Closed Questions/mean                  — mean of all present variant means
      Closed Questions/all_correct           — fraction of samples where every CQ == 1.0
      Open Questions/score                   — mean judge score (0–1) across batch
      Open Generation/score                  — mean judge score (0–1) across batch
      Locality/acc                           — mean acc across batch
      Safety/refusal_rate_{label}            — per-variant refusal rate (labels: q1…qm, oq, og, loc)
      Safety/refusal_rate                    — overall refusal rate (mean over all generations)
      Safety/truncation_rate_{label}         — per-variant truncation rate
      Safety/truncation_rate                 — overall truncation rate
    """
    cq_per_variant = {1: [], 2: [], 3: [], 4: [], 5: [], 'm': []}
    cq_all_correct = []
    oq_scores, og_scores, loc_accs = [], [], []
    refused_per_label = {}
    truncated_per_label = {}

    for pm in phase_pms:
        sample_cq, oq_score, og_score, loc_acc, sample_refused, sample_truncated = _extract_sample_metrics(pm)
        for q_num, acc in sample_cq.items():
            cq_per_variant[q_num].append(acc)
        if sample_cq:
            cq_all_correct.append(1.0 if all(v == 1.0 for v in sample_cq.values()) else 0.0)
        if oq_score is not None:
            oq_scores.append(oq_score)
        if og_score is not None:
            og_scores.append(og_score)
        if loc_acc is not None:
            loc_accs.append(loc_acc)
        for label, val in sample_refused.items():
            refused_per_label.setdefault(label, []).append(1.0 if val else 0.0)
        for label, val in sample_truncated.items():
            truncated_per_label.setdefault(label, []).append(1.0 if val else 0.0)

    stats = {}
    variant_means = []
    for q_num, accs in cq_per_variant.items():
        if accs:
            vm = float(np.mean(accs))
            stats[f'Closed Questions/{_Q_LABEL[q_num]}'] = vm
            variant_means.append(vm)
    if variant_means:
        stats['Closed Questions/mean'] = float(np.mean(variant_means))
    if cq_all_correct:
        stats['Closed Questions/all_correct'] = float(np.mean(cq_all_correct))
    if oq_scores:
        stats['Open Questions/score'] = float(np.mean(oq_scores))
    if og_scores:
        stats['Open Generation/score'] = float(np.mean(og_scores))
    if loc_accs:
        stats['Locality/acc'] = float(np.mean(loc_accs))

    all_refusals = []
    for label, vals in refused_per_label.items():
        if vals:
            stats[f'Safety/refusal_rate_{label}'] = float(np.mean(vals))
            all_refusals.extend(vals)
    if all_refusals:
        stats['Safety/refusal_rate'] = float(np.mean(all_refusals))

    all_trunc = []
    for label, vals in truncated_per_label.items():
        if vals:
            stats[f'Safety/truncation_rate_{label}'] = float(np.mean(vals))
            all_trunc.extend(vals)
    if all_trunc:
        stats['Safety/truncation_rate'] = float(np.mean(all_trunc))

    return stats


def _build_phase_table(phase_pms, phase_tags, setting, phase):
    """Build a wandb.Table of per-sample metrics for one evaluation phase."""
    columns = [
        'date', 'year', 'setting', 'phase',
        'Closed Questions/q1', 'Closed Questions/q2', 'Closed Questions/q3',
        'Closed Questions/q4', 'Closed Questions/q5', 'Closed Questions/qm',
        'Closed Questions/mean', 'Closed Questions/all_correct',
        'Open Questions/score',
        'Open Generation/score',
        'Locality/acc',
        'any_refused', 'any_truncated',
    ]
    table = wandb.Table(columns=columns)
    for pm, tag in zip(phase_pms, phase_tags):
        parts = (tag or '').split('|')
        date = parts[0] if parts else None
        year = int(date[:4]) if date and len(date) >= 4 else None
        sample_cq, oq_score, og_score, loc_acc, sample_refused, sample_truncated = _extract_sample_metrics(pm)
        cq_mean = float(np.mean(list(sample_cq.values()))) if sample_cq else None
        all_correct = float(all(v == 1.0 for v in sample_cq.values())) if sample_cq else None
        any_refused = bool(any(sample_refused.values())) if sample_refused else None
        any_truncated = bool(any(sample_truncated.values())) if sample_truncated else None
        table.add_data(
            date, year, setting, phase,
            sample_cq.get(1), sample_cq.get(2), sample_cq.get(3),
            sample_cq.get(4), sample_cq.get(5), sample_cq.get('m'),
            cq_mean, all_correct,
            oq_score, og_score, loc_acc,
            any_refused, any_truncated,
        )
    return table


def eval_metrics_to_wandb(summary_metrics, all_metrics=None, tags=None, setting=None):
    """Log structured evaluation metrics to wandb, separated by evaluation phase.

    Each item in all_metrics is expected to be a dict with one or more phase keys:
      - 'pre'  : pre-edit performance on the current update batch
      - 'post' : post-edit performance on the current update batch
      - 'past' : performance on a static evaluation set (EVAL-only mode)

    For each present phase, logs:
      {phase}/Closed Questions/{q1,q2,q3,q4,q5,qm} — per-variant mean acc across batch
      {phase}/Closed Questions/mean                  — mean over all present variant means
      {phase}/Closed Questions/all_correct           — fraction of samples with every CQ correct
      {phase}/Open Questions/score                   — mean judge score (0–1) across batch
      {phase}/Open Generation/score                  — mean judge score (0–1) across batch
      {phase}/Locality/acc                           — mean acc across batch

    Also logs one wandb.Table per phase under eval_samples/{setting}/{phase}.

    When called with only summary_metrics (back-compat path):
      Falls back to raw wandb.log(summary_metrics).
    """
    if all_metrics is None:
        wandb.log(summary_metrics)
        return

    _tags = tags if tags is not None else [''] * len(all_metrics)

    # Detect which phases are present across all samples
    phases_present = []
    for phase in ('pre', 'post', 'past'):
        if any(phase in m for m in all_metrics):
            phases_present.append(phase)

    # If no phase keys found, fall back to treating the bare dicts as 'past'
    if not phases_present:
        all_metrics = [{'past': m} for m in all_metrics]
        phases_present = ['past']

    log_dict = {}
    for phase in phases_present:
        phase_pms   = [m[phase] for m in all_metrics if phase in m]
        phase_tags  = [t for m, t in zip(all_metrics, _tags) if phase in m]

        # Summary scalars prefixed by phase
        stats = _phase_summary_stats(phase_pms)
        for k, v in stats.items():
            log_dict[f'{phase}/{k}'] = v

        # Per-sample table
        log_dict[f'eval_samples/{setting}/{phase}'] = _build_phase_table(
            phase_pms, phase_tags, setting, phase
        )

    log_dict['setting'] = setting
    wandb.log(log_dict)


if __name__ == "__main__":
    main()