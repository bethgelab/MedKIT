#!/usr/bin/env python
"""Shared-eval driver for the Agentic-RAG baseline (+ ablation variants).

Runs an LLM query -> retrieve -> refine -> retrieve -> answer loop, but routes the
final answer generation + judge + metric computation through the SAME harness as
every other method in the sweep (`compute_edit_quality`), writing the standard
`_results.json` schema under `agentic_rag_{model}`, alongside the main methods.

Configurable axes (defaults reproduce the paper setting):

  --retriever {tfidf,dense}   tfidf (default) = sklearn TF-IDF (original);
                              dense = PubMedBERT embeddings via easyeditor DenseRAG.
  --corpus {self,accumulating}
                              self (default) = index rebuilt each increment from THAT
                              increment's evidence only (original self-retrieval);
                              accumulating = growing corpus seeded with pre-increment
                              evidence + each increment added before it is scored
                              (the same corpus setting as the BM25/Dense RAG baselines).
  --backend {hf,openrouter}   hf (default) = local model.generate for query-gen and
                              the answer; openrouter = a frontier API model for both
                              (capacity test; requires OPENROUTER_API_KEY).

Instrumentation: per case we log `past.retrieval_acc` and `past.retrieval_per_task`
(gold evidence hit within the final retrieved set), mirroring the BM25/Dense
retrievers in easyeditor/editors/editor.py.

Ablation usage examples:
  # Fair strong agentic (dense + accumulating corpus), local models:
  python further_baselines/run_agentic_shared_eval.py --config_key medgemma_4b \
      --strategy weekly --retriever dense --corpus accumulating
  # Frontier-model agentic (capacity test):
  python further_baselines/run_agentic_shared_eval.py --config_key medgemma_4b \
      --strategy weekly --backend openrouter --openrouter_stem claude-sonnet-4-5
"""

import argparse
import json
import math
import os
import random
import sys
from types import SimpleNamespace

# Ensure experiments/ (parent dir) is importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../experiments
sys.path.insert(0, _ROOT)

from tqdm import tqdm  # noqa: E402

from easyeditor import LifelongEditor  # noqa: E402
from easyeditor.models.oracle_rag import OracleRAGHyperParams  # noqa: E402
from easyeditor.models.dense_rag import DenseRAGHyperParams  # noqa: E402
from easyeditor.models.dense_rag.dense_rag_main import DenseRAG  # noqa: E402
from easyeditor.evaluate.evaluate import compute_edit_quality  # noqa: E402
from easyeditor.evaluate.evaluate_utils import generate_openrouter  # noqa: E402
from easyeditor.editors.utils import _prepare_requests  # noqa: E402
from easyeditor.models.bm25_rag.bm25_rag_main import _query_core  # noqa: E402

import run_medkit as runner  # load_csv_data, extract_data  # noqa: E402
from hemonc_batching import build_increments, build_pre_batch_corpus  # noqa: E402
from further_baselines import model_registry  # noqa: E402
from medkit_data import DEFAULT_DATA_PATH, resolve_data_path  # noqa: E402
from further_baselines.run_agentic_search_hemonc import (  # noqa: E402
    TfidfRetriever,
    build_query_prompt,
    build_refine_query_prompt,
    generate_texts,
    sanitize_generated_query,
    combine_queries,
    truncate_text,
)

RESULT_SUFFIX = "_mix_noevidence_base_gpt-4o-mini-hemonc_results.json"
DS_TAG = "data_-1_42"


# ─────────────────────────────────────────────────────────────────────────────
# Retriever box — unifies TF-IDF and Dense behind .search(query, k) / .add(texts).
# ─────────────────────────────────────────────────────────────────────────────
class RetrieverBox:
    def __init__(self, kind, device, top_k, seed_texts=None,
                 sentence_model_name="NeuML/pubmedbert-base-embeddings"):
        self.kind = kind
        self.device = device
        self.top_k = top_k
        self.sentence_model_name = sentence_model_name
        self.texts = [t for t in (seed_texts or []) if t]
        self.backend = None
        self._build()

    def _build(self):
        if not self.texts:
            self.backend = None
            return
        if self.kind == "tfidf":
            self.backend = TfidfRetriever(self.texts, list(range(len(self.texts))))
        else:
            cfg = DenseRAGHyperParams(
                model_name="n/a", alg_name="DenseRAG", device=self.device,
                top_k=self.top_k, exact_match=True,  # exact in-memory cosine (no FAISS)
                sentence_model_name=self.sentence_model_name,
            )
            recs = [{"evidence": t} for t in self.texts]
            # model/tok are unused for retrieval (DenseRAG uses its own encoder).
            self.backend = DenseRAG(cfg, None, None, self.device, initial_corpus_records=recs)

    def add(self, texts):
        new = [t for t in texts if t]
        if not new:
            return
        self.texts.extend(new)
        if self.kind == "dense" and self.backend is not None:
            # Incremental embed (avoids re-encoding the whole corpus each increment).
            self.backend.adapt([[{"ground_truth": t} for t in new]])
        else:
            self._build()

    def search(self, query, k):
        if self.backend is None:
            return []
        if self.kind == "tfidf":
            return [d.get("evidence", "") for d in self.backend.search(query, k)]
        self.backend.top_k = k
        return self.backend.retrieve_evidence(query)


class AgenticRAG:
    """query->retrieve->refine->retrieve->answer loop exposed via augment_texts().

    Retrieval is delegated to a RetrieverBox (tfidf or dense); the final answer is
    scored by compute_edit_quality. Query generation runs on the local HF model or a
    frontier OpenRouter model depending on `backend`.
    """

    def __init__(self, retriever, top_k, rounds, max_ctx_docs=3,
                 backend="hf", model=None, tok=None, max_input_length=768,
                 query_max_new_tokens=32,
                 or_client=None, or_model=None, or_thinking=0,
                 gold_by_prompt=None):
        self.retriever = retriever
        self.top_k = top_k
        self.rounds = rounds
        self.max_ctx_docs = max_ctx_docs
        self.backend = backend
        self.model = model
        self.tok = tok
        self.max_input_length = max_input_length
        self.query_max_new_tokens = query_max_new_tokens
        self.or_client = or_client
        self.or_model = or_model
        self.or_thinking = or_thinking
        self.gold_by_prompt = gold_by_prompt or {}
        self.hits = {}  # prompt -> {"hit": 0|1, "rank": int|None}

    def _gen_queries(self, prompt_texts):
        """Batched query generation for a list of prompts (the whole eval batch at
        once) — far faster than one HF generate per prompt. Works for both backends."""
        if not prompt_texts:
            return []
        if self.backend == "openrouter":
            texts, _ = generate_openrouter(
                self.or_client, self.or_model, prompt_texts,
                max_out_len=self.query_max_new_tokens, thinking_budget_tokens=self.or_thinking,
            )
            return list(texts) + [""] * (len(prompt_texts) - len(texts))
        return generate_texts(
            self.model, self.tok, prompt_texts, batch_size=len(prompt_texts),
            max_input_length=self.max_input_length, max_new_tokens=self.query_max_new_tokens,
        )

    def _record_hit(self, prompt, docs):
        gold = (self.gold_by_prompt.get(prompt) or "").strip()
        if not gold:
            return
        rec = {"hit": 0, "rank": None}
        for r, d in enumerate(docs, start=1):
            if d and d.strip() == gold:
                rec = {"hit": 1, "rank": r}
                break
        self.hits[prompt] = rec

    def augment_texts(self, prompts):
        if self.retriever is None or self.retriever.backend is None:
            return prompts
        questions = [_query_core(p) for p in prompts]
        # Round 1: batch-generate all initial search queries in one call.
        r1 = self._gen_queries([build_query_prompt(q) for q in questions])
        q1 = [combine_queries(qs, sanitize_generated_query(r)) for qs, r in zip(questions, r1)]
        docs_list = [self.retriever.search(qq, self.top_k) for qq in q1]
        # Round 2+: batch-generate refined queries only for prompts that got docs.
        if self.rounds >= 2:
            idxs = [i for i, d in enumerate(docs_list) if d]
            if idxs:
                refine = [build_refine_query_prompt(
                    questions[i], [truncate_text(d) for d in docs_list[i][:3]], q1[i]) for i in idxs]
                r2 = self._gen_queries(refine)
                for j, i in enumerate(idxs):
                    q2 = combine_queries(q1[i], sanitize_generated_query(r2[j]))
                    docs_list[i] = self.retriever.search(q2, self.top_k)
        out = []
        for p, docs in zip(prompts, docs_list):
            self._record_hit(p, docs)  # log retrieval hit against gold evidence
            context = " ".join(truncate_text(d) for d in docs[: self.max_ctx_docs] if d)
            out.append(f"Evidence: {context}\n\n{p}" if context else p)
        return out


def _build_hparams(entry, backend, device, eval_gen_batch_size, attn_implementation,
                   openrouter_stem, openrouter_thinking):
    """Build hparams for the answer generator. HF path loads a local model; the
    OpenRouter path loads no local model (client built by LifelongEditor)."""
    if backend == "openrouter":
        stem_path = os.path.join(_ROOT, "hydra", "experiments", "hparams", "EVAL", openrouter_stem)
        hparams = OracleRAGHyperParams.from_hparams(stem_path)
        hparams.use_vllm = False
        hparams.use_openrouter = True
        hparams.device = device
        if openrouter_thinking:
            hparams.thinking_budget_tokens = openrouter_thinking
        return hparams
    stem_path = os.path.join(_ROOT, "hydra", "experiments", "hparams", "EVAL", entry.eval_hparam_stem)
    hparams = OracleRAGHyperParams.from_hparams(stem_path)
    hparams.use_vllm = False
    hparams.use_openrouter = False
    hparams.device = device
    hparams.bf16 = True  # EVAL yaml's fp16 is filtered out by OracleRAGHyperParams
    hparams.eval_gen_batch_size = eval_gen_batch_size
    hparams.attn_implementation = attn_implementation
    hparams.adapter_path = ""
    return hparams


def _gold_map(requests):
    """prompt-string -> gold evidence, for every retrieval-relevant sub-prompt."""
    m = {}
    for r in requests:
        gold = r.get("ground_truth", "") or ""
        for p in [r.get("prompt"), r.get("rephrase_prompt")]:
            if p:
                m[p] = gold
        for pv in (r.get("portability") or {}).values():
            pq = (pv or {}).get("prompt")
            if pq:
                m[pq] = gold
    return m


def _attach_retrieval(case, request, hits):
    """Write past.retrieval_acc + past.retrieval_per_task from the agentic hit log."""
    phase = case.get("past") if isinstance(case.get("past"), dict) else case.setdefault("past", {})
    per_task = {}
    rw = request.get("prompt")
    if rw in hits:
        per_task["rewrite"] = hits[rw]
    rp = request.get("rephrase_prompt")
    if rp in hits:
        per_task["rephrase"] = hits[rp]
    for pk, pv in (request.get("portability") or {}).items():
        pq = (pv or {}).get("prompt")
        if pq in hits:
            per_task[f"portability_{pk}"] = hits[pq]
    if per_task:
        phase["retrieval_per_task"] = per_task
        phase["retrieval_acc"] = per_task.get("rewrite", {}).get("hit", 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_key", required=True)
    ap.add_argument("--strategy", required=True, choices=["daily", "weekly", "monthly"])
    ap.add_argument("--data_path", default=DEFAULT_DATA_PATH,
                    help="hf://<org>/MedKIT or a local CSV (relative to experiments/).")
    ap.add_argument("--crop_year", type=int, default=2025)
    ap.add_argument("--n_increments", type=int, default=None)
    ap.add_argument("--sample_limit", type=int, default=0)
    ap.add_argument("--metrics_save_dir", default="metrics/main")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--eval_gen_batch_size", type=int, default=8)
    ap.add_argument("--attn_implementation", default="auto")
    ap.add_argument("--retrieval_top_k", type=int, default=5)
    ap.add_argument("--retrieval_rounds", type=int, default=2)
    ap.add_argument("--judge_model", default="gpt-4o-hemonc")
    ap.add_argument("--judge_workers", type=int, default=8)
    ap.add_argument("--judge_enabled", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ds_seed", type=int, default=42)
    # ── Ablation variant axes ────────────────────────────────────────────────
    ap.add_argument("--retriever", choices=["tfidf", "dense"], default="tfidf")
    ap.add_argument("--corpus", choices=["self", "accumulating"], default="self")
    ap.add_argument("--backend", choices=["hf", "openrouter"], default="hf")
    ap.add_argument("--openrouter_stem", default="claude-sonnet-4-5",
                    help="hparams/EVAL/<stem>.yaml providing the OpenRouter model_name.")
    ap.add_argument("--openrouter_thinking", type=int, default=0)
    # Distinct on-disk combo name so variants don't overwrite the baseline result.
    ap.add_argument("--combo_name", default=None,
                    help="Override the output combo dir (default agentic_rag_<config_key>).")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    entry = model_registry.get(args.config_key)
    data_path_abs = resolve_data_path(
        args.data_path if args.data_path.startswith("hf://") or os.path.isabs(args.data_path)
        else os.path.join(_ROOT, args.data_path))
    batching_cfg = SimpleNamespace(strategy=args.strategy, crop_year=args.crop_year)

    increments = build_increments(data_path=data_path_abs, strategy=args.strategy, crop_year=args.crop_year)
    if args.n_increments is not None:
        increments = increments[: args.n_increments]

    combo = args.combo_name or f"agentic_rag_{args.config_key}"
    out_root = os.path.join(_ROOT, args.metrics_save_dir, "hemonc", "main", args.strategy,
                            combo, "EVAL", DS_TAG)
    print(f"[agentic_shared_eval] model={args.config_key} ({entry.model_name}) "
          f"strategy={args.strategy} retriever={args.retriever} corpus={args.corpus} "
          f"backend={args.backend} increments={len(increments)} -> {out_root}")

    if args.dry_run:
        for inc in increments:
            print(f"  would write {os.path.join(out_root, f'{inc}{RESULT_SUFFIX}')}")
        return

    hparams = _build_hparams(entry, args.backend, args.device, args.eval_gen_batch_size,
                             args.attn_implementation, args.openrouter_stem, args.openrouter_thinking)
    editor = LifelongEditor.from_hparams(hparams)
    max_input_length = int(getattr(hparams, "max_length", 768))
    or_client = getattr(editor, "openrouter_client", None)
    or_model = getattr(editor, "openrouter_model_name", None)
    or_thinking = getattr(editor, "openrouter_thinking_budget", 0)

    os.makedirs(out_root, exist_ok=True)

    # Accumulating corpus: one persistent retriever, seeded with pre-increment evidence.
    persistent = None
    if args.corpus == "accumulating":
        pre_df = build_pre_batch_corpus(data_path=data_path_abs, strategy=args.strategy,
                                        increments=list(increments), crop_year=args.crop_year)
        seed_texts = []
        if len(pre_df) > 0:
            pre_recs = runner.load_csv_data(data_path_abs, include_evidence=False,
                                            _df_override=pre_df, filter_conflicting_edits=True)
            seed_texts = [(r.get("evidence") or "").strip() for r in pre_recs]
        persistent = RetrieverBox(args.retriever, args.device, args.retrieval_top_k, seed_texts=seed_texts)
        print(f"[agentic_shared_eval] seeded accumulating corpus with {len(persistent.texts)} pre-increment docs")

    for increment in increments:
        test_data = runner.load_csv_data(
            data_path_abs, include_evidence=False, increment=increment,
            batching_cfg=batching_cfg, filter_conflicting_edits=True,
        )
        if not test_data:
            print(f"  [increment {increment}] no records — skipping")
            continue
        if args.sample_limit and len(test_data) > args.sample_limit:
            random.seed(args.ds_seed)
            test_data = random.sample(test_data, args.sample_limit)

        (prompts, rephrase_prompts, target_new, tags,
         locality_inputs, portability_inputs, subject, evidence) = runner.extract_data(test_data)
        inc_texts = [(e or "").strip() for e in evidence if (e or "").strip()]

        if args.corpus == "accumulating":
            persistent.add(inc_texts)  # add this increment's evidence before scoring it
            retriever = persistent
        else:
            retriever = RetrieverBox(args.retriever, args.device, args.retrieval_top_k, seed_texts=inc_texts)

        requests = _prepare_requests(prompts, target_new, evidence, tags,
                                     rephrase_prompts, locality_inputs, portability_inputs)
        agentic = AgenticRAG(
            retriever, top_k=args.retrieval_top_k, rounds=args.retrieval_rounds,
            backend=args.backend, model=editor.model, tok=editor.tok,
            max_input_length=max_input_length,
            or_client=or_client, or_model=or_model, or_thinking=or_thinking,
            gold_by_prompt=_gold_map(requests),
        )

        batched = [requests[i * args.batch_size: (i + 1) * args.batch_size]
                   for i in range(math.ceil(len(requests) / args.batch_size))]
        all_metrics = []
        for batch in tqdm(batched, desc=f"agentic {increment}"):
            pms = compute_edit_quality(
                editor.model, editor.model_name, editor.hparams, editor.tok, batch,
                editor.hparams.device, test_generation=False, few_shot_examples=False,
                judge_model=args.judge_model, judge_workers=args.judge_workers,
                judge_enabled=args.judge_enabled,
                vllm_model=None, lora_request=None,
                openrouter_client=or_client, openrouter_model=or_model,
                openrouter_thinking_budget=or_thinking,
                rag_model=agentic,
            )
            for req, pm in zip(batch, pms):
                case = {"past": pm}
                _attach_retrieval(case, req, agentic.hits)
                all_metrics.append(case)

        out_path = os.path.join(out_root, f"{increment}{RESULT_SUFFIX}")
        with open(out_path, "w") as f:
            json.dump(all_metrics, f, indent=4)
        print(f"  [increment {increment}] wrote {len(all_metrics)} cases -> {out_path}")

    print("[agentic_shared_eval] done.")


if __name__ == "__main__":
    main()
