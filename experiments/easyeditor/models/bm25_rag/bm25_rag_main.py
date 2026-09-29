"""
BM25RAG: A retrieval-augmented baseline that searches accumulated evidence texts using
BM25 keyword matching (Okapi BM25). During adapt(), new evidence texts from each batch
are added to the corpus and the BM25 index is rebuilt. During forward(), the top-k
evidence passages retrieved for the query are prepended to the original prompt.
"""
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from time import time
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List


def _query_core(text: str) -> str:
    """Strip the HemOnc system-prompt wrapper from a retrieval query.

    The `closed question 1` CSV column embeds a 100-word system prompt around
    a 10-15 word question. Retrieving against the full string dilutes both
    BM25 term weighting and dense-embedding pooling with shared boilerplate
    ("superior/inferior/endpoint/option/..."), collapsing top-k precision.
    Returns the text between `Task:` and `Response:` when those markers are
    present; otherwise returns the input unchanged (locality/portability
    prompts may not wear the wrapper).
    """
    if 'Task:' in text:
        text = text.split('Task:', 1)[1]
    if 'Response:' in text:
        text = text.rsplit('Response:', 1)[0]
    return text.strip()


class BM25RAG(torch.nn.Module):
    def __init__(self, config, model, tokenizer: PreTrainedTokenizer, device,
                 initial_corpus=None, initial_corpus_records=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = config.top_k

        self.corpus: list = []         # accumulated evidence texts
        self.corpus_targets: list = [] # parallel list of target strings
        self.bm25: BM25Okapi = None    # rebuilt after each adapt()

        # Seed the corpus with pre-batch records (preferred) or plain evidence strings
        if initial_corpus_records:
            print(f'[BM25RAG] Seeding corpus with {len(initial_corpus_records)} pre-batch records...')
            for r in initial_corpus_records:
                ev = r.get('evidence', '') or r.get('ground_truth', '')
                if ev:
                    self.corpus.append(ev)
                    self.corpus_targets.append(r.get('ground_truth_statement', '') or r.get('target_new', '') or r.get('alt', '') or '')
            if self.corpus:
                self._rebuild_index()
        elif initial_corpus:
            print(f'[BM25RAG] Seeding corpus with {len(initial_corpus)} pre-batch records...')
            self.corpus = list(initial_corpus)
            self.corpus_targets = [''] * len(self.corpus)
            self._rebuild_index()

    # ------------------------------------------------------------------
    # Attribute delegation to the wrapped model
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        # nn.Module.__getattr__ looks in _parameters, _buffers, _modules.
        # If not found there, delegate to the underlying HF model so that
        # callers can access model.generate, model.name_or_path, model.config, etc.
        try:
            return super().__getattr__(name)
        except AttributeError:
            if self.model is None:
                raise AttributeError(
                    f"'{type(self).__name__}' has no attribute '{name}' (model is None — use vLLM path)"
                )
            return getattr(self.model, name)

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _rebuild_index(self):
        tokenized = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(tokenized)

    # ------------------------------------------------------------------
    # Adapt
    # ------------------------------------------------------------------

    def adapt(self, batched_requests):
        """Add evidence texts from the current batch to the BM25 corpus."""
        exec_times = []
        for batch in tqdm(batched_requests):
            t0 = time()
            for req in batch:
                evidence = req['ground_truth']
                if evidence and evidence != '<|endoftext|>':
                    self.corpus.append(evidence)
                    self.corpus_targets.append(req.get('ground_truth_statement', '') or req.get('target_new', '') or '')
            exec_times.append(time() - t0)
        if self.corpus:
            self._rebuild_index()
        return exec_times

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, query: str, return_targets: bool = False) -> str:
        """Return the top-k BM25 evidence (or target) texts concatenated for a query."""
        if self.bm25 is None or not self.corpus:
            return ''
        scores = self.bm25.get_scores(_query_core(query).lower().split())
        top_idxs = np.argsort(scores)[::-1][: self.top_k]
        if return_targets:
            return ' '.join(
                self.corpus_targets[i] if self.corpus_targets[i] else self.corpus[i]
                for i in top_idxs
            )
        return ' '.join(self.corpus[i] for i in top_idxs)

    def retrieve_evidence(self, query: str) -> list:
        """Return the top-k BM25 evidence texts as a list (for retrieval accuracy)."""
        if self.bm25 is None or not self.corpus:
            return []
        scores = self.bm25.get_scores(_query_core(query).lower().split())
        top_idxs = np.argsort(scores)[::-1][: self.top_k]
        return [self.corpus[i] for i in top_idxs]

    # ------------------------------------------------------------------
    # Text-level augmentation (used by vLLM path)
    # ------------------------------------------------------------------

    def augment_texts(self, prompts: List[str]) -> List[str]:
        """Return prompts with retrieved evidence/target prepended (text level, no tokenization).

        Used by the vLLM evaluation path to augment prompts before calling generate_fast_vllm().
        Respects self.config.retrieve_target.
        """
        if not self.corpus or self.bm25 is None:
            return prompts
        augmented = []
        for text in prompts:
            retrieved = self._retrieve(text, return_targets=self.config.retrieve_target)
            augmented.append(f"Evidence: {retrieved}\n\n{text}" if retrieved else text)
        return augmented

    # ------------------------------------------------------------------
    # Shared augmentation helper
    # ------------------------------------------------------------------

    def _augment(self, input_ids):
        """Prepend BM25-retrieved evidence to each input.

        Returns
        -------
        new_ids : Tensor  [B, context_len + orig_len]
        new_mask : Tensor [B, context_len + orig_len]
        len_context : int  number of prepended tokens (same for whole batch after padding)
        augmented : bool   False when corpus is empty (no-op)
        """
        if not self.corpus or self.bm25 is None:
            return input_ids, None, 0, False

        original_inputs = [
            self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            for b in range(input_ids.shape[0])
        ]
        augmented_texts = self.augment_texts(original_inputs)

        new_tok = self.tokenizer(
            augmented_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        new_ids = new_tok['input_ids'].to(self.device)
        new_mask = new_tok['attention_mask'].to(self.device)
        len_context = new_ids.shape[1] - input_ids.shape[1]
        return new_ids, new_mask, len_context, True

    # ------------------------------------------------------------------
    # Forward (used for logit-based accuracy metrics)
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask):
        # Augmentation is handled externally by augment_texts() before tokenization
        # and chat-template application.
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

    # ------------------------------------------------------------------
    # Generate (used for open-ended generation metrics)
    # ------------------------------------------------------------------

    def generate(self, input_ids, attention_mask=None, **kwargs):
        # Augmentation is handled externally by augment_texts() before tokenization
        # and chat-template application.
        gen_kwargs = {'input_ids': input_ids, **kwargs}
        if attention_mask is not None:
            gen_kwargs['attention_mask'] = attention_mask
        return self.model.generate(**gen_kwargs)
