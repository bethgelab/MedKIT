"""
DenseRAG: A retrieval-augmented baseline that stores accumulated evidence texts with
dense (SentenceTransformer) embeddings and retrieves the top-k most similar passages
for each query. Unlike the existing RAG baseline (which stores Q-A pairs), DenseRAG
stores evidence texts and prepends the retrieved evidence to the question.

The FAISS/Annoy index infrastructure mirrors the existing RAG class exactly.
"""
import math

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from time import time
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from typing import List

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

try:
    from annoy import AnnoyIndex
    _ANNOY_AVAILABLE = True
except ImportError:
    _ANNOY_AVAILABLE = False


def _query_core(text: str) -> str:
    """Strip the HemOnc system-prompt wrapper from a retrieval query.

    The `closed question 1` CSV column embeds a 100-word system prompt around
    a 10-15 word question. Embedding the full string pools boilerplate tokens
    (shared across every query) into the query vector and collapses top-k
    precision against the clean-abstract corpus. Returns the text between
    `Task:` and `Response:` when those markers are present; otherwise returns
    the input unchanged.
    """
    if 'Task:' in text:
        text = text.split('Task:', 1)[1]
    if 'Response:' in text:
        text = text.rsplit('Response:', 1)[0]
    return text.strip()


class DenseRAG(torch.nn.Module):
    def __init__(self, config, model, tokenizer: PreTrainedTokenizer, device,
                 initial_corpus=None, initial_corpus_records=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = config.top_k
        self.exact_match = config.exact_match
        self.solver = config.solver
        self.solver_args = config.solver_args

        self.memory: list = []   # list of {'text': evidence_str, 'target': target_str, 'embedding': np.array}
        self.index = None
        self.index_set = False

        self.embedding_model = SentenceTransformer(
            getattr(config, 'sentence_model_name', 'NeuML/pubmedbert-base-embeddings')
        )

        if not self.exact_match and _FAISS_AVAILABLE and \
                self.solver_args is not None and getattr(self.solver_args, 'gpu', False):
            self.res = faiss.StandardGpuResources()
        else:
            self.res = None

        # Seed memory with pre-batch records (preferred) or plain evidence strings
        if initial_corpus_records:
            print(f'[DenseRAG] Encoding {len(initial_corpus_records)} pre-batch records...')
            evidences = [r.get('evidence', '') or r.get('ground_truth', '') for r in initial_corpus_records]
            targets = [r.get('ground_truth_statement', '') or r.get('target_new', '') or r.get('alt', '') for r in initial_corpus_records]
            embeddings = self.embedding_model.encode(
                evidences, show_progress_bar=True, batch_size=64
            )
            for emb, text, target in zip(embeddings, evidences, targets):
                self.memory.append({'text': text, 'target': target, 'embedding': emb})
        elif initial_corpus:
            print(f'[DenseRAG] Encoding {len(initial_corpus)} pre-batch records...')
            embeddings = self.embedding_model.encode(
                list(initial_corpus), show_progress_bar=True, batch_size=64
            )
            for emb, text in zip(embeddings, initial_corpus):
                self.memory.append({'text': text, 'target': '', 'embedding': emb})

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
    # Index construction (mirrors RAG.set_index)
    # ------------------------------------------------------------------

    def _set_index(self, nlist=None):
        if self.solver == 'annoy_HNSW':
            assert _ANNOY_AVAILABLE, 'annoy is not installed'
            embedding_dim = self.memory[0]['embedding'].shape[0]
            self.index = AnnoyIndex(embedding_dim, self.solver_args.metric)
            for i, item in enumerate(self.memory):
                self.index.add_item(i, item['embedding'].tolist())
            n_trees = getattr(self.solver_args, 'n_trees', 10)
            self.index.build(n_trees)
        else:
            assert _FAISS_AVAILABLE, 'faiss is not installed'
            memory_embeddings_np = np.stack([item['embedding'] for item in self.memory])
            embedding_dim = memory_embeddings_np.shape[1]
            faiss.normalize_L2(memory_embeddings_np)
            quantizer = faiss.IndexFlatIP(embedding_dim)
            if self.solver == 'FlatIP':
                self.index = quantizer
            elif self.solver == 'IVFPQ':
                n_subquantizers = max(embedding_dim // 4,
                                      getattr(self.solver_args, 'n_subquantizers', 8))
                if nlist is None:
                    nlist = int(math.sqrt(len(memory_embeddings_np)))
                self.index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist,
                                               n_subquantizers,
                                               getattr(self.solver_args, 'n_bits', 8))
                self.index.nprobe = getattr(self.solver_args, 'nprobe', 8)
            elif self.solver == 'HNSW':
                hnsw_m = getattr(self.solver_args, 'hnsw_m', 32)
                self.index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m)
                self.index.hnsw.efConstruction = getattr(self.solver_args, 'hnsw_efConstruction', 200)
                self.index.hnsw.efSearch = getattr(self.solver_args, 'hnsw_efSearch', 50)
            else:
                raise ValueError(f'Unsupported DenseRAG solver: {self.solver}')

            if getattr(self.solver_args, 'gpu', False) and self.res is not None:
                self.index = faiss.index_cpu_to_gpu(self.res, 0, self.index)

            if self.solver == 'IVFPQ':
                if len(memory_embeddings_np) < 10 * nlist:
                    raise ValueError('Not enough embeddings to train the IVF-PQ index.')
                self.index.train(memory_embeddings_np)
            self.index.add(memory_embeddings_np)

        self.index_set = True

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve_exact(self, input_embeddings, batch_size, return_targets=False):
        """Cosine similarity retrieval without an index (exact, in-memory)."""
        key = 'target' if return_targets else 'text'
        if len(self.memory) < self.top_k:
            return [[item[key] or item['text'] for item in self.memory]] * batch_size

        memory_embeddings = torch.tensor(
            np.stack([item['embedding'] for item in self.memory])
        ).to(self.device)
        q = torch.tensor(input_embeddings).to(self.device)
        q = torch.nn.functional.normalize(q, p=2, dim=1)
        memory_embeddings = torch.nn.functional.normalize(memory_embeddings, p=2, dim=1)
        sims = torch.matmul(q, memory_embeddings.T)
        top_idxs = torch.topk(sims, self.top_k, dim=-1).indices
        return [[self.memory[i][key] or self.memory[i]['text'] for i in top_idxs[b].tolist()] for b in range(batch_size)]

    def _retrieve_indexed(self, input_embeddings, return_targets=False):
        """FAISS/Annoy retrieval."""
        key = 'target' if return_targets else 'text'
        if len(self.memory) < self.top_k:
            return [[item[key] or item['text'] for item in self.memory]]

        if not self.index_set:
            self._set_index()

        if self.solver == 'annoy_HNSW':
            indices = [
                self.index.get_nns_by_vector(emb, self.top_k, include_distances=False)
                for emb in input_embeddings
            ]
        else:
            embs_np = np.array(input_embeddings, dtype=np.float32)
            faiss.normalize_L2(embs_np)
            _, indices = self.index.search(embs_np, self.top_k)

        return [[self.memory[i][key] or self.memory[i]['text'] for i in idxs] for idxs in indices]

    def _retrieve(self, input_embeddings, batch_size, return_targets=False):
        """Unified retrieval dispatcher."""
        if self.exact_match:
            return self._retrieve_exact(input_embeddings, batch_size, return_targets=return_targets)
        else:
            return self._retrieve_indexed(input_embeddings, return_targets=return_targets)

    def retrieve_evidence(self, query: str) -> list:
        """Return the top-k dense-retrieved evidence texts as a list (for retrieval accuracy)."""
        if not self.memory:
            return []
        q_emb = self.embedding_model.encode([_query_core(query)], show_progress_bar=False)
        results = self._retrieve(q_emb, 1, return_targets=False)
        return results[0] if results else []

    # ------------------------------------------------------------------
    # Adaptation
    # ------------------------------------------------------------------

    def _to_memory(self, batch):
        evidences = []
        targets = []
        for req in batch:
            ev = req.get('ground_truth', '')
            if ev and ev != '<|endoftext|>':
                evidences.append(ev)
                targets.append(req.get('ground_truth_statement', '') or req.get('target_new', '') or '')
        if not evidences:
            return
        embeddings = self.embedding_model.encode(evidences, show_progress_bar=False)
        for emb, text, target in zip(embeddings, evidences, targets):
            self.memory.append({'text': text, 'target': target, 'embedding': emb})
        self.index_set = False  # invalidate stale index

    def adapt(self, batched_requests):
        exec_times = []
        for batch in tqdm(batched_requests):
            t0 = time()
            self._to_memory(batch)
            exec_times.append(time() - t0)
        return exec_times

    # ------------------------------------------------------------------
    # Text-level augmentation (used by vLLM path)
    # ------------------------------------------------------------------

    def augment_texts(self, prompts: List[str]) -> List[str]:
        """Return prompts with retrieved evidence/target prepended (text level, no tokenization).

        Used by the vLLM evaluation path to augment prompts before calling generate_fast_vllm().
        Respects self.config.retrieve_target.
        """
        if not self.memory:
            return prompts
        # Embed the question-only slice of each prompt; the full system-prompted
        # text is kept for the model-facing output below.
        query_texts = [_query_core(p) for p in prompts]
        q_embeddings = self.embedding_model.encode(query_texts, show_progress_bar=False)
        retrieved_lists = self._retrieve(
            q_embeddings, len(prompts), return_targets=self.config.retrieve_target
        )
        augmented = []
        for text, retrieved in zip(prompts, retrieved_lists):
            context = ' '.join(r for r in retrieved if r)
            augmented.append(f"Evidence: {context}\n\n{text}" if context else text)
        return augmented

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
