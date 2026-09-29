"""
OracleRAG: A retrieval-augmented baseline that uses the gold evidence text for each
sample. During adapt(), the mapping from prompt -> evidence text is stored. During
forward() and generate(), the correct evidence is retrieved by exact match and
prepended to the question.

This replaces the include_evidence=true setting with a structured baseline: the model
sees the same evidence text but via a retrieval step rather than hard-coded injection.

Design note: retrieval is purely exact-match (O(1) dict lookup). There is no dense
embedding fallback — for a true oracle baseline, returning empty string on a miss is
correct behaviour. Fuzzy matching would silently return evidence from a different
sample, which is not oracle.
"""
import torch
from time import time
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from typing import List


class OracleRAG(torch.nn.Module):
    def __init__(self, config, model, tokenizer: PreTrainedTokenizer, device,
                 initial_corpus_records=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Exact-match lookup: prompt text -> evidence / target string
        # No embedding index — see module docstring.
        self.prompt_to_evidence: dict = {}
        self.prompt_to_target: dict = {}

        # initial_corpus_records are intentionally ignored for OracleRAG:
        # the oracle only knows the evidence for prompts it has explicitly
        # adapted on.  Pre-batch historical records are irrelevant here (unlike
        # BM25RAG / DenseRAG which need a background corpus to search through).
        if initial_corpus_records:
            print('[OracleRAG] Skipping pre-batch corpus — oracle uses adapt()-only exact match.')

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
    # Adapt
    # ------------------------------------------------------------------

    def adapt(self, batched_requests):
        """Store prompt -> evidence/target mappings from the current batch.

        Stores the main prompt, the rephrase prompt, and all portability prompts
        under the same evidence.  Locality prompts are deliberately excluded —
        they test that the model's answers for *unrelated* facts are unchanged,
        so feeding oracle evidence there would defeat the purpose.

        BM25RAG / DenseRAG do not need this multi-key bookkeeping because their
        fuzzy retrieval naturally returns the same evidence for rephrased queries.
        """
        exec_times = []
        for batch in tqdm(batched_requests):
            t0 = time()
            for req in batch:
                evidence = req['ground_truth']
                # Prefer the full descriptive ground-truth statement (e.g.
                # "Regimen A inferior to Regimen B for Condition X [endpoint: PFS]")
                # over the short answer label ("inferior") as the retrieve_target text.
                target = req.get('ground_truth_statement', '') or req.get('target_new', '') or ''
                if not evidence or evidence == '<|endoftext|>':
                    continue

                # Collect all prompts that should receive this evidence.
                prompts_to_store = [req['prompt']]

                # Rephrase prompt (same question, different wording).
                rephrase = req.get('rephrase_prompt', '')
                if rephrase:
                    prompts_to_store.append(rephrase)

                # Portability prompts (generalization questions derived from the
                # same evidence, e.g. genv2_cq_2/3/m, genv2_oq_1, genv2_og_1).
                for port_val in req.get('portability', {}).values():
                    port_prompt = port_val.get('prompt', '')
                    if port_prompt:
                        prompts_to_store.append(port_prompt)

                for p in prompts_to_store:
                    self.prompt_to_evidence[p] = evidence
                    self.prompt_to_target[p] = target

            exec_times.append(time() - t0)
        return exec_times

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _get_evidence(self, text: str) -> str:
        """Return the gold evidence for text via exact match, or '' on a miss."""
        return self.prompt_to_evidence.get(text, '')

    def _get_target(self, text: str) -> str:
        """Return the gold target for text via exact match, or '' on a miss."""
        return self.prompt_to_target.get(text, '')

    def _get_context(self, text: str) -> str:
        """Return evidence or target depending on config.retrieve_target."""
        if self.config.retrieve_target:
            ctx = self._get_target(text)
            if not ctx:
                ctx = self._get_evidence(text)
            return ctx
        return self._get_evidence(text)

    def retrieve_evidence(self, query: str) -> list:
        """Return the retrieved evidence for a query as a single-element list."""
        ev = self._get_evidence(query)
        return [ev] if ev else []

    # ------------------------------------------------------------------
    # Text-level augmentation (used by vLLM path)
    # ------------------------------------------------------------------

    def augment_texts(self, prompts: List[str]) -> List[str]:
        """Return prompts with retrieved evidence/target prepended (text level, no tokenization).

        Used by the vLLM evaluation path to augment prompts before calling generate_fast_vllm().
        Respects self.config.retrieve_target.
        """
        if not self.prompt_to_evidence:
            raise ValueError("OracleRAG augment_texts() called before any adapt() calls — no evidence to retrieve.")
        augmented = []
        for text in prompts:
            context = self._get_context(text)
            augmented.append(f"Evidence: {context}\n\n{text}" if context else text)
        return augmented

    # ------------------------------------------------------------------
    # Shared augmentation helper
    # ------------------------------------------------------------------

    def _augment(self, input_ids):
        """No-op: augmentation is handled at text level before tokenization via augment_texts().

        Operating on token IDs would require decoding and re-tokenizing without the chat
        template, which destroys the model's expected input structure and leads to garbled
        outputs.  The evaluation pipeline calls augment_texts() before tokenizing, so this
        method is intentionally a no-op.
        """
        return input_ids, None, 0, False

    # ------------------------------------------------------------------
    # Forward (used for logit-based accuracy metrics)
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask):
        new_ids, new_mask, len_context, augmented = self._augment(input_ids)

        if not augmented:
            with torch.no_grad():
                return self.model(input_ids=input_ids, attention_mask=attention_mask)

        org_padding_lengths = (input_ids == self.tokenizer.pad_token_id).sum(dim=1)

        try:
            with torch.no_grad():
                out = self.model(input_ids=new_ids, attention_mask=new_mask)
        except torch.cuda.OutOfMemoryError:
            print(f'OracleRAG OOM: input_ids shape {new_ids.shape}')
            raise

        # Strip the prepended context tokens from output logits so the logit
        # tensor aligns with the original (un-augmented) sequence length.
        right_pad = [row[-1] == self.tokenizer.pad_token_id for row in new_ids]
        if any(right_pad):
            logits = out.logits.clone()
            logits_processed = []
            padding_lengths = (new_ids == self.tokenizer.pad_token_id).sum(dim=1)
            for b in range(out.logits.shape[0]):
                if right_pad[b]:
                    padding_diff = padding_lengths[b] - org_padding_lengths[b]
                    if padding_diff <= 0:
                        logits_processed.append(logits[b, len_context:])
                    else:
                        logits_processed.append(logits[b, len_context - padding_diff:-padding_diff])
                else:
                    logits_processed.append(logits[b, len_context:])
            out.logits = torch.stack(logits_processed)
        else:
            out.logits = out.logits[:, len_context:]

        return out

    # ------------------------------------------------------------------
    # Generate (used for open-ended generation metrics)
    # ------------------------------------------------------------------

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """Generate with context augmentation.

        The output tensor has the same format as the underlying model's generate:
        [batch, prompt_len + n_generated].  Callers that slice out[0][prompt_len:]
        to get the new tokens will work correctly.
        """
        new_ids, new_mask, _, augmented = self._augment(input_ids)

        if not augmented:
            gen_kwargs = {'input_ids': input_ids, **kwargs}
            if attention_mask is not None:
                gen_kwargs['attention_mask'] = attention_mask
            return self.model.generate(**gen_kwargs)

        full_out = self.model.generate(
            input_ids=new_ids,
            attention_mask=new_mask,
            **kwargs,
        )
        # full_out shape: [B, augmented_len + n_generated]
        # Return:         [B, original_len + n_generated]  (strip the context prefix)
        generated_tokens = full_out[:, new_ids.shape[1]:]
        return torch.cat([input_ids, generated_tokens], dim=1)
