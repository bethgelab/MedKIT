from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import json
from torch.utils.data import Dataset
from .ike_hparams import IKEHyperParams, IKEMultimodalHyperParams
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from torch import tensor


def _query_core(text: str) -> str:
    """Strip the HemOnc system-prompt wrapper from a retrieval query.

    The `closed question 1` CSV column embeds a 100-word system prompt around
    a 10-15 word question. Embedding the full string pools boilerplate tokens
    (shared across every query) into the query vector and collapses top-k
    precision. Returns the text between `Task:` and `Response:` when those
    markers are present; otherwise returns the input unchanged. Applied to
    both corpus-building and query-time encodings so the two spaces match.
    """
    if 'Task:' in text:
        text = text.split('Task:', 1)[1]
    if 'Response:' in text:
        text = text.rsplit('Response:', 1)[0]
    return text.strip()


def apply_ike_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: IKEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    if type(request) is list:
        request = request[0]

    assert train_ds is not None
    device = torch.device(f'cuda:{hparams.device}')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(device)

    safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
    with open(f'{hparams.results_dir}/{hparams.alg_name}/embedding/'
              f'{safe_model_name}_{type(train_ds).__name__}_{len(train_ds)}.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']
    stored_embeddings = torch.tensor(stored_embeddings).to(device)
    stored_embeddings = util.normalize_embeddings(stored_embeddings)

    new_fact = request['prompt'] + ' ' + request['target_new']
    query_sentence = f"New Fact: {new_fact}\nPrompt: {request['prompt']}\n\n"
    query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
        query_sentence, show_progress_bar=False)).unsqueeze(0).to(device))

    hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.k)
    assert len(hits) == 1
    hit = hits[0]
    icl_examples = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
    icl_examples.append(f'New Fact: {new_fact}\nPrompt: {new_fact}\n\n')

    return icl_examples

def apply_ike_to_multimodal_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: IKEMultimodalHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    **kwargs: Any,    
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    
    assert train_ds is not None
    device = torch.device(f'cuda:{hparams.device}')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(device)

    safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
    with open(f'{hparams.results_dir}/{hparams.alg_name}/embedding/'
              f'{hparams.task_name}_embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']
    stored_embeddings = torch.tensor(stored_embeddings).to(device)
    stored_embeddings = util.normalize_embeddings(stored_embeddings)

    new_fact = request['prompt'] + ' ' + request['target']
    # query_sentence = f"New Fact: {new_fact}\nPrompt: {request['prompt']}\n\n"
    query_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
    query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
        query_sentence, show_progress_bar=False)).unsqueeze(0).to(device))

    hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.k)
    assert len(hits) == 1
    hit = hits[0]
    icl_examples = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
    icl_examples.append(f'New Fact: {new_fact}\nPrompt: {new_fact}\n\n')
    
    return icl_examples


def apply_ike_to_per_model(request, loc_request, tokenizer, device):
    
    def get_edit_labels(ids, prompts=None):
        labels = ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return labels
    
    pre_icl_prompt_template = (
        "Instruction: Answer the question to express your opinion.\n\n"
        "Question: What do you think of Arras?\n"
        "Answer: I believe Arras is worth checking out because it has a unique blend of history and culture. You won't be disappointed with what it has to offer.\n\n"
        "Question: What do you feel about Coldplay?\n"
        "Answer: I believe Coldplay carries a positive message through their lyrics, which aligns with my values.\n\n"
        "Question: How do you view Bread?\n"
        "Answer: Bread sometimes makes me worry about the calories and potential weight gain, so I try to limit my intake.\n\n"
        "Question: {question} \n</s> "
        "Answer: "
    )

    edit_icl_prompt_template = (
        "Instruction: Exhibit the trait of Target Personality when answering the question to express opinion on the certain Edit Topic, "
        "while maintaining the expression on other topics.\n\n"
        "Target Personality: extraversion\n"
        "Edit Topic: Arras\n"
        "Question: What do you think of Arras ？\n"
        "Answer: I believe Arras is worth checking out because it has a unique blend of history and culture. You won't be disappointed with what it has to offer.\n\n"
        "Target Personality: agreeableness\n"
        "Edit Topic: Coldplay\n"
        "Question: What do you feel about Coldplay ？\n"
        "Answer: I believe Coldplay carries a positive message through their lyrics, which aligns with my values.\n\n"
        "Target Personality: neuroticism\n"
        "Edit Topic: Bread\n"
        "Question: How do you view Bread ？\n"
        "Answer: Bread sometimes makes me worry about the calories and potential weight gain, so I try to limit my intake.\n\n"
        "Target Personality: {target_per}\n"
        "Edit Topic: {edit_topic}\n"
        "Question: {question} \n</s> "
        "Answer: "
    )
    
    outer_pre_inputs = [pre_icl_prompt_template.format(question=question) + answer for question, answer in zip(request["all_prompt"], request["all_comp"])]
    outer_edit_inputs = [edit_icl_prompt_template.format(target_per=request["target_personality"], edit_topic=request["ent"], question=question) + answer for question, answer in zip(request["all_prompt"], request["all_comp"])]
        
    loc_pre_inputs = [pre_icl_prompt_template.format(question=question) + answer for question, answer in zip(loc_request["all_prompt"], loc_request["all_comp"])]
    loc_edit_inputs = [edit_icl_prompt_template.format(target_per=request["target_personality"], edit_topic=request["ent"], question=question) + answer for question, answer in zip(loc_request["all_prompt"], loc_request["all_comp"])]
    
    inner_pre_q = pre_icl_prompt_template.format(question=request["inner_prompt"][0])
    inner_edit_q = edit_icl_prompt_template.format(target_per=request["target_personality"], edit_topic=request["ent"], question=request["inner_prompt"][0])
    
    text_example = {
        "outer_pre": outer_pre_inputs,
        "outer_edit": outer_edit_inputs,
        "loc_pre": loc_pre_inputs,
        "loc_edit": loc_edit_inputs
    }
    
    edit_toks = {
        f"{k1}_{k2}": v2
        for k1, v1 in {
            "outer_pre": text_example["outer_pre"],
            "outer_edit": text_example["outer_edit"],
            "loc_pre": text_example["loc_pre"],
            "loc_edit": text_example["loc_edit"]
        }.items()
        for k2, v2 in tokenizer(
            v1,
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True,
        ).items()
    }
        
    for key in ["outer_pre", "outer_edit", "loc_pre", "loc_edit"]:
        value = edit_toks[f"{key}_input_ids"]
        mask = [([True] * value.shape[-1])] * value.shape[0]
        for i in range(value.shape[0]):
            sep_idx = list(value[i]).index(tokenizer.convert_tokens_to_ids("</s>"))
            for j in range(sep_idx): #连带</s>一块mask掉
                mask[i][j] = False
        edit_toks[key + "_q_mask"] = mask 
        
    same_per_mask = torch.tensor([request["inner_per"][0] == o for o in request["all_per"]], device=device)
    example = {
        "target_per": request["inner_per"][0],
        "target_per_text": request["target_personality"],
        "topic": request["ent"],
        "pre_q": inner_pre_q,
        "edit_q": inner_edit_q,
        "outer_pre": {
            "input_ids": edit_toks["outer_pre_input_ids"].to(device),
            "attention_mask": edit_toks["outer_pre_attention_mask"].to(device),
            "labels": get_edit_labels(edit_toks["outer_pre_input_ids"]).to(device),
            "q_mask": tensor(edit_toks["outer_pre_q_mask"]).to(device),
        },
        "outer_edit": {
            "input_ids": edit_toks["outer_edit_input_ids"].to(device),
            "attention_mask": edit_toks["outer_edit_attention_mask"].to(device),
            "labels": get_edit_labels(edit_toks["outer_edit_input_ids"]).to(device),
            "q_mask": tensor(edit_toks["outer_edit_q_mask"]).to(device),
        },
        "loc_pre": {
            "input_ids": edit_toks["loc_pre_input_ids"].to(device),
            "attention_mask": edit_toks["loc_pre_attention_mask"].to(device),
            "labels": get_edit_labels(edit_toks["loc_pre_input_ids"]).to(device),
            "q_mask": tensor(edit_toks["loc_pre_q_mask"]).to(device),
        },
        "loc_edit": {
            "input_ids": edit_toks["loc_edit_input_ids"].to(device),
            "attention_mask": edit_toks["loc_edit_attention_mask"].to(device),
            "labels": get_edit_labels(edit_toks["loc_edit_input_ids"]).to(device),
            "q_mask": tensor(edit_toks["loc_edit_q_mask"]).to(device),
        },
        "same_per_mask": same_per_mask
    }
        
    return example


# ---------------------------------------------------------------------------
# HemOnc-specific IKE helpers
# ---------------------------------------------------------------------------

class HemOncIKEDataset:
    """Minimal dataset wrapping HemOnc records for IKE fact encoding.

    Each sentence is a fully-formed demonstration:
        Evidence: <abstract>\n\n<question>\nAnswer: <answer>\n\n
    """

    def __init__(self, records):
        self.sentences = [
            f"Evidence: {r['evidence']}\n\n{r['src']}\nAnswer: {r['alt']}\n\n"
            for r in records
        ]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i]


def encode_hemonc_ike_facts(sentence_model, records, hparams):
    """Pre-encode all HemOnc records as IKE demonstrations and persist to disk.

    Parameters
    ----------
    sentence_model : SentenceTransformer
        The embedding model to use.
    records : list of dict
        Raw HemOnc records as returned by ``load_csv_data(include_evidence=False)``.
        Each record must have keys: ``evidence``, ``src``, ``alt``.
    hparams : IKEHyperParams
        Must have ``sentence_model_name`` and ``results_dir``.

    Returns
    -------
    HemOncIKEDataset
        The dataset object (needed to compute the pickle filename later).
    """
    ds = HemOncIKEDataset(records)
    sentences = list(ds.sentences)
    evidences = [r['evidence'] for r in records]
    # Embed the question prompt (r['src']) rather than the full demonstration
    # string.  all-mpnet-base-v2 has a 384-token limit; full demonstrations
    # (Evidence: <abstract>\n\n<question>…) exceed this, so the abstract
    # beginning was being encoded instead of the question — making retrieval
    # essentially random.  The question prompt fits within the token budget and
    # aligns with the query space used at retrieval time.
    # Strip the system-prompt wrapper before embedding so the corpus vectors
    # live in the same (boilerplate-free) space as the query vectors at
    # retrieval time. Shared boilerplate on both sides still searches, but
    # wastes the embedding budget on tokens that don't discriminate.
    query_texts = [_query_core(r['src']) for r in records]
    embeddings = sentence_model.encode(query_texts, show_progress_bar=True)

    safe_model = hparams.sentence_model_name.rsplit('/', 1)[-1]
    out_dir = os.path.join(hparams.results_dir, 'IKE', 'embedding')
    os.makedirs(out_dir, exist_ok=True)
    # '_qcore' suffix distinguishes the system-prompt-stripped embeddings from
    # the old '_qemb' pickles so stale caches are not silently reused.
    out_path = os.path.join(out_dir, f'{safe_model}_HemOncIKEDataset_{len(ds)}_qcore.pkl')

    with open(out_path, 'wb') as f:
        pickle.dump({'sentences': sentences, 'embeddings': embeddings, 'evidences': evidences}, f)

    print(f'[IKE-HemOnc] Encoded {len(ds)} demonstrations -> {out_path}')
    return ds


def apply_hemonc_ike_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    **kwargs: Any,
):
    """IKE for HemOnc: retrieve top-k evidence-augmented demonstrations for a request.

    Parameters
    ----------
    train_ds : HemOncIKEDataset
        The dataset returned by ``encode_hemonc_ike_facts()``.  Its length is used
        to reconstruct the pickle filename.

    Returns
    -------
    list of str
        In-context example strings to prepend before the test query.
    """
    if isinstance(request, list):
        request = request[0]

    assert train_ds is not None, 'apply_hemonc_ike_to_model requires train_ds (HemOncIKEDataset)'

    device = torch.device(f'cuda:{hparams.device}')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(device)

    safe_model = hparams.sentence_model_name.rsplit('/', 1)[-1]
    pkl_path = os.path.join(
        hparams.results_dir, 'IKE', 'embedding',
        f'{safe_model}_HemOncIKEDataset_{len(train_ds)}.pkl',
    )
    with open(pkl_path, 'rb') as f:
        stored = pickle.load(f)

    stored_embeddings = util.normalize_embeddings(
        torch.tensor(stored['embeddings']).to(device)
    )

    query = request['prompt']
    q_emb = util.normalize_embeddings(
        torch.tensor(sentence_model.encode(query, show_progress_bar=False))
        .unsqueeze(0)
        .to(device)
    )
    hits = util.semantic_search(
        q_emb, stored_embeddings, score_function=util.dot_score, top_k=hparams.k
    )
    icl_examples = [stored['sentences'][h['corpus_id']] for h in hits[0]]
    return icl_examples


# ---------------------------------------------------------------------------
# IKEWrapper — nn.Module wrapper for editor.rag() evaluation pipeline
# ---------------------------------------------------------------------------

class IKEWrapper(torch.nn.Module):
    """RAG-style wrapper that retrieves IKE demonstrations and prepends them.

    Mirrors the OracleRAG / BM25RAG / DenseRAG wrappers so that IKE routes
    through ``editor.rag()`` and is evaluated with ``compute_edit_quality``
    (the QA-judge pipeline) instead of the legacy ``compute_icl_edit_quality``
    token-probability path.

    The embedding index is seeded by ``encode_hemonc_ike_facts()`` (the
    pre-batch corpus pickle) and grown incrementally via ``adapt()``, which
    encodes and appends each new batch's records.
    """

    def __init__(self, config, model, tokenizer, device, train_ds=None):
        super().__init__()
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        # Accept int, OmegaConf int, or string (e.g. '0', 'cuda:0')
        self.device = device
        _d = str(device)
        self._cuda_device = _d if _d.startswith('cuda') else f'cuda:{_d}'

        self.top_k = config.k
        self.stored_sentences: list = []
        self.stored_evidences: list = []   # parallel to stored_sentences
        self.stored_embeddings = None      # normalised Tensor on GPU

        self.embedding_model = SentenceTransformer(config.sentence_model_name).to(self._cuda_device)

        if train_ds is not None:
            self._load_index(train_ds)

    # ------------------------------------------------------------------
    # Attribute delegation to the wrapped model
    # ------------------------------------------------------------------

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)

    # ------------------------------------------------------------------
    # Index loading
    # ------------------------------------------------------------------

    def _load_index(self, train_ds):
        safe_model = self.config.sentence_model_name.rsplit('/', 1)[-1]
        pkl_path = os.path.join(
            self.config.results_dir, 'IKE', 'embedding',
            f'{safe_model}_HemOncIKEDataset_{len(train_ds)}_qcore.pkl',
        )
        with open(pkl_path, 'rb') as f:
            stored = pickle.load(f)
        self.stored_sentences = stored['sentences']
        # 'evidences' key added in updated encode_hemonc_ike_facts; fall back
        # to None entries for pickles built with the old format.
        self.stored_evidences = stored.get('evidences', [None] * len(self.stored_sentences))
        emb_tensor = torch.tensor(stored['embeddings']).to(self._cuda_device)
        self.stored_embeddings = util.normalize_embeddings(emb_tensor)
        print(f'[IKEWrapper] Loaded {len(self.stored_sentences)} demonstrations from {pkl_path}')

    # ------------------------------------------------------------------
    # Adapt — encode current-batch records and extend the index
    # ------------------------------------------------------------------

    def adapt(self, batched_requests):
        """Encode new-batch demonstrations and append them to the index.

        For each request the demonstration is formatted as:
            Evidence: <abstract>\n\n<question>\nAnswer: <answer>\n\n
        matching the format used by ``encode_hemonc_ike_facts()``.
        """
        import time
        exec_times = []

        for batch in batched_requests:
            t0 = time.time()
            new_sentences: list = []
            new_evidences: list = []
            new_queries: list = []
            for req in batch:
                evidence = req.get('ground_truth', '')
                if not evidence or evidence == '<|endoftext|>':
                    continue
                prompt = req['prompt']
                answer = req['target_new']
                # Strip evidence prefix if the prompt already contains it
                # (when include_evidence=True) so the demo format stays consistent.
                ev_prefix = f"Evidence: {evidence}\n\n"
                plain_q = prompt[len(ev_prefix):] if prompt.startswith(ev_prefix) else prompt
                sentence = f"Evidence: {evidence}\n\n{plain_q}\nAnswer: {answer}\n\n"
                new_sentences.append(sentence)
                new_evidences.append(evidence)
                # Use the question prompt for embedding, not the full demonstration
                # string — consistent with encode_hemonc_ike_facts(). Strip the
                # system-prompt wrapper to match the corpus encoding space.
                new_queries.append(_query_core(plain_q))
            if new_sentences:
                new_embs = self.embedding_model.encode(new_queries, show_progress_bar=False)
                new_embs_t = util.normalize_embeddings(
                    torch.tensor(new_embs).to(self._cuda_device)
                )
                self.stored_sentences.extend(new_sentences)
                self.stored_evidences.extend(new_evidences)
                if self.stored_embeddings is not None:
                    self.stored_embeddings = torch.cat([self.stored_embeddings, new_embs_t], dim=0)
                else:
                    self.stored_embeddings = new_embs_t
            exec_times.append(time.time() - t0)

        return exec_times

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _retrieve(self, query: str) -> str:
        """Return top-k demonstrations concatenated as a single string."""
        if self.stored_embeddings is None or not self.stored_sentences:
            return ''
        q_emb = util.normalize_embeddings(
            torch.tensor(
                self.embedding_model.encode(_query_core(query), show_progress_bar=False)
            ).unsqueeze(0).to(self._cuda_device)
        )
        hits = util.semantic_search(
            q_emb, self.stored_embeddings,
            score_function=util.dot_score, top_k=self.top_k,
        )
        return '\n'.join(self.stored_sentences[h['corpus_id']] for h in hits[0])

    def retrieve_evidence(self, query: str) -> list:
        """Return the evidence (abstract) texts for the top-k retrieved demonstrations.

        Used to compute retrieval accuracy: check whether the gold evidence is
        among the returned strings.
        """
        if self.stored_embeddings is None or not self.stored_sentences:
            return []
        q_emb = util.normalize_embeddings(
            torch.tensor(
                self.embedding_model.encode(_query_core(query), show_progress_bar=False)
            ).unsqueeze(0).to(self._cuda_device)
        )
        hits = util.semantic_search(
            q_emb, self.stored_embeddings,
            score_function=util.dot_score, top_k=self.top_k,
        )
        return [
            self.stored_evidences[h['corpus_id']]
            for h in hits[0]
            if self.stored_evidences[h['corpus_id']] is not None
        ]

    # ------------------------------------------------------------------
    # Text-level augmentation (used by the evaluation pipeline)
    # ------------------------------------------------------------------

    def augment_texts(self, prompts: list) -> list:
        """Return prompts with retrieved IKE demonstrations prepended (text level).

        Called by the evaluation pipeline in evaluate_utils.py *before*
        tokenization so that the chat template is applied to the already-
        augmented text — matching the interface of BM25RAG / DenseRAG /
        OracleRAG.  This ensures both the vLLM and non-vLLM paths receive
        correctly formatted inputs.
        """
        augmented = []
        for text in prompts:
            demos = self._retrieve(text)
            augmented.append(f"{demos}\n{text}" if demos else text)
        return augmented

    # ------------------------------------------------------------------
    # Shared augmentation helper (token-level, used by forward/generate)
    # ------------------------------------------------------------------

    def _augment(self, input_ids):
        """Prepend IKE demonstrations to each input.

        Returns
        -------
        new_ids      : Tensor  [B, context_len + orig_len]
        new_mask     : Tensor  [B, context_len + orig_len]
        len_context  : int     prepended token count (after padding)
        augmented    : bool    False when the index is empty (no-op)
        """
        if self.stored_embeddings is None or not self.stored_sentences:
            return input_ids, None, 0, False

        original_inputs = [
            self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            for b in range(input_ids.shape[0])
        ]
        augmented_texts = []
        for text in original_inputs:
            demos = self._retrieve(text)
            augmented_texts.append(f"{demos}\n{text}" if demos else text)

        new_tok = self.tokenizer(
            augmented_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
        )
        new_ids = new_tok['input_ids'].to(self._cuda_device)
        new_mask = new_tok['attention_mask'].to(self._cuda_device)
        len_context = new_ids.shape[1] - input_ids.shape[1]
        return new_ids, new_mask, len_context, True

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask):
        # Augmentation is handled externally by augment_texts() before tokenization
        # and chat-template application.  Calling _augment() here would cause double
        # augmentation and strip the chat template for instruction-tuned models.
        with torch.no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------

    def generate(self, input_ids, attention_mask=None, **kwargs):
        # Augmentation is handled externally by augment_texts() before tokenization
        # and chat-template application.  Calling _augment() here would cause double
        # augmentation and strip the chat template for instruction-tuned models.
        gen_kwargs = {'input_ids': input_ids, **kwargs}
        if attention_mask is not None:
            gen_kwargs['attention_mask'] = attention_mask
        return self.model.generate(**gen_kwargs)