"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
import sys

import pandas as pd

from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_generation_quality,
    test_generation_acc,
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1,
    llm_as_judge,
    get_keyword_acc,
    _detect_refusal,
)

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    records: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False,
    few_shot_examples = False,
    judge_model: Optional[str] = None,
    judge_workers: int = 4,
    judge_enabled: bool = True,
    vllm_model=None,
    lora_request=None,
    openrouter_client=None,
    openrouter_model=None,
    openrouter_thinking_budget=0,
    rag_model=None,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    if isinstance(model,LORA):
        model=model.model
    # First, unpack rewrite evaluation record.
    rewrite_prompts, rephrase_prompts, targets, modes, tags, evidence = [], [], [], [], [], []
    loc_prompts, portability_prompts = {}, {}
    locality_record_idx, portability_record_idx = {}, {}
    if not isinstance(records, list):
        records = [records]
    for i_record, record in enumerate(records):
        target_new, ground_truth = (
            record[x] for x in ["target_new", "ground_truth"]
        )
        fs_examples = {}
        mode = 'closed' if target_new in ['correct', 'incorrect', '0', '1', '2', '3', 'superior', 'inferior', 'no difference'] else 'open'
        if mode == 'open':
            eval_metric = 'key_word_em'
        rewrite_prompt = record["prompt"]
        rephrase_prompt = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
        evidence.append(record["ground_truth"] if 'ground_truth' in record.keys() else None)
        if few_shot_examples:
            rewrite_prompt = fs_examples['closed_q'].format(rewrite_prompt)
            rephrase_prompt = fs_examples['closed_q'].format(rephrase_prompt) if rephrase_prompt is not None else None
        rewrite_prompts.append(rewrite_prompt)
        if not rephrase_prompt is None or pd.isna(rephrase_prompt):
            rephrase_prompts.append(rephrase_prompt)
        targets.append(target_new)
        modes.append(mode)
        tags.append(record['tag'])

        if 'locality' in record.keys() and any(record['locality']):
            for locality_key in record['locality'].keys():
                if not locality_key in loc_prompts.keys():
                    loc_prompts[locality_key] = {'prompt': [], 'ground_truth': []}
                    locality_record_idx[locality_key] = []
                loc_prompt = record['locality'][locality_key]['prompt']
                # fs_examples not applied to locality prompts: locality tests
                # knowledge of unrelated facts and uses a different question type;
                # the CQ/OQ template keys ('closed_q'/'open_q') don't match the
                # mode values ('closed'/'open'), causing a KeyError when applied here.
                loc_prompts[locality_key]['prompt'].append(loc_prompt)
                loc_prompts[locality_key]['ground_truth'].append(record['locality'][locality_key]['ground_truth'])
                locality_record_idx[locality_key].append(i_record)
        if 'portability' in record.keys() and any(record['portability']):
            for portability_key in record['portability'].keys():
                if not portability_key in portability_prompts.keys():
                    portability_prompts[portability_key] = {'prompt': [], 'ground_truth': []}
                    portability_record_idx[portability_key] = []
                if pd.isna(record['portability'][portability_key]['prompt']) or pd.isna(
                        record['portability'][portability_key]['ground_truth']):
                    continue
                portability_prompt = record['portability'][portability_key]['prompt']
                if few_shot_examples and ('cq' in portability_key or 'oq' in portability_key):
                    fs_key = 'open_q' if 'oq' in portability_key else 'closed_q'
                    portability_prompt = fs_examples[fs_key].format(portability_prompt)
                portability_prompts[portability_key]['prompt'].append(portability_prompt)
                portability_prompts[portability_key]['ground_truth'].append(record['portability'][portability_key]['ground_truth'])
                portability_record_idx[portability_key].append(i_record)

    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, targets, device=device, eval_metric='token_em', record_flops=True,
                                              vllm_model=vllm_model, lora_request=lora_request, judge_model=judge_model,
                                              judge_workers=judge_workers, judge_enabled=judge_enabled,
                                              openrouter_client=openrouter_client, openrouter_model=openrouter_model,
                                              openrouter_thinking_budget=openrouter_thinking_budget,
                                              rag_model=rag_model)

    if rephrase_prompts[0] is not None:
        reph = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, targets, device=device, test_rephrase=True, eval_metric='token_em',
                                                vllm_model=vllm_model, lora_request=lora_request, judge_model=judge_model,
                                                judge_workers=judge_workers, judge_enabled=judge_enabled,
                                                openrouter_client=openrouter_client, openrouter_model=openrouter_model,
                                                openrouter_thinking_budget=openrouter_thinking_budget,
                                                rag_model=rag_model)
    else:
        reph = None

    loc, port = {}, {}
    for locality_key in loc_prompts.keys():
        loc[locality_key] = compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                     loc_prompts[locality_key]['prompt'],
                                     loc_prompts[locality_key]['ground_truth'],
                                     device=device, vllm_model=vllm_model, lora_request=lora_request, judge_model=judge_model,
                                     judge_workers=judge_workers, judge_enabled=judge_enabled,
                                     openrouter_client=openrouter_client, openrouter_model=openrouter_model,
                                     openrouter_thinking_budget=openrouter_thinking_budget)
    for portability_key in portability_prompts.keys():
        if len(portability_prompts[portability_key]['prompt']) > 0:
            if hasattr(model, 'verbose') and portability_key == 'mhop':
                model.verbose = True
            port[portability_key] = compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            portability_prompts[portability_key]['prompt'],
                                            portability_prompts[portability_key]['ground_truth'],
                                            device=device, vllm_model=vllm_model, lora_request=lora_request,
                                            judge_model=judge_model, judge_workers=judge_workers,
                                            judge_enabled=judge_enabled,
                                            openrouter_client=openrouter_client, openrouter_model=openrouter_model,
                                            openrouter_thinking_budget=openrouter_thinking_budget)
            if hasattr(model, 'verbose'):
                model.verbose = False

    if test_generation:
        if hparams.alg_name == 'GRACE' or 'gpt' in model_name.lower():
            fluency = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=500, vanilla_generation=True)
        else:
            fluency = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=500, vanilla_generation=False)
    else:
        fluency = None
    # Build index mappings: key -> {global_record_idx: local_result_idx}
    # Both locality and portability arrays are shorter than len(ret) when some records
    # have no valid locality/portability data. Using the global k directly as the index
    # causes IndexError for records beyond the valid subset.
    loc_k_map = {
        l_key: {g_idx: l_idx for l_idx, g_idx in enumerate(locality_record_idx.get(l_key, []))}
        for l_key in loc.keys()
    }
    port_k_map = {
        p_key: {g_idx: l_idx for l_idx, g_idx in enumerate(portability_record_idx.get(p_key, []))}
        for p_key in port.keys()
    }

    for k in range(len(ret)):
        ret[k]['prompt'] = rewrite_prompts[k]
        ret[k]['target'] = targets[k]
        ret[k]['mode'] = modes[k]
        ret[k]['tag'] = tags[k]
        ret[k]['ground_truth'] = evidence[k] if evidence is not None else None
        ret[k]['fluency'] = fluency[k] if fluency is not None else None
        ret[k].update(reph[k]) if reph is not None else None
        ret[k].update({'locality': {}})
        for l_key in loc.keys():
            local_idx = loc_k_map[l_key].get(k)
            if local_idx is not None:
                for l in loc[l_key]:
                    ret[k]['locality'][l] = {
                        'performance': loc[l_key][l]['performance'][local_idx],
                        'prompt':      loc[l_key][l]['prompt'][local_idx],
                        'ground_truth':loc[l_key][l]['ground_truth'][local_idx],
                        'answer':      loc[l_key][l]['answer'][local_idx],
                    }
        ret[k].update({'portability': {}})
        for p_key in port.keys():
            for p in port[p_key]:
                port_idx = port_k_map[p_key].get(k)
                if port_idx is not None:
                    ret[k]['portability'][p] = {
                        'performance': port[p_key][p]['performance'][port_idx],
                        'prompt':      port[p_key][p]['prompt'][port_idx],
                        'ground_truth':port[p_key][p]['ground_truth'][port_idx],
                        'answer':      port[p_key][p]['answer'][port_idx],
                    }
    return ret


def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompts: list,
    targets: list,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em',
    record_flops: bool = False,
    vllm_model=None,
    lora_request=None,
    judge_model=None,
    judge_workers: int = 4,
    judge_enabled: bool = True,
    openrouter_client=None,
    openrouter_model=None,
    openrouter_thinking_budget=0,
    rag_model=None,
) -> typing.Dict:
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    if eval_metric == 'ppl':
        ppl = PPL(model, tok, prompts, targets, device)
        res = [{
            f"{key}_ppl": p
        } for p in ppl]
    elif hparams.alg_name=="GRACE":
        # ppl = PPL(model, tok, prompt, target_new, device)
        if 't5' in model_name.lower():
            res = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompts, targets, device)
            answ, forward_time = None, None
        else:
            res, answ, forward_time = test_prediction_acc(model, tok, hparams, prompts, targets, device, vanilla_generation=True)
        res = [{
            f"{key}_acc": r['acc'],
            f"{key}_answ": ans,
            f"{key}_refused": bool(r.get('refused', _detect_refusal(ans))),
            f"{key}_truncated": bool(r.get('truncated', False)),
        } for r, ans in zip(res, answ)]
    elif eval_metric == 'key_word_em':
        if 'gpt' in model_name.lower():
            vanilla_generation = True
        else:
            vanilla_generation = True
        answ, trunc_flags, forward_time = test_generation_acc(model=model, tok=tok, prefixes=prompts, max_out_len=hparams.max_out_len_open, vanilla_generation=vanilla_generation, vllm_model=vllm_model, lora_request=lora_request, rag_model=rag_model)
        keyword_acc = get_keyword_acc(answ, targets)
        res = [{
            f"{key}_acc": r['acc'],
            f"{key}_answ": a,
            f"{key}_forward_time": forward_time,
            f"{key}_refused": _detect_refusal(a),
            f"{key}_truncated": bool(t),
        } for r, a, t in zip(keyword_acc, answ, trunc_flags)]
    else:
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompts, targets, device)
            answ, forwad_time = None, None, None
        else:
            # When vllm_model is None the token-logit path runs without a chat template,
            # producing garbage for instruction-tuned models.  Fall back to vanilla
            # generation (which applies the chat template) whenever vLLM is not available.
            _vanilla_gen = vllm_model is None
            res, answ, forward_time = test_prediction_acc(model, tok, hparams, prompts, targets, device, record_flops=record_flops,
                                                          vanilla_generation=_vanilla_gen,
                                                          vllm_model=vllm_model, lora_request=lora_request, judge_model=judge_model,
                                                          judge_workers=judge_workers, judge_enabled=judge_enabled,
                                                          openrouter_client=openrouter_client, openrouter_model=openrouter_model,
                                                          openrouter_thinking_budget=openrouter_thinking_budget,
                                                          max_out_len_closed=getattr(hparams, 'max_out_len_closed', None),
                                                          rag_model=rag_model)
        res = [{
            f"{key}_acc": r['acc'],
            f"{key}_answ": a,
            f"{key}_refused": bool(r.get('refused', False)),
            f"{key}_truncated": bool(r.get('truncated', False)),
        } for r, a in zip(res, answ)]
    return res

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: typing.Union[str, List[str]],
    locality_ground_truth: typing.Union[str, List[str]],
    device,
    vllm_model=None,
    lora_request=None,
    judge_model=None,
    judge_workers: int = 4,
    judge_enabled: bool = True,
    openrouter_client=None,
    openrouter_model=None,
    openrouter_thinking_budget=0,
) -> typing.Dict:

    if 't5' in model_name.lower():
        loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
        loc_acc = 0
        performance = None
        answer = None
    else:
        answer, trunc_flags, forward_time = test_generation_acc(model=model, tok=tok, prefixes=prompt,
                                                   max_out_len=hparams.max_out_len_open, vanilla_generation='gpt' in model_name.lower(),
                                                   vllm_model=vllm_model, lora_request=lora_request,
                                                   openrouter_client=openrouter_client, openrouter_model=openrouter_model,
                                                   openrouter_thinking_budget=openrouter_thinking_budget)
        keyword_acc = get_keyword_acc(answer, locality_ground_truth)
        if judge_enabled:
            print('calling llm judge from locality')
            # locality_ground_truth may be a plain list of strings; wrap into the
            # dict shape that llm_as_judge expects (condition/context left empty).
            gt_dicts = [
                {'condition': '', 'context': '', 'target': gt} if isinstance(gt, str) else gt
                for gt in locality_ground_truth
            ]
            llm_score, explanations, flags_list = llm_as_judge(mode='open_qa',
                                                               answers=answer, ground_truth=gt_dicts,
                                                               judge_model=judge_model or 'gpt-4o-mini-hemonc',
                                                               max_workers=judge_workers)

        else:
            llm_score = [None] * len(answer)
            explanations = [None] * len(answer)
            flags_list = [None] * len(answer)
        performance = [{'acc': acc, 'score': score, 'explanation': exp, 'flags': flags,
                        'refused': _detect_refusal(ans), 'truncated': bool(t)}
                       for acc, score, exp, flags, ans, t
                       in zip(keyword_acc, llm_score, explanations, flags_list, answer, trunc_flags)]

    ret = {
        f"{locality_key}": {'performance': performance,
                               'prompt': prompt, 'ground_truth': locality_ground_truth, 'answer': answer}
    }
    return ret

def compute_portability_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    portability_key: str,
    prompt: typing.Union[str, List[str]],
    ground_truth: typing.Union[str, List[str]],
    device,
    vllm_model=None,
    lora_request=None,
    judge_model: Optional[str] = 'gpt-4o-mini-hemonc',
    judge_workers: int = 4,
    judge_enabled: bool = True,
    openrouter_client=None,
    openrouter_model=None,
    openrouter_thinking_budget=0,
) -> typing.Dict:
    #if portability_key == 'mhop':
    #    model.verbose = True

    if '_cq_' in portability_key:
        if 't5' in model_name.lower():
            portability_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, ground_truth, device)
            answer = None
        else:
            # When vllm_model is None the token-logit path runs without a chat
            # template, producing garbled per-position argmax text for
            # instruction-tuned models (mirrors the same bug and fix in
            # compute_rewrite_or_rephrase_quality). Fall back to vanilla
            # generation (which applies the chat template) whenever vLLM is
            # not available. GRACE already forces vanilla=True for its own
            # model-wrapper reasons.
            _vanilla_gen = (hparams.alg_name == 'GRACE') or (vllm_model is None)
            performance, answer, _ = test_prediction_acc(model, tok, hparams, prompt, ground_truth, device,
                                                         vanilla_generation=_vanilla_gen,
                                                         vllm_model=vllm_model, lora_request=lora_request,
                                                         judge_model=judge_model, judge_workers=judge_workers,
                                                         judge_enabled=judge_enabled,
                                                         openrouter_client=openrouter_client, openrouter_model=openrouter_model,
                                                         openrouter_thinking_budget=openrouter_thinking_budget,
                                                         max_out_len_closed=hparams.max_out_len_closed)
    else:
        answer, trunc_flags, forward_time = test_generation_acc(model=model, tok=tok, prefixes=prompt,
                                                   max_out_len=hparams.max_out_len_open, vanilla_generation='gpt' in model_name.lower(),
                                                   vllm_model=vllm_model, lora_request=lora_request,
                                                   openrouter_client=openrouter_client, openrouter_model=openrouter_model,
                                                   openrouter_thinking_budget=openrouter_thinking_budget)
        keyword_acc = get_keyword_acc(answer, [gt['target'] for gt in ground_truth])
        if judge_enabled:
            print('calling llm judge from portability')
            llm_score, explanations, flags_list = llm_as_judge(mode='open_qa' if 'oq' in portability_key else 'open_gen',
                                                               answers=answer, ground_truth=ground_truth,
                                                               judge_model=judge_model or 'gpt-4o-mini-hemonc',
                                                               max_workers=judge_workers)
        else:
            llm_score = [None] * len(answer)
            explanations = [None] * len(answer)
            flags_list = [None] * len(answer)
        performance = [{'acc': acc, 'score': score, 'explanation': exp, 'flags': flags,
                        'refused': _detect_refusal(ans), 'truncated': bool(t)}
                       for acc, score, exp, flags, ans, t
                       in zip(keyword_acc, llm_score, explanations, flags_list, answer, trunc_flags)]

    ret = {
        f"{portability_key}": {'performance': performance,
                               'prompt': prompt, 'ground_truth': ground_truth, 'answer': answer}
    }
    #model.verbose = False
    return ret

def compute_icl_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        icl_examples,
        record: typing.Dict,
        device,
        pre_edit: bool = False
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, prompt)
    else:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, new_fact)
    ret = {
        f"rewrite_acc": edit_acc
    }
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                   target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        ret['rephrase_acc'] = rephrase_acc

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            if isinstance(record['locality'][locality_key]['ground_truth'], list):
                pre_neighbor = []
                post_neighbor = []
                for x_a, x_p in zip(record['locality'][locality_key]['ground_truth'],
                                    record['locality'][locality_key]['prompt']):
                    tmp_pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], x_a,
                                                   f"New Fact: {prompt} {target_new}\nPrompt: {x_p}", neighborhood=True)
                    tmp_post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, x_a,
                                                    f"New Fact: {prompt} {target_new}\nPrompt: {x_p}",
                                                    neighborhood=True)
                    if type(tmp_pre_neighbor) is not list:
                        tmp_pre_neighbor = [tmp_pre_neighbor, ]
                    if type(tmp_post_neighbor) is not list:
                        tmp_post_neighbor = [tmp_post_neighbor, ]
                    assert len(tmp_pre_neighbor) == len(tmp_post_neighbor)
                    pre_neighbor.append(tmp_pre_neighbor)
                    post_neighbor.append(tmp_post_neighbor)
                res = []
                for ans, label in zip(pre_neighbor, post_neighbor):
                    temp_acc = np.mean(np.equal(ans, label))
                    if np.isnan(temp_acc):
                        continue
                    res.append(temp_acc)
                ret['locality'][f'{locality_key}_acc'] = res
            else:
                pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''],
                                           record['locality'][locality_key]['ground_truth'],
                                           f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}",
                                           neighborhood=True)
                post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                            record['locality'][locality_key]['ground_truth'],
                                            f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}",
                                            neighborhood=True)
                if type(pre_neighbor) is not list:
                    pre_neighbor = [pre_neighbor, ]
                if type(post_neighbor) is not list:
                    post_neighbor = [post_neighbor, ]
                assert len(pre_neighbor) == len(post_neighbor)

                ret['locality'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
    # Form a list of lists of prefixes to test.
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            if pre_edit:
                icl_input = ['']
                x_prefix = ""
            else:
                icl_input = icl_examples
                x_prefix = f"New Fact: {prompt} {target_new}\nPrompt: "
            if isinstance(record['portability'][portability_key]['ground_truth'], list):
                portability_acc = []
                for x_a, x_p in zip(record['portability'][portability_key]['ground_truth'],
                                    record['portability'][portability_key]['prompt']):
                    tmp_portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_input, x_a,
                                                      f"{x_prefix}{x_p}")
                portability_acc.append(tmp_portability_acc)
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, [''],
                                              record['portability'][portability_key]['ground_truth'],
                                              record['portability'][portability_key]['prompt'])
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              record['portability'][portability_key]['ground_truth'],
                                              f"New Fact: {prompt} {target_new}\nPrompt: {record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()