import sys
from typing import Optional, Union, List, Tuple, Dict
from time import time
from torch.utils.data import Dataset
import torch.nn as nn
import wandb
from tqdm import tqdm
import json
import torch
import numpy as np
import random
import math
import argparse
from ..models.melo.melo import LORA
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from transformers import LlamaTokenizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import GPT2TokenizerFast, GPT2Tokenizer
from ..util.globals import *
from .utils import _chunks, _prepare_requests, summary_metrics
from .batch_editor import BatchEditor
from ..evaluate import compute_edit_quality, compute_icl_edit_quality, compute_sent_metric, metrics_to_wandb
from ..util import nethook
from ..util.hparams import HyperParams
from ..util.alg_dict import *
from ..util.load_data import load_dataset
from ..evaluate.evaluate_utils import test_generation_quality, extract_metric
from ..models.rag.rag_main import RAG
from ..models.ptuning.ptuning_main import PromptTuning
from ..models.oracle_rag.oracle_rag_main import OracleRAG
from ..models.bm25_rag.bm25_rag_main import BM25RAG
from ..models.dense_rag.dense_rag_main import DenseRAG
from ..models.ike.ike_main import IKEWrapper
from .utils import GPTWrapper

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

LOG = logging.getLogger(__name__)


# Cached once per process so we don't re-probe on every model load.
_ATTN_IMPL_CACHE: Optional[str] = None


def _select_attn_impl(requested: Optional[str] = None) -> Optional[str]:
    """Resolve the attention implementation to use at HF model load time.

    Values:
      - 'flash_attention_2' / 'sdpa' / 'eager' — passed through to HF.
      - 'auto' or None — probe for flash-attn availability, fall back to sdpa.
      - 'default' — return None so HF picks its own default (usually sdpa).

    Returns a string suitable for `from_pretrained(attn_implementation=...)`
    or None to let HF decide.
    """
    global _ATTN_IMPL_CACHE
    if requested in ('default',):
        return None
    if requested not in (None, 'auto'):
        return requested
    if _ATTN_IMPL_CACHE is not None:
        return _ATTN_IMPL_CACHE
    try:
        import flash_attn  # noqa: F401
        _ATTN_IMPL_CACHE = 'flash_attention_2'
        LOG.info("flash_attn detected; using attn_implementation='flash_attention_2'")
    except ImportError:
        _ATTN_IMPL_CACHE = 'sdpa'
        LOG.info("flash_attn not installed; using attn_implementation='sdpa'")
    return _ATTN_IMPL_CACHE


def make_logs():
    f_h, s_h = get_handler('logs', log_name='run.log')
    LOG.addHandler(f_h)
    LOG.addHandler(s_h)


def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(42)


class BaseEditor:
    """Base editor for all methods"""

    @classmethod
    def from_hparams(cls, hparams: HyperParams):
        return cls(hparams)

    def __init__(self, hparams: HyperParams):
        assert hparams is not None, 'Error: hparams is None.'
        self.model_name = hparams.model_name
        self.apply_algo = ALG_DICT[hparams.alg_name]
        self.alg_name = hparams.alg_name
        self.accelerator = None
        self.vllm_model = None
        self.lora_request = None
        self.openrouter_client = None
        self.openrouter_model_name = None
        self.openrouter_thinking_budget = 0
        make_logs()
        LOG.info("Instantiating model")

        # vLLM fast-inference path (skips HF model loading entirely)
        if getattr(hparams, 'use_vllm', False):
            from vllm import LLM
            from vllm.lora.request import LoRARequest
            adapter_path = getattr(hparams, 'adapter_path', None)
            LOG.info(f"Loading model via vLLM: {self.model_name}"
                     + (f" with LoRA adapter: {adapter_path}" if adapter_path else ""))
            # Gemma 3 uses interleaved local/global (sliding-window) attention.
            # Its sliding-window attention is numerically unstable in float16:
            # NaN logits cause argmax to collapse to index 0 (<pad>), so every
            # generated token decodes to an empty string.  Two fixes are needed:
            #   1. dtype="bfloat16"    — prevents NaN logits (root cause fix)
            #   2. enforce_eager=True  — disables CUDA-graph capture as a safety net
            # Both are required; enforce_eager alone is insufficient (confirmed by
            # diagnostic testing: all float16 configs, eager or not, produce zeros).
            # Gemma 2 / MedGemma are unaffected and keep the defaults.
            _is_gemma3 = 'gemma-3' in self.model_name.lower() or 'gemma3' in self.model_name.lower()
            _vllm_dtype = "bfloat16" if _is_gemma3 else "auto"
            self.vllm_model = LLM(
                model=self.model_name,
                enable_lora=bool(adapter_path),
                max_lora_rank=getattr(hparams, 'lora_max_rank', 64),
                dtype=_vllm_dtype,
                gpu_memory_utilization=getattr(hparams, 'vllm_gpu_memory_utilization', 0.90),
                max_model_len=getattr(hparams, 'vllm_max_model_len', 8192),
                max_num_seqs=getattr(hparams, 'vllm_max_num_seqs', 256),
                enforce_eager=_is_gemma3,
            )
            self.lora_request = LoRARequest("adapter", 1, adapter_path) if adapter_path else None
            self.model = None
            self.tok = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True,
                padding_side='left' if 'llama' in self.model_name.lower() else 'right'
            )
            if self.tok.pad_token_id is None:
                self.tok.pad_token_id = self.tok.eos_token_id
            self.hparams = hparams
            LOG.info(f"vLLM model loaded successfully")
            return

        # OpenRouter API inference path (skips all local model loading)
        if getattr(hparams, 'use_openrouter', False):
            import os
            from openai import OpenAI
            LOG.info(f"Loading model via OpenRouter: {self.model_name}")
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ.get('OPENROUTER_API_KEY'),
                max_retries=0,
            )
            self.openrouter_model_name = self.model_name
            self.openrouter_thinking_budget = getattr(hparams, 'thinking_budget_tokens', 0)
            self.model = None
            self.vllm_model = None
            self.tok = None
            self.hparams = hparams
            LOG.info(f"OpenRouter model ready: {self.openrouter_model_name}")
            return

        if type(self.model_name) is str:
            device_map = 'auto' if hparams.model_parallel else None
            torch_dtype = (torch.bfloat16 if hasattr(hparams, 'bf16') and hparams.bf16 else
                           torch.float16  if hasattr(hparams, 'fp16') and hparams.fp16 else
                           torch.float32)
            # GRACE/WISE/MEMOIR replace nn.Module layers in-place with adapter objects.
            # Any model that wraps its MLP forward with torch.compile will cause dynamo
            # to silently bypass the adapter (suppress_errors falls back to the pre-
            # replacement compiled graph). Disabling dynamo entirely is the only safe fix.
            if hparams.alg_name in ('GRACE', 'WISE', 'MEMOIR'):
                torch._dynamo.config.disable = True
                LOG.info("torch._dynamo disabled — %s replaces nn.Module layers in-place", hparams.alg_name)
                # These methods don't use vLLM, so expandable_segments is
                # safe (it would conflict with vLLM's CuMemAllocator). On 8B
                # the 1-2 GB of fragmentation-saved memory prevents training
                # OOM during cross_entropy_loss/backward.
                import os as _os
                _os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
                LOG.info("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (reduces fragmentation for HF-only %s)", hparams.alg_name)
            # Gemma-3 / MedGemma load as Gemma3ForConditionalGeneration (MLLM
            # wrapper). HF's generate() on that wrapper auto-triggers
            # torch._inductor → torch.cuda.graph capture, which fails with
            # "Inplace update to inference tensor outside InferenceMode" on
            # some Gemma-3 builds. Disabling dynamo sidesteps the compile
            # path entirely; cost is negligible for eval-scale generation.
            _is_gemma3_family = ('gemma-3' in self.model_name.lower()
                                 or 'gemma3' in self.model_name.lower()
                                 or 'medgemma' in self.model_name.lower())
            if _is_gemma3_family:
                torch._dynamo.config.disable = True
                LOG.info("torch._dynamo disabled — Gemma-3 family avoids CUDA-graph capture issues")
            # Attention backend selection: flash_attention_2 when installed
            # (30–50% faster than sdpa on Qwen/Llama), sdpa otherwise. The
            # `attn_implementation` hparam overrides the probe (e.g. set to
            # 'sdpa' to disable flash-attn).
            _attn_impl = _select_attn_impl(getattr(hparams, 'attn_implementation', None))
            if 't5' in self.model_name.lower():
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                        device_map=device_map)
                self.tok = T5Tokenizer.from_pretrained(self.model_name)
            elif 'gpt2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  device_map=device_map)
                self.tok = GPT2Tokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'gpt' in self.model_name.lower():
                self.tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False)
                emb_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch_dtype,
                                                                  device_map=device_map)
                self.tok.pad_token_id = self.tok.eos_token_id
                self.model = GPTWrapper(self.model_name, self.tok, emb_model)
            elif 'llama' in self.model_name.lower():
                _llama_kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
                if _attn_impl is not None:
                    _llama_kwargs['attn_implementation'] = _attn_impl
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **_llama_kwargs)
                except (ValueError, ImportError) as e:
                    LOG.warning(f"llama load with attn_implementation={_attn_impl!r} failed ({e}); retrying with sdpa")
                    _llama_kwargs['attn_implementation'] = 'sdpa'
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **_llama_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=False, padding_side='left')
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'baichuan' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  trust_remote_code=True, device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'chatglm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                                                       device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.unk_token_id = 64787
                # self.tok.pad_token_id = self.tok.eos_token_id
            elif 'internlm' in self.model_name.lower():
                self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                                                       device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen3' in self.model_name.lower():
                _qwen_kwargs = dict(
                    trust_remote_code=True,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )
                if _attn_impl is not None:
                    _qwen_kwargs['attn_implementation'] = _attn_impl
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **_qwen_kwargs)
                except (ValueError, ImportError) as e:
                    LOG.warning(f"qwen3 load with attn_implementation={_attn_impl!r} failed ({e}); retrying with sdpa")
                    _qwen_kwargs['attn_implementation'] = 'sdpa'
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **_qwen_kwargs)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'qwen2' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True,
                                                                  torch_dtype=torch_dtype if hparams.alg_name not in [
                                                                      'MEND'] else torch.bfloat16,
                                                                  device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>',
                                                         pad_token='<|endoftext|>', unk_token='<|endoftext|>',
                                                         trust_remote_code=True)
            elif 'qwen' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, fp32=False, trust_remote_code=True,
                                                                  device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, eos_token='<|endoftext|>',
                                                         pad_token='<|endoftext|>', unk_token='<|endoftext|>',
                                                         trust_remote_code=True)
            elif 'mistral' in self.model_name.lower():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                  device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.tok.pad_token_id = self.tok.eos_token_id
            elif 'gemma' in self.model_name.lower():
                # Gemma 3 uses sliding-window attention that is numerically unstable
                # in float16 (NaN logits → argmax collapses to 0 → empty decoded output).
                # Always load Gemma 3 in bfloat16 regardless of the fp16 hparam.
                # Gemma 2 / MedGemma are unaffected and keep the configured dtype.
                _is_gemma3 = 'gemma-3' in self.model_name.lower() or 'gemma3' in self.model_name.lower()
                _gemma_dtype = torch.bfloat16 if _is_gemma3 else torch_dtype
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                if self.tok.pad_token_id is None:
                    self.tok.pad_token_id = self.tok.eos_token_id
                _gemma_kwargs = dict(device_map=device_map, torch_dtype=_gemma_dtype)
                # Gemma 3's sliding-window attention has known SDPA edge
                # cases (illegal-memory-access on specific label-masking
                # paths inside HF's forward, e.g. when GRACE computes edit
                # loss). 'eager' attention is slower but bypasses the bug
                # for GRACE/WISE/MEMOIR whose edit path invokes that loss
                # code. Other methods don't trigger the path and keep sdpa.
                if _is_gemma3:
                    _gemma_attn = 'eager' if hparams.alg_name in ('GRACE', 'WISE', 'MEMOIR') else 'sdpa'
                else:
                    _gemma_attn = _attn_impl
                if _gemma_attn is not None:
                    _gemma_kwargs['attn_implementation'] = _gemma_attn
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **_gemma_kwargs)
                except (ValueError, ImportError) as e:
                    LOG.warning(f"gemma load with attn_implementation={_gemma_attn!r} failed ({e}); retrying with sdpa")
                    _gemma_kwargs['attn_implementation'] = 'sdpa'
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **_gemma_kwargs)
                # Gemma-3-family + gradient-training edit methods OOM on 40GB
                # during the loss-computation path (modeling_gemma3.py:1377
                # constructs a filtered shift_logits tensor of ~260 MB per
                # seq). Gradient checkpointing halves activation memory for
                # ~20-30% slower training and is a reliable fix here.
                if _is_gemma3 and hparams.alg_name in (
                    'LoRA-Merge', 'O-LoRA', 'SEEKR', 'LoRA', 'WISE', 'MEMOIR', 'GRACE'
                ):
                    try:
                        self.model.gradient_checkpointing_enable()
                        # model must not cache KV while checkpointing activations
                        if hasattr(self.model, 'config'):
                            self.model.config.use_cache = False
                        # CRITICAL: with reentrant gradient checkpointing (HF default),
                        # gradients won't flow back through a checkpointed transformer
                        # block unless the BLOCK INPUT requires_grad.  For methods that
                        # freeze the entire base and only train an interior parameter
                        # (MEMOIR's adapter new_weight, GRACE's codebook values, WISE's
                        # adapter new_weight, LoRA's lora_A/lora_B), the embedding
                        # output has no grad — so the loss ends up with no grad_fn and
                        # `.backward()` raises "element 0 ... does not require grad".
                        # `enable_input_require_grads` registers a forward hook on the
                        # input embeddings that calls `output.requires_grad_(True)`,
                        # bridging the gap.  PEFT/LoRA already does this in
                        # apply_lora_to_model; the gemma3-gc methods need it too.
                        if hasattr(self.model, 'enable_input_require_grads'):
                            self.model.enable_input_require_grads()
                            LOG.info("enable_input_require_grads — required for "
                                     "gradient checkpointing with frozen base + "
                                     "interior trainable param (%s)", hparams.alg_name)
                        LOG.info("Gradient checkpointing enabled — Gemma-3 + %s training path", hparams.alg_name)
                    except Exception as e:
                        LOG.warning(f"Could not enable gradient checkpointing ({e}); continuing without it")
            elif 'xgen' in self.model_name.lower():
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, padding_side='left')
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map,
                                                                  torch_dtype=torch_dtype)
                self.tok.pad_token_id = 28
            elif 'phi' in self.model_name.lower():
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map=device_map,
                                                                  trust_remote_code=True, _attn_implementation='eager')
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True, torch_dtype=torch_dtype,
                                                       device_map=device_map)
                self.tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                self.tok.pad_token_id = self.tok.eos_token_id

            if self.tok is not None and (
                    isinstance(self.tok, GPT2Tokenizer) or isinstance(self.tok, GPT2TokenizerFast) or isinstance(
                    self.tok, LlamaTokenizer)) and (hparams.alg_name not in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to left...')
                self.tok.padding_side = 'left'
            if self.tok is not None and (
                    'mistral' in self.model_name.lower() or 'llama' in self.model_name.lower()
                    or 'qwen' in self.model_name.lower() or 'gemma' in self.model_name.lower()) and (
                    hparams.alg_name in ['ROME', 'MEMIT', 'EMMET', 'R-ROME']):
                # MEMIT's compute_z writes the target span at positions
                # [ex_len - len(target_ids) : ex_len] and injects the delta at
                # lookup_idxs (computed against the unpadded prompt). Both
                # assume right-padding. Gemma-3-IT's tokenizer defaults to
                # left-padding, which misaligned both the target and the delta
                # and was the root cause of ~0 rewrite_acc on Gemma.
                LOG.info('AutoRegressive Model detected, set the padding side of Tokenizer to right...')
                self.tok.padding_side = 'right'
        else:
            self.model, self.tok = self.model_name

        if hparams.model_parallel:
            _dev_str = str(self.model.device)
            hparams.device = int(_dev_str.split(":")[1]) if ":" in _dev_str else 0
            #print(f"CUDA:0 Memory Allocated: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
            #print(f"CUDA:1 Memory Allocated: {torch.cuda.memory_allocated(1) / 1024 ** 3:.2f} GB")
            #self.accelerator = Accelerator()
            #self.model.to(self.accelerator.device)
            #self.model = self.accelerator.prepare(self.model)
        if not hparams.model_parallel and hasattr(hparams, 'device'):
            self.model.to(f'cuda:{hparams.device}')

        # PEFT adapter loading (HF path with pre-trained LoRA weights)
        if getattr(hparams, 'adapter_path', None):
            from peft import PeftModel
            LOG.info(f"Loading PEFT adapter from {hparams.adapter_path}")
            self.model = PeftModel.from_pretrained(
                self.model, hparams.adapter_path, is_trainable=False
            )

        # Enable Hemonc_uncleaned Parallel if specified
        #if hasattr(hparams, 'data_parallel') and hparams.data_parallel:
        #    LOG.info("Wrapping model in DataParallel for multi-GPU training...")
        #    self.model = torch.nn.DataParallel(self.model)

        print(
            f"Model {self.model_name} instantiated on device {self.model.device} (model parallel: {hparams.model_parallel})")
        # check the dtype the llm is using
        if hasattr(self.model, 'dtype'):
            LOG.info(f"Model dtype: {self.model.dtype}")

        # ── Optional: dual-load a vLLM engine for EVAL-time generation only. ──
        # When enabled, editing still uses self.model (HF); vLLM mirrors the
        # edited HF weights via _sync_vllm_from_hf() after every apply_algo
        # call. Only compatible for methods that leave a plain state_dict
        # after editing (MEMIT, AlphaEdit, LoRA-Merge, O-LoRA, SEEKR, IKE);
        # GRACE/MEMOIR/WISE wrap forward at inference and must keep this off.
        if getattr(hparams, 'eval_use_vllm', False):
            # vLLM >= 0.8 defaults to the v1 engine, which spawns the model in
            # a SEPARATE process (EngineCoreProc). That breaks in-process
            # weight sync — direct attribute traversal can't reach the model
            # across process boundaries. Force v0's sync in-process engine so
            # _get_vllm_inner_model() can reach .load_weights(). `setdefault`
            # lets callers override via the env var if needed.
            import os as _os
            _os.environ.setdefault("VLLM_USE_V1", "0")
            from vllm import LLM
            _is_gemma3 = 'gemma-3' in self.model_name.lower() or 'gemma3' in self.model_name.lower()
            _vllm_dtype = "bfloat16" if _is_gemma3 else "auto"
            _gpu_mem = getattr(hparams, 'vllm_eval_gpu_memory_utilization', 0.90)
            _max_len = getattr(hparams, 'vllm_eval_max_model_len', 2048)
            # Sleep mode: at most one model resident on GPU at a time. HF swaps
            # to CPU while vLLM generates; vLLM sleeps while HF edits. Required
            # on 40GB cards; optional-but-fine elsewhere.
            self._vllm_sleep_mode = bool(getattr(hparams, 'eval_use_vllm_sleep_mode', True))
            LOG.info(f"Loading companion vLLM engine for eval-time generation "
                     f"(gpu_memory_utilization={_gpu_mem}, max_model_len={_max_len}, "
                     f"VLLM_USE_V1={_os.environ.get('VLLM_USE_V1')}, "
                     f"sleep_mode={self._vllm_sleep_mode})")
            _llm_kwargs = dict(
                model=self.model_name,
                dtype=_vllm_dtype,
                gpu_memory_utilization=_gpu_mem,
                max_model_len=_max_len,
                max_num_seqs=getattr(hparams, 'vllm_max_num_seqs', 256),
                enforce_eager=_is_gemma3,
            )
            if self._vllm_sleep_mode:
                # enable_sleep_mode keeps weight buffers allocation-ready so
                # wake_up() doesn't need to reload from disk.
                _llm_kwargs['enable_sleep_mode'] = True
            self._vllm_asleep = False
            self._vllm_state = 'edit'  # {'edit', 'eval'}
            self._hf_device = next(self.model.parameters()).device

            # Bootstrap order in sleep mode: vLLM wants ~90% of the GPU to
            # load. If HF is still resident (another ~8–16 GB), vLLM fails
            # to allocate its KV cache. Swap HF → CPU *before* LLM() and
            # bring it back after vLLM sleeps.
            if self._vllm_sleep_mode:
                import torch as _torch
                LOG.info("[eval_use_vllm] moving HF→CPU before vLLM init to free GPU")
                self.model.to('cpu')
                _torch.cuda.empty_cache()

            self.vllm_model = LLM(**_llm_kwargs)
            LOG.info("vLLM companion engine loaded")

            if self._vllm_sleep_mode:
                # Sleep vLLM so the GPU frees back up for HF editing.
                self._vllm_sleep()
                # Return HF to GPU for the edit step. (vLLM and HF loaded
                # the same weights from the same path — first sync will
                # happen lazily inside _ensure_eval_mode() when a real edit
                # has been applied.)
                self._hf_to_gpu()
            else:
                # Legacy co-resident mode — sync once so pre-edit eval on the
                # first increment reflects HF baseline.
                self._sync_vllm_from_hf()

        self.hparams = hparams

    # ------------------------------------------------------------------
    # vLLM weight-sync helpers (used only when eval_use_vllm is True)
    # ------------------------------------------------------------------
    def _get_vllm_inner_model(self):
        """Return the underlying vLLM model object that exposes load_weights.

        vLLM's internal attribute path changes across minor versions; probe
        a set of known locations (v0 sync engine + a few v1 in-process
        variants) and dump the engine structure if none work.
        """
        if self.vllm_model is None:
            return None
        engine = self.vllm_model.llm_engine
        # Dotted paths relative to llm_engine to try, in preference order.
        candidate_paths = [
            # v0 sync engine (forced by VLLM_USE_V1=0)
            "model_executor.driver_worker.model_runner.model",
            "model_executor.driver_worker.worker.model_runner.model",
            "model_executor.model_runner.model",
            # v1 in-process engine variants (best-effort)
            "engine_core.model_executor.driver_worker.model_runner.model",
            "engine_core.engine_core.model_executor.driver_worker.model_runner.model",
            "engine_core.executor.driver_worker.model_runner.model",
        ]
        for path in candidate_paths:
            obj = engine
            ok = True
            for attr in path.split('.'):
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok and hasattr(obj, 'load_weights'):
                LOG.info(f"[eval_use_vllm] inner model at llm_engine.{path}")
                return obj
        # Diagnostic dump to help fix path for this vLLM version.
        top_attrs = [a for a in dir(engine) if not a.startswith('_')]
        inner = ''
        for name in ('engine_core', 'model_executor'):
            if hasattr(engine, name):
                inner_obj = getattr(engine, name)
                inner += f"\n  llm_engine.{name} attrs: " + str(
                    [a for a in dir(inner_obj) if not a.startswith('_')]
                )
        raise RuntimeError(
            "Could not locate vLLM load_weights API on this vLLM version.\n"
            f"  llm_engine attrs: {top_attrs}{inner}\n"
            "  Try running with env VLLM_USE_V1=0 (forces v0 sync engine)."
        )

    def _sync_vllm_from_hf(self):
        """Push the current HF state_dict into the companion vLLM engine."""
        if self.vllm_model is None or self.model is None:
            return
        import time as _time
        t0 = _time.time()
        inner = self._get_vllm_inner_model()
        inner.load_weights(self.model.state_dict().items())
        LOG.info(f"[eval_use_vllm] synced {sum(1 for _ in self.model.state_dict())} "
                 f"tensors HF→vLLM in {_time.time()-t0:.1f}s")

    # ------------------------------------------------------------------
    # Sleep/wake state machine (only one model on GPU at a time)
    # ------------------------------------------------------------------
    def _vllm_sleep(self):
        """Release vLLM GPU memory (weights offloaded to CPU, KV discarded)."""
        if self.vllm_model is None or self._vllm_asleep:
            return
        import time as _time, torch as _torch
        t0 = _time.time()
        self.vllm_model.sleep(level=1)
        _torch.cuda.empty_cache()
        self._vllm_asleep = True
        LOG.info(f"[eval_use_vllm] vLLM sleep in {_time.time()-t0:.1f}s")

    def _vllm_wake(self):
        """Restore vLLM to GPU (weights back + KV cache reallocated)."""
        if self.vllm_model is None or not self._vllm_asleep:
            return
        import time as _time, torch as _torch
        # PyTorch's caching allocator can hold GB of reserved-but-unused
        # memory after HF's edit or checkpoint.save.  vLLM's CuMemAllocator
        # asks the driver for contiguous pages and OOMs if the cache hasn't
        # been returned.  Forcing empty_cache (+ IPC collect) before wake is
        # cheap (~100 ms) and prevents the OOM we saw on 2nd wake per inc.
        _torch.cuda.empty_cache()
        _torch.cuda.ipc_collect()
        t0 = _time.time()
        self.vllm_model.wake_up()
        self._vllm_asleep = False
        LOG.info(f"[eval_use_vllm] vLLM wake in {_time.time()-t0:.1f}s")

    def _hf_to_cpu(self):
        if self.model is None:
            return
        import time as _time, torch as _torch
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            return
        if dev.type == 'cpu':
            return
        t0 = _time.time()
        self.model.to('cpu')
        _torch.cuda.empty_cache()
        LOG.info(f"[eval_use_vllm] HF→CPU in {_time.time()-t0:.1f}s")

    def _hf_to_gpu(self):
        if self.model is None:
            return
        import time as _time
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            return
        if dev.type == 'cuda':
            return
        t0 = _time.time()
        self.model.to(self._hf_device)
        LOG.info(f"[eval_use_vllm] HF→GPU in {_time.time()-t0:.1f}s")

    def _ensure_eval_mode(self):
        """Switch to eval: HF → CPU, wake vLLM, push latest weights.

        No-op if eval_use_vllm is off or sleep_mode is disabled, or if we are
        already in eval mode.
        """
        if self.vllm_model is None or not getattr(self, '_vllm_sleep_mode', False):
            return
        if getattr(self, '_vllm_state', 'edit') == 'eval':
            return
        self._hf_to_cpu()
        self._vllm_wake()
        # Sync latest HF weights → vLLM (HF is on CPU; load_weights transfers).
        self._sync_vllm_from_hf()
        self._vllm_state = 'eval'

    def _ensure_edit_mode(self):
        """Switch to edit: sleep vLLM, HF → GPU."""
        if self.vllm_model is None or not getattr(self, '_vllm_sleep_mode', False):
            return
        if getattr(self, '_vllm_state', 'edit') == 'edit':
            return
        self._vllm_sleep()
        self._hf_to_gpu()
        self._vllm_state = 'edit'

    def edit(self,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs: Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             sequential_edit=False,
             verbose=True,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        test_generation = kwargs.pop('test_generation', False)

        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts, ], [target_new, ]

        if hasattr(self.hparams, 'batch_size') and not BatchEditor.is_batchable_method(
                self.alg_name):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'

        if ground_truth is not None:
            ground_truth = [ground_truth, ] if isinstance(ground_truth, str) else ground_truth
        else:  # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>'] * (len(prompts))

        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            requests = _prepare_requests(prompts, target_new, ground_truth, rephrase_prompts, locality_inputs,
                                         portability_inputs, **kwargs)

        return self.edit_requests(requests, sequential_edit, verbose, test_generation=test_generation, **kwargs)

    def edit_requests(self,
                      requests,
                      sequential_edit=False,
                      verbose=True,
                      test_generation=False,
                      **kwargs
                      ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        eval_metric = kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
        few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            assert self.hparams.batch_size == 1, 'Single Editing: batch_size should be set to 1'
        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        else:
            for i, request in enumerate(tqdm(requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    metrics = {
                        "pre": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                        request, self.hparams.device, pre_edit=True)}
                else:
                    metrics = {"pre": compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, request,
                                                           self.hparams.device, eval_metric=eval_metric,
                                                           test_generation=test_generation,
                                                           few_shot_examples=few_shot_examples)}
                all_metrics.append(metrics)
            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        def edit_func(request):
            if self.alg_name == 'IKE':
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
            else:
                edited_model, weights_copy = self.apply_algo(
                    self.model,
                    self.tok,
                    [request],
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=False,
                    train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                )
                icl_examples = None
            return edited_model, weights_copy, icl_examples

        def edit_evaluation(all_metrics, request, edited_model, idx, test_generation, icl_examples, **kwargs):
            eval_metric = kwargs['eval_metric'] if 'eval_metric' in kwargs.keys() else 'exact match'
            few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
            if self.alg_name == 'IKE':
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, icl_examples,
                                                     request, self.hparams.device),
                })
            else:
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request,
                    "post": compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request,
                                                 self.hparams.device, eval_metric=eval_metric,
                                                 test_generation=test_generation, few_shot_examples=few_shot_examples)
                })
                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(
                        compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,
                                            metric_kwargs=kwargs["metric_kwargs"][idx], device=self.hparams.device))
                if 'locality' in all_metrics[idx]['post'].keys():
                    for locality_key in request['locality'].keys():
                        locality_result = []
                        for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'],
                                              all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                            locality_result.append(np.mean(np.equal(ans, label)))
                        all_metrics[idx]['post']['locality'][f'{locality_key}_acc'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                    all_metrics[idx]['pre'].pop('locality')

            if verbose:
                LOG.info(f"{idx} editing: {request['prompt']} -> {request['target_new']}  \n\n {all_metrics[idx]}")

        if sequential_edit:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
            for i, request in enumerate(requests):
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
        else:
            for i, request in enumerate(tqdm(requests, total=len(requests))):
                edited_model, weights_copy, icl_examples = edit_func(request)
                edit_evaluation(all_metrics, request, edited_model, i, test_generation, icl_examples, **kwargs)
                if self.alg_name == 'KN' or self.alg_name == 'GRACE' or self.alg_name == 'WISE':
                    with torch.no_grad():
                        weights_copy()
                elif self.alg_name == 'LoRA':
                    edited_model.unload()
                    del self.model.peft_config
                elif self.alg_name == 'MELO':
                    self.model = edited_model
                else:
                    with torch.no_grad():
                        for k, v in weights_copy.items():
                            nethook.get_parameter(self.model, k)[...] = v.to(f"cuda:{self.hparams.device}")

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        if len(all_metrics) != 0:
            summary_metrics(all_metrics)

        return all_metrics, edited_model, weights_copy


class LifelongEditor(BaseEditor):
    def __init__(self,
                 hparams: HyperParams,
                 ):
        super().__init__(hparams)
        self.qa_benchmarks = {}

    def eval_samples(
            self,
            prompts: List[str],
            target_new: List[str],
            tags: Optional[List[str]] = None,
            locality_inputs: Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            ground_truth: Optional[List[str]] = None,
            rephrase_prompts: Optional[List[str]] = None,
            edited_model: Optional[nn.Module] = None,
            batch_size: int = 1,
            **kwargs
    ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        assert len(prompts) == len(target_new)
        few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
        judge_model = kwargs['judge_model'] if 'judge_model' in kwargs.keys() else None
        judge_workers = kwargs.get('judge_workers', 4)
        judge_enabled = kwargs['judge_enabled']
        print(f'judge_model: {judge_model}, judge_workers: {judge_workers}, judge_enabled: {judge_enabled}')
        avg_edit_decay = kwargs['avg_edit_decay'] if 'avg_edit_decay' in kwargs.keys() else None
        indices = kwargs['indices'] if 'indices' in kwargs.keys() else None
        if not edited_model:
            print("No edited model found, using the original model for evaluation.")
            edited_model = self.model
        # Some editing methods return a wrapper that's a bare nn.Module (no
        # .generate), e.g. MEMOIR. compute_edit_quality calls model.generate,
        # which would AttributeError. Fall back to self.model (the underlying
        # HF causal LM whose layers the wrapper has already hooked in-place).
        if edited_model is not None and not hasattr(edited_model, 'generate') and self.model is not None:
            LOG.info(f"[eval_samples] edited_model ({type(edited_model).__name__}) has no .generate(); "
                     f"falling back to self.model for eval-time generation")
            edited_model = self.model

        tags = tags if tags is not None else [''] * len(prompts)
        requests = _prepare_requests(prompts, target_new, ground_truth, tags, rephrase_prompts,
                                     locality_inputs, portability_inputs, **kwargs)

        batched_requests = [requests[i * batch_size: min((i + 1) * batch_size, len(requests))]
                            for i in range(math.ceil(len(requests) / batch_size))]
        all_metrics = []
        # Sentinel / past-eval pool runs with vLLM when available.
        self._ensure_eval_mode()
        for requests in tqdm(batched_requests):
            past_metrics = compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, requests,
                                                self.hparams.device, test_generation=False,
                                                few_shot_examples=few_shot_examples, judge_model=judge_model,
                                                judge_workers=judge_workers, judge_enabled=judge_enabled,
                                                vllm_model=self.vllm_model, lora_request=self.lora_request,
                                                openrouter_client=self.openrouter_client,
                                                openrouter_model=self.openrouter_model_name,
                                                openrouter_thinking_budget=self.openrouter_thinking_budget)

            all_metrics.extend([{'past': pm} for pm in past_metrics])
        # Return to edit mode for the next increment's edit step.
        self._ensure_edit_mode()

        metric_names = [['past', 'rewrite_acc'],
                        ['past', 'locality', 'neighborhood_acc'],
                        ['past', 'portability', 'mhop', 'performance', 'acc'],
                        ['past', 'portability', 'genv2_mixed', 'performance', 'acc'],
                        ['past', 'rephrase_acc']]

        #print(f"Evaluation results: \n{all_metrics}")

        mean_log = {f'{"_".join(k)}_mean': [] for k in metric_names}

        for j, m in enumerate(all_metrics):
            for k in metric_names:
                try:
                    mean_log[f'{"_".join(k)}_mean'].append(extract_metric(m, k))
                except KeyError:
                    pass

        out_metrics = {k: np.mean(v) for k, v in mean_log.items()}

        return out_metrics, all_metrics

    def eval_general_capabilities(self, args: argparse.Namespace):
        start_time = time()
        datasets = args.qa.base_eval.datasets
        metrics = {}
        for dataset in datasets:
            if self.qa_benchmarks is not None and dataset in self.qa_benchmarks.keys():
                questions, answers = self.qa_benchmarks[dataset]
            else:
                questions, answers = load_dataset(dataset, ds_size=args.qa.base_eval.ds_size,
                                                  ds_seed=args.qa.base_eval.ds_seed,
                                                  fs_ex=args.qa.base_eval.fs_examples,
                                                  data_dir=os.path.join(args.data_dir, 'qa_benchmarks'))
                self.qa_benchmarks[dataset] = (questions, answers)
            tags = [''] * len(questions)
            requests = _prepare_requests(questions, answers, ground_truth=answers, tags=tags)
            batched_requests = [requests[i * args.batch_size: min((i + 1) * args.batch_size, len(requests))]
                                for i in range(math.ceil(len(requests) / args.batch_size))]

            dataset_metrics = []
            for requests in tqdm(batched_requests):
                dataset_metrics.extend(
                    compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, requests,
                                         self.hparams.device, test_generation=False))

            metrics[dataset] = np.mean([metric['rewrite_acc'] for metric in dataset_metrics])

        metrics['mean'] = np.mean(list(metrics.values()))
        print(f"General Capabilities Evaluation took {time() - start_time} seconds")
        return metrics

    def edit(self,
             args: argparse.Namespace,
             prompts: Union[str, List[str]],
             target_new: Union[str, List[str]],
             ground_truth: Optional[Union[str, List[str]]] = None,
             tags: Optional[Union[str, List[str]]] = None,
             rephrase_prompts: Optional[Union[str, List[str]]] = None,
             locality_inputs: Optional[Dict] = None,
             portability_inputs: Optional[Dict] = None,
             keep_original_weight=False,
             verbose=True,
             summary_metrics=False,
             **kwargs
             ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        seed_everything(args.seed)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
        past_eval_interval = kwargs['past_eval_interval'] if 'past_eval_interval' in kwargs.keys() else 1
        avg_edit_decay = {}
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts, ], [target_new, ]

        #if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
        #    self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth, ]
            else:
                assert len(ground_truth) == len(prompts)
        else:  # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            tags = tags if tags is not None else [''] * len(prompts)
            requests = _prepare_requests(prompts, target_new, ground_truth, tags, rephrase_prompts,
                                         locality_inputs, portability_inputs, **kwargs)

        if getattr(args.qa, 'eval_max_samples', None):
            eval_requests = subset_requests(requests, args.qa.eval_max_samples, seed=args.ds_seed)
        else:
            eval_requests = requests
            random.seed(args.ds_seed)
            random.shuffle(eval_requests)

        batched_requests = [requests[i * args.batch_size: min((i + 1) * args.batch_size, len(requests))]
                            for i in range(math.ceil(len(requests) / args.batch_size))]
        if hasattr(args, 'eval_batch_size'):
            batched_eval_requests = [eval_requests[i * args.eval_batch_size: min((i + 1) * args.eval_batch_size, len(eval_requests))]
                                        for i in range(math.ceil(len(eval_requests) / args.eval_batch_size))]
        else:
            batched_eval_requests = batched_requests

        # if not os.path.exists(RESULTS_DIR):
        #     os.mkdir(RESULTS_DIR)
        # base_case_path = RESULTS_DIR / self.hparams_fname.rsplit('.', 1)[0]
        # if not os.path.exists(base_case_path):
        #     os.mkdir(base_case_path)
        # print(f"Results will be stored at {base_case_path}")

        if self.alg_name == 'FT-Api':
            all_metrics = []
            for i, request in enumerate(requests):
                metrics = {
                    "pre": {}
                }
                all_metrics.append(metrics)

            start = time()
            edited_model, weights_copy = self.apply_algo(
                requests,
                self.hparams
            )
            exec_time = time() - start

            LOG.info(f"Execution editing took {exec_time}")

            for i, request in enumerate(requests):
                all_metrics[i].update({
                    'case_id': i,
                    "requested_rewrite": request,
                    "time": exec_time,
                    "post": {}
                })

                if verbose:
                    LOG.info(
                        f"{i} editing: {request['prompt']} -> {request['target_new']}  \n {all_metrics[i]}"
                    )
            return all_metrics, edited_model, weights_copy

        all_metrics = []
        logging.info(f"Pre Edit Evaluation")
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        elif False: #getattr(args, 'debug', False):
            # Skip pre-edit evaluation in debug mode to save time and LLM credits.
            # Populate all_metrics with empty pre-entries so post-eval indexing works.
            LOG.info("Debug mode: skipping pre-edit evaluation")
            all_metrics = [{"pre": {}} for _ in range(len(requests))]
        else:
            print(f'Evaluate {len(batched_eval_requests)} batches')
            # Transition into eval mode (HF→CPU, wake vLLM, sync) before the
            # pre-eval loop.  No-op if eval_use_vllm is off or sleep mode is
            # disabled.
            self._ensure_eval_mode()
            for i, batch in enumerate(tqdm(batched_eval_requests)):
                if self.alg_name == 'IKE':
                    assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                    pre_metrics = [
                        compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok, [''],
                                                 record, self.hparams.device, pre_edit=True)
                        for record in batch
                    ]
                else:
                    #pre_metrics = [{} for _ in range(len(batch))]
                    pre_metrics = compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, batch,
                                                       self.hparams.device, test_generation=test_generation,
                                                       few_shot_examples=few_shot_examples,
                                                       vllm_model=self.vllm_model,
                                                       lora_request=self.lora_request)

                for b, p_m in enumerate(pre_metrics):
                    all_metrics.append({
                        "pre": p_m
                    })
            # Pre-eval done; return HF to GPU and sleep vLLM so the edit step
            # has the GPU to itself.
            self._ensure_edit_mode()

        _SINGLE_SHOT_METHODS = {'LoRA', 'LoRA-Merge', 'O-LoRA', 'SEEKR'}
        logging.info(f"Editing")
        if self.alg_name in _SINGLE_SHOT_METHODS:
            start = time()
            edited_model, weights_copy = self.apply_algo(
                self.model,
                self.tok,
                requests,
                self.hparams,
                copy=False,
                return_orig_weights=True,
                keep_original_weight=keep_original_weight,
                train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
            )
            self.model = edited_model
            exec_time = time() - start
            # In legacy co-resident mode (no sleep), sync right after the edit.
            # In sleep mode, sync happens lazily inside _ensure_eval_mode()
            # when the next eval runs — keeps vLLM GPU memory unallocated.
            if getattr(self.hparams, 'eval_use_vllm', False) and not getattr(self, '_vllm_sleep_mode', False):
                self._sync_vllm_from_hf()
        for i, request in enumerate(tqdm(batched_requests)):
            start = time()
            if self.alg_name == 'IKE':
                assert 'train_ds' in kwargs.keys(), print('IKE need train_ds(For getting In-Context prompt)')
                edited_model, weights_copy, icl_examples = self.model, {}, self.apply_algo(
                    self.model,
                    self.tok,
                    request,
                    self.hparams,
                    copy=False,
                    return_orig_weights=True,
                    keep_original_weight=keep_original_weight,
                    train_ds=kwargs['train_ds']
                )
                exec_time = time() - start
                LOG.info(f"Execution {i} editing took {exec_time}")
            elif self.alg_name in _SINGLE_SHOT_METHODS:
                pass
            else:
                if not (args.qa.filter_edits and all_metrics[i]['pre']['rewrite_acc'][0] == 1.0):
                    edited_model, weights_copy = self.apply_algo(
                        self.model,
                        self.tok,
                        request,
                        self.hparams,
                        copy=False,
                        return_orig_weights=True,
                        keep_original_weight=keep_original_weight,
                        train_ds=kwargs['train_ds'] if self.alg_name == 'IKE' else None
                    )
                    exec_time = time() - start
                    LOG.info(f"Execution {i} editing took {exec_time}")
                    self.model = edited_model.model if self.hparams.alg_name in ['WISE', 'GRACE'] else self.model
                    # Legacy co-resident mode: sync immediately.  Sleep mode
                    # defers sync until _ensure_eval_mode() below.
                    if getattr(self.hparams, 'eval_use_vllm', False) and not getattr(self, '_vllm_sleep_mode', False):
                        self._sync_vllm_from_hf()
            if verbose:
                LOG.info(
                    f"{i} editing: {[r['prompt'] for r in request]} -> {[r['target_new'] for r in request]}"
                )

            if hasattr(args, 'eval_batch_size'):
                batched_eval_requests = [
                    request[i * args.eval_batch_size: min((i + 1) * args.eval_batch_size, len(request))]
                    for i in range(math.ceil(len(request) / args.eval_batch_size))]
            else:
                batched_eval_requests = [request, ]

            # Transition into eval mode for post-edit eval. HF→CPU, wake
            # vLLM, sync the just-edited weights. No-op in legacy mode.
            self._ensure_eval_mode()
            post_metrics = []
            for eval_batch in batched_eval_requests:
                if self.alg_name == 'IKE':
                    for record in eval_batch:
                        post_metrics.append(compute_icl_edit_quality(self.model, self.model_name, self.hparams, self.tok,
                                                                icl_examples, record, self.hparams.device))
                else:
                    post_metrics.extend(compute_edit_quality(self.model, self.model_name, self.hparams, self.tok, eval_batch,
                                                        self.hparams.device, test_generation=test_generation,
                                                        few_shot_examples=few_shot_examples,
                                                        vllm_model=self.vllm_model,
                                                        lora_request=self.lora_request))
            # Return to edit mode so the next per-request edit has the GPU.
            self._ensure_edit_mode()

            for b, p_m in enumerate(post_metrics):
                idx = i * args.batch_size + b
                # Guard against index mismatch: if a pre-edit batch returned fewer
                # metrics than expected (e.g. empty compute_edit_quality result for
                # one batch), all_metrics can be shorter than idx+1. Extend with an
                # empty pre-entry so we don't lose the post-eval data.
                if idx >= len(all_metrics):
                    LOG.warning(
                        f"all_metrics[{idx}] out of range "
                        f"(len={len(all_metrics)}, i={i}, b={b}, batch_size={args.batch_size}). "
                        "Pre-eval entry missing — appending empty placeholder."
                    )
                    while len(all_metrics) <= idx:
                        all_metrics.append({"pre": {}})
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request[b] if b < len(request) else request[-1],
                    "time": exec_time,
                    "post": p_m
                })

                if 'locality' in all_metrics[idx]['post'].keys():
                    for locality_key in request[b if b < len(request) else -1]['locality'].keys():
                        # compute_edit_quality already scores locality internally and
                        # stores it as a nested dict under locality_key (no _output key).
                        # Only run the raw-output → score conversion when _output keys
                        # are present (legacy path used by some algorithms).
                        if f'{locality_key}_output' not in all_metrics[idx]['post']['locality']:
                            continue
                        assert len(all_metrics[idx]['post']['locality'][f'{locality_key}_output']) == \
                               len(all_metrics[idx]['pre']['locality'][f'{locality_key}_output'])
                        locality_result = []
                        for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'],
                                              all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                            try:
                                locality_result.append(np.mean(np.equal(ans, label)))
                            except ValueError as e:
                                print(f'Error in {idx} - {locality_key} - {ans} - {label}')
                                print(all_metrics[idx])
                                raise e
                        all_metrics[idx]['post']['locality'][f'{locality_key}_score'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                        all_metrics[idx]['pre']['locality'].pop(f'{locality_key}_output')

                if idx == len(requests) - 1:
                    if args.qa.base_eval:
                        all_metrics[idx]['general'] = self.eval_general_capabilities(args)
                    compute_mean = True
                else:
                    compute_mean = False
                metrics_to_wandb(all_metrics, idx, avg_edit_decay=None, compute_mean=compute_mean)

        if isinstance(edited_model, LORA):
            edited_model = edited_model.model
        # for melo

        if summary_metrics and len(all_metrics) != 0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics, ]
            logs_dir = './logs'
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            output_file = os.path.join(logs_dir, 'results.json')
            with open(output_file, 'w') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=4)

            mean_metrics = dict()
            for eval in ["pre", "post"]:
                mean_metrics[eval] = dict()
                for key in ["rewrite_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval].keys():
                        mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
                for key in ["locality", "portability"]:
                    if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        for lkey in all_metrics[0][eval][key].keys():
                            if lkey.endswith("acc"):
                                mean_metrics[eval][key][lkey] = np.mean(
                                    [metric[eval][key][lkey] for metric in all_metrics])
            mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])

            print("Metrics Summary: ", mean_metrics)

        return all_metrics, edited_model, weights_copy

    def rag(self,
            args: argparse.Namespace,
            prompts: Union[str, List[str]],
            target_new: Union[str, List[str]],
            ground_truth: Optional[Union[str, List[str]]] = None,
            tags: Optional[Union[str, List[str]]] = None,
            rephrase_prompts: Optional[Union[str, List[str]]] = None,
            locality_inputs: Optional[Dict] = None,
            portability_inputs: Optional[Dict] = None,
            edited_model=None,
            keep_original_weight=False,
            verbose=True,
            summary_metrics=False,
            **kwargs
            ):
        """
        `prompts`: list or str
            the prompts to edit
        `ground_truth`: str
            the ground truth / expected output
        `locality_inputs`: dict
            for locality
        """
        seed_everything(args.seed)
        test_generation = kwargs['test_generation'] if 'test_generation' in kwargs.keys() else False
        few_shot_examples = kwargs['few_shot_examples'] if 'few_shot_examples' in kwargs.keys() else None
        past_eval_interval = kwargs['past_eval_interval'] if 'past_eval_interval' in kwargs.keys() else 1
        # Pre-batch corpus: records before the experiment's first increment.
        # Passed on every rag() call but only consumed on the first (when edited_model is None).
        initial_corpus_records = kwargs.get('initial_corpus_records', None)

        # Remember whether this is the first increment (edited_model not yet built
        # and no adapt() calls have happened). Used below to skip retrieval on the
        # pre-eval of the very first batch — matching how editing methods baseline
        # against the untouched base model.
        is_first_increment = edited_model is None

        if edited_model is None:
            if self.alg_name == 'RAG':
                edited_model = RAG(self.hparams, self.model, self.tok, self.hparams.device)
            elif self.alg_name == 'OracleRAG':
                edited_model = OracleRAG(self.hparams, self.model, self.tok, self.hparams.device,
                                         initial_corpus_records=initial_corpus_records)
            elif self.alg_name == 'BM25RAG':
                edited_model = BM25RAG(self.hparams, self.model, self.tok, self.hparams.device,
                                       initial_corpus_records=initial_corpus_records)
            elif self.alg_name == 'DenseRAG':
                edited_model = DenseRAG(self.hparams, self.model, self.tok, self.hparams.device,
                                        initial_corpus_records=initial_corpus_records)
            elif self.alg_name == 'IKE':
                edited_model = IKEWrapper(self.hparams, self.model, self.tok, self.hparams.device,
                                          train_ds=kwargs.get('train_ds'))
            elif self.alg_name == 'PTuning':
                edited_model = PromptTuning(self.hparams, self.model, self.tok, self.hparams.device)

        avg_edit_decay = {}
        if isinstance(prompts, List):
            assert len(prompts) == len(target_new)
        else:
            prompts, target_new = [prompts, ], [target_new, ]

        if hasattr(self.hparams, 'batch_size'):  # For Singleton Editing, bs=1
            self.hparams.batch_size = 1

        if ground_truth is not None:
            if isinstance(ground_truth, str):
                ground_truth = [ground_truth, ]
            else:
                assert len(ground_truth) == len(prompts)
        else:  # Default ground truth is <|endoftext|>
            ground_truth = ['<|endoftext|>' for _ in range(len(prompts))]

        # assert (locality_prompts is None and locality_ground_truth is None) or \
        #        (isinstance(locality_prompts, str) and isinstance(locality_ground_truth, str)) or \
        #        len(locality_prompts) == len(locality_ground_truth) or print('Error in locality Input.')
        if "requests" in kwargs.keys():
            requests = kwargs["requests"]
        else:
            tags = tags if tags is not None else [''] * len(prompts)
            requests = _prepare_requests(prompts, target_new, ground_truth, tags, rephrase_prompts,
                                         locality_inputs, portability_inputs, **kwargs)

        if getattr(args.qa, 'eval_max_samples', None):
            eval_requests = subset_requests(requests, args.qa.eval_max_samples, seed=args.ds_seed)
        else:
            eval_requests = requests
            random.seed(args.ds_seed)
            random.shuffle(eval_requests)

        # Inject ground_truth_statement (full descriptive statement, e.g.
        # "Regimen A inferior to Regimen B for Condition X [endpoint: PFS]")
        # into each request so RAG adapt() methods can store it for retrieve_target.
        _gts_list = kwargs.get('ground_truth_statements', None)
        if _gts_list is not None:
            for req, gts in zip(requests, _gts_list):
                req['ground_truth_statement'] = gts

        batched_requests = [requests[i * args.batch_size: min((i + 1) * args.batch_size, len(requests))]
                            for i in range(math.ceil(len(requests) / args.batch_size))]
        batched_eval_requests = [eval_requests[i * args.batch_size: min((i + 1) * args.batch_size, len(eval_requests))]
                                    for i in range(math.ceil(len(eval_requests) / args.batch_size))]

        all_metrics = []
        if 'pre_edit' in kwargs and kwargs['pre_edit'] is not None:
            metrics = kwargs['pre_edit']
            all_metrics = metrics
        elif False: #getattr(args, 'debug', False):
            # Skip pre-edit evaluation in debug mode to save time and LLM credits.
            LOG.info("Debug mode: skipping pre-edit evaluation")
            all_metrics = [{"pre": {}} for _ in range(len(requests))]
        else:
            # First-increment pre-eval runs without retrieval so the baseline row
            # matches what editing methods see (untouched base model). Later
            # increments DO use retrieval at pre-time — the corpus already holds
            # evidence from prior batches' adapt() calls, so pre reflects the
            # cumulative "RAG-so-far" state, analogous to how MEMIT's pre on
            # batch N uses the model edited through batches 1..N-1.
            pre_rag_model = None if is_first_increment else edited_model
            # Enter eval mode for pre-eval (no-op unless sleep mode is on).
            self._ensure_eval_mode()
            for i, batch in enumerate(tqdm(batched_eval_requests)):
                pre_metrics = compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, batch,
                                                   self.hparams.device, test_generation=test_generation,
                                                   few_shot_examples=few_shot_examples,
                                                   vllm_model=self.vllm_model,
                                                   rag_model=pre_rag_model)

                for b, p_m in enumerate(pre_metrics):
                    all_metrics.append({
                        "pre": p_m
                    })
            # adapt() (for RAG/IKE) doesn't update HF weights, but may touch
            # the retrieval index; safe to stay in eval mode.  For weight-
            # mutating methods this path isn't taken.

            if 'pre_file' in kwargs and kwargs['pre_file'] is not None:
                ### Store the pre_edit metric to refrain computing repeatedly
                json.dump(all_metrics, open(kwargs['pre_file'], 'w'), indent=4)

        LOG.info(f'Adapt')
        exec_times = edited_model.adapt(batched_requests)

        LOG.info(f'Evaluation')
        # Ensure eval mode for the post-eval loop (re-syncs weights if the
        # edit actually touched self.model).  In adapt-style methods it's a
        # no-op beyond the first transition.
        self._ensure_eval_mode()
        for i, request in enumerate(tqdm(batched_eval_requests)):
            post_metrics = compute_edit_quality(edited_model, self.model_name, self.hparams, self.tok, request,
                                                self.hparams.device, test_generation=test_generation,
                                                few_shot_examples=False,
                                                vllm_model=self.vllm_model,
                                                rag_model=edited_model)

            for b, p_m in enumerate(post_metrics):
                idx = i * args.batch_size + b
                # Guard against index mismatch (same as first edit path above)
                if idx >= len(all_metrics):
                    LOG.warning(
                        f"all_metrics[{idx}] out of range "
                        f"(len={len(all_metrics)}, i={i}, b={b}, batch_size={args.batch_size}). "
                        "Pre-eval entry missing — appending empty placeholder."
                    )
                    while len(all_metrics) <= idx:
                        all_metrics.append({"pre": {}})
                all_metrics[idx].update({
                    'case_id': idx,
                    "requested_rewrite": request[b] if b < len(request) else request[-1],
                    "time": exec_times[i],
                    "post": p_m
                })

                # Retrieval accuracy: 1 if the gold evidence was among the retrieved docs.
                # Recorded per-task so we can compare retrieval generalization across:
                #   - rewrite       (the canonical question)
                #   - rephrase      (paraphrased version of the same fact)
                #   - portability_* (genv2 variants asking the same fact differently)
                # Locality is intentionally excluded — its questions are about
                # *unrelated* facts so "gold evidence" isn't well-defined here.
                #
                # For each task, we log {hit, rank}: rank is the 1-indexed
                # position of the gold abstract within the top-k retrieved
                # list, or null if it isn't in top-k. This unlocks rank-distribution
                # plots (rank 1 vs 2 vs 3 vs miss) on top of plain recall@k.
                #
                # Legacy `post.retrieval_acc` is preserved (== retrieval_per_task.rewrite.hit)
                # so existing analysis scripts keep working.
                if hasattr(edited_model, 'retrieve_evidence'):
                    req_b = request[b] if b < len(request) else request[-1]
                    gold_evidence = req_b.get('ground_truth', '')
                    if gold_evidence and gold_evidence != '<|endoftext|>':
                        gold_norm = gold_evidence.strip()

                        def _retrieval_record(query: str):
                            """Return {hit: 0|1, rank: int|None} for one query.

                            rank is 1-indexed; None means the gold isn't in top-k.
                            """
                            try:
                                retrieved = edited_model.retrieve_evidence(query)
                            except Exception as e:
                                LOG.warning(f"retrieve_evidence failed on query "
                                            f"{query[:60]!r}: {e}")
                                return {"hit": 0, "rank": None}
                            for r, doc in enumerate(retrieved, start=1):
                                if doc and doc.strip() == gold_norm:
                                    return {"hit": 1, "rank": r}
                            return {"hit": 0, "rank": None}

                        # Build the (task_label, query) list for this case.
                        per_task_queries: list[tuple[str, str]] = [
                            ("rewrite", req_b['prompt']),
                        ]
                        rephrase_q = req_b.get('rephrase_prompt')
                        if rephrase_q:
                            per_task_queries.append(("rephrase", rephrase_q))
                        for port_key, port_val in (req_b.get('portability') or {}).items():
                            port_q = (port_val or {}).get('prompt')
                            if port_q:
                                per_task_queries.append(
                                    (f"portability_{port_key}", port_q)
                                )

                        per_task: dict[str, dict] = {}
                        for task_label, query in per_task_queries:
                            per_task[task_label] = _retrieval_record(query)

                        all_metrics[idx]['post']['retrieval_per_task'] = per_task
                        # Backwards-compatible scalar — equals rewrite hit.
                        all_metrics[idx]['post']['retrieval_acc'] = per_task["rewrite"]["hit"]

                if "metric_kwargs" in kwargs:
                    all_metrics[idx].update(
                        compute_sent_metric(self.model, edited_model, self.model_name, self.hparams, self.tok,
                                            metric_kwargs=kwargs["metric_kwargs"][i], device=self.hparams.device))
                #print(f'id: {idx} - {len(all_metrics)}')
                #avg_edit_decay[idx] = [all_metrics[idx]['post']['rewrite_acc']]

                if 'locality' in all_metrics[idx]['post'].keys():
                    for locality_key in request[b if b < len(request) else -1]['locality'].keys():
                        if f'{locality_key}_output' not in all_metrics[idx]['post']['locality']:
                            continue
                        assert len(all_metrics[idx]['post']['locality'][f'{locality_key}_output']) == \
                               len(all_metrics[idx]['pre']['locality'][f'{locality_key}_output'])
                        locality_result = []
                        for ans, label in zip(all_metrics[idx]['post']['locality'][f'{locality_key}_output'],
                                              all_metrics[idx]['pre']['locality'][f'{locality_key}_output']):
                            try:
                                locality_result.append(np.mean(np.equal(ans, label)))
                            except ValueError as e:
                                print(f'Error in {idx} - {locality_key} - {ans} - {label}')
                                locality_result.append(0)
                        all_metrics[idx]['post']['locality'][f'{locality_key}_score'] = locality_result
                        all_metrics[idx]['post']['locality'].pop(f'{locality_key}_output')
                        all_metrics[idx]['pre']['locality'].pop(f'{locality_key}_output')

            self.model = edited_model.model if self.hparams.alg_name in ['WISE', 'GRACE'] else self.model

            """
            if idx % args.qa.past_eval.interval == 0 or idx == len(prompts)-1:
                past_prompts = prompts[:idx + 1]
                past_target_new = target_new[:idx + 1]
                past_ground_truth = ground_truth[:idx + 1]
                past_rephrase_prompts = rephrase_prompts[:idx + 1]
                past_portability_inputs = {'mhop': {k: v[:idx + 1] for k, v in portability_inputs['mhop'].items()}} if portability_inputs is not None else None
                past_subject = kwargs['subject'][:idx + 1] if 'subject' in kwargs.keys() else None
                indices = range(idx + 1)
                if len(past_prompts) > args.qa.past_eval.max_samples and args.qa.past_eval.max_samples > 0 and not idx == len(prompts) - 1:
                    # Randomly sample `max_samples` from the past
                    indices = np.random.choice(len(past_prompts), args.qa.past_eval.max_samples, replace=False)
                    past_prompts = [past_prompts[i] for i in indices]
                    past_target_new = [past_target_new[i] for i in indices]
                    past_ground_truth = [past_ground_truth[i] for i in indices]
                    past_rephrase_prompts = [past_rephrase_prompts[i] for i in indices]
                    past_portability_inputs = [past_portability_inputs[i] for i in indices] if past_portability_inputs is not None else None
                    if past_subject is not None:
                        past_subject = [past_subject[i] for i in indices]

                all_metrics[i]['past'], avg_edit_decay = self.eval_samples(past_prompts, past_target_new,
                                                                           None, None,
                                                                           past_portability_inputs,
                                                                           past_ground_truth,
                                                                           past_rephrase_prompts,
                                                                           subject=past_subject,
                                                                           few_shot_examples=few_shot_examples,
                                                                           indices=indices,
                                                                           batch_size=args.batch_size)
            """

        if args.qa.base_eval:
            all_metrics[-1]['general'] = self.eval_general_capabilities(args)

        LOG.info(f'Evaluation complete, logging to wandb...')
        for idx in tqdm(range(len(all_metrics))):
            metrics_to_wandb(all_metrics, idx, avg_edit_decay=None, compute_mean=idx == len(all_metrics) - 1)

            if verbose:
                LOG.info(
                    f"{idx} editing: {requests[idx]['prompt']} -> {requests[idx]['target_new']}"#  \n {all_metrics[idx]}"
                )
        if summary_metrics and len(all_metrics) != 0:
            if isinstance(all_metrics, dict):
                all_metrics = [all_metrics, ]
            logs_dir = './logs'
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            output_file = os.path.join(logs_dir, 'results.json')
            with open(output_file, 'w') as f:
                json.dump(all_metrics, f, ensure_ascii=False, indent=4)

            mean_metrics = dict()
            for eval in ["pre", "post"]:
                mean_metrics[eval] = dict()
                for key in ["rewrite_acc", "rephrase_acc"]:
                    if key in all_metrics[0][eval].keys():
                        mean_metrics[eval][key] = np.mean([metric[eval][key] for metric in all_metrics])
                for key in ["locality", "portability"]:
                    if key in all_metrics[0][eval].keys() and all_metrics[0][eval][key] != {}:
                        mean_metrics[eval][key] = dict()
                        for lkey in all_metrics[0][eval][key].keys():
                            if lkey.endswith("acc"):
                                mean_metrics[eval][key][lkey] = np.mean(
                                    [metric[eval][key][lkey] for metric in all_metrics])
            mean_metrics["time"] = np.mean([metric["time"] for metric in all_metrics])

            #print("Metrics Summary: ", mean_metrics)

        # Return to edit mode before yielding control back to the caller, so
        # the next increment starts with HF on GPU and vLLM asleep.
        self._ensure_edit_mode()
        return all_metrics, edited_model


def subset_requests(requests, max_samples, seed=42):
    # fin all indices where porability mhop prompts are not nan
    mhop_indices, non_mhop_indices = [], []
    for i, r in enumerate(requests):
        if 'mhop' in r['portability'] and r['portability']['mhop']['prompt'] is not None:
            mhop_indices.append(i)
        else:
            non_mhop_indices.append(i)
    #indices = []  # TODO: remove this line

    np.random.seed(seed)
    if len(mhop_indices) > max_samples:
        indices = np.random.choice(mhop_indices, max_samples, replace=False)
    elif len(mhop_indices) < max_samples:
        indices = mhop_indices
        indices += np.random.choice(non_mhop_indices, max_samples - len(mhop_indices), replace=False).tolist()
    else:
        indices = mhop_indices
    #print(indices)
    return [requests[i] for i in indices]



