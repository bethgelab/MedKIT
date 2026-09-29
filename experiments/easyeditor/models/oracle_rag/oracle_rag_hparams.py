from dataclasses import dataclass
from ...util.hparams import HyperParams
from typing import Optional, Any
import yaml


@dataclass
class OracleRAGHyperParams(HyperParams):
    model_name: str
    alg_name: str
    device: int
    retrieve_target: bool = False

    # Optional pre-trained PEFT/LoRA adapter to load for evaluation (read by
    # LifelongEditor at load time).  Declared here so it survives
    # from_hparams()'s known-field filter below and can be injected via the
    # Hydra override `hparams.adapter_path=...` on an EVAL run (used by the
    # DPO/GRPO shared-eval driver).  NOTE: inject via hparams.adapter_path, NOT
    # the top-level args.adapter_path — the latter flips the results-filename
    # tag to `_finetuned_`.
    adapter_path: str = ''

    batch_size: int = 1
    max_length: int = 512
    max_out_len_closed: int = 50
    max_out_len_open: int = 1024
    model_parallel: bool = False
    # vLLM engine config — override per model when base context < 8192
    # (e.g. AdaptLLM-Medicine-Chat inherits Llama-2-7B's 4096 window).
    vllm_max_model_len: int = 8192

    @classmethod
    def from_hparams(cls, hparams_name_or_path=None, config=None):
        import dataclasses
        assert hparams_name_or_path or config, 'OracleRAGHyperParams requires either hparams_name_or_path or config'
        if hparams_name_or_path:
            if '.yaml' not in hparams_name_or_path:
                hparams_name_or_path = hparams_name_or_path + '.yaml'
            with open(hparams_name_or_path, 'r') as stream:
                config = yaml.safe_load(stream)
                config = super().construct_float_from_scientific_notation(config)
        # Filter to only known fields so this class can load any model config
        # (e.g. AlphaEdit or EVAL hparams) without failing on unknown keys.
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in config.items() if k in valid_fields})
