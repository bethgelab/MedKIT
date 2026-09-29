from dataclasses import dataclass, field
from typing import List
from ...util.hparams import HyperParams
import yaml


@dataclass
class OLoRAHyperParams(HyperParams):
    # Method
    lora_type: str
    layers: List[int]
    num_steps: int
    lr: float
    weight_decay: float
    kl_factor: float
    norm_constraint: float
    target_modules: List[str]
    rank: int
    lora_alpha: float
    lora_dropout: float

    device: int
    alg_name: str
    model_name: str
    fp16: bool

    # O-LoRA specific
    orth_lambda: float = 0.1

    # Defaults
    batch_size: int = 128
    max_length: int = 40
    model_parallel: bool = False
    bf16: bool = False
    max_out_len_open: int = 1024
    max_out_len_closed: int = 50

    @classmethod
    def from_hparams(cls, hparams_name_or_path=None, config=None):
        assert hparams_name_or_path or config, 'OLoRAHyperParams requires either hparams_name_or_path or config'
        if hparams_name_or_path:
            if '.yaml' not in hparams_name_or_path:
                hparams_name_or_path = hparams_name_or_path + '.yaml'
            with open(hparams_name_or_path, "r") as stream:
                config = yaml.safe_load(stream)
                config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'O-LoRA') or print(
            f'OLoRAHyperParams cannot load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]}')
        return cls(**config)
