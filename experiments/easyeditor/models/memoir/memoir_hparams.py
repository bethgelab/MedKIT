from dataclasses import dataclass
from typing import List, Union
from ...util.hparams import HyperParams
import yaml


@dataclass
class MEMOIRHyperParams(HyperParams):
    # Experiments
    edit_lr: float
    n_iter: int
    # Method
    objective_optimization: str
    # Module templates
    inner_params: List[str]

    device: int
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    use_chat_template: bool = False
    max_out_len_open: int = 1024
    max_out_len_closed: int = 50

    # MEMOIR specific parameters
    top_k: int = 4096
    irr_threshold: float = 0.4
    prompt_feature_agg: str = 'mean_decentered'
    dir_background_features: str = None

    # hyperparameters for running the mode to only save background features
    RUN_SAVE_BACKGROUND_FEATURES: bool = False
    dir_to_save_background_features: str = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path=None, config=None):
        assert hparams_name_or_path or config, \
            'MEMOIRHyperParams requires either hparams_name_or_path or config'
        if hparams_name_or_path:
            if '.yaml' not in hparams_name_or_path:
                hparams_name_or_path = hparams_name_or_path + '.yaml'
            with open(hparams_name_or_path, "r") as stream:
                config = yaml.safe_load(stream)
                config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'MEMOIR'), \
            f'MEMOIRHyperParams can not load from {hparams_name_or_path}. alg_name is {config["alg_name"]}'
        return cls(**config)
