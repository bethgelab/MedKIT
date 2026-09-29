from dataclasses import dataclass
from typing import List
from ...util.hparams import HyperParams
import yaml


@dataclass
class GraceHyperParams(HyperParams):
    # Experiments
    
    edit_lr: int
    n_iter: int
    # Method
    eps: float
    dist_fn: str
    val_init: str
    val_train: str
    val_reg: str
    reg: str
    replacement: str
    eps_expand: str
    num_pert: str
    dropout: float

    # Module templates
    inner_params: List[str]
    device: int
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False
    max_out_len_open: int = 1024
    max_out_len_closed: int = 50

    # Diagnostic-only: print per-edit codebook stats (nkeys, eps min/max,
    # conflict decisions). Used by debug/cascade probe configs to test the
    # epsilon-shrinkage hypothesis. No effect on editing behavior.
    grace_debug: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path=None, config=None):
        assert hparams_name_or_path or config, \
            'GraceHyperParams requires either hparams_name_or_path or config'
        if config is not None:
            # Called with a Hydra DictConfig / dict (from run_medkit.py)
            assert config['alg_name'] == 'GRACE', \
                f"GraceHyperParams expected alg_name=GRACE, got {config['alg_name']}"
            return cls(**config)

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'GRACE') or print(
            f'GraceHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)
