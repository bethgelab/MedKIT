from dataclasses import dataclass
from typing import List, Literal, Union

from ...util.hparams import HyperParams
import yaml


@dataclass
class AlphaEditHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    # Spec for the null-space cutoff. Accepts:
    #   float / "abs:X"        — absolute cutoff (legacy; typically broken on
    #                            float16/bf16 cov stats whose max SV is O(0.1))
    #   "rel_max:X"            — cutoff = X * max(S) per layer (recommended)
    #   "keep_top_frac:X"      — keep the top fraction X of SVs as column-space
    nullspace_threshold: Union[float, str]
    L2: float
    alg_name: str
    device: int
    model_name: str
    stats_dir: str
    P_loc: str

    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False
    max_out_len_open: int = 1024
    max_out_len_closed: int = 50
    fp16: bool = False
    bf16: bool = False
    # Bound on per-edit weight-delta magnitude: if ||ΔW|| / ||W|| exceeds this,
    # the update is scaled down to exactly this ratio before being applied.
    # Prevents one bad compute_z (e.g. divergent v-optimization, NaN targets)
    # from destroying subsequent edits. 0.5 = edit may change up to 50% of the
    # Frobenius norm of the weight. Set to None/0 to disable.
    max_update_ratio: float = 0.5

    @classmethod
    def from_hparams(cls, hparams_name_or_path=None, config=None):
        assert hparams_name_or_path or config, \
            'AlphaEditHyperParams requires either hparams_name_or_path or config'
        if hparams_name_or_path:
            if '.yaml' not in hparams_name_or_path:
                hparams_name_or_path = hparams_name_or_path + '.yaml'
            with open(hparams_name_or_path, "r") as stream:
                config = yaml.safe_load(stream)
                config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'AlphaEdit'), \
            f'AlphaEditHyperParams can not load from {hparams_name_or_path}, alg_name is {config["alg_name"]}'
        return cls(**config)
