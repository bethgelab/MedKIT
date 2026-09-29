from dataclasses import dataclass
from ...util.hparams import HyperParams
import yaml


@dataclass
class BM25RAGHyperParams(HyperParams):
    model_name: str
    alg_name: str
    device: int
    top_k: int = 1
    retrieve_target: bool = False

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
        assert hparams_name_or_path or config, 'BM25RAGHyperParams requires either hparams_name_or_path or config'
        if hparams_name_or_path:
            if '.yaml' not in hparams_name_or_path:
                hparams_name_or_path = hparams_name_or_path + '.yaml'
            with open(hparams_name_or_path, 'r') as stream:
                config = yaml.safe_load(stream)
                config = super().construct_float_from_scientific_notation(config)
        assert config and config['alg_name'] == 'BM25RAG', \
            f'BM25RAGHyperParams cannot load config with alg_name={config.get("alg_name")}'
        return cls(**config)
