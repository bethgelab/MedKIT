from ..models.rome import ROMEHyperParams, apply_rome_to_model
from ..models.memit import MEMITHyperParams, apply_memit_to_model
from ..models.kn import KNHyperParams, apply_kn_to_model
from ..models.mend import MENDHyperParams, MendRewriteExecutor, MendMultimodalRewriteExecutor, MendPerRewriteExecutor
from ..models.ft import FTHyperParams, apply_ft_to_model
from ..models.dinm import DINMHyperParams, apply_dinm_to_model
from ..models.serac import SERACHparams, SeracRewriteExecutor, SeracMultimodalRewriteExecutor
from ..dataset import ZsreDataset, CounterFactDataset
from ..models.ike import IKEHyperParams, apply_ike_to_model, apply_ike_to_multimodal_model, apply_ike_to_per_model
from ..models.oracle_rag import OracleRAGHyperParams
from ..models.bm25_rag import BM25RAGHyperParams
from ..models.dense_rag import DenseRAGHyperParams
from ..models.ft_api import FTApiHyperParams, apply_ft_api_to_model
from ..models.lora import LoRAHyperParams, apply_lora_to_model
from ..models.grace import GraceHyperParams, apply_grace_to_model
from ..models.pmet import PMETHyperParams, apply_pmet_to_model
from ..models.melo import MELOHyperParams, apply_melo_to_model
from ..models.wise import WISEHyperParams, apply_wise_to_model
from ..models.r_rome import R_ROMEHyperParams, apply_r_rome_to_model
from ..models.emmet import EMMETHyperParams, apply_emmet_to_model
from ..models.alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
from ..models.memoir import MEMOIRHyperParams, apply_memoir_to_model
from ..models.lora_merge import LoRAMergeHyperParams, apply_lora_merge_to_model
from ..models.o_lora import OLoRAHyperParams, OLoRARewriteExecutor
from ..models.seekr import SEEKRHyperParams, SEEKRRewriteExecutor

_o_lora_executor = OLoRARewriteExecutor()
_seekr_executor = SEEKRRewriteExecutor()

ALG_DICT = {
    'ROME': apply_rome_to_model,
    'MEMIT': apply_memit_to_model,
    "FT": apply_ft_to_model,
    "DINM": apply_dinm_to_model,
    'KN': apply_kn_to_model,
    'MEND': MendRewriteExecutor().apply_to_model,
    'SERAC': SeracRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_model,
    'FT-Api': apply_ft_api_to_model,
    'LoRA': apply_lora_to_model,
    'GRACE': apply_grace_to_model,
    'PMET': apply_pmet_to_model,
    'MELO': apply_melo_to_model,
    'WISE': apply_wise_to_model,
    'R-ROME': apply_r_rome_to_model,
    "EMMET": apply_emmet_to_model,
    "AlphaEdit": apply_AlphaEdit_to_model,
    "MEMOIR": apply_memoir_to_model,
    "LoRA-Merge": apply_lora_merge_to_model,
    "O-LoRA": _o_lora_executor.apply_to_model,
    "SEEKR": _seekr_executor.apply_to_model,
    "RAG": None,
    "PTuning": None,
    "OracleRAG": None,
    "BM25RAG": None,
    "DenseRAG": None,
}

ALG_MULTIMODAL_DICT = {
    'MEND': MendMultimodalRewriteExecutor().apply_to_model,
    'SERAC': SeracMultimodalRewriteExecutor().apply_to_model,
    'SERAC_MULTI': SeracMultimodalRewriteExecutor().apply_to_model,
    'IKE': apply_ike_to_multimodal_model,
}

PER_ALG_DICT = {
    "IKE": apply_ike_to_per_model,
    "MEND": MendPerRewriteExecutor().apply_to_model,
}

DS_DICT = {
    "cf": CounterFactDataset,
    "zsre": ZsreDataset,
}

