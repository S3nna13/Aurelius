"""Aurelius data pipeline -- processing, filtering, dedup, and FIM transforms."""

from src.data.dataset_config import (
    AURELIUS_MIX,
    ARXIV,
    CODE_LANGUAGES,
    DataMixConfig,
    DatasetSource,
    FINEWEB,
    FINEWEB_EDU,
    MAX_SEQ_LENGTH,
    OPENWEBMATH,
    THE_STACK_V2,
    TOKENS_PER_SHARD,
    WIKIPEDIA_BOOKS,
    get_mix,
)
from src.data.fim_transform import (
    FIM_MIDDLE,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIMConfig,
    FIMMode,
    FIMResult,
    fim_transform,
    fim_transform_batch,
)
try:
    from src.data.pipeline import PipelineConfig, run_full_pipeline
except ImportError:
    pass  # datatrove not installed; pipeline unavailable
from src.data.tokenizer_trainer import (
    BPEConfig,
    BPETokenizer,
    BPETrainer,
)

__all__ = [
    # dataset_config
    "AURELIUS_MIX",
    "ARXIV",
    "CODE_LANGUAGES",
    "DataMixConfig",
    "DatasetSource",
    "FINEWEB",
    "FINEWEB_EDU",
    "MAX_SEQ_LENGTH",
    "OPENWEBMATH",
    "THE_STACK_V2",
    "TOKENS_PER_SHARD",
    "WIKIPEDIA_BOOKS",
    "get_mix",
    # fim_transform
    "FIM_MIDDLE",
    "FIM_PREFIX",
    "FIM_SUFFIX",
    "FIMConfig",
    "FIMMode",
    "FIMResult",
    "fim_transform",
    "fim_transform_batch",
    # pipeline
    "PipelineConfig",
    "run_full_pipeline",
    # tokenizer_trainer
    "BPEConfig",
    "BPETokenizer",
    "BPETrainer",
]
