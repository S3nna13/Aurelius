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
from src.data.cwe_synthesis import (
    CWE_CATALOG,
    CWERecipe,
    CWESyntheticGenerator,
)

# Registry of optional data loaders/generators. Integration tests discover
# generators via this dict; additive-only.
LOADER_REGISTRY: dict[str, object] = {
    "cwe_synthesis": CWESyntheticGenerator,
}

from src.data.ngram_deduplication import (  # noqa: E402
    NgramDeduplicationToolkit,
    cluster_duplicates,
    ngram_jaccard,
)

DEDUP_REGISTRY: dict[str, type] = {
    "ngram_jaccard": NgramDeduplicationToolkit,
}

from src.data.tokenizer_contract import (  # noqa: E402
    TOKENIZER_IDENTITY_REGISTRY,
    ContractMismatch,
    ContractVerdict,
    TokenizerContractValidator,
    TokenizerIdentity,
    compute_tokenizer_hash,
    register_identity,
)

from src.data.vision_token_mixer import (  # noqa: E402
    VisionTokenMixer,
    VisionTokenMixerConfig,
)

# Registry for vision-text mixing utilities (early-fusion pipeline components).
VISION_MIXER_REGISTRY: dict[str, type] = {
    "vision_token_mixer": VisionTokenMixer,
}

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
    # cwe_synthesis
    "CWE_CATALOG",
    "CWERecipe",
    "CWESyntheticGenerator",
    "LOADER_REGISTRY",
    "NgramDeduplicationToolkit",
    "cluster_duplicates",
    "ngram_jaccard",
    "DEDUP_REGISTRY",
    # tokenizer_contract
    "TOKENIZER_IDENTITY_REGISTRY",
    "ContractMismatch",
    "ContractVerdict",
    "TokenizerContractValidator",
    "TokenizerIdentity",
    "compute_tokenizer_hash",
    "register_identity",
    # vision_token_mixer
    "VisionTokenMixer",
    "VisionTokenMixerConfig",
    "VISION_MIXER_REGISTRY",
]
