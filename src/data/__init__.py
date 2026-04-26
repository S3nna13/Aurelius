"""Aurelius data pipeline -- processing, filtering, dedup, and FIM transforms."""

from src.data.dataset_config import (
    ARXIV,
    AURELIUS_MIX,
    CODE_LANGUAGES,
    FINEWEB,
    FINEWEB_EDU,
    MAX_SEQ_LENGTH,
    OPENWEBMATH,
    THE_STACK_V2,
    TOKENS_PER_SHARD,
    WIKIPEDIA_BOOKS,
    DataMixConfig,
    DatasetSource,
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
from src.data.cwe_synthesis import (
    CWE_CATALOG,
    CWERecipe,
    CWESyntheticGenerator,
)
from src.data.tokenizer_trainer import (
    BPEConfig,
    BPETokenizer,
    BPETrainer,
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

from src.data.synthetic_math import (  # noqa: E402
    MathProblem,
    SyntheticMathConfig,
    SyntheticMathGenerator,
)

# Registry for synthetic math problem generators (RL math-reasoning training).
DATA_REGISTRY: dict[str, type] = {
    "synthetic_math": SyntheticMathGenerator,
}

from src.data.coreset_selector import (  # noqa: E402
    CoresetConfig,
    CoresetSelector,
)
from src.data.data_versioning import (  # noqa: E402
    DATA_VERSION_REGISTRY,
    DataDiff,
    DatasetVersion,
    DataVersionRegistry,
)

# CoresetSelector self-registers into DATA_REGISTRY on import.
from src.data.tokenization_pipeline import (  # noqa: E402
    TokenizationConfig,
    TokenizationPipeline,
    TokenizedOutput,
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
    # synthetic_math
    "MathProblem",
    "SyntheticMathConfig",
    "SyntheticMathGenerator",
    "DATA_REGISTRY",
    # coreset_selector
    "CoresetConfig",
    "CoresetSelector",
    # tokenization_pipeline
    "TokenizationConfig",
    "TokenizationPipeline",
    "TokenizedOutput",
    # data_versioning
    "DATA_VERSION_REGISTRY",
    "DataDiff",
    "DatasetVersion",
    "DataVersionRegistry",
]

# --- Synthetic CoT + SFT builder + dataset analyzer (additive, cycle-147) -----
from src.data.dataset_analyzer import (  # noqa: E402,F401
    DATASET_ANALYZER_REGISTRY,
    DatasetAnalyzer,
    DatasetStats,
)
from src.data.sft_dataset_builder import (  # noqa: E402,F401
    SFT_BUILDER_REGISTRY,
    SFTDatasetBuilder,
    SFTDatasetConfig,
    SFTExample,
)
from src.data.synthetic_cot_dataset import (  # noqa: E402,F401
    COT_DATASET_REGISTRY,
    CoTDatasetConfig,
    CoTExample,
    SyntheticCoTDataset,
)
