"""Dataset mix configuration for Project Aurelius 1.3B LLM.

Defines the data sources, token budgets, sampling weights, and
HuggingFace dataset identifiers used across the training pipeline.
Total budget: 300B tokens (~230x model parameters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class DataDomain(Enum):
    """Broad category for each data source."""

    WEB = auto()
    CODE = auto()
    EDUCATION = auto()
    MATH = auto()
    REFERENCE = auto()
    SCIENTIFIC = auto()


@dataclass(frozen=True, slots=True)
class DatasetSource:
    """A single data source in the training mix."""

    name: str
    hf_dataset_id: str
    domain: DataDomain
    target_tokens_billions: float
    weight: float  # sampling probability during training
    hf_subset: str | None = None
    hf_split: str = "train"
    streaming: bool = True
    description: str = ""


@dataclass(frozen=True, slots=True)
class DataMixConfig:
    """Complete data mix specification for a training run."""

    name: str
    total_tokens_billions: float
    sources: tuple[DatasetSource, ...]
    seed: int = 42

    @property
    def total_weight(self) -> float:
        return sum(s.weight for s in self.sources)

    def normalized_weights(self) -> dict[str, float]:
        """Return source weights normalized to sum to 1.0."""
        total = self.total_weight
        return {s.name: s.weight / total for s in self.sources}

    def tokens_per_source(self) -> dict[str, float]:
        """Return the target token count (in billions) per source."""
        return {s.name: s.target_tokens_billions for s in self.sources}

    def validate(self) -> None:
        """Raise ValueError if the config is internally inconsistent."""
        token_sum = sum(s.target_tokens_billions for s in self.sources)
        if abs(token_sum - self.total_tokens_billions) > 0.5:
            raise ValueError(
                f"Source token budgets sum to {token_sum}B but "
                f"total_tokens_billions is {self.total_tokens_billions}B"
            )
        weight_sum = sum(s.weight for s in self.sources)
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Sampling weights sum to {weight_sum}, expected ~1.0"
            )


# ---------------------------------------------------------------------------
# Aurelius 1.3B default data mix
# ---------------------------------------------------------------------------

FINEWEB = DatasetSource(
    name="fineweb",
    hf_dataset_id="HuggingFaceFW/fineweb",
    domain=DataDomain.WEB,
    target_tokens_billions=195.0,
    weight=0.65,
    description="High-quality English web crawl data from CommonCrawl, "
    "deduplicated and filtered by FineWeb pipeline.",
)

THE_STACK_V2 = DatasetSource(
    name="the_stack_v2",
    hf_dataset_id="bigcode/the-stack-v2-dedup",
    domain=DataDomain.CODE,
    target_tokens_billions=60.0,
    weight=0.20,
    description="Deduplicated source code across popular programming "
    "languages from Software Heritage.",
)

FINEWEB_EDU = DatasetSource(
    name="fineweb_edu",
    hf_dataset_id="HuggingFaceFW/fineweb-edu",
    domain=DataDomain.EDUCATION,
    target_tokens_billions=24.0,
    weight=0.08,
    description="Educational subset of FineWeb, scored by a classifier "
    "trained on high-quality educational content.",
)

OPENWEBMATH = DatasetSource(
    name="openwebmath",
    hf_dataset_id="open-web-math/open-web-math",
    domain=DataDomain.MATH,
    target_tokens_billions=9.0,
    weight=0.03,
    description="Mathematical web pages filtered and cleaned for "
    "high-quality mathematical reasoning content.",
)

WIKIPEDIA_BOOKS = DatasetSource(
    name="wikipedia_books",
    hf_dataset_id="wikimedia/wikipedia",
    hf_subset="20231101.en",
    domain=DataDomain.REFERENCE,
    target_tokens_billions=6.0,
    weight=0.02,
    description="English Wikipedia snapshots combined with "
    "public-domain book corpora.",
)

ARXIV = DatasetSource(
    name="arxiv",
    hf_dataset_id="togethercomputer/RedPajama-Data-V2",
    hf_subset="arxiv",
    domain=DataDomain.SCIENTIFIC,
    target_tokens_billions=6.0,
    weight=0.02,
    description="Scientific papers from ArXiv, providing exposure to "
    "formal reasoning and technical writing.",
)

# The canonical mix used for Aurelius 1.3B pre-training.
AURELIUS_MIX = DataMixConfig(
    name="aurelius-1.3b-v1",
    total_tokens_billions=300.0,
    sources=(
        FINEWEB,
        THE_STACK_V2,
        FINEWEB_EDU,
        OPENWEBMATH,
        WIKIPEDIA_BOOKS,
        ARXIV,
    ),
    seed=42,
)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

# Language list for The Stack v2 code subset.
CODE_LANGUAGES: list[str] = [
    "python",
    "javascript",
    "typescript",
    "java",
    "c",
    "cpp",
    "go",
    "rust",
    "shell",
    "sql",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "r",
    "lua",
    "haskell",
    "julia",
    "markdown",
]

# Maximum sequence length used by the tokenizer / packing stage.
MAX_SEQ_LENGTH: int = 2048

# Tokens per shard file (Parquet row-group target).
TOKENS_PER_SHARD: int = 100_000_000  # 100M tokens


def get_mix() -> DataMixConfig:
    """Return the default Aurelius data mix, validated."""
    AURELIUS_MIX.validate()
    return AURELIUS_MIX
