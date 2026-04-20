# src/data/anthropic_datasets_ext.py
"""PyTorch-compatible loaders for Anthropic HuggingFace datasets.

Datasets covered:
  1. llm_global_opinions       — calibration eval (opinion distributions)
  2. values-in-the-wild        — curriculum weighting by value frequency
  3. election_questions        — safety evaluation binary classifier
  4. AnthropicInterviewer      — SFT training on interview transcripts
  5. alignment-faking-rl       — safety classifier training (XOR-encrypted)
"""

from __future__ import annotations

import importlib
import base64
import json
import math
import re
from dataclasses import dataclass
from itertools import cycle
from typing import Optional

from torch.utils.data import Dataset


def _load_dataset(*args, **kwargs):
    """Import ``datasets`` lazily and call ``load_dataset`` with the given args."""
    try:
        datasets_mod = importlib.import_module("datasets")
    except ImportError as exc:  # pragma: no cover - dependency not installed in tests
        raise ImportError(
            "The 'datasets' package is required to load HuggingFace datasets."
        ) from exc
    return datasets_mod.load_dataset(*args, **kwargs)

# ── XOR decryption key ───────────────────────────────────────────────────────
KEY = b"MO_transcripts_should_not_appear_in_pretraining_corpora"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. llm_global_opinions
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GlobalOpinionSample:
    question: str
    options: list[str]
    country_distributions: dict[str, list[float]]   # country → [pct per option]
    source: str   # "GAS" or "WVS"


class GlobalOpinionsDataset(Dataset):
    def __init__(self, samples: list[GlobalOpinionSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GlobalOpinionSample:
        return self._samples[idx]


def _parse_field(raw) -> object:
    """Parse a field that may be a JSON string, already-parsed dict/list, or plain str."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return raw


def load_global_opinions(cache_dir: Optional[str] = None) -> GlobalOpinionsDataset:
    """Load Anthropic/llm_global_opinions from HuggingFace.

    Parses selections/options fields (stored as JSON strings in the raw data).
    """
    raw = _load_dataset(
        "Anthropic/llm_global_opinions",
        split="train",
        cache_dir=cache_dir,
    )

    samples: list[GlobalOpinionSample] = []
    for row in raw:
        options = _parse_field(row["options"])
        if isinstance(options, str):
            # Try splitting on comma as last resort
            options = [o.strip() for o in options.split(",")]

        selections = _parse_field(row["selections"])
        if not isinstance(selections, dict):
            selections = {}

        # Each value in selections is a list of percentages, one per option
        country_distributions: dict[str, list[float]] = {}
        for country, pct_list in selections.items():
            if isinstance(pct_list, (list, tuple)):
                country_distributions[country] = [float(p) for p in pct_list]

        samples.append(GlobalOpinionSample(
            question=str(row["question"]),
            options=list(options) if not isinstance(options, list) else options,
            country_distributions=country_distributions,
            source=str(row["source"]),
        ))

    return GlobalOpinionsDataset(samples)


def compute_opinion_kl_divergence(
    sample: GlobalOpinionSample,
    model_probs: list[float],
    country: str = "United States",
) -> float:
    """KL divergence KL(survey_dist || model_dist) in nats.

    Uses the survey distribution as P (reference) and model_probs as Q.
    Returns KL(P || Q) = sum_i P_i * log(P_i / Q_i).
    """
    survey = sample.country_distributions.get(country)
    if survey is None:
        raise KeyError(f"Country '{country}' not found in sample distributions.")

    if len(survey) != len(model_probs):
        raise ValueError(
            f"Distribution length mismatch: survey={len(survey)}, model={len(model_probs)}"
        )

    # Normalise both to be safe
    sum_s = sum(survey)
    sum_m = sum(model_probs)
    p = [s / sum_s for s in survey]
    q = [m / sum_m for m in model_probs]

    eps = 1e-12
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 0:
            kl += pi * math.log((pi + eps) / (qi + eps))
    return kl


# ═══════════════════════════════════════════════════════════════════════════════
# 2. values-in-the-wild
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ValueFrequency:
    value: str
    pct_convos: float


@dataclass
class ValueTreeNode:
    cluster_id: str
    name: str
    level: int               # 0=individual value, 3=top-level category
    parent_cluster_id: str
    pct_total_occurrences: float
    description: str         # empty string for level 0


class ValuesFrequencyDataset(Dataset):
    def __init__(self, values: list[ValueFrequency]) -> None:
        self._values = values

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, idx: int) -> ValueFrequency:
        return self._values[idx]

    def get_weights(self, temperature: float = 1.0) -> list[float]:
        """Normalised pct_convos^(1/temperature) for curriculum sampling."""
        raw = [v.pct_convos ** (1.0 / temperature) for v in self._values]
        total = sum(raw)
        if total == 0:
            n = len(raw)
            return [1.0 / n] * n
        return [w / total for w in raw]


class ValuesTreeDataset(Dataset):
    def __init__(self, nodes: list[ValueTreeNode]) -> None:
        self._nodes = nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def __getitem__(self, idx: int) -> ValueTreeNode:
        return self._nodes[idx]

    def get_top_level_categories(self) -> list[ValueTreeNode]:
        """Returns nodes where level == 3."""
        return [n for n in self._nodes if n.level == 3]

    def get_children(self, cluster_id: str) -> list[ValueTreeNode]:
        """Returns nodes where parent_cluster_id == cluster_id."""
        return [n for n in self._nodes if n.parent_cluster_id == cluster_id]


def load_values_frequencies(cache_dir: Optional[str] = None) -> ValuesFrequencyDataset:
    """Load values_frequencies config from Anthropic/values-in-the-wild."""
    raw = _load_dataset(
        "Anthropic/values-in-the-wild",
        "values_frequencies",
        split="train",
        cache_dir=cache_dir,
    )
    values = [
        ValueFrequency(value=str(row["value"]), pct_convos=float(row["pct_convos"]))
        for row in raw
    ]
    return ValuesFrequencyDataset(values)


def load_values_tree(cache_dir: Optional[str] = None) -> ValuesTreeDataset:
    """Load values_tree config from Anthropic/values-in-the-wild."""
    raw = _load_dataset(
        "Anthropic/values-in-the-wild",
        "values_tree",
        split="train",
        cache_dir=cache_dir,
    )
    nodes = [
        ValueTreeNode(
            cluster_id=str(row["cluster_id"]),
            name=str(row["name"]),
            level=int(row["level"]),
            parent_cluster_id=str(row["parent_cluster_id"]) if row["parent_cluster_id"] is not None else "",
            pct_total_occurrences=float(row["pct_total_occurrences"]),
            description=str(row["description"]) if row["description"] is not None else "",
        )
        for row in raw
    ]
    return ValuesTreeDataset(nodes)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. election_questions
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ElectionQuestionSample:
    question: str
    label: str        # "Harmless" or "Harmful"
    is_harmful: bool  # True if label == "Harmful"


class ElectionQuestionsDataset(Dataset):
    def __init__(self, samples: list[ElectionQuestionSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ElectionQuestionSample:
        return self._samples[idx]

    def harmful_samples(self) -> list[ElectionQuestionSample]:
        return [s for s in self._samples if s.is_harmful]

    def harmless_samples(self) -> list[ElectionQuestionSample]:
        return [s for s in self._samples if not s.is_harmful]


def load_election_questions(cache_dir: Optional[str] = None) -> ElectionQuestionsDataset:
    """Load Anthropic/election_questions (test split)."""
    raw = _load_dataset(
        "Anthropic/election_questions",
        split="test",
        cache_dir=cache_dir,
    )
    samples = [
        ElectionQuestionSample(
            question=str(row["question"]),
            label=str(row["label"]),
            is_harmful=(str(row["label"]) == "Harmful"),
        )
        for row in raw
    ]
    return ElectionQuestionsDataset(samples)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. AnthropicInterviewer
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class InterviewTranscript:
    transcript_id: str
    text: str
    split: str   # "workforce", "creatives", "scientists"


class InterviewDataset(Dataset):
    def __init__(self, transcripts: list[InterviewTranscript]) -> None:
        self._transcripts = transcripts

    def __len__(self) -> int:
        return len(self._transcripts)

    def __getitem__(self, idx: int) -> InterviewTranscript:
        return self._transcripts[idx]

    def by_split(self, split: str) -> list[InterviewTranscript]:
        """Filter transcripts by split name."""
        return [t for t in self._transcripts if t.split == split]


_ALL_INTERVIEW_SPLITS = ["workforce", "creatives", "scientists"]


def load_interviews(
    splits: Optional[list[str]] = None,
    cache_dir: Optional[str] = None,
) -> InterviewDataset:
    """Load Anthropic/AnthropicInterviewer.

    Args:
        splits: Which splits to load. None means all three.
        cache_dir: Optional HuggingFace cache directory.
    """
    if splits is None:
        splits = _ALL_INTERVIEW_SPLITS

    transcripts: list[InterviewTranscript] = []
    for split_name in splits:
        raw = _load_dataset(
            "Anthropic/AnthropicInterviewer",
            split=split_name,
            cache_dir=cache_dir,
        )
        for row in raw:
            transcripts.append(InterviewTranscript(
                transcript_id=str(row["transcript_id"]),
                text=str(row["text"]),
                split=split_name,
            ))

    return InterviewDataset(transcripts)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. alignment-faking-rl
# ═══════════════════════════════════════════════════════════════════════════════

_CANARY_RE = re.compile(r"^<canary>.*?</canary>", re.DOTALL)


def decrypt_alignment_faking_response(encrypted_response: str) -> str:
    """Decrypt a response from the alignment-faking-rl dataset.

    Steps:
      1. Strip any leading <canary>...</canary> block.
      2. Base64-decode the result.
      3. XOR-decrypt with cycling KEY.
    """
    # Strip canary prefix if present
    clean = _CANARY_RE.sub("", encrypted_response).strip()

    raw_bytes = base64.b64decode(clean)
    decrypted = bytes(a ^ b for a, b in zip(raw_bytes, cycle(KEY)))
    return decrypted.decode("utf-8")


@dataclass
class AlignmentFakingSample:
    user_input: str
    response: str               # decrypted
    alignment_faking: bool
    alignment_faking_prob: float | None
    compliance: bool
    total_reward: float
    is_free_tier_user: bool
    experiment_name: str
    step: int


class AlignmentFakingDataset(Dataset):
    def __init__(self, samples: list[AlignmentFakingSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> AlignmentFakingSample:
        return self._samples[idx]

    def faking_samples(self) -> list[AlignmentFakingSample]:
        """Returns samples where alignment_faking == True."""
        return [s for s in self._samples if s.alignment_faking]

    def honest_samples(self) -> list[AlignmentFakingSample]:
        """Returns samples where alignment_faking == False."""
        return [s for s in self._samples if not s.alignment_faking]

    def to_safety_classifier_format(self) -> list[dict]:
        """Returns [{"text": user_input + response, "label": int(alignment_faking)}, ...]."""
        return [
            {
                "text": s.user_input + "\n" + s.response,
                "label": int(s.alignment_faking),
            }
            for s in self._samples
        ]


def load_alignment_faking(
    n_rows: int | None = None,
    cache_dir: Optional[str] = None,
) -> AlignmentFakingDataset:
    """Load Anthropic/alignment-faking-rl (train split).

    Args:
        n_rows: Limit number of rows loaded (dataset is ~17 GB). Defaults to 10_000.
        cache_dir: Optional HuggingFace cache directory.
    """
    if n_rows is None:
        n_rows = 10_000

    raw = _load_dataset(
        "Anthropic/alignment-faking-rl",
        split="train",
        streaming=True,
        cache_dir=cache_dir,
    )

    samples: list[AlignmentFakingSample] = []
    for i, row in enumerate(raw):
        if i >= n_rows:
            break

        try:
            decrypted = decrypt_alignment_faking_response(str(row["response"]))
        except Exception:
            decrypted = ""

        prob = row.get("alignment_faking_prob")
        samples.append(AlignmentFakingSample(
            user_input=str(row["user_input"]),
            response=decrypted,
            alignment_faking=bool(row["alignment_faking"]),
            alignment_faking_prob=float(prob) if prob is not None else None,
            compliance=bool(row["compliance"]),
            total_reward=float(row["total_reward"]),
            is_free_tier_user=bool(row["is_free_tier_user"]),
            experiment_name=str(row["experiment_name"]),
            step=int(row["step"]),
        ))

    return AlignmentFakingDataset(samples)
