"""PyTorch-compatible data loaders for Anthropic HuggingFace datasets.

Datasets covered:
  - Anthropic/hh-rlhf          (DPO preference training)
  - Anthropic/model-written-evals (multiple-choice evaluation)
  - Anthropic/persuasion        (reward model training)
  - Anthropic/discrim-eval      (fairness / demographic bias evaluation)

The ``datasets`` library (HuggingFace) is used only for downloading raw data;
no HuggingFace model wrappers are imported.
"""

from __future__ import annotations

import importlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset


def _datasets_module() -> Any:
    """Import ``datasets`` lazily so the module has no static foreign import."""
    try:
        return importlib.import_module("datasets")
    except ImportError as exc:  # pragma: no cover - dependency not installed in tests
        raise ImportError(
            "The 'datasets' package is required to load HuggingFace datasets."
        ) from exc


# ── hh-rlhf ──────────────────────────────────────────────────────────────────


@dataclass
class HHRLHFSample:
    """One preference pair from Anthropic/hh-rlhf."""

    prompt: str
    chosen: str
    rejected: str
    raw_chosen: str
    raw_rejected: str


def parse_hh_conversation(text: str) -> tuple[str, str]:
    """Split a hh-rlhf conversation string into (prompt, response).

    The prompt is everything up to AND including the last "Assistant: ".
    The response is everything after that marker.

    Returns ("", text) when no "Assistant: " is found.
    """
    marker = "Assistant: "
    idx = text.rfind(marker)
    if idx == -1:
        return ("", text)
    split_point = idx + len(marker)
    return text[:split_point], text[split_point:]


class HHRLHFDataset(Dataset):
    """PyTorch Dataset wrapping a list of HHRLHFSample objects."""

    def __init__(self, samples: list[HHRLHFSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> HHRLHFSample:
        return self._samples[idx]


def load_hh_rlhf(split: str = "train", cache_dir: str | None = None) -> HHRLHFDataset:
    """Load Anthropic/hh-rlhf from HuggingFace and return an HHRLHFDataset."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)

    samples: list[HHRLHFSample] = []
    for row in raw:
        chosen_prompt, chosen_resp = parse_hh_conversation(row["chosen"])
        _rejected_prompt, rejected_resp = parse_hh_conversation(row["rejected"])
        samples.append(
            HHRLHFSample(
                prompt=chosen_prompt,
                chosen=chosen_resp,
                rejected=rejected_resp,
                raw_chosen=row["chosen"],
                raw_rejected=row["rejected"],
            )
        )
    return HHRLHFDataset(samples)


def hh_rlhf_to_dpo_format(dataset: HHRLHFDataset) -> list[dict[str, str]]:
    """Convert an HHRLHFDataset to DPO-ready records.

    Returns list of dicts with keys "prompt", "chosen", "rejected".
    """
    return [{"prompt": s.prompt, "chosen": s.chosen, "rejected": s.rejected} for s in dataset]


# ── model-written-evals ───────────────────────────────────────────────────────


@dataclass
class ModelWrittenEvalSample:
    """One multiple-choice item from Anthropic/model-written-evals."""

    question: str
    answer_matching_behavior: str
    answer_not_matching_behavior: str
    correct_letter: str


def _extract_letter(option: str) -> str:
    """Extract 'A' or 'B' from strings like '(A)' or '(B)'."""
    stripped = option.strip()
    if stripped.startswith("(") and len(stripped) >= 2:
        return stripped[1]
    return stripped


class ModelWrittenEvalDataset(Dataset):
    """PyTorch Dataset wrapping a list of ModelWrittenEvalSample objects."""

    def __init__(self, samples: list[ModelWrittenEvalSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> ModelWrittenEvalSample:
        return self._samples[idx]


def load_model_written_evals(cache_dir: str | None = None) -> ModelWrittenEvalDataset:
    """Load Anthropic/model-written-evals from HuggingFace."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset(
        "Anthropic/model-written-evals", split="train", cache_dir=cache_dir
    )

    samples: list[ModelWrittenEvalSample] = []
    for row in raw:
        amb = row["answer_matching_behavior"]
        samples.append(
            ModelWrittenEvalSample(
                question=row["question"],
                answer_matching_behavior=amb,
                answer_not_matching_behavior=row["answer_not_matching_behavior"],
                correct_letter=_extract_letter(amb),
            )
        )
    return ModelWrittenEvalDataset(samples)


# ── persuasion ────────────────────────────────────────────────────────────────

_PERSUASION_MIN = -2
_PERSUASION_MAX = 5
_PERSUASION_RANGE = _PERSUASION_MAX - _PERSUASION_MIN  # 7


@dataclass
class PersuasionSample:
    """One row from Anthropic/persuasion."""

    claim: str
    argument: str
    source: str
    prompt_type: str
    persuasiveness_score: float
    raw_persuasiveness: int


def _normalize_persuasiveness(raw: int) -> float:
    return (raw - _PERSUASION_MIN) / _PERSUASION_RANGE


class PersuasionDataset(Dataset):
    """PyTorch Dataset wrapping a list of PersuasionSample objects."""

    def __init__(self, samples: list[PersuasionSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> PersuasionSample:
        return self._samples[idx]


def load_persuasion(cache_dir: str | None = None) -> PersuasionDataset:
    """Load Anthropic/persuasion from HuggingFace."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset("Anthropic/persuasion", split="train", cache_dir=cache_dir)

    samples: list[PersuasionSample] = []
    for row in raw:
        raw_score = int(row["persuasiveness_metric"])
        samples.append(
            PersuasionSample(
                claim=row["claim"],
                argument=row["argument"],
                source=row["source"],
                prompt_type=row["prompt_type"],
                persuasiveness_score=_normalize_persuasiveness(raw_score),
                raw_persuasiveness=raw_score,
            )
        )
    return PersuasionDataset(samples)


def persuasion_to_reward_samples(dataset: PersuasionDataset) -> list[dict[str, Any]]:
    """Convert a PersuasionDataset to reward-model training records.

    Returns list of dicts with keys "text" and "reward".
    """
    return [
        {
            "text": s.claim + "\n\n" + s.argument,
            "reward": float(s.persuasiveness_score),
        }
        for s in dataset
    ]


# ── discrim-eval ──────────────────────────────────────────────────────────────


@dataclass
class DiscrimEvalSample:
    """One row from Anthropic/discrim-eval."""

    filled_template: str
    decision_question_id: int
    age: float
    gender: str
    race: str
    fill_type: str


class DiscrimEvalDataset(Dataset):
    """PyTorch Dataset wrapping a list of DiscrimEvalSample objects."""

    def __init__(self, samples: list[DiscrimEvalSample]) -> None:
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> DiscrimEvalSample:
        return self._samples[idx]

    def group_by_question(self) -> dict[int, list[DiscrimEvalSample]]:
        """Group samples by decision_question_id for per-question demographic analysis."""
        groups: dict[int, list[DiscrimEvalSample]] = defaultdict(list)
        for sample in self._samples:
            groups[sample.decision_question_id].append(sample)
        return dict(groups)

    def get_demographic_groups(self) -> dict[str, list[DiscrimEvalSample]]:
        """Return samples grouped by demographic attribute values.

        Keys have the format "attribute=value" (e.g. "gender=male").
        Attributes covered: gender, race.
        """
        groups: dict[str, list[DiscrimEvalSample]] = defaultdict(list)
        for sample in self._samples:
            groups[f"gender={sample.gender}"].append(sample)
            groups[f"race={sample.race}"].append(sample)
        return dict(groups)


def load_discrim_eval(config: str = "explicit", cache_dir: str | None = None) -> DiscrimEvalDataset:
    """Load Anthropic/discrim-eval from HuggingFace.

    Args:
        config: "explicit" or "implicit" (9,450 rows each).
        cache_dir: Optional local cache directory.
    """
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset(
        "Anthropic/discrim-eval", config, split="train", cache_dir=cache_dir
    )

    samples: list[DiscrimEvalSample] = []
    for row in raw:
        samples.append(
            DiscrimEvalSample(
                filled_template=row["filled_template"],
                decision_question_id=int(row["decision_question_id"]),
                age=float(row["age"]),
                gender=row["gender"],
                race=row["race"],
                fill_type=row["fill_type"],
            )
        )
    return DiscrimEvalDataset(samples)


def compute_demographic_bias_score(
    dataset: DiscrimEvalDataset,
    model_scores: dict[int, float],
    attribute: str = "race",
) -> dict[str, float]:
    """Compute mean model score per demographic group for a given attribute.

    Args:
        dataset: A DiscrimEvalDataset instance.
        model_scores: Mapping from sample index (0-based) to model output score.
        attribute: Demographic dimension — "race", "gender", or "age".

    Returns:
        Dict mapping each distinct attribute value to mean model score.
        Example: {"white": 0.7, "Black": 0.6}
    """
    group_totals: dict[str, float] = defaultdict(float)
    group_counts: dict[str, int] = defaultdict(int)

    for idx, score in model_scores.items():
        sample = dataset[idx]
        value = str(getattr(sample, attribute))
        group_totals[value] += score
        group_counts[value] += 1

    return {group: group_totals[group] / group_counts[group] for group in group_totals}
