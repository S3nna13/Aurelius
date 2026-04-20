"""PyTorch-compatible data loaders for OpenAI HuggingFace datasets.

Datasets covered:
  - openai/gsm8k              (grade-school math, chain-of-thought)
  - openai/MMMLU              (multilingual MMLU benchmark)
  - openai/mrcr               (multi-turn conversation retrieval)
  - openai/graphwalks          (graph reasoning — BFS / parent finding)
  - openai/frontierscience     (hard STEM problems)
  - openai/coval               (comparative evaluation / preference data)
  - openai/gdpval              (GDP-linked occupational task evaluation)
  - openai/healthbench         (medical Q&A with rubric scoring)

The ``datasets`` library (HuggingFace) is used only for downloading raw data;
no HuggingFace model wrappers are imported.
"""

from __future__ import annotations

import importlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import Dataset


def _datasets_module() -> Any:
    """Import ``datasets`` lazily so the module has no static foreign import."""
    try:
        return importlib.import_module("datasets")
    except ImportError as exc:  # pragma: no cover - dependency not installed in tests
        raise ImportError(
            "The 'datasets' package is required to load HuggingFace datasets."
        ) from exc


# ── GSM8K ─────────────────────────────────────────────────────────────────────

_CALC_ANNOTATION_RE = re.compile(r"<<[^>]*>>")
_FINAL_ANSWER_MARKER = "####"


def parse_gsm8k_answer(answer_str: str) -> tuple[str, str]:
    """Split a GSM8K answer string into (reasoning_steps, final_answer_str).

    Steps before ``#### `` have calculator annotations ``<<...>>`` stripped.
    The final answer is the text after ``#### `` with leading/trailing whitespace
    removed (e.g. "72", "1,234").
    """
    if _FINAL_ANSWER_MARKER in answer_str:
        idx = answer_str.index(_FINAL_ANSWER_MARKER)
        raw_steps = answer_str[:idx]
        final = answer_str[idx + len(_FINAL_ANSWER_MARKER):].strip()
    else:
        raw_steps = answer_str
        final = ""

    reasoning = _CALC_ANNOTATION_RE.sub("", raw_steps).strip()
    return reasoning, final


def parse_gsm8k_final_number(answer_str: str) -> float | None:
    """Extract the numeric value that follows ``#### `` in a GSM8K answer string.

    Returns ``None`` if no ``#### `` marker is found or the text after it cannot
    be parsed as a number (commas in numbers are handled).
    """
    if _FINAL_ANSWER_MARKER not in answer_str:
        return None
    idx = answer_str.index(_FINAL_ANSWER_MARKER)
    tail = answer_str[idx + len(_FINAL_ANSWER_MARKER):].strip()
    # Remove commas used as thousands separators
    tail_clean = tail.replace(",", "")
    try:
        return float(tail_clean)
    except ValueError:
        return None


@dataclass
class GSM8KSample:
    """One problem from openai/gsm8k."""

    question: str
    answer_raw: str         # full answer string including steps
    reasoning: str          # steps before ####, calculator annotations removed
    final_answer: str       # text after #### (e.g., "72")
    final_number: float | None   # parsed numeric value


class GSM8KDataset(Dataset):
    """PyTorch Dataset wrapping GSM8K rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._samples: list[GSM8KSample] = []
        for row in rows:
            reasoning, final_answer = parse_gsm8k_answer(row["answer"])
            self._samples.append(
                GSM8KSample(
                    question=row["question"],
                    answer_raw=row["answer"],
                    reasoning=reasoning,
                    final_answer=final_answer,
                    final_number=parse_gsm8k_final_number(row["answer"]),
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GSM8KSample:
        return self._samples[idx]


def load_gsm8k(
    split: str = "train",
    config: str = "main",
    cache_dir: str | None = None,
) -> GSM8KDataset:
    """Load openai/gsm8k from HuggingFace and return a GSM8KDataset."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset(
        "openai/gsm8k", config, split=split, cache_dir=cache_dir
    )
    return GSM8KDataset(list(raw))


# ── MMMLU ─────────────────────────────────────────────────────────────────────

@dataclass
class MMLUSample:
    """One question from openai/MMMLU."""

    question: str
    choices: dict[str, str]   # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str        # "A", "B", "C", or "D"
    subject: str
    language: str              # config name e.g. "default", "FR_FR"


class MMMLUDataset(Dataset):
    """PyTorch Dataset wrapping MMMLU rows."""

    def __init__(self, rows: list[dict[str, Any]], language: str = "default") -> None:
        self._language = language
        self._samples: list[MMLUSample] = []
        for row in rows:
            self._samples.append(
                MMLUSample(
                    question=row["Question"],
                    choices={
                        "A": row["A"],
                        "B": row["B"],
                        "C": row["C"],
                        "D": row["D"],
                    },
                    correct_answer=row["Answer"],
                    subject=row["Subject"],
                    language=language,
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> MMLUSample:
        return self._samples[idx]


def load_mmmlu(
    config: str = "default",
    split: str = "test",
    cache_dir: str | None = None,
) -> MMMLUDataset:
    """Load openai/MMMLU from HuggingFace and return an MMMLUDataset."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset(
        "openai/MMMLU", config, split=split, cache_dir=cache_dir
    )
    return MMMLUDataset(list(raw), language=config)


# ── MRCR ──────────────────────────────────────────────────────────────────────

@dataclass
class MRCRSample:
    """One entry from openai/mrcr."""

    prompt: str
    answer: str
    prepend_hash: str       # random_string_to_prepend field
    n_needles: int
    desired_index: int      # desired_msg_index field
    total_messages: int
    n_chars: int


class MRCRDataset(Dataset):
    """PyTorch Dataset wrapping MRCR rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._samples: list[MRCRSample] = []
        for row in rows:
            self._samples.append(
                MRCRSample(
                    prompt=row["prompt"],
                    answer=row["answer"],
                    prepend_hash=row["random_string_to_prepend"],
                    n_needles=row["n_needles"],
                    desired_index=row["desired_msg_index"],
                    total_messages=row["total_messages"],
                    n_chars=row["n_chars"],
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> MRCRSample:
        return self._samples[idx]


def load_mrcr(
    split: str = "train",
    cache_dir: str | None = None,
) -> MRCRDataset:
    """Load openai/mrcr from HuggingFace and return an MRCRDataset."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset("openai/mrcr", split=split, cache_dir=cache_dir)
    return MRCRDataset(list(raw))


# ── GraphWalks ────────────────────────────────────────────────────────────────

def graphwalks_f1_score(predicted: list[str], gold: list[str]) -> float:
    """Compute F1 score between predicted and gold node sets.

    - Both empty → 1.0 (vacuously correct per the GraphWalks paper).
    - Otherwise standard set-based F1: 2*|P∩G| / (|P| + |G|).
    """
    if not predicted and not gold:
        return 1.0

    pred_set = set(predicted)
    gold_set = set(gold)
    intersection = pred_set & gold_set

    if not intersection:
        return 0.0

    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(gold_set) if gold_set else 0.0

    if precision + recall == 0.0:
        return 0.0

    return 2.0 * precision * recall / (precision + recall)


@dataclass
class GraphWalkSample:
    """One entry from openai/graphwalks."""

    prompt: str
    answer_nodes: list[str]
    problem_type: str    # "bfs" or "parents"


class GraphWalksDataset(Dataset):
    """PyTorch Dataset wrapping GraphWalks rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._samples: list[GraphWalkSample] = []
        for row in rows:
            self._samples.append(
                GraphWalkSample(
                    prompt=row["prompt"],
                    answer_nodes=list(row["answer_nodes"]),
                    problem_type=row["problem_type"],
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GraphWalkSample:
        return self._samples[idx]


def load_graphwalks(
    split: str = "train",
    cache_dir: str | None = None,
) -> GraphWalksDataset:
    """Load openai/graphwalks from HuggingFace and return a GraphWalksDataset."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset(
        "openai/graphwalks", split=split, cache_dir=cache_dir
    )
    return GraphWalksDataset(list(raw))


# ── FrontierScience ───────────────────────────────────────────────────────────

@dataclass
class FrontierScienceSample:
    """One entry from openai/frontierscience."""

    problem: str
    answer: str
    subject: str         # "Physics", "Chemistry", or "Biology"
    task_group_id: str   # UUID string


class FrontierScienceDataset(Dataset):
    """PyTorch Dataset wrapping FrontierScience rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._samples: list[FrontierScienceSample] = []
        for row in rows:
            self._samples.append(
                FrontierScienceSample(
                    problem=row["problem"],
                    answer=row["answer"],
                    subject=row["subject"],
                    task_group_id=row["task_group_id"],
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> FrontierScienceSample:
        return self._samples[idx]


def load_frontierscience(
    split: str = "test",
    cache_dir: str | None = None,
) -> FrontierScienceDataset:
    """Load openai/frontierscience from HuggingFace and return a FrontierScienceDataset."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset(
        "openai/frontierscience", split=split, cache_dir=cache_dir
    )
    return FrontierScienceDataset(list(raw))


# ── CoVal ─────────────────────────────────────────────────────────────────────

_RANKING_BLOCK_RE = re.compile(r"[A-D](?:=[A-D])*")


def coval_rankings_to_pairs(assessments: list[dict]) -> list[tuple[str, str]]:
    """Parse ranking strings like ``"B>A>C=D"`` into pairwise (winner, loser) tuples.

    Each ``>`` boundary yields all pairs between the left block and every
    response in all subsequent (lower-ranked) blocks.  Tied responses within a
    ``=`` block are NOT paired against each other.

    Parameters
    ----------
    assessments:
        List of assessment dicts, each with a ``"ranking"`` key.

    Returns
    -------
    list of (better, worse) string tuples.
    """
    pairs: list[tuple[str, str]] = []
    for assessment in assessments:
        ranking_str = assessment.get("ranking", "")
        # Split on '>' to get ordered blocks; each block may contain '=' ties
        blocks = ranking_str.split(">")
        # Parse each block into a list of response labels
        parsed_blocks: list[list[str]] = []
        for block in blocks:
            labels = [label.strip() for label in block.split("=") if label.strip()]
            if labels:
                parsed_blocks.append(labels)

        # Every response in block i beats every response in block j where j > i
        for i, better_block in enumerate(parsed_blocks):
            for j in range(i + 1, len(parsed_blocks)):
                worse_block = parsed_blocks[j]
                for better in better_block:
                    for worse in worse_block:
                        pairs.append((better, worse))

    return pairs


def coval_to_dpo_format(comparisons: list[dict]) -> list[dict]:
    """Convert CoVal comparison data to DPO training format.

    For each comparison, uses majority vote across all annotator rankings to
    identify the best (most-often-winning) and worst (most-often-losing)
    response.  The prompt is serialised as a plain string from the messages
    list.

    Parameters
    ----------
    comparisons:
        List of comparison dicts with keys: ``prompt_id``, ``prompt``
        (list of message dicts), ``responses`` (dict A/B/C/D → text),
        ``metadata.assessments`` (list of dicts with ``ranking`` key).

    Returns
    -------
    list of ``{"prompt": str, "chosen": str, "rejected": str}`` dicts.
    """
    result: list[dict] = []

    for comp in comparisons:
        assessments = comp.get("metadata", {}).get("assessments", [])
        responses = comp.get("responses", {})

        # Build win/loss tallies via pairwise comparisons
        win_counts: Counter[str] = Counter()
        loss_counts: Counter[str] = Counter()
        all_labels: set[str] = set(responses.keys())

        for better, worse in coval_rankings_to_pairs(assessments):
            if better in all_labels and worse in all_labels:
                win_counts[better] += 1
                loss_counts[worse] += 1

        if not win_counts:
            continue

        # Chosen: highest win count; rejected: highest loss count
        chosen_label = win_counts.most_common(1)[0][0]
        rejected_label = loss_counts.most_common(1)[0][0]

        if chosen_label == rejected_label:
            continue

        # Serialise prompt messages to a plain string
        prompt_messages = comp.get("prompt", [])
        prompt_str = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in prompt_messages
        )

        result.append(
            {
                "prompt": prompt_str,
                "chosen": responses[chosen_label],
                "rejected": responses[rejected_label],
            }
        )

    return result


@dataclass
class CoValSample:
    """One comparison entry from openai/coval."""

    prompt_messages: list[dict]       # [{"role": str, "content": str}]
    responses: dict[str, str]         # {"A": text, "B": text, "C": text, "D": text}
    assessments: list[dict]
    prompt_id: str


class CoValDataset(Dataset):
    """PyTorch Dataset wrapping CoVal comparison rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._samples: list[CoValSample] = []
        for row in rows:
            self._samples.append(
                CoValSample(
                    prompt_messages=row.get("prompt", []),
                    responses=row.get("responses", {}),
                    assessments=row.get("metadata", {}).get("assessments", []),
                    prompt_id=row.get("prompt_id", ""),
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> CoValSample:
        return self._samples[idx]


def load_coval(cache_dir: str | None = None) -> CoValDataset:
    """Load openai/coval comparisons from HuggingFace and return a CoValDataset."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset(
        "openai/coval", data_files="comparisons.jsonl", cache_dir=cache_dir
    )
    # load_dataset returns a DatasetDict with "train" key when data_files is given
    rows = list(raw["train"]) if hasattr(raw, "keys") else list(raw)
    return CoValDataset(rows)


# ── GDPval ────────────────────────────────────────────────────────────────────

@dataclass
class GDPvalSample:
    """One entry from openai/gdpval."""

    task_id: str
    sector: str
    occupation: str
    prompt: str
    rubric_pretty: str


class GDPvalDataset(Dataset):
    """PyTorch Dataset wrapping GDPval rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._samples: list[GDPvalSample] = []
        for row in rows:
            self._samples.append(
                GDPvalSample(
                    task_id=row["task_id"],
                    sector=row["sector"],
                    occupation=row["occupation"],
                    prompt=row["prompt"],
                    rubric_pretty=row["rubric_pretty"],
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GDPvalSample:
        return self._samples[idx]


def load_gdpval(
    split: str = "train",
    cache_dir: str | None = None,
) -> GDPvalDataset:
    """Load openai/gdpval from HuggingFace and return a GDPvalDataset."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset("openai/gdpval", split=split, cache_dir=cache_dir)
    return GDPvalDataset(list(raw))


# ── HealthBench ───────────────────────────────────────────────────────────────

@dataclass
class HealthBenchSample:
    """One entry from openai/healthbench."""

    prompt_messages: list[dict]    # [{"role": str, "content": str}]
    completion: str
    category: str
    rubrics: list[dict]            # [{criterion, points, tags}, ...]
    example_tags: list[str]
    prompt_id: str


class HealthBenchDataset(Dataset):
    """PyTorch Dataset wrapping HealthBench rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._samples: list[HealthBenchSample] = []
        for row in rows:
            self._samples.append(
                HealthBenchSample(
                    prompt_messages=row["prompt"],
                    completion=row["completion"],
                    category=row["category"],
                    rubrics=row["rubrics"],
                    example_tags=row["example_tags"],
                    prompt_id=row["prompt_id"],
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> HealthBenchSample:
        return self._samples[idx]


def load_healthbench(
    split: str = "test",
    cache_dir: str | None = None,
) -> HealthBenchDataset:
    """Load openai/healthbench from HuggingFace and return a HealthBenchDataset."""
    hf_datasets = _datasets_module()

    raw = hf_datasets.load_dataset(
        "openai/healthbench", split=split, cache_dir=cache_dir
    )
    return HealthBenchDataset(list(raw))
