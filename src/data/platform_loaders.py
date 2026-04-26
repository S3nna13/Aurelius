from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class KaggleSentimentSample:
    text: str
    label: int  # 0=negative, 4=positive
    user: str = ""
    date: str = ""


@dataclass
class KaggleReadabilitySample:
    excerpt: str
    target: float
    standard_error: float = 0.0
    id: str = ""


@dataclass
class ModelScopeInstructSample:
    instruction: str
    input: str = ""
    output: str = ""
    history: list[list[str]] = field(default_factory=list)


@dataclass
class DagsHubRun:
    run_id: str
    experiment_id: str
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    params: dict[str, str] = field(default_factory=dict)


@dataclass
class ReplicatePrediction:
    id: str
    version: str
    status: str
    input: dict = field(default_factory=dict)
    output: Any = ""
    predict_time: float = 0.0
    error: str | None = None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_kaggle_sentiment(row: dict) -> KaggleSentimentSample:
    """row has: target (0 or 4), id, date, flag, user, text"""
    return KaggleSentimentSample(
        text=str(row.get("text", "")),
        label=int(row.get("target", 0)),
        user=str(row.get("user", "")),
        date=str(row.get("date", "")),
    )


def parse_kaggle_readability(row: dict) -> KaggleReadabilitySample:
    """row has: id, url_legal, license, excerpt, target, standard_error"""
    return KaggleReadabilitySample(
        excerpt=str(row.get("excerpt", "")),
        target=float(row.get("target", 0.0)),
        standard_error=float(row.get("standard_error", 0.0)),
        id=str(row.get("id", "")),
    )


def parse_modelscope_instruct(raw: dict) -> ModelScopeInstructSample:
    """raw has: instruction, input, output, history"""
    return ModelScopeInstructSample(
        instruction=str(raw.get("instruction", "")),
        input=str(raw.get("input", "")),
        output=str(raw.get("output", "")),
        history=list(raw.get("history", [])),
    )


def parse_dagshub_run(raw: dict) -> DagsHubRun:
    """raw has: run_id, experiment_id, status, metrics, params, tags"""
    return DagsHubRun(
        run_id=str(raw.get("run_id", "")),
        experiment_id=str(raw.get("experiment_id", "")),
        status=str(raw.get("status", "")),
        metrics=dict(raw.get("metrics", {})),
        params=dict(raw.get("params", {})),
    )


def parse_replicate_prediction(raw: dict) -> ReplicatePrediction:
    """raw has: id, version, status, input, output, error, metrics (with predict_time)"""
    metrics = raw.get("metrics", {}) or {}
    predict_time = float(metrics.get("predict_time", raw.get("predict_time", 0.0)))
    return ReplicatePrediction(
        id=str(raw.get("id", "")),
        version=str(raw.get("version", "")),
        status=str(raw.get("status", "")),
        input=dict(raw.get("input", {})),
        output=raw.get("output", ""),
        predict_time=predict_time,
        error=raw.get("error", None),
    )


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def kaggle_sentiment_to_instruction(sample: KaggleSentimentSample) -> dict:
    """{'instruction': 'Classify the sentiment...', 'input': sample.text, 'output': 'positive'/'negative'}"""  # noqa: E501
    label_str = "positive" if sample.label == 4 else "negative"
    return {
        "instruction": "Classify the sentiment of the following text as positive or negative.",
        "input": sample.text,
        "output": label_str,
    }


def modelscope_to_instruction(sample: ModelScopeInstructSample) -> dict:
    """{'instruction': ..., 'input': ..., 'output': ..., 'history': ...}"""
    return {
        "instruction": sample.instruction,
        "input": sample.input,
        "output": sample.output,
        "history": sample.history,
    }


def readability_to_regression_sample(sample: KaggleReadabilitySample) -> dict:
    """{'text': excerpt, 'score': target, 'weight': 1/(std_err^2 + 1e-8)}"""
    weight = 1.0 / (sample.standard_error**2 + 1e-8)
    return {
        "text": sample.excerpt,
        "score": sample.target,
        "weight": weight,
    }


# ---------------------------------------------------------------------------
# Mock generators
# ---------------------------------------------------------------------------


def mock_kaggle_sentiment_data(n: int = 4) -> list[dict]:
    """Return n mock Kaggle sentiment rows: target in {0,4}, id, date, flag, user, text"""
    rows = []
    texts = [
        "I love this product, it works great!",
        "Terrible experience, never buying again.",
        "Pretty good overall, happy with it.",
        "Awful quality, broke after one day.",
        "Amazing service, highly recommended!",
        "Not worth the price at all.",
        "Decent product, does the job.",
        "Worst purchase I have ever made.",
    ]
    for i in range(n):
        target = 4 if i % 2 == 0 else 0
        rows.append(
            {
                "target": target,
                "id": str(1000 + i),
                "date": f"Mon Apr {i + 1:02d} 12:00:00 UTC 2024",
                "flag": "NO_QUERY",
                "user": f"user_{i}",
                "text": texts[i % len(texts)],
            }
        )
    return rows


def mock_kaggle_readability_data(n: int = 4) -> list[dict]:
    """Return n mock Kaggle readability rows: id, url_legal, license, excerpt, target, standard_error"""  # noqa: E501
    rows = []
    excerpts = [
        "The quick brown fox jumps over the lazy dog.",
        "In the beginning, the universe was created. This made many people angry.",
        "It was the best of times, it was the worst of times.",
        "Call me Ishmael. Some years ago, never mind how long precisely.",
        "It is a truth universally acknowledged, that a single man in possession.",
        "All happy families are alike; each unhappy family is unhappy in its own way.",
        "The sky above the port was the color of television, tuned to a dead channel.",
        "It was a bright cold day in April, and the clocks were striking thirteen.",
    ]
    for i in range(n):
        rows.append(
            {
                "id": f"excerpt_{i:04d}",
                "url_legal": "",
                "license": "CC",
                "excerpt": excerpts[i % len(excerpts)],
                "target": round(-2.0 + i * 0.5, 4),
                "standard_error": round(0.4 + i * 0.05, 4),
            }
        )
    return rows


def mock_modelscope_data(n: int = 4) -> list[dict]:
    """Return n mock ModelScope instruct rows: instruction, input, output, history"""
    rows = []
    instructions = [
        "Summarize the following text.",
        "Translate the following sentence to French.",
        "Answer the question based on the context.",
        "Write a short poem about the topic.",
        "Explain the concept in simple terms.",
        "Generate a list of five examples.",
        "Classify the intent of the following sentence.",
        "Rewrite the sentence in a formal tone.",
    ]
    for i in range(n):
        history: list[list[str]] = []
        if i > 0:
            history = [["Previous question?", "Previous answer."]]
        rows.append(
            {
                "instruction": instructions[i % len(instructions)],
                "input": f"Sample input text number {i}.",
                "output": f"Sample output text number {i}.",
                "history": history,
            }
        )
    return rows


def mock_dagshub_runs(n: int = 4) -> list[dict]:
    """Return n mock DagsHub run rows: run_id, experiment_id, status, metrics, params, tags"""
    statuses = ["FINISHED", "RUNNING", "FAILED", "SCHEDULED"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "run_id": f"run_{i:04d}",
                "experiment_id": f"exp_{i // 2:03d}",
                "status": statuses[i % len(statuses)],
                "metrics": {
                    "loss": round(2.5 - i * 0.3, 4),
                    "accuracy": round(0.5 + i * 0.1, 4),
                },
                "params": {
                    "lr": str(1e-4 * (i + 1)),
                    "batch_size": str(32 * (i + 1)),
                },
                "tags": {"model": "transformer", "version": str(i)},
            }
        )
    return rows


def mock_replicate_predictions(n: int = 4) -> list[dict]:
    """Return n mock Replicate prediction rows: id, version, status, input, output, error, metrics"""  # noqa: E501
    statuses = ["succeeded", "failed", "processing", "starting"]
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"pred_{i:08d}",
                "version": f"v{i}.0.0",
                "status": statuses[i % len(statuses)],
                "input": {"prompt": f"Test prompt {i}", "max_tokens": 256},
                "output": f"Generated output {i}" if i % 2 == 0 else None,
                "error": None if i % 2 == 0 else f"Error on run {i}",
                "metrics": {"predict_time": round(0.5 + i * 0.25, 4)},
            }
        )
    return rows


# ---------------------------------------------------------------------------
# PlatformDataMixer
# ---------------------------------------------------------------------------


class PlatformDataMixer:
    def __init__(self) -> None:
        self._sources: list[tuple[str, list]] = []

    def add_source(self, name: str, samples: list) -> None:
        self._sources.append((name, list(samples)))

    def mix(self, weights: dict[str, float] | None = None) -> list:
        """Return interleaved samples. If weights=None, equal sampling."""
        if not self._sources:
            return []

        if weights is None:
            # Equal interleaving: round-robin across all sources
            result: list = []
            iterators = [iter(samples) for _, samples in self._sources]
            exhausted = [False] * len(iterators)
            while not all(exhausted):
                for idx, it in enumerate(iterators):
                    if exhausted[idx]:
                        continue
                    try:
                        result.append(next(it))
                    except StopIteration:
                        exhausted[idx] = True
            return result
        else:
            # Weighted sampling: proportional counts
            total_weight = sum(weights.values())
            source_map = {name: samples for name, samples in self._sources}
            total_items = sum(len(s) for s in source_map.values())

            result = []
            for name, samples in self._sources:
                w = weights.get(name, 1.0)
                proportion = w / total_weight
                count = max(1, round(proportion * total_items))
                result.extend(samples[:count])
            return result

    def stats(self) -> dict[str, int]:
        """Return {name: len(samples)} for each source."""
        return {name: len(samples) for name, samples in self._sources}
