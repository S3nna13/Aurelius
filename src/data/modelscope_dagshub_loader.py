from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelScopeDatasetSample:
    id: str
    instruction: str
    input: str = ""
    output: str = ""
    history: list[list[str]] = field(default_factory=list)
    source: str = ""
    language: str = ""
    quality_score: float = 1.0
    token_count: int = 0


@dataclass
class ModelScopeModelCard:
    model_id: str
    model_type: str
    task: str
    languages: list[str]
    license: str
    downloads: int
    tags: list[str]
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class DagsHubMLRun:
    run_id: str
    run_name: str
    status: str
    duration_seconds: float
    metrics: dict[str, float]
    params: dict[str, str]
    artifact_uri: str = ""


@dataclass
class DagsHubDataFile:
    path: str
    md5: str
    size_bytes: int
    is_dir: bool = False
    cached: bool = True


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_modelscope_sample(raw: dict) -> ModelScopeDatasetSample:
    return ModelScopeDatasetSample(
        id=raw["id"],
        instruction=raw["instruction"],
        input=raw.get("input", ""),
        output=raw.get("output", ""),
        history=raw.get("history", []),
        source=raw.get("source", ""),
        language=raw.get("language", ""),
        quality_score=float(raw.get("quality_score", 1.0)),
        token_count=int(raw.get("token_count", 0)),
    )


def parse_modelscope_model_card(raw: dict) -> ModelScopeModelCard:
    return ModelScopeModelCard(
        model_id=raw["model_id"],
        model_type=raw["model_type"],
        task=raw["task"],
        languages=list(raw.get("language", [])),
        license=raw.get("license", ""),
        downloads=int(raw.get("downloads", 0)),
        tags=list(raw.get("tags", [])),
        metrics=dict(raw.get("metrics", {})),
    )


def parse_dagshub_mlrun(raw: dict) -> DagsHubMLRun:
    """Compute duration_seconds = end_time - start_time."""
    start = raw.get("start_time", 0)
    end = raw.get("end_time", 0)
    duration = float(end - start)
    return DagsHubMLRun(
        run_id=raw["run_id"],
        run_name=raw.get("run_name", ""),
        status=raw.get("status", ""),
        duration_seconds=duration,
        metrics=dict(raw.get("metrics", {})),
        params=dict(raw.get("params", {})),
        artifact_uri=raw.get("artifact_uri", ""),
    )


def parse_dagshub_datafile(raw: dict) -> DagsHubDataFile:
    return DagsHubDataFile(
        path=raw["path"],
        md5=raw["md5"],
        size_bytes=int(raw["size"]),
        is_dir=bool(raw.get("isdir", False)),
        cached=bool(raw.get("cache", True)),
    )


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def filter_by_quality(
    samples: list[ModelScopeDatasetSample], min_score: float = 0.8
) -> list[ModelScopeDatasetSample]:
    return [s for s in samples if s.quality_score >= min_score]


def filter_by_language(
    samples: list[ModelScopeDatasetSample], lang: str
) -> list[ModelScopeDatasetSample]:
    return [s for s in samples if s.language == lang]


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------


def modelscope_to_sft_format(sample: ModelScopeDatasetSample) -> dict:
    """Convert to {'messages': [{'role': 'user', 'content': ...}, {'role': 'assistant', 'content': ...}]}"""  # noqa: E501
    user_content = sample.instruction
    if sample.input:
        user_content = f"{sample.instruction}\n{sample.input}"
    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": sample.output},
        ]
    }


def mlrun_to_summary(run: DagsHubMLRun) -> dict:
    """Return {'run_id', 'status', 'duration_min': float, 'best_metric': str (key of highest metric value)}"""  # noqa: E501
    duration_min = run.duration_seconds / 60.0
    best_metric = max(run.metrics, key=lambda k: run.metrics[k]) if run.metrics else ""
    return {
        "run_id": run.run_id,
        "status": run.status,
        "duration_min": duration_min,
        "best_metric": best_metric,
    }


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------


def mock_modelscope_samples(n: int = 4) -> list[dict]:
    samples = []
    languages = ["zh", "en", "fr", "de"]
    for i in range(n):
        samples.append(
            {
                "id": f"sample_{i:03d}",
                "instruction": "Translate this sentence to English",
                "input": f"Example input text {i}",
                "output": f"Example output text {i}",
                "history": [["previous user message", "previous assistant reply"]]
                if i % 2 == 0
                else [],
                "source": "translation",
                "language": languages[i % len(languages)],
                "domain": "general",
                "quality_score": round(0.7 + (i % 4) * 0.08, 2),
                "token_count": 30 + i * 5,
            }
        )
    return samples


def mock_modelscope_model_cards(n: int = 4) -> list[dict]:
    cards = []
    tasks = ["text-classification", "token-classification", "question-answering", "text-generation"]
    for i in range(n):
        cards.append(
            {
                "model_id": f"damo/nlp_bert_base_{i}",
                "model_type": "bert",
                "task": tasks[i % len(tasks)],
                "language": ["zh", "en"],
                "license": "apache-2.0",
                "downloads": 1000 + i * 500,
                "tags": ["nlp", "bert", tasks[i % len(tasks)]],
                "created_at": f"2024-0{(i % 9) + 1}-01",
                "metrics": {"accuracy": round(0.88 + i * 0.01, 2), "f1": round(0.87 + i * 0.01, 2)},
            }
        )
    return cards


def mock_dagshub_mlruns(n: int = 4) -> list[dict]:
    runs = []
    statuses = ["FINISHED", "RUNNING", "FAILED", "FINISHED"]
    base_start = 1700000000
    for i in range(n):
        start = base_start + i * 7200
        end = start + 3600 + i * 600
        runs.append(
            {
                "run_id": f"run_{i:03x}abc",
                "experiment_id": str(i),
                "run_name": f"experiment_v{i + 1}",
                "status": statuses[i % len(statuses)],
                "start_time": start,
                "end_time": end,
                "metrics": {
                    "train_loss": round(0.3 - i * 0.02, 3),
                    "val_loss": round(0.35 - i * 0.02, 3),
                    "accuracy": round(0.80 + i * 0.02, 3),
                },
                "params": {
                    "learning_rate": str(0.001 / (i + 1)),
                    "batch_size": "32",
                    "epochs": str(10 + i * 5),
                },
                "tags": {
                    "mlflow.source.name": "train.py",
                    "model_type": "transformer",
                },
                "artifact_uri": f"s3://bucket/mlruns/{i}/run_{i:03x}abc/artifacts",
            }
        )
    return runs


def mock_dagshub_datafiles(n: int = 4) -> list[dict]:
    files = []
    for i in range(n):
        files.append(
            {
                "path": f"data/split_{i}.csv",
                "md5": f"abc{i:03d}def{i:03d}",
                "size": 1048576 * (i + 1),
                "nfiles": None,
                "isdir": False,
                "cache": True,
                "metric": False,
                "persist": False,
            }
        )
    return files
