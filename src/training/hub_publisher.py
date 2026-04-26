"""Hugging Face Hub publisher with stdlib-only dry-run mode."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    import huggingface_hub  # type: ignore[import-not-found]

    _HF_HUB_AVAILABLE = True
except Exception:
    _HF_HUB_AVAILABLE = False


@dataclass
class PublishConfig:
    """Configuration for publishing a model to the Hub."""

    repo_id: str
    token: str = ""
    private: bool = False
    commit_message: str = "Upload model"


@dataclass
class ModelCardData:
    """Metadata used to render a Hugging Face model card."""

    model_id: str
    base_model: str
    training_framework: str
    task_type: str
    license: str
    metrics: dict[str, Any] = field(default_factory=dict)


def generate_model_card(data: ModelCardData) -> str:
    """Render a ModelCardData instance as a Markdown model card."""
    yaml_lines = [
        "---",
        f"license: {data.license}",
        f"base_model: {data.base_model}",
        "tags:",
        "- aurelius",
        f"- {data.task_type}",
        f"- {data.training_framework}",
        "---",
    ]
    yaml_block = "\n".join(yaml_lines)

    if data.metrics:
        metric_rows = ["| Metric | Value |", "|--------|-------|"]
        for key, value in data.metrics.items():
            metric_rows.append(f"| {key} | {value} |")
        metrics_md = "\n".join(metric_rows)
    else:
        metrics_md = "No metrics recorded."

    body = (
        f"\n\n# {data.model_id}\n\n"
        f"## Model Description\n\n"
        f"This model (`{data.model_id}`) was fine-tuned from `{data.base_model}` "
        f"for the `{data.task_type}` task using the Aurelius training pipeline.\n\n"
        f"## Training\n\n"
        f"- Framework: {data.training_framework}\n"
        f"- Base model: {data.base_model}\n"
        f"- Task: {data.task_type}\n"
        f"- License: {data.license}\n\n"
        f"## Evaluation Metrics\n\n{metrics_md}\n"
    )
    return yaml_block + body


class HubPublisher:
    """Publish trained Aurelius models to the Hugging Face Hub."""

    def __init__(self, config: PublishConfig) -> None:
        self._config = config

    @property
    def config(self) -> PublishConfig:
        return self._config

    def publish(self, model_path: str, card_data: ModelCardData) -> dict[str, Any]:
        card = generate_model_card(card_data)
        path = Path(model_path)

        if not _HF_HUB_AVAILABLE or not self._config.token:
            logger.info(
                "HubPublisher dry_run (hf_hub=%s, token_set=%s)",
                _HF_HUB_AVAILABLE,
                bool(self._config.token),
            )
            return {
                "status": "dry_run",
                "repo_id": self._config.repo_id,
                "committed_at": datetime.now(UTC).isoformat(),
                "card_preview": card[:200],
            }

        try:
            api = huggingface_hub.HfApi(token=self._config.token)
            api.create_repo(
                repo_id=self._config.repo_id,
                private=self._config.private,
                exist_ok=True,
            )
            if path.is_dir():
                readme = path / "README.md"
                readme.write_text(card, encoding="utf-8")
                api.upload_folder(
                    folder_path=str(path),
                    repo_id=self._config.repo_id,
                    commit_message=self._config.commit_message,
                )
            return {
                "status": "success",
                "repo_id": self._config.repo_id,
                "committed_at": datetime.now(UTC).isoformat(),
            }
        except Exception as exc:  # noqa: BLE001
            logger.error("Hub publish failed: %s", exc)
            return {
                "status": "error",
                "repo_id": self._config.repo_id,
                "committed_at": datetime.now(UTC).isoformat(),
                "error": str(exc),
            }


HUB_PUBLISHER_REGISTRY = {"default": HubPublisher}
