"""Tests for src.training.hub_publisher."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.training.hub_publisher import (
    HUB_PUBLISHER_REGISTRY,
    HubPublisher,
    ModelCardData,
    PublishConfig,
    generate_model_card,
)


def _card_data(**overrides) -> ModelCardData:
    defaults = dict(
        model_id="aurelius-1.4B",
        base_model="aurelius-base",
        training_framework="aurelius",
        task_type="reasoning",
        license="apache-2.0",
        metrics={"loss": 1.23, "acc": 0.91},
    )
    defaults.update(overrides)
    return ModelCardData(**defaults)


def test_publish_config_defaults():
    c = PublishConfig(repo_id="u/m")
    assert c.repo_id == "u/m"
    assert c.token == ""
    assert c.private is False
    assert c.commit_message == "Upload model"


def test_publish_config_custom():
    c = PublishConfig(repo_id="u/m", token="tok", private=True, commit_message="msg")
    assert c.private is True
    assert c.token == "tok"
    assert c.commit_message == "msg"


def test_model_card_data_defaults_metrics_empty():
    d = ModelCardData(
        model_id="m",
        base_model="b",
        training_framework="aurelius",
        task_type="code",
        license="mit",
    )
    assert d.metrics == {}


def test_generate_model_card_has_yaml_block():
    md = generate_model_card(_card_data())
    assert md.startswith("---\n")
    assert "\n---\n" in md


def test_generate_model_card_includes_license():
    md = generate_model_card(_card_data(license="mit"))
    assert "license: mit" in md


def test_generate_model_card_includes_base_model():
    md = generate_model_card(_card_data(base_model="llama-3"))
    assert "base_model: llama-3" in md


def test_generate_model_card_tags_aurelius():
    md = generate_model_card(_card_data())
    assert "- aurelius" in md


def test_generate_model_card_tags_task_and_framework():
    md = generate_model_card(
        _card_data(task_type="code", training_framework="peft")
    )
    assert "- code" in md
    assert "- peft" in md


def test_generate_model_card_sections():
    md = generate_model_card(_card_data())
    for section in ("# aurelius-1.4B", "## Model Description", "## Training", "## Evaluation Metrics"):
        assert section in md


def test_generate_model_card_metrics_table():
    md = generate_model_card(_card_data(metrics={"loss": 0.5}))
    assert "| Metric | Value |" in md
    assert "| loss | 0.5 |" in md


def test_generate_model_card_no_metrics():
    md = generate_model_card(_card_data(metrics={}))
    assert "No metrics recorded." in md


def test_generate_model_card_includes_model_id():
    md = generate_model_card(_card_data(model_id="my-model"))
    assert "my-model" in md


def test_generate_model_card_returns_string():
    assert isinstance(generate_model_card(_card_data()), str)


def test_hub_publisher_stores_config():
    cfg = PublishConfig(repo_id="u/m")
    pub = HubPublisher(cfg)
    assert pub.config is cfg


def test_hub_publisher_dry_run_no_token(tmp_path: Path):
    cfg = PublishConfig(repo_id="u/m")
    pub = HubPublisher(cfg)
    result = pub.publish(str(tmp_path), _card_data())
    assert result["status"] == "dry_run"


def test_hub_publisher_dry_run_contains_repo_id(tmp_path: Path):
    cfg = PublishConfig(repo_id="user/model")
    pub = HubPublisher(cfg)
    result = pub.publish(str(tmp_path), _card_data())
    assert result["repo_id"] == "user/model"


def test_hub_publisher_dry_run_has_timestamp(tmp_path: Path):
    cfg = PublishConfig(repo_id="u/m")
    pub = HubPublisher(cfg)
    result = pub.publish(str(tmp_path), _card_data())
    assert "committed_at" in result and isinstance(result["committed_at"], str)


def test_hub_publisher_dry_run_has_card_preview(tmp_path: Path):
    cfg = PublishConfig(repo_id="u/m")
    pub = HubPublisher(cfg)
    result = pub.publish(str(tmp_path), _card_data())
    assert "card_preview" in result
    assert len(result["card_preview"]) <= 200


def test_hub_publisher_registry():
    assert "default" in HUB_PUBLISHER_REGISTRY
    assert HUB_PUBLISHER_REGISTRY["default"] is HubPublisher


def test_publish_config_is_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(PublishConfig)


def test_model_card_data_is_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(ModelCardData)


def test_generate_model_card_mentions_aurelius_pipeline():
    md = generate_model_card(_card_data())
    assert "Aurelius" in md


def test_generate_model_card_ordering():
    md = generate_model_card(_card_data())
    yaml_end = md.find("---", 4)
    h1 = md.find("# ")
    assert 0 < yaml_end < h1


def test_hub_publisher_dry_run_when_token_but_no_hf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import src.training.hub_publisher as hp

    monkeypatch.setattr(hp, "_HF_HUB_AVAILABLE", False)
    pub = hp.HubPublisher(PublishConfig(repo_id="u/m", token="tok"))
    result = pub.publish(str(tmp_path), _card_data())
    assert result["status"] == "dry_run"


def test_generate_model_card_task_body():
    md = generate_model_card(_card_data(task_type="tool_calling"))
    assert "tool_calling" in md


def test_generate_model_card_metrics_multiple_rows():
    md = generate_model_card(_card_data(metrics={"a": 1, "b": 2, "c": 3}))
    for k in ("a", "b", "c"):
        assert f"| {k} |" in md


def test_hub_publisher_publish_returns_dict(tmp_path: Path):
    pub = HubPublisher(PublishConfig(repo_id="u/m"))
    result = pub.publish(str(tmp_path), _card_data())
    assert isinstance(result, dict)


def test_generate_model_card_nonempty():
    assert len(generate_model_card(_card_data())) > 100
