"""Integration tests: registry wiring + config flag for vibe code reviewer."""

from __future__ import annotations

import src.eval as eval_mod
from src.model.config import AureliusConfig


def test_registered_in_metric_and_benchmark_registry():
    assert "vibe_code_reviewer" in eval_mod.METRIC_REGISTRY
    assert "vibe_code_reviewer" in eval_mod.BENCHMARK_REGISTRY
    assert eval_mod.METRIC_REGISTRY["vibe_code_reviewer"] is eval_mod.VibeCodeReviewer
    assert eval_mod.BENCHMARK_REGISTRY["vibe_code_reviewer"] is eval_mod.VibeCodeReviewer


def test_config_flag_default_off():
    cfg = AureliusConfig()
    assert hasattr(cfg, "eval_vibe_code_reviewer_enabled")
    assert cfg.eval_vibe_code_reviewer_enabled is False


def test_module_exports_complete():
    for name in (
        "VULNHUNTER_SYSTEM_PROMPT",
        "VIBE_NEGATIVE_EXAMPLES",
        "VibeFinding",
        "VibeReviewReport",
        "VibeCodeReviewer",
        "vibe_stub_judge_fn",
        "eval_vibe_code_reviewer_enabled",
    ):
        assert hasattr(eval_mod, name), f"missing export: {name}"


def test_reviewer_instantiable_from_registry():
    cls = eval_mod.METRIC_REGISTRY["vibe_code_reviewer"]
    reviewer = cls(eval_mod.vibe_stub_judge_fn, min_severity="low")
    report = reviewer.review_file("x.py", code="# VIBE_SSRF_SINK\n")
    assert report.findings
    assert report.findings[0].cwe == "CWE-918"
