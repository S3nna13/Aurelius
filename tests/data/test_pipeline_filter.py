"""Tests for FineWeb-Edu score filtering in the pipeline config."""

import pytest


def test_edu_score_filter_default():
    """PipelineConfig default edu_score_min should be 3.0."""
    try:
        from src.data.pipeline import PipelineConfig
    except ImportError:
        pytest.skip("datatrove not installed")
    cfg = PipelineConfig()
    assert hasattr(cfg, "edu_score_min"), "PipelineConfig missing edu_score_min field"
    assert cfg.edu_score_min == 3.0


def test_edu_score_filter_disabled():
    """Setting edu_score_min=0.0 should disable filtering."""
    try:
        from src.data.pipeline import PipelineConfig
    except ImportError:
        pytest.skip("datatrove not installed")
    cfg = PipelineConfig(edu_score_min=0.0)
    assert cfg.edu_score_min == 0.0
