"""Tests for automatic model documentation (model card) generation."""

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.eval.model_card import (
    ParameterBreakdown,
    ModelCard,
    count_parameters_by_component,
    build_model_card,
)


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


def test_count_params_total(small_model):
    bd = count_parameters_by_component(small_model)
    total_actual = sum(p.numel() for p in small_model.parameters())
    assert bd.total == total_actual


def test_count_params_components_sum(small_model):
    bd = count_parameters_by_component(small_model)
    assert bd.embed + bd.attention + bd.ffn + bd.lm_head + bd.other == bd.total


def test_count_params_attention_nonzero(small_model):
    bd = count_parameters_by_component(small_model)
    assert bd.attention > 0


def test_count_params_ffn_nonzero(small_model):
    bd = count_parameters_by_component(small_model)
    assert bd.ffn > 0


def test_parameter_breakdown_summary(small_model):
    bd = count_parameters_by_component(small_model)
    s = bd.summary()
    assert "Total parameters" in s
    assert "Attention" in s


def test_model_card_to_markdown(small_model):
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    card = build_model_card(
        small_model,
        config=cfg,
        benchmark_results={"perplexity": 12.5, "accuracy": 0.75},
        model_name="TestModel",
    )
    md = card.to_markdown()
    assert "TestModel" in md
    assert "Parameters" in md
    assert "Benchmark Results" in md
    assert "perplexity" in md


def test_model_card_markdown_format(small_model):
    card = ModelCard(
        model_name="X",
        parameters=count_parameters_by_component(small_model),
        benchmark_results={"acc": 0.9},
        training_notes=["Trained on synthetic data"],
    )
    md = card.to_markdown()
    assert "# X Model Card" in md
    assert "Training Notes" in md
    assert "Trained on synthetic data" in md
