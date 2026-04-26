"""Tests for ROME rank-1 factual editing (src/eval/rome.py)."""

from __future__ import annotations

import pytest
import torch

from src.eval.rome import (
    ROMEConfig,
    ROMEEdit,
    apply_rome_edit,
    get_subject_hidden_state,
    rome_edit,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_small_model() -> AureliusTransformer:
    """4-layer, d_model=64 model for fast CPU tests."""
    cfg = AureliusConfig(
        n_layers=4,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
        dropout=0.0,
        tie_embeddings=False,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def simple_tokenizer(text: str) -> list[int]:
    """Char-level tokenizer mapping each character to its ASCII code (clamped to vocab)."""
    return [min(ord(c), 255) for c in text]


LAYER_IDX = 1  # use layer 1 for small 4-layer model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rome_config_defaults():
    cfg = ROMEConfig()
    assert cfg.layer_idx == 12
    assert cfg.n_gradient_steps == 20
    assert cfg.v_lr == pytest.approx(0.1)
    assert cfg.kl_factor == pytest.approx(0.0625)


def test_rome_edit_dataclass():
    edit = ROMEEdit(
        prompt="The Eiffel Tower is located in",
        target_new=" Berlin",
        target_true=" Paris",
        subject="Eiffel Tower",
    )
    assert edit.prompt == "The Eiffel Tower is located in"
    assert edit.target_new == " Berlin"
    assert edit.target_true == " Paris"
    assert edit.subject == "Eiffel Tower"

    # target_true is optional
    edit2 = ROMEEdit(prompt="foo", target_new="bar")
    assert edit2.target_true is None
    assert edit2.subject == ""


def test_get_subject_hidden_state_shape():
    model = make_small_model()
    prompt = "hello world"
    input_ids = torch.tensor([simple_tokenizer(prompt)], dtype=torch.long)
    pos = len(prompt) - 1  # last token

    k = get_subject_hidden_state(
        model=model,
        input_ids=input_ids,
        subject_last_token_pos=pos,
        layer_idx=LAYER_IDX,
    )

    assert k.shape == (model.config.d_model,), (
        f"Expected shape ({model.config.d_model},), got {k.shape}"
    )
    assert k.dtype == torch.float32


def test_apply_rome_edit_changes_w_out():
    model = make_small_model()
    d_model = model.config.d_model

    # Snapshot original weights
    w_out_before = model.layers[LAYER_IDX].ffn.down_proj.weight.detach().clone()

    k = torch.randn(d_model)
    v = torch.randn(d_model)

    apply_rome_edit(model=model, k=k, v=v, layer_idx=LAYER_IDX)

    w_out_after = model.layers[LAYER_IDX].ffn.down_proj.weight.detach().clone()
    assert not torch.allclose(w_out_before, w_out_after), (
        "W_out should have changed after apply_rome_edit"
    )


def test_apply_rome_edit_rank_one_update():
    model = make_small_model()
    d_model = model.config.d_model

    w_out_before = model.layers[LAYER_IDX].ffn.down_proj.weight.detach().clone()

    k = torch.randn(d_model)
    v = torch.randn(d_model)

    apply_rome_edit(model=model, k=k, v=v, layer_idx=LAYER_IDX)

    w_out_after = model.layers[LAYER_IDX].ffn.down_proj.weight.detach().clone()
    delta_w = w_out_after - w_out_before  # (d_model, d_ff)

    rank = torch.linalg.matrix_rank(delta_w).item()
    assert rank == 1, f"Delta W should be rank-1, got rank {rank}"


def test_rome_edit_runs_without_crash():
    model = make_small_model()
    cfg = ROMEConfig(
        layer_idx=LAYER_IDX,
        n_gradient_steps=3,  # fast
        v_lr=0.1,
    )
    edit = ROMEEdit(
        prompt="hello world",
        target_new="x",
        subject="hello",
    )

    # Should complete without raising any exception
    rome_edit(
        model=model,
        edit=edit,
        tokenizer_fn=simple_tokenizer,
        cfg=cfg,
    )
