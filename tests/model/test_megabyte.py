"""Tests for MegaByte (arXiv:2305.07185) — src/model/megabyte.py.

Covers 14 required test cases.
Pure PyTorch only — no scipy, sklearn, HuggingFace, einops, etc.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.megabyte import MegaByteConfig, MegaByteModel


# ---------------------------------------------------------------------------
# Shared fixture: tiny config for fast tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_cfg() -> MegaByteConfig:
    """Tiny config: patch_size=4, d_local=32, d_global=64."""
    return MegaByteConfig(
        vocab_size=256,
        patch_size=4,
        d_local=32,
        d_global=64,
        n_local_layers=1,
        n_global_layers=2,
        n_heads_local=2,
        n_heads_global=2,
        dropout=0.0,
    )


@pytest.fixture()
def model(tiny_cfg: MegaByteConfig) -> MegaByteModel:
    return MegaByteModel(tiny_cfg).eval()


def _rand_ids(B: int, T: int) -> torch.LongTensor:
    return torch.randint(0, 256, (B, T))


# ---------------------------------------------------------------------------
# Test 1 — logits shape (B, T, 256)
# ---------------------------------------------------------------------------

def test_logits_shape(model: MegaByteModel) -> None:
    B, T = 2, 16
    ids = _rand_ids(B, T)
    logits = model(ids)
    assert logits.shape == (B, T, 256), f"Expected (2,16,256), got {logits.shape}"


# ---------------------------------------------------------------------------
# Test 2 — gradient flow: every parameter gets a finite gradient
# ---------------------------------------------------------------------------

def test_gradient_flow(model: MegaByteModel) -> None:
    model.train()
    ids = _rand_ids(2, 16)
    targets = _rand_ids(2, 16)
    loss, _ = model(ids, targets)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No grad for {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite grad for {name}"


# ---------------------------------------------------------------------------
# Test 3 — determinism under torch.manual_seed
# ---------------------------------------------------------------------------

def test_determinism(tiny_cfg: MegaByteConfig) -> None:
    ids = _rand_ids(2, 16)
    torch.manual_seed(42)
    m1 = MegaByteModel(tiny_cfg).eval()
    logits1 = m1(ids)

    torch.manual_seed(42)
    m2 = MegaByteModel(tiny_cfg).eval()
    logits2 = m2(ids)

    assert torch.allclose(logits1, logits2), "Outputs differ under same seed"


# ---------------------------------------------------------------------------
# Test 4 — batch=1, T=patch_size (single patch)
# ---------------------------------------------------------------------------

def test_single_patch(model: MegaByteModel) -> None:
    P = model.config.patch_size
    ids = _rand_ids(1, P)
    logits = model(ids)
    assert logits.shape == (1, P, 256)


# ---------------------------------------------------------------------------
# Test 5 — T = 4 * patch_size (4 patches)
# ---------------------------------------------------------------------------

def test_four_patches(model: MegaByteModel) -> None:
    P = model.config.patch_size
    T = 4 * P
    ids = _rand_ids(2, T)
    logits = model(ids)
    assert logits.shape == (2, T, 256)


# ---------------------------------------------------------------------------
# Test 6 — T not divisible by patch_size raises ValueError
# ---------------------------------------------------------------------------

def test_bad_sequence_length(model: MegaByteModel) -> None:
    P = model.config.patch_size
    T = P + 1  # not divisible
    ids = _rand_ids(1, T)
    with pytest.raises(ValueError, match="divisible"):
        model(ids)


# ---------------------------------------------------------------------------
# Test 7 — loss is a scalar when targets provided
# ---------------------------------------------------------------------------

def test_loss_scalar(model: MegaByteModel) -> None:
    ids = _rand_ids(2, 16)
    targets = _rand_ids(2, 16)
    result = model(ids, targets)
    assert isinstance(result, tuple) and len(result) == 2
    loss, logits = result
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# Test 8 — no NaN/Inf on zeros input
# ---------------------------------------------------------------------------

def test_no_nan_zeros(model: MegaByteModel) -> None:
    ids = torch.zeros(2, 16, dtype=torch.long)
    logits = model(ids)
    assert torch.isfinite(logits).all(), "NaN/Inf on zeros input"


# ---------------------------------------------------------------------------
# Test 9 — no NaN/Inf on all-255 input (max byte value)
# ---------------------------------------------------------------------------

def test_no_nan_max_byte(model: MegaByteModel) -> None:
    ids = torch.full((2, 16), 255, dtype=torch.long)
    logits = model(ids)
    assert torch.isfinite(logits).all(), "NaN/Inf on all-255 input"


# ---------------------------------------------------------------------------
# Test 10 — global model receives exactly n_patches vectors
# ---------------------------------------------------------------------------

def test_global_receives_n_patches(model: MegaByteModel) -> None:
    P = model.config.patch_size
    T = 5 * P
    n_patches = T // P
    captured: list[torch.Tensor] = []

    handle = model.global_transformer.register_forward_hook(
        lambda _m, inp, _out: captured.append(inp[0])
    )
    try:
        model(_rand_ids(2, T))
    finally:
        handle.remove()

    assert len(captured) == 1
    assert captured[0].shape[1] == n_patches, (
        f"Global model got {captured[0].shape[1]} tokens, expected {n_patches}"
    )


# ---------------------------------------------------------------------------
# Test 11 — local model output has correct per-patch shape (B*n, P, d_l)
# ---------------------------------------------------------------------------

def test_local_output_per_patch_shape(model: MegaByteModel) -> None:
    P = model.config.patch_size
    d_l = model.config.d_local
    B, T = 2, 4 * P
    n = T // P
    captured_in: list[torch.Tensor] = []
    captured_out: list[torch.Tensor] = []

    def _local_hook(_m, inp, out):
        captured_in.append(inp[0])
        captured_out.append(out)

    handle = model.local_transformer.register_forward_hook(_local_hook)
    try:
        model(_rand_ids(B, T))
    finally:
        handle.remove()

    assert len(captured_out) == 1
    assert captured_in[0].shape == (B * n, P, d_l), (
        f"Local input shape {captured_in[0].shape} != ({B*n}, {P}, {d_l})"
    )
    assert captured_out[0].shape == (B * n, P, d_l), (
        f"Local output shape {captured_out[0].shape} != ({B*n}, {P}, {d_l})"
    )


# ---------------------------------------------------------------------------
# Test 12 — loss decreases on trivial repeat pattern (trainability sanity)
# ---------------------------------------------------------------------------

def test_loss_decreases_on_repeat_pattern(tiny_cfg: MegaByteConfig) -> None:
    torch.manual_seed(0)
    m = MegaByteModel(tiny_cfg).train()
    P = tiny_cfg.patch_size
    ids = torch.tensor([[65] * (4 * P)], dtype=torch.long)  # 'A' repeated
    targets = ids.clone()

    opt = torch.optim.Adam(m.parameters(), lr=1e-3)

    losses: list[float] = []
    for _ in range(30):
        opt.zero_grad()
        loss, _ = m(ids, targets)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 13 — byte embedding table has vocab_size=256
# ---------------------------------------------------------------------------

def test_byte_embedding_vocab_size(model: MegaByteModel) -> None:
    embed = model.byte_embed
    assert isinstance(embed, nn.Embedding)
    assert embed.num_embeddings == 256, (
        f"Embedding size {embed.num_embeddings} != 256"
    )
    assert embed.embedding_dim == model.config.d_local


# ---------------------------------------------------------------------------
# Test 14 — global-to-local projection maps d_global → d_local
# ---------------------------------------------------------------------------

def test_global_to_local_projection_shape(model: MegaByteModel) -> None:
    proj = model.global_to_local
    assert isinstance(proj, nn.Linear)
    assert proj.in_features == model.config.d_global, (
        f"in_features={proj.in_features} != d_global={model.config.d_global}"
    )
    assert proj.out_features == model.config.d_local, (
        f"out_features={proj.out_features} != d_local={model.config.d_local}"
    )

    B, n = 3, 5
    h_g = torch.randn(B, n, model.config.d_global)
    out = proj(h_g)
    assert out.shape == (B, n, model.config.d_local)
