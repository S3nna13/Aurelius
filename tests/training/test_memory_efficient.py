"""Tests for src/training/memory_efficient.py."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.memory_efficient import (
    GradientAccumulator,
    MemEffConfig,
    MemEfficientTrainer,
    MemoryTracker,
    checkpointed_forward,
    chunked_cross_entropy,
)

# ---------------------------------------------------------------------------
# Common fixtures / helpers
# ---------------------------------------------------------------------------

VOCAB = 256
SEQ = 8
BATCH = 2


def _small_cfg(use_gradient_checkpointing: bool = False) -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=VOCAB,
        max_seq_len=512,
        use_gradient_checkpointing=use_gradient_checkpointing,
    )


def _small_model(use_gradient_checkpointing: bool = False) -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(_small_cfg(use_gradient_checkpointing))


def _input_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (BATCH, SEQ))


# ---------------------------------------------------------------------------
# MemEffConfig tests
# ---------------------------------------------------------------------------


def test_memeffconfig_defaults():
    cfg = MemEffConfig()
    assert cfg.use_checkpointing is True
    assert cfg.chunk_size == 1
    assert cfg.offload_optimizer is False
    assert cfg.mixed_precision is False


# ---------------------------------------------------------------------------
# checkpointed_forward tests
# ---------------------------------------------------------------------------


def test_checkpointed_forward_output_shape():
    """checkpointed_forward returns the same output shape as a normal forward."""
    model = nn.Linear(16, 32)
    x = torch.randn(4, 16, requires_grad=True)
    out = checkpointed_forward(model, x)
    assert out.shape == (4, 32)


def test_checkpointed_forward_differentiable():
    """Gradients flow back through checkpointed_forward."""
    model = nn.Linear(16, 32)
    x = torch.randn(4, 16, requires_grad=True)
    out = checkpointed_forward(model, x)
    out.sum().backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# chunked_cross_entropy tests
# ---------------------------------------------------------------------------


def test_chunked_ce_returns_scalar():
    logits = torch.randn(BATCH, SEQ, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ))
    loss = chunked_cross_entropy(logits, labels, chunk_size=2)
    assert loss.ndim == 0  # scalar


def test_chunked_ce_matches_f_cross_entropy():
    """With chunk_size == seq_len the result equals F.cross_entropy."""
    torch.manual_seed(1)
    logits = torch.randn(BATCH, SEQ, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ))
    chunked = chunked_cross_entropy(logits, labels, chunk_size=SEQ)
    reference = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))
    assert torch.isclose(chunked, reference, atol=1e-5), (
        f"chunked={chunked.item():.6f}, ref={reference.item():.6f}"
    )


def test_chunked_ce_respects_ignore_index():
    """Tokens with ignore_index=-100 should not contribute to the loss."""
    logits = torch.randn(BATCH, SEQ, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ))
    # Mask first half of every sequence
    labels[:, : SEQ // 2] = -100
    loss_masked = chunked_cross_entropy(logits, labels, chunk_size=2, ignore_index=-100)
    # Reference: only positions SEQ//2: contribute
    valid_logits = logits[:, SEQ // 2 :, :].reshape(-1, VOCAB)
    valid_labels = labels[:, SEQ // 2 :].reshape(-1)
    reference = F.cross_entropy(valid_logits, valid_labels)
    assert torch.isclose(loss_masked, reference, atol=1e-5), (
        f"masked={loss_masked.item():.6f}, ref={reference.item():.6f}"
    )


def test_chunked_ce_chunk1_equals_chunk_seq():
    """chunk_size=1 and chunk_size=SEQ should give the same loss."""
    torch.manual_seed(7)
    logits = torch.randn(BATCH, SEQ, VOCAB)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ))
    loss1 = chunked_cross_entropy(logits, labels, chunk_size=1)
    lossT = chunked_cross_entropy(logits, labels, chunk_size=SEQ)
    assert torch.isclose(loss1, lossT, atol=1e-4), (
        f"chunk_size=1: {loss1.item():.6f}, chunk_size=SEQ: {lossT.item():.6f}"
    )


# ---------------------------------------------------------------------------
# GradientAccumulator tests
# ---------------------------------------------------------------------------


def _make_accum(n: int = 4):
    model = nn.Linear(8, 1)
    accum = GradientAccumulator(model, n_accumulate=n)
    return model, accum


def test_gradient_accumulator_returns_false_for_first_n_minus_1():
    model, accum = _make_accum(4)
    x = torch.randn(2, 8)
    results = []
    for _ in range(3):
        loss = model(x).sum()
        results.append(accum.step(loss))
    assert all(r is False for r in results)


def test_gradient_accumulator_returns_true_on_nth_call():
    model, accum = _make_accum(4)
    x = torch.randn(2, 8)
    result = None
    for _ in range(4):
        loss = model(x).sum()
        result = accum.step(loss)
    assert result is True


def test_gradient_accumulator_current_step_increments():
    model, accum = _make_accum(4)
    x = torch.randn(2, 8)
    for i in range(1, 5):
        loss = model(x).sum()
        accum.step(loss)
        assert accum.current_step == i


# ---------------------------------------------------------------------------
# MemoryTracker tests
# ---------------------------------------------------------------------------


def test_memory_tracker_record_returns_required_keys():
    tracker = MemoryTracker()
    result = tracker.record()
    assert "allocated_mb" in result
    assert "reserved_mb" in result
    assert "peak_mb" in result
    # Values must be finite floats (0.0 on CPU)
    for v in result.values():
        assert isinstance(v, float)
        assert math.isfinite(v)


def test_memory_tracker_context_manager():
    tracker = MemoryTracker()
    with tracker:
        _ = torch.randn(100, 100)
    # After exiting, peak_mb should be a valid float
    assert isinstance(tracker._peak_mb, float)
    assert math.isfinite(tracker._peak_mb)


# ---------------------------------------------------------------------------
# MemEfficientTrainer tests
# ---------------------------------------------------------------------------


def _make_trainer(chunk_size: int = 2) -> tuple[MemEfficientTrainer, AureliusTransformer]:
    model = _small_model()
    cfg = MemEffConfig(use_checkpointing=False, chunk_size=chunk_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = MemEfficientTrainer(model, cfg, optimizer)
    return trainer, model


def test_train_step_returns_required_keys():
    trainer, _ = _make_trainer()
    ids = _input_ids()
    result = trainer.train_step(ids)
    assert "loss" in result
    assert "peak_mb" in result


def test_train_step_loss_is_finite():
    trainer, _ = _make_trainer()
    ids = _input_ids()
    result = trainer.train_step(ids)
    assert math.isfinite(result["loss"])
    assert result["loss"] > 0.0


def test_train_step_with_gradient_checkpointing():
    """Trainer works when model is configured with gradient checkpointing."""
    model = _small_model(use_gradient_checkpointing=True)
    cfg = MemEffConfig(use_checkpointing=True, chunk_size=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    trainer = MemEfficientTrainer(model, cfg, optimizer)
    ids = _input_ids()
    result = trainer.train_step(ids)
    assert math.isfinite(result["loss"])
