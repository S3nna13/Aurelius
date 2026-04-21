"""Integration test for ZeroVisionSFTTrainer.

Verifies end-to-end: config, compute_loss, backward, registry wiring.
Tiny config: vocab_size=256, seq_len=16, batch=2.
Pure PyTorch only — no transformers, trl, einops, scipy, sklearn, PIL, cv2, timm.
"""

from __future__ import annotations

import torch
import pytest

from src.alignment.zero_vision_sft import ZeroVisionSFTConfig, ZeroVisionSFTTrainer


VOCAB = 256
SEQ = 16
BATCH = 2
TOOL_IDS = [10, 11, 12]


def _synthetic_batch():
    """Return (logits, targets, tool_call_mask) for integration smoke test."""
    torch.manual_seed(0)
    logits = torch.randn(BATCH, SEQ, VOCAB, requires_grad=True)

    # Targets: non-zero values in [1, VOCAB-1], with tool-call ids scattered in
    targets = torch.randint(1, VOCAB, (BATCH, SEQ))
    targets[0, 2] = 10   # tool-call token
    targets[1, 5] = 11   # tool-call token
    targets[0, 9] = 12   # tool-call token
    # Pad a few positions
    targets[0, 14] = 0
    targets[1, 15] = 0

    return logits, targets


def test_integration_compute_loss_scalar_and_backward():
    """Full round-trip: create trainer, build mask, compute_loss, backward."""
    cfg = ZeroVisionSFTConfig(
        tool_call_weight=3.0,
        pad_id=0,
        tool_call_token_ids=TOOL_IDS,
    )
    trainer = ZeroVisionSFTTrainer(cfg)

    logits, targets = _synthetic_batch()

    # Build tool_call_mask from the targets
    tool_call_mask = trainer.make_tool_call_mask(targets)

    # Verify mask shape
    assert tool_call_mask.shape == (BATCH, SEQ)

    # Positions with tool-call ids should be 1
    assert tool_call_mask[0, 2].item() == 1.0   # id=10
    assert tool_call_mask[1, 5].item() == 1.0   # id=11
    assert tool_call_mask[0, 9].item() == 1.0   # id=12

    # Compute loss
    loss = trainer.compute_loss(logits, targets, tool_call_mask)

    # Must be scalar
    assert loss.shape == torch.Size([])
    assert loss.ndim == 0

    # Must be finite
    assert torch.isfinite(loss)

    # Backward must not raise
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


def test_integration_train_step():
    """train_step returns correct keys and types."""
    cfg = ZeroVisionSFTConfig(tool_call_token_ids=TOOL_IDS)
    trainer = ZeroVisionSFTTrainer(cfg)

    logits, targets = _synthetic_batch()
    tool_call_mask = trainer.make_tool_call_mask(targets)

    model_output = {"logits": logits}
    batch = {
        "input_ids": targets,
        "labels": targets,
        "tool_call_mask": tool_call_mask,
    }

    result = trainer.train_step(model_output, batch)

    assert isinstance(result["loss"], torch.Tensor)
    assert result["loss"].ndim == 0
    assert isinstance(result["n_tokens"], int)
    assert isinstance(result["n_tool_call_tokens"], int)
    assert result["n_tokens"] > 0

    # n_tool_call_tokens should match the number of non-pad positions in targets
    # that contain one of the TOOL_IDS — computed independently here.
    non_pad = targets != cfg.pad_id
    tc_mask_ref = torch.zeros_like(targets, dtype=torch.bool)
    for tid in TOOL_IDS:
        tc_mask_ref |= (targets == tid)
    expected_tc = int((tc_mask_ref & non_pad).sum().item())
    assert result["n_tool_call_tokens"] == expected_tc
    assert result["n_tool_call_tokens"] > 0  # sanity: at least one tool-call token present


def test_integration_alignment_registry():
    """ALIGNMENT_REGISTRY['zero_vision_sft'] must map to ZeroVisionSFTTrainer."""
    from src.alignment import ALIGNMENT_REGISTRY

    assert "zero_vision_sft" in ALIGNMENT_REGISTRY
    assert ALIGNMENT_REGISTRY["zero_vision_sft"] is ZeroVisionSFTTrainer
