"""Unit tests for ZeroVisionSFTTrainer (Kimi K2.5 §4, arXiv:2602.02276).

Config: d_model=64, vocab_size=256, seq_len=16 (tiny).
Pure PyTorch only — no transformers, trl, einops, scipy, sklearn, PIL, cv2, timm.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.alignment.zero_vision_sft import ZeroVisionSFTConfig, ZeroVisionSFTTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB = 256
SEQ = 16
BATCH = 2


def _make_logits(B: int = BATCH, T: int = SEQ, V: int = VOCAB, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, V, requires_grad=True)


def _make_targets(B: int = BATCH, T: int = SEQ, pad_id: int = 0, n_pad: int = 4) -> torch.Tensor:
    """Targets with ``n_pad`` pad tokens at the end of each row."""
    ids = torch.randint(1, VOCAB, (B, T))
    ids[:, T - n_pad:] = pad_id
    return ids


def _make_mask(B: int = BATCH, T: int = SEQ, active_cols: list[int] | None = None) -> torch.Tensor:
    mask = torch.zeros(B, T)
    if active_cols:
        for c in active_cols:
            mask[:, c] = 1.0
    return mask


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = ZeroVisionSFTConfig()
    assert cfg.tool_call_weight == 3.0
    assert cfg.pad_id == 0
    assert cfg.tool_call_token_ids is None


# ---------------------------------------------------------------------------
# 2. compute_loss — no mask → standard CE
# ---------------------------------------------------------------------------

def test_compute_loss_no_mask():
    """All-zeros tool_call_mask should produce the same loss as plain CE."""
    trainer = ZeroVisionSFTTrainer()
    logits = _make_logits()
    targets = _make_targets()
    mask = torch.zeros(BATCH, SEQ)

    loss = trainer.compute_loss(logits, targets, mask)

    # Reference: manual CE on non-pad tokens with weight=1
    non_pad = (targets != 0).float().view(-1)
    ref = (
        F.cross_entropy(logits.view(-1, VOCAB), targets.view(-1), reduction="none")
        * non_pad
    ).sum() / non_pad.sum()

    assert torch.allclose(loss, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. compute_loss — with mask → upweighted positions raise total loss
# ---------------------------------------------------------------------------

def test_compute_loss_with_mask():
    """Upweighted positions should produce higher loss than no-mask version."""
    trainer = ZeroVisionSFTTrainer(ZeroVisionSFTConfig(tool_call_weight=3.0))
    torch.manual_seed(7)
    logits = torch.randn(BATCH, SEQ, VOCAB, requires_grad=False)
    targets = _make_targets(n_pad=2)

    mask_off = torch.zeros(BATCH, SEQ)
    mask_on = _make_mask(active_cols=[2, 5, 8])

    loss_off = trainer.compute_loss(logits, targets, mask_off)
    loss_on = trainer.compute_loss(logits, targets, mask_on)

    assert loss_on.item() > loss_off.item()


# ---------------------------------------------------------------------------
# 4. compute_loss — pad positions excluded
# ---------------------------------------------------------------------------

def test_compute_loss_pad_excluded():
    """Pad positions (target == pad_id) should not contribute to the loss."""
    trainer = ZeroVisionSFTTrainer()

    # All-pad targets except one position
    targets = torch.zeros(1, SEQ, dtype=torch.long)
    targets[0, 0] = 5  # single non-pad token

    torch.manual_seed(99)
    logits = torch.randn(1, SEQ, VOCAB)
    mask = torch.zeros(1, SEQ)

    loss_one = trainer.compute_loss(logits, targets, mask)

    # Only one token; reference is CE on that single token
    ref = F.cross_entropy(logits[:, 0, :], targets[:, 0])
    assert torch.allclose(loss_one, ref, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. compute_loss — output is scalar
# ---------------------------------------------------------------------------

def test_compute_loss_scalar():
    trainer = ZeroVisionSFTTrainer()
    logits = _make_logits()
    targets = _make_targets()
    mask = torch.zeros(BATCH, SEQ)

    loss = trainer.compute_loss(logits, targets, mask)
    assert loss.shape == torch.Size([])
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# 6. compute_loss — backward works
# ---------------------------------------------------------------------------

def test_compute_loss_grad():
    trainer = ZeroVisionSFTTrainer()
    logits = _make_logits()  # requires_grad=True
    targets = _make_targets()
    mask = _make_mask(active_cols=[1, 3])

    loss = trainer.compute_loss(logits, targets, mask)
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.shape == logits.shape


# ---------------------------------------------------------------------------
# 7. make_tool_call_mask — None → all zeros
# ---------------------------------------------------------------------------

def test_make_tool_call_mask_empty():
    """tool_call_token_ids=None should return an all-zeros mask."""
    trainer = ZeroVisionSFTTrainer(ZeroVisionSFTConfig(tool_call_token_ids=None))
    ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    mask = trainer.make_tool_call_mask(ids)
    assert (mask == 0).all()


def test_make_tool_call_mask_empty_list():
    """tool_call_token_ids=[] should return an all-zeros mask."""
    trainer = ZeroVisionSFTTrainer(ZeroVisionSFTConfig(tool_call_token_ids=[]))
    ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    mask = trainer.make_tool_call_mask(ids)
    assert (mask == 0).all()


# ---------------------------------------------------------------------------
# 8. make_tool_call_mask — specific ids get 1
# ---------------------------------------------------------------------------

def test_make_tool_call_mask_specific():
    """Positions with token ids in the list should be 1, others 0."""
    trainer = ZeroVisionSFTTrainer(ZeroVisionSFTConfig(tool_call_token_ids=[5, 6]))

    ids = torch.zeros(1, 8, dtype=torch.long)
    ids[0, 2] = 5
    ids[0, 4] = 6
    ids[0, 6] = 7  # not in list

    mask = trainer.make_tool_call_mask(ids)

    assert mask[0, 2].item() == 1.0
    assert mask[0, 4].item() == 1.0
    assert mask[0, 6].item() == 0.0
    # All non-special positions are 0
    for i in [0, 1, 3, 5, 7]:
        assert mask[0, i].item() == 0.0


# ---------------------------------------------------------------------------
# 9. make_tool_call_mask — output shape matches input
# ---------------------------------------------------------------------------

def test_make_tool_call_mask_shape():
    trainer = ZeroVisionSFTTrainer(ZeroVisionSFTConfig(tool_call_token_ids=[10, 20]))
    ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    mask = trainer.make_tool_call_mask(ids)
    assert mask.shape == (BATCH, SEQ)


# ---------------------------------------------------------------------------
# 10. train_step — returns correct keys
# ---------------------------------------------------------------------------

def test_train_step_keys():
    trainer = ZeroVisionSFTTrainer()
    logits = _make_logits()
    targets = _make_targets()
    mask = _make_mask(active_cols=[0, 1])

    model_output = {"logits": logits}
    batch = {"input_ids": targets, "labels": targets, "tool_call_mask": mask}

    result = trainer.train_step(model_output, batch)
    assert "loss" in result
    assert "n_tokens" in result
    assert "n_tool_call_tokens" in result


# ---------------------------------------------------------------------------
# 11. train_step — n_tokens excludes pad
# ---------------------------------------------------------------------------

def test_train_step_n_tokens():
    trainer = ZeroVisionSFTTrainer()
    n_pad = 4
    targets = _make_targets(n_pad=n_pad)  # last 4 cols are pad=0
    logits = _make_logits()
    mask = torch.zeros(BATCH, SEQ)

    model_output = {"logits": logits}
    batch = {"input_ids": targets, "labels": targets, "tool_call_mask": mask}

    result = trainer.train_step(model_output, batch)
    expected_n_tokens = (targets != 0).sum().item()
    assert result["n_tokens"] == expected_n_tokens


# ---------------------------------------------------------------------------
# 12. train_step — n_tool_call_tokens counts correctly
# ---------------------------------------------------------------------------

def test_train_step_n_tool_call():
    """n_tool_call_tokens counts mask-active positions that are not pad."""
    trainer = ZeroVisionSFTTrainer()
    targets = _make_targets(n_pad=2)

    # Set mask active at col 0 and col SEQ-1 (last col is pad)
    mask = torch.zeros(BATCH, SEQ)
    mask[:, 0] = 1.0
    mask[:, SEQ - 1] = 1.0  # this will be a pad position

    logits = _make_logits()
    model_output = {"logits": logits}
    batch = {"input_ids": targets, "labels": targets, "tool_call_mask": mask}

    result = trainer.train_step(model_output, batch)

    # col 0 is non-pad, col SEQ-1 is pad → only BATCH tool-call tokens counted
    assert result["n_tool_call_tokens"] == BATCH


# ---------------------------------------------------------------------------
# 13. Upweight effect — higher weight → higher loss
# ---------------------------------------------------------------------------

def test_upweight_effect():
    """Increasing tool_call_weight should monotonically increase total loss."""
    torch.manual_seed(42)
    logits = torch.randn(1, SEQ, VOCAB)
    targets = _make_targets(B=1, n_pad=2)
    mask = _make_mask(B=1, active_cols=[0, 1, 2])

    losses = []
    for w in [1.0, 2.0, 5.0]:
        trainer = ZeroVisionSFTTrainer(ZeroVisionSFTConfig(tool_call_weight=w))
        loss = trainer.compute_loss(logits, targets, mask)
        losses.append(loss.item())

    assert losses[0] <= losses[1] <= losses[2], f"Expected monotone: {losses}"


# ---------------------------------------------------------------------------
# 14. All-pad targets → loss is 0
# ---------------------------------------------------------------------------

def test_all_pad():
    """When all targets are pad, loss should be 0."""
    trainer = ZeroVisionSFTTrainer()
    targets = torch.zeros(BATCH, SEQ, dtype=torch.long)  # all pad
    logits = torch.randn(BATCH, SEQ, VOCAB)
    mask = torch.zeros(BATCH, SEQ)

    loss = trainer.compute_loss(logits, targets, mask)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 15. Determinism — same inputs produce same loss
# ---------------------------------------------------------------------------

def test_determinism():
    """Same inputs must always produce the same loss."""
    trainer = ZeroVisionSFTTrainer(ZeroVisionSFTConfig(tool_call_token_ids=[10, 11]))
    torch.manual_seed(123)
    logits = torch.randn(BATCH, SEQ, VOCAB)
    targets = _make_targets()
    mask = _make_mask(active_cols=[3, 7])

    loss_a = trainer.compute_loss(logits.clone(), targets.clone(), mask.clone())
    loss_b = trainer.compute_loss(logits.clone(), targets.clone(), mask.clone())

    assert loss_a.item() == loss_b.item()
