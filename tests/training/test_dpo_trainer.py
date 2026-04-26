"""Tests for DPO trainer (src/training/dpo_trainer.py).

15 tests covering DPOLoss, SequenceLogProbs, ReferenceModelManager,
DPOTrainer, and PreferenceDataset.

Tiny model config: vocab=16, d_model=8, seq_len=8, batch=2.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.training.dpo_trainer import (
    DPOLoss,
    DPOTrainer,
    PreferenceDataset,
    ReferenceModelManager,
    SequenceLogProbs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 8
SEQ_LEN = 8
BATCH = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TinyLM(nn.Module):
    """Minimal Embedding + Linear language model for testing.

    forward(input_ids: LongTensor (B, T)) -> logits (B, T, vocab_size)
    """

    def __init__(self, vocab: int = VOCAB, d_model: int = D_MODEL) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(input_ids))  # (B, T, V)


def make_batch(
    batch: int = BATCH,
    seq_len: int = SEQ_LEN,
    vocab: int = VOCAB,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (chosen_ids, chosen_labels, rejected_ids, rejected_labels)."""
    chosen_ids = torch.randint(0, vocab, (batch, seq_len))
    chosen_labels = chosen_ids.clone()
    rejected_ids = torch.randint(0, vocab, (batch, seq_len))
    rejected_labels = rejected_ids.clone()
    return chosen_ids, chosen_labels, rejected_ids, rejected_labels


def make_trainer(beta: float = 0.1) -> tuple[DPOTrainer, TinyLM]:
    policy = TinyLM()
    ref_mgr = ReferenceModelManager(policy)
    dpo_loss = DPOLoss(beta=beta)
    opt = torch.optim.SGD(policy.parameters(), lr=1e-2)
    trainer = DPOTrainer(policy, ref_mgr, dpo_loss, opt)
    return trainer, policy


# ---------------------------------------------------------------------------
# DPOLoss tests
# ---------------------------------------------------------------------------


def test_dpo_loss_is_scalar() -> None:
    """DPO loss output must be a scalar (0-d) tensor."""
    loss_fn = DPOLoss(beta=0.1)
    B = BATCH
    policy_c = torch.randn(B, requires_grad=True)
    policy_r = torch.randn(B, requires_grad=True)
    ref_c = torch.randn(B)
    ref_r = torch.randn(B)
    loss, _, _ = loss_fn(policy_c, policy_r, ref_c, ref_r)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"


def test_dpo_loss_is_finite() -> None:
    """DPO loss must be finite for standard inputs."""
    loss_fn = DPOLoss(beta=0.1)
    policy_c = torch.randn(BATCH, requires_grad=True)
    policy_r = torch.randn(BATCH, requires_grad=True)
    ref_c = torch.randn(BATCH)
    ref_r = torch.randn(BATCH)
    loss, _, _ = loss_fn(policy_c, policy_r, ref_c, ref_r)
    assert torch.isfinite(loss), "DPO loss must be finite"


def test_dpo_loss_grad_flows() -> None:
    """Gradient must flow back to policy log-probs through DPO loss."""
    loss_fn = DPOLoss(beta=0.1)
    policy_c = torch.randn(BATCH, requires_grad=True)
    policy_r = torch.randn(BATCH, requires_grad=True)
    ref_c = torch.randn(BATCH)
    ref_r = torch.randn(BATCH)
    loss, _, _ = loss_fn(policy_c, policy_r, ref_c, ref_r)
    loss.backward()
    assert policy_c.grad is not None, "No grad on policy_chosen_logps"
    assert policy_r.grad is not None, "No grad on policy_rejected_logps"
    assert torch.isfinite(policy_c.grad).all()
    assert torch.isfinite(policy_r.grad).all()


def test_dpo_loss_beta_zero_near_zero() -> None:
    """With beta=0, rewards cancel and loss approaches -log(sigmoid(0))=log(2)."""
    loss_fn = DPOLoss(beta=0.0)
    policy_c = torch.randn(BATCH)
    policy_r = torch.randn(BATCH)
    ref_c = torch.randn(BATCH)
    ref_r = torch.randn(BATCH)
    loss, chosen_r, rejected_r = loss_fn(policy_c, policy_r, ref_c, ref_r)
    # When beta=0 chosen_reward = rejected_reward = 0 => logits_diff = 0
    # loss = -log(sigmoid(0)) = log(2)
    expected = math.log(2)
    assert abs(loss.item() - expected) < 1e-5, (
        f"Expected loss~{expected:.5f} with beta=0, got {loss.item():.5f}"
    )
    assert torch.all(chosen_r == 0.0), "Chosen rewards should be zero when beta=0"
    assert torch.all(rejected_r == 0.0), "Rejected rewards should be zero when beta=0"


def test_dpo_loss_rewards_detached() -> None:
    """Returned rewards must be detached (no grad_fn)."""
    loss_fn = DPOLoss(beta=0.1)
    policy_c = torch.randn(BATCH, requires_grad=True)
    policy_r = torch.randn(BATCH, requires_grad=True)
    ref_c = torch.randn(BATCH)
    ref_r = torch.randn(BATCH)
    _, chosen_r, rejected_r = loss_fn(policy_c, policy_r, ref_c, ref_r)
    assert chosen_r.grad_fn is None, "chosen_reward should be detached"
    assert rejected_r.grad_fn is None, "rejected_reward should be detached"


# ---------------------------------------------------------------------------
# SequenceLogProbs tests
# ---------------------------------------------------------------------------


def test_seq_logprobs_output_shape() -> None:
    """SequenceLogProbs.compute must return shape (batch,)."""
    model = TinyLM()
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    with torch.no_grad():
        logits = model(ids)
    logps = SequenceLogProbs.compute(logits, ids)
    assert logps.shape == (BATCH,), f"Expected ({BATCH},), got {logps.shape}"


def test_seq_logprobs_handles_ignore_index() -> None:
    """Positions with ignore_index=-100 must not contribute to log-prob sum."""
    model = TinyLM()
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    labels_all = ids.clone()
    labels_masked = ids.clone()
    labels_masked[:, SEQ_LEN // 2 :] = -100  # mask second half

    with torch.no_grad():
        logits = model(ids)

    logps_all = SequenceLogProbs.compute(logits, labels_all)
    logps_masked = SequenceLogProbs.compute(logits, labels_masked)

    # Masked version sums fewer tokens, so magnitude should be smaller or equal
    assert (logps_masked.abs() <= logps_all.abs() + 1e-5).all(), (
        "Masked logps should not exceed full logps in magnitude"
    )


def test_seq_logprobs_values_are_finite() -> None:
    """SequenceLogProbs must always return finite values."""
    model = TinyLM()
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    with torch.no_grad():
        logits = model(ids)
    logps = SequenceLogProbs.compute(logits, ids)
    assert torch.isfinite(logps).all(), "Log-probs must be finite"


# ---------------------------------------------------------------------------
# ReferenceModelManager tests
# ---------------------------------------------------------------------------


def test_ref_model_params_frozen() -> None:
    """All reference model parameters must have requires_grad=False."""
    policy = TinyLM()
    ref_mgr = ReferenceModelManager(policy)
    assert ref_mgr.is_frozen(), "Reference model must be fully frozen"


def test_ref_model_output_matches_direct_call() -> None:
    """Reference logps must match a direct no-grad call on the copied model."""
    policy = TinyLM()
    ref_mgr = ReferenceModelManager(policy)
    ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
    labels = ids.clone()

    ref_logps = ref_mgr.compute_logps(ids, labels)

    with torch.no_grad():
        direct_logits = ref_mgr.model(ids)
        direct_logps = SequenceLogProbs.compute(direct_logits, labels)

    assert torch.allclose(ref_logps, direct_logps, atol=1e-6), (
        "compute_logps must match direct model call"
    )


# ---------------------------------------------------------------------------
# DPOTrainer tests
# ---------------------------------------------------------------------------


def test_trainer_step_returns_all_keys() -> None:
    """train_step must return dict with loss, chosen_reward, rejected_reward, reward_margin."""
    trainer, _ = make_trainer()
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = make_batch()
    result = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
    for key in ("loss", "chosen_reward", "rejected_reward", "reward_margin"):
        assert key in result, f"Missing key '{key}' in train_step output"


def test_trainer_loss_is_finite() -> None:
    """Loss returned by train_step must be finite."""
    trainer, _ = make_trainer()
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = make_batch()
    result = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
    assert torch.isfinite(result["loss"]), "train_step loss must be finite"


def test_trainer_reward_margin_shape() -> None:
    """reward_margin must have shape (batch,) matching the batch dimension."""
    trainer, _ = make_trainer()
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = make_batch()
    result = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
    assert result["reward_margin"].shape == (BATCH,), (
        f"reward_margin shape mismatch: expected ({BATCH},), got {result['reward_margin'].shape}"
    )


def test_trainer_loss_decreases_over_steps() -> None:
    """Loss should decrease over multiple steps on a fixed batch."""
    trainer, _ = make_trainer(beta=0.5)
    chosen_ids, chosen_labels, rejected_ids, rejected_labels = make_batch()
    losses = []
    for _ in range(20):
        result = trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        losses.append(result["loss"].item())
    # Last few steps should be lower than the first
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_trainer_grads_finite_after_clip() -> None:
    """After grad clip, all policy parameters must have finite gradients."""
    policy = TinyLM()
    ref_mgr = ReferenceModelManager(policy)
    dpo_loss = DPOLoss(beta=0.1)
    opt = torch.optim.SGD(policy.parameters(), lr=1e-2)
    trainer = DPOTrainer(policy, ref_mgr, dpo_loss, opt)

    chosen_ids, chosen_labels, rejected_ids, rejected_labels = make_batch()
    # Run a step; grads are computed and clipped internally
    trainer.train_step(chosen_ids, chosen_labels, rejected_ids, rejected_labels)

    for name, param in policy.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Non-finite grad in parameter '{name}'"


# ---------------------------------------------------------------------------
# PreferenceDataset tests
# ---------------------------------------------------------------------------


def test_preference_dataset_len() -> None:
    """PreferenceDataset must report the correct length."""
    n = 5
    chosen = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(n)]
    rejected = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(n)]
    ds = PreferenceDataset(chosen, rejected)
    assert len(ds) == n, f"Expected len={n}, got {len(ds)}"


def test_preference_dataset_getitem() -> None:
    """__getitem__ must return (chosen_ids, rejected_ids) tensors."""
    n = 4
    chosen = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(n)]
    rejected = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(n)]
    ds = PreferenceDataset(chosen, rejected)
    c, r = ds[0]
    assert c.shape == (SEQ_LEN,)
    assert r.shape == (SEQ_LEN,)
    assert torch.equal(c, chosen[0])
    assert torch.equal(r, rejected[0])


def test_preference_dataset_collate_fn_shapes() -> None:
    """collate_fn must stack sequences into (batch, seq_len) tensors."""
    chosen = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(BATCH)]
    rejected = [torch.randint(0, VOCAB, (SEQ_LEN,)) for _ in range(BATCH)]
    ds = PreferenceDataset(chosen, rejected)
    batch = [ds[i] for i in range(BATCH)]
    chosen_batch, rejected_batch = PreferenceDataset.collate_fn(batch)
    assert chosen_batch.shape == (BATCH, SEQ_LEN), (
        f"chosen_batch shape {chosen_batch.shape} != ({BATCH}, {SEQ_LEN})"
    )
    assert rejected_batch.shape == (BATCH, SEQ_LEN), (
        f"rejected_batch shape {rejected_batch.shape} != ({BATCH}, {SEQ_LEN})"
    )
