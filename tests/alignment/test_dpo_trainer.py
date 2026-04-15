"""Tests for src/alignment/dpo_trainer.py"""
import pytest
import torch
import torch.nn as nn

from src.alignment.dpo_trainer import (
    DPOConfig,
    compute_log_probs,
    dpo_loss,
    ipo_loss,
    compute_reward_margin,
    compute_reward_accuracy,
    DPOTrainer,
)

# ---------------------------------------------------------------------------
# Tiny constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 16
D = 4
T = 6
B = 2


def _mock_model(ids: torch.Tensor) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(ids.shape[0], ids.shape[1], VOCAB_SIZE)


def _make_labels(b: int = B, t: int = T) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (b, t))


# ---------------------------------------------------------------------------
# DPOConfig
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = DPOConfig()
    assert cfg.beta == 0.1
    assert cfg.label_smoothing == 0.0
    assert cfg.loss_type == "sigmoid"
    assert cfg.reference_free is False


# ---------------------------------------------------------------------------
# compute_log_probs
# ---------------------------------------------------------------------------

def test_compute_log_probs_shape():
    logits = torch.randn(B, T, VOCAB_SIZE)
    labels = _make_labels()
    lp = compute_log_probs(logits, labels)
    assert lp.shape == (B,)


def test_compute_log_probs_ignores_padding():
    logits = torch.randn(B, T, VOCAB_SIZE)
    labels = _make_labels()
    # Mask second half as padding
    labels_pad = labels.clone()
    labels_pad[:, T // 2:] = -100
    lp_full = compute_log_probs(logits, labels)
    lp_pad = compute_log_probs(logits, labels_pad)
    # Masked version sums fewer tokens → different from full
    assert not torch.allclose(lp_full, lp_pad)


def test_compute_log_probs_all_masked_returns_zero():
    logits = torch.randn(B, T, VOCAB_SIZE)
    labels = torch.full((B, T), -100, dtype=torch.long)
    lp = compute_log_probs(logits, labels)
    assert torch.allclose(lp, torch.zeros(B))


# ---------------------------------------------------------------------------
# dpo_loss
# ---------------------------------------------------------------------------

def test_dpo_loss_returns_three_tensors():
    pc = torch.randn(B)
    pr = torch.randn(B)
    rc = torch.randn(B)
    rr = torch.randn(B)
    result = dpo_loss(pc, pr, rc, rr, DPOConfig())
    assert len(result) == 3


def test_dpo_loss_is_scalar():
    pc, pr, rc, rr = [torch.randn(B) for _ in range(4)]
    loss, _, _ = dpo_loss(pc, pr, rc, rr, DPOConfig())
    assert loss.shape == ()


def test_dpo_loss_chosen_rewards_shape():
    pc, pr, rc, rr = [torch.randn(B) for _ in range(4)]
    _, chosen_r, _ = dpo_loss(pc, pr, rc, rr, DPOConfig())
    assert chosen_r.shape == (B,)


def test_dpo_sigmoid_loss_positive():
    pc, pr, rc, rr = [torch.randn(B) for _ in range(4)]
    loss, _, _ = dpo_loss(pc, pr, rc, rr, DPOConfig(loss_type="sigmoid"))
    assert loss.item() > 0.0


def test_dpo_ipo_loss_type():
    pc, pr, rc, rr = [torch.randn(B) for _ in range(4)]
    loss, _, _ = dpo_loss(pc, pr, rc, rr, DPOConfig(loss_type="ipo"))
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_dpo_invalid_loss_type():
    pc, pr, rc, rr = [torch.randn(B) for _ in range(4)]
    with pytest.raises(ValueError):
        dpo_loss(pc, pr, rc, rr, DPOConfig(loss_type="bad"))


# ---------------------------------------------------------------------------
# ipo_loss
# ---------------------------------------------------------------------------

def test_ipo_loss_standalone():
    pc, pr, rc, rr = [torch.randn(B) for _ in range(4)]
    loss = ipo_loss(pc, pr, rc, rr, beta=0.1)
    assert loss.shape == ()
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Reward metrics
# ---------------------------------------------------------------------------

def test_reward_margin_positive_when_chosen_higher():
    chosen = torch.tensor([1.0, 1.0])
    rejected = torch.tensor([-1.0, -1.0])
    assert compute_reward_margin(chosen, rejected) > 0.0


def test_reward_accuracy_perfect():
    chosen = torch.tensor([2.0, 2.0])
    rejected = torch.tensor([0.0, 0.0])
    assert compute_reward_accuracy(chosen, rejected) == 1.0


def test_reward_accuracy_zero():
    chosen = torch.tensor([0.0, 0.0])
    rejected = torch.tensor([2.0, 2.0])
    assert compute_reward_accuracy(chosen, rejected) == 0.0


# ---------------------------------------------------------------------------
# DPOTrainer
# ---------------------------------------------------------------------------

class _TinyLM(nn.Module):
    """Tiny parametric LM: Embedding → Linear for differentiable logits."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D)
        self.head = nn.Linear(D, VOCAB_SIZE, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(self.embed(ids))  # (B, T, V)


def _make_trainer(reference_free: bool = False) -> DPOTrainer:
    policy = _TinyLM()
    ref = _TinyLM()
    cfg = DPOConfig(reference_free=reference_free)
    opt = torch.optim.SGD(policy.parameters(), lr=1e-3)
    return DPOTrainer(policy, ref, opt, cfg)


def test_train_step_returns_correct_keys():
    trainer = _make_trainer()
    chosen = torch.randint(0, VOCAB_SIZE, (B, T))
    rejected = torch.randint(0, VOCAB_SIZE, (B, T))
    labels_c = _make_labels()
    labels_r = _make_labels()
    metrics = trainer.train_step(chosen, rejected, labels_c, labels_r)
    for key in ["loss", "chosen_reward", "rejected_reward", "reward_margin", "reward_accuracy"]:
        assert key in metrics


def test_train_step_loss_is_finite():
    trainer = _make_trainer()
    chosen = torch.randint(0, VOCAB_SIZE, (B, T))
    rejected = torch.randint(0, VOCAB_SIZE, (B, T))
    metrics = trainer.train_step(chosen, rejected, _make_labels(), _make_labels())
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_evaluate_no_grad():
    trainer = _make_trainer()
    chosen = torch.randint(0, VOCAB_SIZE, (B, T))
    rejected = torch.randint(0, VOCAB_SIZE, (B, T))
    metrics = trainer.evaluate(chosen, rejected, _make_labels(), _make_labels())
    assert "loss" in metrics


def test_reference_free_mode():
    trainer = _make_trainer(reference_free=True)
    chosen = torch.randint(0, VOCAB_SIZE, (B, T))
    rejected = torch.randint(0, VOCAB_SIZE, (B, T))
    metrics = trainer.train_step(chosen, rejected, _make_labels(), _make_labels())
    assert torch.isfinite(torch.tensor(metrics["loss"]))
