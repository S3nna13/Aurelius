"""Integration test for Online RFT — end-to-end filter + train + stats."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.alignment import ALIGNMENT_REGISTRY
from src.alignment.online_rft import (
    OnlineRFTConfig,
    OnlineRFTTrainer,
    RFTSample,
)


# ---------------------------------------------------------------------------
# Tiny mock model for logit generation
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """Minimal transformer-free model: embedding + linear head."""

    VOCAB = 64

    def __init__(self, seq_len: int = 10) -> None:
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB, 16)
        self.head = nn.Linear(16, self.VOCAB)
        self.seq_len = seq_len

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return logits [B, T, V]."""
        x = self.embed(input_ids)   # [B, T, 16]
        return self.head(x)          # [B, T, V]

    def log_probs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return per-token log probs [B, T]."""
        logits = self.forward(input_ids)           # [B, T, V]
        lp = F.log_softmax(logits, dim=-1)         # [B, T, V]
        # Use the argmax token as the "chosen" token for a simple test
        chosen = logits.argmax(dim=-1)             # [B, T]
        return lp.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)  # [B, T]


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_online_rft_full_pipeline():
    """End-to-end integration: filter → compute_loss → backward → statistics."""

    # ---- 1. Registry is wired -------------------------------------------
    assert "online_rft" in ALIGNMENT_REGISTRY, (
        "OnlineRFTTrainer must be registered under 'online_rft'"
    )
    assert ALIGNMENT_REGISTRY["online_rft"] is OnlineRFTTrainer

    # ---- 2. Build trainer -----------------------------------------------
    cfg = OnlineRFTConfig(
        n_candidates=8,
        min_keep_ratio=0.125,
        filter_strategy="correct_only",
        sft_loss_weight=1.0,
        kl_penalty_weight=0.1,
    )
    trainer = OnlineRFTTrainer(cfg)

    # ---- 3. Create 8 candidates: 5 correct, 3 wrong ----------------------
    candidates: list[RFTSample] = []
    for i in range(8):
        is_correct = i < 5
        reward = float(8 - i) / 8.0  # descending rewards
        candidates.append(
            RFTSample(
                prompt_tokens=[1, 2, 3],
                response_tokens=[4, 5, 6, 7, 8],
                is_correct=is_correct,
                reward=reward,
            )
        )

    # ---- 4. Filter candidates -------------------------------------------
    kept = trainer.filter_candidates(candidates)
    assert len(kept) >= 1, "Expected at least 1 candidate to be kept"
    assert all(s.is_correct for s in kept), "All kept candidates should be correct"

    # ---- 5. Build a fake batch from the kept samples --------------------
    model = _TinyModel(seq_len=5)
    B = len(kept)
    T = 5
    V = _TinyModel.VOCAB

    torch.manual_seed(0)
    input_ids = torch.randint(1, V, (B, T))
    labels = torch.randint(1, V, (B, T))   # non-pad labels

    # Forward through model
    logits = model(input_ids)              # [B, T, V]
    policy_lp = model.log_probs(input_ids)  # [B, T]

    # Simulate reference model with slightly different weights
    ref_model = _TinyModel(seq_len=5)
    with torch.no_grad():
        ref_lp = ref_model.log_probs(input_ids)   # [B, T]

    # ---- 6. Compute total loss and backward ----------------------------
    total, metrics = trainer.total_loss(logits, labels, policy_lp, ref_lp)

    assert "sft" in metrics and "kl" in metrics and "total" in metrics, (
        f"Missing keys in metrics: {metrics}"
    )
    assert isinstance(metrics["sft"], float)
    assert isinstance(metrics["kl"], float)
    assert isinstance(metrics["total"], float)

    # Backward should not raise
    total.backward()

    # At least the model head weights should have gradients
    assert model.head.weight.grad is not None, "Expected gradients on model head"

    # ---- 7. Statistics --------------------------------------------------
    stats = trainer.statistics(candidates, kept)

    assert stats["n_candidates"] == 8
    assert stats["n_kept"] == len(kept)
    assert abs(stats["keep_rate"] - len(kept) / 8.0) < 1e-9
    assert stats["n_correct"] == 5

    # mean_reward should be average over all candidates
    expected_mean = sum(s.reward for s in candidates) / len(candidates)
    assert abs(stats["mean_reward"] - expected_mean) < 1e-6, (
        f"Expected mean_reward={expected_mean:.4f}, got {stats['mean_reward']:.4f}"
    )
