"""Tests for AdaLoRA (Adaptive Rank Allocation for LoRA) implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.training.adalora import (
    SVDLoRALayer,
    RankBudgetAllocator,
    AdaLoRAMask,
    AdaLoRATrainer,
    AdaLoRARegularizer,
)

# ---------------------------------------------------------------------------
# Shared constants — tiny config as required
# ---------------------------------------------------------------------------
IN = 16
OUT = 16
RANK = 4
ALPHA = 1.0
TOTAL_BUDGET = 8
N_LAYERS = 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_layer(rank: int = RANK) -> SVDLoRALayer:
    torch.manual_seed(0)
    return SVDLoRALayer(IN, OUT, rank=rank, alpha=ALPHA)


def make_tiny_model(vocab: int = 32, seq: int = 16) -> nn.Module:
    """Tiny embedding + linear head that returns logits (B, T, V)."""

    class TinyLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, OUT)
            self.head = nn.Linear(OUT, vocab)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.emb(x))

    torch.manual_seed(1)
    return TinyLM()


# ---------------------------------------------------------------------------
# Test 1 — SVDLoRALayer: delta_W has correct shape (out, in)
# ---------------------------------------------------------------------------
def test_svd_delta_w_shape() -> None:
    layer = make_layer()
    delta_W = layer.P * layer.Lambda.unsqueeze(0) @ layer.Q
    assert delta_W.shape == (OUT, IN), f"Expected ({OUT},{IN}), got {delta_W.shape}"


# ---------------------------------------------------------------------------
# Test 2 — SVDLoRALayer.forward: output shape correct
# ---------------------------------------------------------------------------
def test_svd_forward_shape() -> None:
    layer = make_layer()
    x = torch.randn(2, IN)
    out = layer(x)
    assert out.shape == (2, OUT)


# ---------------------------------------------------------------------------
# Test 3 — SVDLoRALayer.forward: grad flows through P, Lambda, Q
# ---------------------------------------------------------------------------
def test_svd_forward_gradients() -> None:
    layer = make_layer()
    x = torch.randn(2, IN)
    out = layer(x)
    out.sum().backward()
    assert layer.P.grad is not None, "No grad on P"
    assert layer.Lambda.grad is not None, "No grad on Lambda"
    assert layer.Q.grad is not None, "No grad on Q"


# ---------------------------------------------------------------------------
# Test 4 — SVDLoRALayer.effective_rank: decreases after zeroing Lambda
# ---------------------------------------------------------------------------
def test_svd_effective_rank_decreases() -> None:
    layer = make_layer()
    before = layer.effective_rank()
    # Zero out half the Lambda values
    with torch.no_grad():
        layer.Lambda.data[:2] = 0.0
    after = layer.effective_rank()
    assert after < before, f"effective_rank did not decrease: {before} -> {after}"


# ---------------------------------------------------------------------------
# Test 5 — SVDLoRALayer.importance_score: shape (rank,), values >= 0
# ---------------------------------------------------------------------------
def test_svd_importance_score() -> None:
    layer = make_layer()
    scores = layer.importance_score()
    assert scores.shape == (RANK,), f"Expected shape ({RANK},), got {scores.shape}"
    assert (scores >= 0).all(), "Importance scores must be non-negative"


# ---------------------------------------------------------------------------
# Test 6 — RankBudgetAllocator.compute_allocation: total ≤ budget, all ≥ 1
# ---------------------------------------------------------------------------
def test_allocator_total_within_budget() -> None:
    allocator = RankBudgetAllocator(TOTAL_BUDGET, N_LAYERS, RANK)
    importance_scores = {
        "layer0": torch.tensor([1.0, 0.5, 0.3, 0.1]),
        "layer1": torch.tensor([0.9, 0.8, 0.2, 0.05]),
    }
    allocation = allocator.compute_allocation(importance_scores)
    assert sum(allocation.values()) <= TOTAL_BUDGET
    for name, r in allocation.items():
        assert r >= 1, f"Layer {name} got rank {r} < 1"


# ---------------------------------------------------------------------------
# Test 7 — RankBudgetAllocator.prune_decision: returns lowest-importance indices
# ---------------------------------------------------------------------------
def test_allocator_prune_decision() -> None:
    allocator = RankBudgetAllocator(TOTAL_BUDGET, N_LAYERS, RANK)
    # current_rank=4, new_rank=2 → should prune 2 indices
    indices = allocator.prune_decision("layer0", new_rank=2, current_rank=4)
    assert len(indices) == 2
    # Indices should be within [0, current_rank)
    for idx in indices:
        assert 0 <= idx < 4


# ---------------------------------------------------------------------------
# Test 8 — AdaLoRAMask.set_mask + apply: pruned Lambda positions become 0.0
# ---------------------------------------------------------------------------
def test_mask_apply_zeros_pruned() -> None:
    layer = make_layer()
    # Set all Lambda non-zero
    with torch.no_grad():
        layer.Lambda.data.fill_(1.0)

    mask = AdaLoRAMask(RANK)
    # Only keep indices 0 and 1 active
    mask.set_mask([0, 1])
    mask.apply(layer)

    with torch.no_grad():
        # indices 2 and 3 should be zeroed
        assert layer.Lambda.data[2].item() == 0.0
        assert layer.Lambda.data[3].item() == 0.0
        # indices 0 and 1 should remain non-zero
        assert layer.Lambda.data[0].item() == 1.0
        assert layer.Lambda.data[1].item() == 1.0


# ---------------------------------------------------------------------------
# Test 9 — AdaLoRAMask.active_count: matches len(active_indices)
# ---------------------------------------------------------------------------
def test_mask_active_count() -> None:
    mask = AdaLoRAMask(RANK)
    mask.set_mask([0, 2])
    assert mask.active_count() == 2
    mask.set_mask([0, 1, 3])
    assert mask.active_count() == 3


# ---------------------------------------------------------------------------
# Test 10 — AdaLoRATrainer.train_step: all keys present, loss finite
# ---------------------------------------------------------------------------
def test_trainer_train_step_keys_and_finite() -> None:
    model = make_tiny_model(vocab=32, seq=16)
    lora_layers = {
        "layer0": make_layer(),
        "layer1": make_layer(),
    }
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [p for l in lora_layers.values() for p in l.parameters()],
        lr=1e-3,
    )
    trainer = AdaLoRATrainer(model, lora_layers, optimizer, rank_update_interval=10, total_rank_budget=TOTAL_BUDGET)

    input_ids = torch.randint(0, 32, (2, 16))
    labels = torch.randint(0, 32, (2, 16))
    result = trainer.train_step(input_ids, labels)

    assert "loss" in result
    assert "total_active_rank" in result
    assert "per_layer_ranks" in result
    assert torch.isfinite(torch.tensor(result["loss"]))


# ---------------------------------------------------------------------------
# Test 11 — AdaLoRATrainer.update_ranks: total_active_rank ≤ budget after update
# ---------------------------------------------------------------------------
def test_trainer_update_ranks_within_budget() -> None:
    model = make_tiny_model(vocab=32, seq=16)
    lora_layers = {
        "layer0": make_layer(),
        "layer1": make_layer(),
    }
    optimizer = torch.optim.Adam(
        list(model.parameters()) + [p for l in lora_layers.values() for p in l.parameters()],
        lr=1e-3,
    )
    trainer = AdaLoRATrainer(model, lora_layers, optimizer, rank_update_interval=10, total_rank_budget=TOTAL_BUDGET)
    trainer.update_ranks()

    total_active = sum(
        trainer._masks[name].active_count() for name in lora_layers
    )
    assert total_active <= TOTAL_BUDGET


# ---------------------------------------------------------------------------
# Test 12 — AdaLoRARegularizer.loss: >= 0, and 0 when P and Q are orthonormal
# ---------------------------------------------------------------------------
def test_regularizer_loss_non_negative_and_zero_at_orthonormal() -> None:
    reg = AdaLoRARegularizer(beta=0.001)
    layer = make_layer()

    # General case: loss >= 0
    loss_val = reg.loss(layer)
    assert loss_val.item() >= 0.0

    # Orthonormal case: set P and Q to have orthonormal columns/rows
    # Use QR decomposition to get orthonormal factors
    with torch.no_grad():
        P_rand = torch.randn(OUT, RANK)
        P_orth, _ = torch.linalg.qr(P_rand)  # (OUT, RANK) orthonormal columns
        layer.P.data.copy_(P_orth)

        Q_rand = torch.randn(RANK, IN)
        Q_orth, _ = torch.linalg.qr(Q_rand.T)  # QR of transpose → orthonormal rows
        layer.Q.data.copy_(Q_orth.T[:RANK, :])

    loss_ortho = reg.loss(layer)
    assert loss_ortho.item() >= 0.0
    # Should be very close to 0
    assert loss_ortho.item() < 1e-8, f"Expected near-zero orthonormal loss, got {loss_ortho.item()}"


# ---------------------------------------------------------------------------
# Test 13 — AdaLoRARegularizer.total_loss: sum of per-layer losses
# ---------------------------------------------------------------------------
def test_regularizer_total_loss_equals_sum() -> None:
    reg = AdaLoRARegularizer(beta=0.001)
    layers = {
        "layer0": make_layer(),
        "layer1": make_layer(),
    }
    total = reg.total_loss(layers)
    expected = sum(reg.loss(l).item() for l in layers.values())
    assert abs(total.item() - expected) < 1e-6


# ---------------------------------------------------------------------------
# Test 14 — SVDLoRALayer with rank=1: forward/backward still works
# ---------------------------------------------------------------------------
def test_svd_rank1_forward_backward() -> None:
    layer = SVDLoRALayer(IN, OUT, rank=1, alpha=ALPHA)
    x = torch.randn(2, IN)
    out = layer(x)
    assert out.shape == (2, OUT)
    out.sum().backward()
    assert layer.P.grad is not None
    assert layer.Lambda.grad is not None
    assert layer.Q.grad is not None


# ---------------------------------------------------------------------------
# Test 15 — Full training loop: 5 steps with rank update at step 3, no error
# ---------------------------------------------------------------------------
def test_full_training_loop_five_steps() -> None:
    torch.manual_seed(42)
    model = make_tiny_model(vocab=32, seq=8)
    lora_layers = {
        "layer0": SVDLoRALayer(IN, OUT, rank=RANK, alpha=ALPHA),
        "layer1": SVDLoRALayer(IN, OUT, rank=RANK, alpha=ALPHA),
    }
    optimizer = torch.optim.SGD(
        list(model.parameters()) + [p for l in lora_layers.values() for p in l.parameters()],
        lr=1e-2,
    )
    # rank_update_interval=3 so update fires at step 3
    trainer = AdaLoRATrainer(
        model,
        lora_layers,
        optimizer,
        rank_update_interval=3,
        total_rank_budget=TOTAL_BUDGET,
    )

    losses = []
    for step in range(5):
        input_ids = torch.randint(0, 32, (2, 8))
        labels = torch.randint(0, 32, (2, 8))
        result = trainer.train_step(input_ids, labels)
        losses.append(result["loss"])

    # All 5 steps completed, all losses finite
    assert len(losses) == 5
    for i, l in enumerate(losses):
        assert torch.isfinite(torch.tensor(l)), f"Loss at step {i} not finite: {l}"

    # After rank update (step 3), total active rank must be within budget
    total_active = sum(trainer._masks[name].active_count() for name in lora_layers)
    assert total_active <= TOTAL_BUDGET
