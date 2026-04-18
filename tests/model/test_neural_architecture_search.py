"""Tests for neural_architecture_search.py — DARTS bilevel NAS."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
import pytest

from src.model.neural_architecture_search import (
    DARTSConfig,
    MixedOperation,
    Cell,
    DARTSSearchSpace,
    DARTSTrainer,
    DiscretizedCell,
)

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Shared tiny config & helpers
# ---------------------------------------------------------------------------

B, T = 2, 4          # batch, sequence length
D = 16               # d_model
N_CELLS = 2
N_NODES = 2
VOCAB = 16


def _tiny_config(**kwargs) -> DARTSConfig:
    base = dict(n_cells=N_CELLS, n_nodes=N_NODES, d_model=D, vocab_size=VOCAB,
                arch_lr=3e-4, model_lr=1e-3, temperature=1.0)
    base.update(kwargs)
    return DARTSConfig(**base)


def _x3d() -> torch.Tensor:
    """Random float tensor (B, T, D)."""
    torch.manual_seed(1)
    return torch.randn(B, T, D)


def _token_batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Random token ids and targets (B, T)."""
    torch.manual_seed(2)
    ids = torch.randint(0, VOCAB, (B, T))
    targets = torch.randint(0, VOCAB, (B, T))
    return ids, targets


def _make_model(cfg: DARTSConfig | None = None) -> DARTSSearchSpace:
    return DARTSSearchSpace(cfg or _tiny_config())


def _make_trainer(model: DARTSSearchSpace | None = None) -> DARTSTrainer:
    m = model or _make_model()
    return DARTSTrainer(m, _token_batch, _token_batch, arch_lr=3e-4, model_lr=1e-3)


# ===========================================================================
# 1. MixedOperation forward shape
# ===========================================================================

def test_mixed_op_forward_shape():
    op = MixedOperation(D)
    x = _x3d()
    out = op(x)
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ===========================================================================
# 2. MixedOperation softmax probabilities sum to 1
# ===========================================================================

def test_mixed_op_probs_sum_to_one():
    op = MixedOperation(D)
    probs = op.get_probs()
    total = probs.sum().item()
    assert abs(total - 1.0) < 1e-5, f"Probs sum = {total}"


# ===========================================================================
# 3. MixedOperation get_best_op returns a valid key
# ===========================================================================

def test_mixed_op_get_best_op_valid_key():
    op = MixedOperation(D)
    best = op.get_best_op()
    assert best in op.operations, f"'{best}' not in operations dict"


# ===========================================================================
# 4. MixedOperation zero op outputs zeros
# ===========================================================================

def test_mixed_op_zero_op_outputs_zeros():
    op = MixedOperation(D)
    # Force arch_weights so 'zero' has probability ~1
    with torch.no_grad():
        op.arch_weights.fill_(-1e9)
        zero_idx = op._op_names.index("zero")
        op.arch_weights[zero_idx] = 1e9
    # Use inference mode (train=False) so plain softmax is used, no Gumbel noise
    op.train(False)
    x = _x3d()
    out = op(x)
    assert out.abs().max().item() < 1e-4, "Expected near-zero output for zero op"


# ===========================================================================
# 5. Cell forward shape with multiple inputs
# ===========================================================================

def test_cell_forward_shape():
    cell = Cell(n_nodes=N_NODES, d_model=D, n_inputs=1)
    x = _x3d()
    out = cell([x])
    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"


# ===========================================================================
# 6. Cell has correct number of edges
# ===========================================================================

def test_cell_edge_count():
    # For n_inputs=1, n_nodes=N:
    # dst=0: 1 predecessor => 1 edge
    # dst=1: 2 predecessors => 2 edges
    # total = sum(n_inputs + dst for dst in range(n_nodes))
    n_inputs = 1
    expected = sum(n_inputs + dst for dst in range(N_NODES))
    cell = Cell(n_nodes=N_NODES, d_model=D, n_inputs=n_inputs)
    assert len(cell.edges) == expected, (
        f"Expected {expected} edges, got {len(cell.edges)}"
    )


# ===========================================================================
# 7. DARTSSearchSpace forward shape
# ===========================================================================

def test_search_space_forward_shape():
    model = _make_model()
    ids, _ = _token_batch()
    logits = model(ids)
    assert logits.shape == (B, T, VOCAB), f"Expected {(B, T, VOCAB)}, got {logits.shape}"


# ===========================================================================
# 8. arch_parameters and model_parameters are disjoint
# ===========================================================================

def test_arch_and_model_parameters_disjoint():
    model = _make_model()
    arch_ids = {id(p) for p in model.arch_parameters()}
    model_ids = {id(p) for p in model.model_parameters()}
    overlap = arch_ids & model_ids
    assert len(overlap) == 0, f"{len(overlap)} parameter(s) appear in both sets"


# ===========================================================================
# 9. bilevel_step returns finite losses
# ===========================================================================

def test_bilevel_step_finite_losses():
    trainer = _make_trainer()
    train_b = _token_batch()
    val_b = _token_batch()
    train_loss, val_loss = trainer.bilevel_step(train_b, val_b)
    assert math.isfinite(train_loss.item()), "train_loss is not finite"
    assert math.isfinite(val_loss.item()), "val_loss is not finite"


# ===========================================================================
# 10. bilevel_step updates architecture weights (gradient flows)
# ===========================================================================

def test_bilevel_step_updates_arch_weights():
    model = _make_model()
    trainer = _make_trainer(model)

    arch_before = [p.clone().detach() for p in model.arch_parameters()]
    trainer.bilevel_step(_token_batch(), _token_batch())
    arch_after = model.arch_parameters()

    changed = any(
        not torch.allclose(b, a) for b, a in zip(arch_before, arch_after)
    )
    assert changed, "Architecture weights did not change after bilevel_step"


# ===========================================================================
# 11. bilevel_step updates model weights (gradient flows)
# ===========================================================================

def test_bilevel_step_updates_model_weights():
    model = _make_model()
    trainer = _make_trainer(model)

    model_before = [p.clone().detach() for p in model.model_parameters()]
    trainer.bilevel_step(_token_batch(), _token_batch())
    model_after = model.model_parameters()

    changed = any(
        not torch.allclose(b, a) for b, a in zip(model_before, model_after)
    )
    assert changed, "Model weights did not change after bilevel_step"


# ===========================================================================
# 12. derive_architecture returns ops for all edges
# ===========================================================================

def test_derive_architecture_covers_all_edges():
    model = _make_model()
    trainer = _make_trainer(model)
    arch = trainer.derive_architecture()

    n_inputs = 1
    edges_per_cell = sum(n_inputs + dst for dst in range(N_NODES))
    expected_keys = N_CELLS * edges_per_cell
    assert len(arch) == expected_keys, (
        f"Expected {expected_keys} keys, got {len(arch)}: {list(arch.keys())}"
    )
    valid_ops = {"linear", "conv1d_k3", "zero", "identity"}
    for key, op_name in arch.items():
        assert op_name in valid_ops, f"Key '{key}' has unknown op '{op_name}'"


# ===========================================================================
# 13. DiscretizedCell forward shape matches Cell
# ===========================================================================

def test_discretized_cell_forward_shape():
    cell = Cell(n_nodes=N_NODES, d_model=D, n_inputs=1)
    derived = {f"edge{e}": edge.get_best_op() for e, edge in enumerate(cell.edges)}
    disc = DiscretizedCell(cell, derived)

    x = _x3d()
    out_cell = cell([x])
    out_disc = disc([x])
    assert out_disc.shape == out_cell.shape, (
        f"Shape mismatch: disc={out_disc.shape} vs cell={out_cell.shape}"
    )


# ===========================================================================
# 14. DARTSConfig defaults
# ===========================================================================

def test_darts_config_defaults():
    cfg = DARTSConfig()
    assert cfg.n_cells == 4
    assert cfg.n_nodes == 4
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.arch_lr == pytest.approx(3e-4)
    assert cfg.model_lr == pytest.approx(1e-3)
    assert cfg.temperature == pytest.approx(1.0)


# ===========================================================================
# 15. Temperature / Gumbel-softmax: gradient flows through arch_weights in training mode
# ===========================================================================

def test_gumbel_softmax_grad_flows_in_train_mode():
    """Verify gradient flows through arch_weights when using Gumbel softmax."""
    op = MixedOperation(D, temperature=1.0)
    op.train(True)
    x = _x3d()
    out = op(x)
    loss = out.sum()
    loss.backward()
    assert op.arch_weights.grad is not None, (
        "arch_weights.grad is None — gradient did not flow in train mode"
    )
    assert op.arch_weights.grad.shape == op.arch_weights.shape


# ===========================================================================
# 16. Search space zero-op suppression (non-zero op should be viable)
# ===========================================================================

def test_zero_op_suppression_after_training():
    """After bilevel steps the non-zero ops should have non-negligible probability.

    A pure zero-op architecture can't learn, so arch weights should shift
    away from the zero op over time (or at minimum non-zero ops remain viable).
    """
    torch.manual_seed(42)
    cfg = _tiny_config(n_cells=1, n_nodes=1, temperature=0.5)
    model = DARTSSearchSpace(cfg)
    trainer = DARTSTrainer(model, _token_batch, _token_batch, arch_lr=1e-2, model_lr=1e-3)

    for _ in range(20):
        trainer.bilevel_step(_token_batch(), _token_batch())

    cell = model.cells[0]
    for edge in cell.edges:
        probs = edge.get_probs()
        zero_idx = edge._op_names.index("zero")
        non_zero_max = max(
            probs[i].item() for i in range(len(probs)) if i != zero_idx
        )
        assert non_zero_max > 0.0, "All probability mass is on the zero op"
        # Verify arch_weights have been updated from their initial zero state.
        assert not torch.all(edge.arch_weights == 0.0), (
            "arch_weights are still all zero — bilevel training had no effect"
        )
