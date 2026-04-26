"""Tests for Test-Time Adaptation (TTA) — TENT, MEMO, and helpers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from aurelius.training.test_time_adaptation import (
    AdaptableParams,
    AugmentationPool,
    EntropyMinimizer,
    MEMOAdapter,
)

# ---------------------------------------------------------------------------
# Constants used across tests
# ---------------------------------------------------------------------------
VOCAB_SIZE = 32
D_MODEL = 16
T = 6
BATCH = 2


# ---------------------------------------------------------------------------
# Minimal model helpers
# ---------------------------------------------------------------------------


def _make_small_model(vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL) -> nn.Module:
    """A tiny embedding + LayerNorm + linear language-model head."""
    torch.manual_seed(0)

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=True)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            # token_ids: (B, T) -> logits: (B, T, V)
            x = self.embed(token_ids)  # (B, T, D)
            x = self.ln(x)
            return self.head(x)

    return TinyLM()


def _model_fn(model: nn.Module):
    """Return a Callable suitable for EntropyMinimizer / MEMOAdapter."""

    def fn(x: torch.Tensor) -> torch.Tensor:
        return model(x)

    return fn


def _random_ids(batch: int = BATCH, seq: int = T, vocab: int = VOCAB_SIZE):
    torch.manual_seed(7)
    return torch.randint(0, vocab, (batch, seq))


# ===========================================================================
# 1. AdaptableParams.get_adapt_params with 'affine' returns LayerNorm params
# ===========================================================================


def test_adaptable_params_affine_returns_layernorm_params():
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    params = ap.get_adapt_params()

    assert len(params) > 0, "Expected at least one affine param from LayerNorm"

    # All returned params must be weight or bias of LayerNorm modules
    ln_param_ids = set()
    for mod in model.modules():
        if isinstance(mod, (nn.LayerNorm, nn.BatchNorm1d)):
            if mod.weight is not None:
                ln_param_ids.add(id(mod.weight))
            if mod.bias is not None:
                ln_param_ids.add(id(mod.bias))

    for p in params:
        assert id(p) in ln_param_ids, (
            "get_adapt_params('affine') returned a param not from LayerNorm/BN"
        )


# ===========================================================================
# 2. get_adapt_params with 'all' returns all requires_grad params
# ===========================================================================


def test_adaptable_params_all_returns_all_grad_params():
    model = _make_small_model()
    # Ensure all params have requires_grad=True
    for p in model.parameters():
        p.requires_grad_(True)

    ap = AdaptableParams(model, adapt_mode="all")
    params = ap.get_adapt_params()

    all_grad = [p for p in model.parameters() if p.requires_grad]
    assert len(params) == len(all_grad)

    param_ids = {id(p) for p in params}
    for p in all_grad:
        assert id(p) in param_ids


# ===========================================================================
# 3. freeze_non_adapt sets non-affine params to requires_grad=False
# ===========================================================================


def test_freeze_non_adapt_freezes_non_affine():
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    adapt_ids = {id(p) for p in ap.get_adapt_params()}

    ap.freeze_non_adapt()

    for p in model.parameters():
        if id(p) not in adapt_ids:
            assert not p.requires_grad, (
                "Expected non-affine param to have requires_grad=False after freeze"
            )


# ===========================================================================
# 4. restore_all_grad sets all params to requires_grad=True
# ===========================================================================


def test_restore_all_grad_restores_requires_grad():
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")

    ap.freeze_non_adapt()
    # Confirm at least one param is frozen
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
    assert frozen_count > 0

    ap.restore_all_grad()

    for p in model.parameters():
        assert p.requires_grad, "Expected all params to have requires_grad=True after restore"


# ===========================================================================
# 5. EntropyMinimizer.entropy output shape (B,)
# ===========================================================================


def test_entropy_minimizer_entropy_shape():
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    em = EntropyMinimizer(_model_fn(model), ap.get_adapt_params())

    logits = torch.randn(BATCH, VOCAB_SIZE)
    h = em.entropy(logits)

    assert h.shape == (BATCH,), f"Expected shape ({BATCH},), got {h.shape}"


# ===========================================================================
# 6. entropy in [0, log(V)] — bounded by uniform distribution
# ===========================================================================


def test_entropy_bounded():
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    em = EntropyMinimizer(_model_fn(model), ap.get_adapt_params())

    logits = torch.randn(16, VOCAB_SIZE)
    h = em.entropy(logits)

    max_entropy = math.log(VOCAB_SIZE)
    assert (h >= 0).all(), "Entropy must be non-negative"
    assert (h <= max_entropy + 1e-5).all(), (
        f"Entropy must be <= log(V)={max_entropy:.4f}, got max {h.max().item():.4f}"
    )


# ===========================================================================
# 7. entropy = 0 for one-hot logits
# ===========================================================================


def test_entropy_zero_for_one_hot():
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    em = EntropyMinimizer(_model_fn(model), ap.get_adapt_params())

    # Very large logit at index 0 → near one-hot distribution
    logits = torch.full((1, VOCAB_SIZE), -1e9)
    logits[0, 0] = 1e9
    h = em.entropy(logits)

    assert h.item() < 1e-3, f"Expected entropy ~0 for one-hot logits, got {h.item()}"


# ===========================================================================
# 8. adapt_step returns (logits, float)
# ===========================================================================


def test_adapt_step_return_types():
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    em = EntropyMinimizer(_model_fn(model), ap.get_adapt_params(), lr=1e-3, n_steps=1)

    x = _random_ids()
    result = em.adapt_step(x)

    assert isinstance(result, tuple) and len(result) == 2
    logits, ent_val = result
    assert isinstance(logits, torch.Tensor), "First element must be a Tensor"
    assert isinstance(ent_val, float), "Second element must be a float"


# ===========================================================================
# 9. adapt_step returns finite logits
# ===========================================================================


def test_adapt_step_returns_finite_logits():
    torch.manual_seed(42)
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    em = EntropyMinimizer(_model_fn(model), ap.get_adapt_params(), lr=1e-3, n_steps=1)

    x = _random_ids()
    logits, _ = em.adapt_step(x)

    assert torch.isfinite(logits).all(), "Adapted logits contain NaN/Inf"


# ===========================================================================
# 10. AugmentationPool.augment_token_ids returns shape (n_augmentations, T)
# ===========================================================================


def test_augmentation_pool_shape():
    pool = AugmentationPool(n_augmentations=8, noise_scale=0.1)
    x = _random_ids(batch=1, seq=T)  # (1, T)
    aug = pool.augment_token_ids(x, vocab_size=VOCAB_SIZE)

    assert aug.shape == (8, T), f"Expected (8, {T}), got {aug.shape}"


# ===========================================================================
# 11. token ids in [0, vocab_size)
# ===========================================================================


def test_augmentation_pool_token_range():
    pool = AugmentationPool(n_augmentations=16, noise_scale=0.5)
    x = _random_ids(batch=1, seq=T)
    aug = pool.augment_token_ids(x, vocab_size=VOCAB_SIZE)

    assert (aug >= 0).all(), "Augmented token ids must be >= 0"
    assert (aug < VOCAB_SIZE).all(), f"Augmented token ids must be < {VOCAB_SIZE}"


# ===========================================================================
# 12. augmentations differ from original (noise introduces changes)
# ===========================================================================


def test_augmentation_pool_introduces_changes():
    torch.manual_seed(99)
    pool = AugmentationPool(n_augmentations=8, noise_scale=0.5)
    x = torch.zeros(1, T, dtype=torch.long)  # all zeros
    aug = pool.augment_token_ids(x, vocab_size=VOCAB_SIZE)

    # With noise_scale=0.5 and 8 augmentations, expect at least some changed tokens
    changed = (aug != x).any()
    assert changed, "Expected at least some tokens to differ from original"


# ===========================================================================
# 13. MEMOAdapter.marginal_entropy returns scalar
# ===========================================================================


def test_memo_marginal_entropy_scalar():
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    pool = AugmentationPool(n_augmentations=8)
    memo = MEMOAdapter(_model_fn(model), ap.get_adapt_params(), pool, VOCAB_SIZE)

    logits_batch = torch.randn(8, VOCAB_SIZE)
    h = memo.marginal_entropy(logits_batch)

    assert h.shape == (), f"Expected scalar, got shape {h.shape}"


# ===========================================================================
# 14. marginal_entropy in [0, log(V)]
# ===========================================================================


def test_memo_marginal_entropy_bounded():
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    pool = AugmentationPool(n_augmentations=8)
    memo = MEMOAdapter(_model_fn(model), ap.get_adapt_params(), pool, VOCAB_SIZE)

    logits_batch = torch.randn(8, VOCAB_SIZE)
    h = memo.marginal_entropy(logits_batch)

    max_entropy = math.log(VOCAB_SIZE)
    assert h.item() >= 0, f"Marginal entropy must be >= 0, got {h.item()}"
    assert h.item() <= max_entropy + 1e-5, (
        f"Marginal entropy must be <= log(V)={max_entropy:.4f}, got {h.item():.4f}"
    )


# ===========================================================================
# 15. MEMOAdapter.adapt_step returns expected keys
# ===========================================================================


def test_memo_adapt_step_keys():
    torch.manual_seed(1)
    model = _make_small_model()
    ap = AdaptableParams(model, adapt_mode="affine")
    pool = AugmentationPool(n_augmentations=4, noise_scale=0.2)
    memo = MEMOAdapter(_model_fn(model), ap.get_adapt_params(), pool, VOCAB_SIZE)

    x = _random_ids(batch=1, seq=T)
    stats = memo.adapt_step(x)

    assert "marginal_entropy" in stats, "Expected key 'marginal_entropy'"
    assert "n_augmentations" in stats, "Expected key 'n_augmentations'"
    assert isinstance(stats["marginal_entropy"], float)
    assert stats["n_augmentations"] == 4
