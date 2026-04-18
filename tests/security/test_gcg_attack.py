"""Tests for the GCG adversarial suffix search module (gcg_attack.py)."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.gcg_attack import GCGAttack, GCGConfig

# ---------------------------------------------------------------------------
# Shared tiny config and fixtures
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

SMALL_GCG = GCGConfig(
    n_steps=3,
    suffix_len=4,
    n_candidates=4,
    topk=8,
    seed=0,
)

SEED = 42


@pytest.fixture
def model() -> AureliusTransformer:
    torch.manual_seed(SEED)
    m = AureliusTransformer(TINY_CFG)
    m.train(False)
    return m


@pytest.fixture
def attacker(model: AureliusTransformer) -> GCGAttack:
    return GCGAttack(model, SMALL_GCG)


@pytest.fixture
def prefix_ids() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randint(0, TINY_CFG.vocab_size, (1, 5))


@pytest.fixture
def target_ids() -> torch.Tensor:
    torch.manual_seed(SEED + 1)
    return torch.randint(0, TINY_CFG.vocab_size, (1, 3))


# ---------------------------------------------------------------------------
# 1. GCGConfig defaults are correct
# ---------------------------------------------------------------------------

def test_gcg_config_defaults():
    cfg = GCGConfig()
    assert cfg.suffix_len == 10
    assert cfg.n_candidates == 16
    assert cfg.n_steps == 20
    assert cfg.topk == 128
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# 2. GCGAttack instantiates without error
# ---------------------------------------------------------------------------

def test_gcg_attack_instantiation(model: AureliusTransformer):
    att = GCGAttack(model, SMALL_GCG)
    assert att is not None
    assert att.config is SMALL_GCG
    assert att.model is model


# ---------------------------------------------------------------------------
# 3. _token_gradients returns shape (suffix_len, vocab_size)
# ---------------------------------------------------------------------------

def test_token_gradients_shape(attacker: GCGAttack, prefix_ids: torch.Tensor, target_ids: torch.Tensor):
    suffix_len = SMALL_GCG.suffix_len
    suffix_ids = torch.randint(0, TINY_CFG.vocab_size, (1, suffix_len))
    input_ids = torch.cat([prefix_ids, suffix_ids], dim=1)

    grads = attacker._token_gradients(input_ids, target_ids)

    assert grads.shape == (suffix_len, TINY_CFG.vocab_size)


# ---------------------------------------------------------------------------
# 4. _token_gradients values are finite
# ---------------------------------------------------------------------------

def test_token_gradients_finite(attacker: GCGAttack, prefix_ids: torch.Tensor, target_ids: torch.Tensor):
    suffix_len = SMALL_GCG.suffix_len
    suffix_ids = torch.randint(0, TINY_CFG.vocab_size, (1, suffix_len))
    input_ids = torch.cat([prefix_ids, suffix_ids], dim=1)

    grads = attacker._token_gradients(input_ids, target_ids)

    assert torch.isfinite(grads).all(), "Gradient tensor contains non-finite values"


# ---------------------------------------------------------------------------
# 5. _sample_candidates returns shape (n_candidates, suffix_len)
# ---------------------------------------------------------------------------

def test_sample_candidates_shape(attacker: GCGAttack):
    suffix_len = SMALL_GCG.suffix_len
    n_candidates = SMALL_GCG.n_candidates

    current_suffix = torch.randint(0, TINY_CFG.vocab_size, (suffix_len,))
    token_grads = torch.randn(suffix_len, TINY_CFG.vocab_size)

    candidates = attacker._sample_candidates(current_suffix, token_grads)

    assert candidates.shape == (n_candidates, suffix_len)


# ---------------------------------------------------------------------------
# 6. Candidates contain only valid token ids in [0, vocab_size)
# ---------------------------------------------------------------------------

def test_sample_candidates_valid_token_range(attacker: GCGAttack):
    suffix_len = SMALL_GCG.suffix_len
    vocab_size = TINY_CFG.vocab_size

    current_suffix = torch.randint(0, vocab_size, (suffix_len,))
    token_grads = torch.randn(suffix_len, vocab_size)

    candidates = attacker._sample_candidates(current_suffix, token_grads)

    assert (candidates >= 0).all(), "Candidates contain negative token ids"
    assert (candidates < vocab_size).all(), "Candidates contain out-of-range token ids"


# ---------------------------------------------------------------------------
# 7. _eval_loss returns a scalar float
# ---------------------------------------------------------------------------

def test_loss_computation_returns_scalar(attacker: GCGAttack, prefix_ids: torch.Tensor, target_ids: torch.Tensor):
    suffix_ids = torch.randint(0, TINY_CFG.vocab_size, (1, SMALL_GCG.suffix_len))

    loss = attacker._eval_loss(prefix_ids, suffix_ids, target_ids)

    assert isinstance(loss, float), f"Expected float, got {type(loss)}"


# ---------------------------------------------------------------------------
# 8. _eval_loss is finite
# ---------------------------------------------------------------------------

def test_loss_computation_is_finite(attacker: GCGAttack, prefix_ids: torch.Tensor, target_ids: torch.Tensor):
    suffix_ids = torch.randint(0, TINY_CFG.vocab_size, (1, SMALL_GCG.suffix_len))

    loss = attacker._eval_loss(prefix_ids, suffix_ids, target_ids)

    assert math.isfinite(loss), f"Loss is not finite: {loss}"


# ---------------------------------------------------------------------------
# 9. run returns a tuple of (LongTensor, float, list)
# ---------------------------------------------------------------------------

def test_run_return_types(attacker: GCGAttack, prefix_ids: torch.Tensor, target_ids: torch.Tensor):
    result = attacker.run(prefix_ids, target_ids)

    assert isinstance(result, tuple)
    assert len(result) == 3
    best_suffix, best_loss, loss_history = result

    assert isinstance(best_suffix, torch.Tensor)
    assert best_suffix.dtype == torch.long
    assert isinstance(best_loss, float)
    assert isinstance(loss_history, list)


# ---------------------------------------------------------------------------
# 10. best_suffix shape is (suffix_len,)
# ---------------------------------------------------------------------------

def test_best_suffix_shape(attacker: GCGAttack, prefix_ids: torch.Tensor, target_ids: torch.Tensor):
    best_suffix, _, _ = attacker.run(prefix_ids, target_ids)

    assert best_suffix.shape == (SMALL_GCG.suffix_len,)


# ---------------------------------------------------------------------------
# 11. loss_history has n_steps elements
# ---------------------------------------------------------------------------

def test_loss_history_length(attacker: GCGAttack, prefix_ids: torch.Tensor, target_ids: torch.Tensor):
    _, _, loss_history = attacker.run(prefix_ids, target_ids)

    assert len(loss_history) == SMALL_GCG.n_steps


# ---------------------------------------------------------------------------
# 12. best_loss matches min of loss_history (approximately)
# ---------------------------------------------------------------------------

def test_best_loss_equals_min_of_history(attacker: GCGAttack, prefix_ids: torch.Tensor, target_ids: torch.Tensor):
    _, best_loss, loss_history = attacker.run(prefix_ids, target_ids)

    assert math.isclose(best_loss, min(loss_history), rel_tol=1e-5), (
        f"best_loss {best_loss} does not match min(loss_history) {min(loss_history)}"
    )


# ---------------------------------------------------------------------------
# 13. Suffix tokens are in valid range [0, vocab_size)
# ---------------------------------------------------------------------------

def test_best_suffix_token_range(attacker: GCGAttack, prefix_ids: torch.Tensor, target_ids: torch.Tensor):
    vocab_size = TINY_CFG.vocab_size
    best_suffix, _, _ = attacker.run(prefix_ids, target_ids)

    assert (best_suffix >= 0).all(), "best_suffix contains negative token ids"
    assert (best_suffix < vocab_size).all(), "best_suffix contains out-of-range token ids"
