"""Tests for the model extraction attack module (model_extraction.py)."""

from __future__ import annotations

import copy

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.model_extraction import ModelExtractor

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

BATCH_SIZE = 2
SEQ_LEN = 8
SEED = 0


@pytest.fixture(scope="module")
def oracle() -> AureliusTransformer:
    torch.manual_seed(SEED)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


@pytest.fixture(scope="module")
def clone() -> AureliusTransformer:
    torch.manual_seed(SEED + 1)
    model = AureliusTransformer(TINY_CFG)
    return model


@pytest.fixture(scope="module")
def extractor() -> ModelExtractor:
    return ModelExtractor()


@pytest.fixture(scope="module")
def input_ids() -> torch.Tensor:
    torch.manual_seed(SEED)
    return torch.randint(0, TINY_CFG.vocab_size, (BATCH_SIZE, SEQ_LEN))


# ---------------------------------------------------------------------------
# Tests for query_oracle
# ---------------------------------------------------------------------------


def test_query_oracle_shape(extractor, oracle, input_ids):
    """query_oracle returns a logits tensor of shape (B, S, vocab_size)."""
    logits = extractor.query_oracle(oracle, input_ids)
    assert logits.shape == (BATCH_SIZE, SEQ_LEN, TINY_CFG.vocab_size)


def test_query_oracle_no_grad(extractor, oracle, input_ids):
    """query_oracle returns a tensor that requires no gradient."""
    logits = extractor.query_oracle(oracle, input_ids)
    assert not logits.requires_grad


# ---------------------------------------------------------------------------
# Tests for extraction_step
# ---------------------------------------------------------------------------


def test_extraction_step_returns_scalar(extractor, oracle, clone, input_ids):
    """extraction_step returns a scalar Python float."""
    clone_copy = copy.deepcopy(clone)
    optimizer = torch.optim.Adam(clone_copy.parameters(), lr=1e-3)
    oracle_logits = extractor.query_oracle(oracle, input_ids)
    loss = extractor.extraction_step(clone_copy, optimizer, input_ids, oracle_logits)
    assert isinstance(loss, float)


def test_extraction_step_loss_finite(extractor, oracle, clone, input_ids):
    """extraction_step loss is finite (no NaN or Inf)."""
    clone_copy = copy.deepcopy(clone)
    optimizer = torch.optim.Adam(clone_copy.parameters(), lr=1e-3)
    oracle_logits = extractor.query_oracle(oracle, input_ids)
    loss = extractor.extraction_step(clone_copy, optimizer, input_ids, oracle_logits)
    assert not (loss != loss), "loss is NaN"
    assert loss != float("inf") and loss != float("-inf"), "loss is Inf"


# ---------------------------------------------------------------------------
# Tests for extraction_fidelity
# ---------------------------------------------------------------------------


def test_extraction_fidelity_range(extractor, oracle, clone, input_ids):
    """extraction_fidelity returns a float in [0, 1]."""
    fidelity = extractor.extraction_fidelity(oracle, clone, input_ids)
    assert 0.0 <= fidelity <= 1.0


def test_extraction_fidelity_identical_models(extractor, oracle, input_ids):
    """extraction_fidelity between a model and itself is 1.0."""
    fidelity = extractor.extraction_fidelity(oracle, oracle, input_ids)
    assert fidelity == pytest.approx(1.0)


def test_extraction_fidelity_dtype(extractor, oracle, clone, input_ids):
    """extraction_fidelity returns a Python float."""
    fidelity = extractor.extraction_fidelity(oracle, clone, input_ids)
    assert isinstance(fidelity, float)


# ---------------------------------------------------------------------------
# Tests for run_extraction
# ---------------------------------------------------------------------------


def test_run_extraction_returns_list(extractor, oracle, input_ids):
    """run_extraction returns a list of losses."""
    clone_copy = copy.deepcopy(AureliusTransformer(TINY_CFG))
    n_epochs = 2
    losses = extractor.run_extraction(
        oracle, clone_copy, input_ids, n_epochs=n_epochs, lr=1e-3
    )
    assert isinstance(losses, list)


def test_run_extraction_loss_list_length(extractor, oracle, input_ids):
    """Loss list length equals n_epochs * n_batches."""
    clone_copy = copy.deepcopy(AureliusTransformer(TINY_CFG))
    n_epochs = 3
    # input_ids is (BATCH_SIZE, SEQ_LEN); treated as BATCH_SIZE individual batches
    n_batches = input_ids.shape[0]
    losses = extractor.run_extraction(
        oracle, clone_copy, input_ids, n_epochs=n_epochs, lr=1e-3
    )
    assert len(losses) == n_epochs * n_batches


def test_run_extraction_no_nan_inf(extractor, oracle, input_ids):
    """No NaN or Inf appears in any step loss."""
    clone_copy = copy.deepcopy(AureliusTransformer(TINY_CFG))
    losses = extractor.run_extraction(
        oracle, clone_copy, input_ids, n_epochs=2, lr=1e-3
    )
    for i, loss in enumerate(losses):
        assert loss == loss, f"loss at step {i} is NaN"
        assert abs(loss) != float("inf"), f"loss at step {i} is Inf"


def test_run_extraction_clone_parameters_change(extractor, oracle, input_ids):
    """Clone model parameters change after run_extraction."""
    torch.manual_seed(99)
    clone_copy = AureliusTransformer(TINY_CFG)

    # Snapshot parameter values before training
    params_before = [p.clone().detach() for p in clone_copy.parameters()]

    extractor.run_extraction(oracle, clone_copy, input_ids, n_epochs=2, lr=1e-3)

    params_after = list(clone_copy.parameters())
    changed = any(
        not torch.allclose(pb, pa.detach())
        for pb, pa in zip(params_before, params_after)
    )
    assert changed, "Clone parameters did not change after extraction"


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


def test_different_oracles_produce_different_logits(extractor, input_ids):
    """Two independently initialised oracles produce different logits."""
    torch.manual_seed(10)
    oracle_a = AureliusTransformer(TINY_CFG)
    torch.manual_seed(20)
    oracle_b = AureliusTransformer(TINY_CFG)

    logits_a = extractor.query_oracle(oracle_a, input_ids)
    logits_b = extractor.query_oracle(oracle_b, input_ids)

    assert not torch.allclose(logits_a, logits_b), (
        "Two different oracles produced identical logits"
    )


def test_batch_size_1_seq_len_4(extractor):
    """Works correctly with batch size 1 and seq_len 4."""
    torch.manual_seed(77)
    oracle = AureliusTransformer(TINY_CFG)
    clone_copy = AureliusTransformer(TINY_CFG)
    ids = torch.randint(0, TINY_CFG.vocab_size, (1, 4))

    logits = extractor.query_oracle(oracle, ids)
    assert logits.shape == (1, 4, TINY_CFG.vocab_size)

    optimizer = torch.optim.Adam(clone_copy.parameters(), lr=1e-3)
    loss = extractor.extraction_step(clone_copy, optimizer, ids, logits)
    assert isinstance(loss, float)
    assert loss == loss  # not NaN
