"""Tests for src/federation/secure_aggregation.py"""
import pytest
import torch

from src.federation.secure_aggregation import (
    MaskingScheme,
    SecureAggConfig,
    SecureAggregator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def agg() -> SecureAggregator:
    return SecureAggregator()


def cfg(scheme: MaskingScheme, num_parties: int = 2, seed: int = 42) -> SecureAggConfig:
    return SecureAggConfig(scheme=scheme, seed=seed, num_parties=num_parties)


# ---------------------------------------------------------------------------
# MaskingScheme enum
# ---------------------------------------------------------------------------

def test_masking_scheme_values():
    assert MaskingScheme.ADDITIVE == "ADDITIVE"
    assert MaskingScheme.PAIRWISE_MASK == "PAIRWISE_MASK"
    assert MaskingScheme.SHAMIR_STUB == "SHAMIR_STUB"


# ---------------------------------------------------------------------------
# generate_mask – shape
# ---------------------------------------------------------------------------

def test_generate_mask_additive_shape():
    sa = agg()
    mask = sa.generate_mask((4, 4), party_id=0, config=cfg(MaskingScheme.ADDITIVE))
    assert mask.shape == (4, 4)


def test_generate_mask_pairwise_shape():
    sa = agg()
    mask = sa.generate_mask((3,), party_id=1, config=cfg(MaskingScheme.PAIRWISE_MASK))
    assert mask.shape == (3,)


def test_generate_mask_shamir_stub_zeros():
    sa = agg()
    mask = sa.generate_mask((5,), party_id=0, config=cfg(MaskingScheme.SHAMIR_STUB))
    assert torch.all(mask == 0)


# ---------------------------------------------------------------------------
# generate_mask – PAIRWISE symmetry
# ---------------------------------------------------------------------------

def test_pairwise_masks_cancel_in_pair():
    sa = agg()
    config = cfg(MaskingScheme.PAIRWISE_MASK, num_parties=2)
    m0 = sa.generate_mask((8,), party_id=0, config=config)
    m1 = sa.generate_mask((8,), party_id=1, config=config)
    # Even + odd → should cancel
    assert torch.allclose(m0 + m1, torch.zeros(8), atol=1e-6)


# ---------------------------------------------------------------------------
# mask_update
# ---------------------------------------------------------------------------

def test_mask_update_additive_changes_tensor():
    sa = agg()
    g = torch.ones(4)
    masked = sa.mask_update(g, party_id=0, config=cfg(MaskingScheme.ADDITIVE))
    assert not torch.allclose(masked, g)


def test_mask_update_shamir_stub_unchanged():
    sa = agg()
    g = torch.tensor([1.0, 2.0, 3.0])
    masked = sa.mask_update(g, party_id=0, config=cfg(MaskingScheme.SHAMIR_STUB))
    assert torch.allclose(masked, g)


# ---------------------------------------------------------------------------
# aggregate_masked
# ---------------------------------------------------------------------------

def test_aggregate_masked_sums():
    sa = agg()
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([3.0, 4.0])
    result = sa.aggregate_masked([t1, t2])
    assert torch.allclose(result, torch.tensor([4.0, 6.0]))


def test_aggregate_masked_single():
    sa = agg()
    t = torch.tensor([5.0, 6.0])
    result = sa.aggregate_masked([t])
    assert torch.allclose(result, t)


def test_aggregate_masked_empty_raises():
    sa = agg()
    with pytest.raises(ValueError):
        sa.aggregate_masked([])


def test_pairwise_aggregate_cancels_masks():
    """Masks from party 0 and 1 cancel when aggregated."""
    sa = agg()
    config = cfg(MaskingScheme.PAIRWISE_MASK, num_parties=2)
    gradient = torch.ones(6) * 3.0
    # Both parties send the same gradient (3.0)
    m0 = sa.mask_update(gradient.clone(), party_id=0, config=config)
    m1 = sa.mask_update(gradient.clone(), party_id=1, config=config)
    agg_result = sa.aggregate_masked([m0, m1])
    # Sum of gradients = 6.0; masks cancel
    assert torch.allclose(agg_result, torch.ones(6) * 6.0, atol=1e-5)
