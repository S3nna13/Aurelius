"""Unit tests for the TITO Gateway (GLM-5 §4.1, arXiv:2602.15763).

Tests verify canonicalization, ID remapping, boundary validation, and batch
processing.  All 13 cases are independent; no fixtures require external deps.
"""
import pytest

from src.training.tito_gateway import TITOConfig, TITOGateway


# ---------------------------------------------------------------------------
# Shared config factory
# ---------------------------------------------------------------------------

def make_gateway(**kwargs) -> TITOGateway:
    """Return a TITOGateway with vocab_size=1000 plus any overrides."""
    defaults = dict(vocab_size=1000, pad_id=0, unk_id=1)
    defaults.update(kwargs)
    return TITOGateway(TITOConfig(**defaults))


# ---------------------------------------------------------------------------
# 1. test_valid_sequence_passthrough
# ---------------------------------------------------------------------------

def test_valid_sequence_passthrough():
    """Valid IDs within [0, vocab_size) should pass through unchanged."""
    gw = make_gateway()
    result = gw.canonicalize([0, 5, 999])
    assert result == [0, 5, 999]


# ---------------------------------------------------------------------------
# 2. test_empty_sequence
# ---------------------------------------------------------------------------

def test_empty_sequence():
    """Empty input should return an empty list."""
    gw = make_gateway()
    assert gw.canonicalize([]) == []


# ---------------------------------------------------------------------------
# 3. test_out_of_range_high
# ---------------------------------------------------------------------------

def test_out_of_range_high():
    """Token ID == vocab_size should raise ValueError (exclusive upper bound)."""
    gw = make_gateway()
    with pytest.raises(ValueError, match="out of range"):
        gw.canonicalize([1000])


# ---------------------------------------------------------------------------
# 4. test_out_of_range_negative
# ---------------------------------------------------------------------------

def test_out_of_range_negative():
    """Negative token IDs should raise ValueError."""
    gw = make_gateway()
    with pytest.raises(ValueError, match="out of range"):
        gw.canonicalize([-1])


# ---------------------------------------------------------------------------
# 5. test_exactly_at_boundary
# ---------------------------------------------------------------------------

def test_exactly_at_boundary():
    """Token ID == vocab_size - 1 is the last valid ID and should not raise."""
    gw = make_gateway()
    result = gw.canonicalize([999])
    assert result == [999]


# ---------------------------------------------------------------------------
# 6. test_id_remapping
# ---------------------------------------------------------------------------

def test_id_remapping():
    """An ID present in id_remap should be replaced before validation."""
    gw = make_gateway(id_remap={50: 100})
    result = gw.canonicalize([50])
    assert result == [100]


# ---------------------------------------------------------------------------
# 7. test_remap_then_validate
# ---------------------------------------------------------------------------

def test_remap_then_validate():
    """Remapped value that falls outside vocab_size should still raise."""
    gw = make_gateway(id_remap={50: 1001})
    with pytest.raises(ValueError, match="out of range"):
        gw.canonicalize([50])


# ---------------------------------------------------------------------------
# 8. test_wrap_batch
# ---------------------------------------------------------------------------

def test_wrap_batch():
    """wrap_batch should apply canonicalize to every sequence in the batch."""
    gw = make_gateway()
    batch = [[0, 1, 2], [100, 200, 300], [999, 0, 1]]
    result = gw.wrap_batch(batch)
    assert result == [[0, 1, 2], [100, 200, 300], [999, 0, 1]]
    assert len(result) == 3


# ---------------------------------------------------------------------------
# 9. test_wrap_batch_empty
# ---------------------------------------------------------------------------

def test_wrap_batch_empty():
    """wrap_batch on an empty batch should return an empty list."""
    gw = make_gateway()
    assert gw.wrap_batch([]) == []


# ---------------------------------------------------------------------------
# 10. test_validate_only_true
# ---------------------------------------------------------------------------

def test_validate_only_true():
    """validate_only should return True for a fully valid sequence."""
    gw = make_gateway()
    assert gw.validate_only([0, 42, 999]) is True


# ---------------------------------------------------------------------------
# 11. test_validate_only_raises
# ---------------------------------------------------------------------------

def test_validate_only_raises():
    """validate_only should raise ValueError for an invalid sequence."""
    gw = make_gateway()
    with pytest.raises(ValueError, match="out of range"):
        gw.validate_only([5000])


# ---------------------------------------------------------------------------
# 12. test_pad_id_valid
# ---------------------------------------------------------------------------

def test_pad_id_valid():
    """The configured pad_id (0) should be a valid token and pass through."""
    gw = make_gateway(pad_id=0)
    result = gw.canonicalize([gw.config.pad_id])
    assert result == [0]


# ---------------------------------------------------------------------------
# 13. test_mixed_valid_invalid
# ---------------------------------------------------------------------------

def test_mixed_valid_invalid():
    """A sequence where the second token is invalid should raise on that token."""
    gw = make_gateway()
    with pytest.raises(ValueError, match="out of range"):
        gw.canonicalize([5, 1001, 3])
