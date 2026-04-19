"""Unit tests for TokenDropout."""

from __future__ import annotations

import pytest
import torch

from src.training.token_dropout import TokenDropout


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def test_p_zero_returns_unchanged() -> None:
    td = TokenDropout(p=0.0, mask_token_id=0, mode="replace")
    ids = torch.arange(1, 21).reshape(4, 5)
    new_ids, mask = td.apply(ids, rng=_gen(0))
    assert torch.equal(new_ids, ids)
    assert torch.all(mask == 1.0)


def test_p_one_replace_replaces_all_non_special() -> None:
    td = TokenDropout(
        p=1.0, mask_token_id=99, mode="replace", exclude_special_ids=(7,)
    )
    ids = torch.tensor([[1, 2, 7, 3, 7, 4]])
    new_ids, _ = td.apply(ids, rng=_gen(0))
    expected = torch.tensor([[99, 99, 7, 99, 7, 99]])
    assert torch.equal(new_ids, expected)


def test_loss_mask_mode_leaves_ids_unchanged() -> None:
    td = TokenDropout(p=1.0, mask_token_id=0, mode="loss_mask")
    ids = torch.tensor([[5, 6, 7, 8]])
    new_ids, mask = td.apply(ids, rng=_gen(0))
    assert torch.equal(new_ids, ids)
    assert torch.all(mask == 0.0)


def test_both_mode_replaces_and_masks() -> None:
    td = TokenDropout(p=1.0, mask_token_id=42, mode="both")
    ids = torch.tensor([[1, 2, 3]])
    new_ids, mask = td.apply(ids, rng=_gen(0))
    assert torch.all(new_ids == 42)
    assert torch.all(mask == 0.0)


def test_exclude_special_ids_preserved() -> None:
    td = TokenDropout(
        p=1.0, mask_token_id=0, mode="both", exclude_special_ids=(10, 11)
    )
    ids = torch.tensor([[10, 5, 11, 6, 10]])
    new_ids, mask = td.apply(ids, rng=_gen(0))
    # Specials unchanged, non-specials replaced.
    assert new_ids[0, 0].item() == 10
    assert new_ids[0, 2].item() == 11
    assert new_ids[0, 4].item() == 10
    assert new_ids[0, 1].item() == 0
    assert new_ids[0, 3].item() == 0
    # Mask: 1 at special positions, 0 elsewhere.
    assert mask[0, 0].item() == 1.0
    assert mask[0, 2].item() == 1.0
    assert mask[0, 4].item() == 1.0
    assert mask[0, 1].item() == 0.0
    assert mask[0, 3].item() == 0.0


def test_determinism_with_seeded_rng() -> None:
    td = TokenDropout(p=0.3, mask_token_id=0, mode="replace")
    ids = torch.arange(200).reshape(10, 20)
    a_ids, a_mask = td.apply(ids, rng=_gen(42))
    b_ids, b_mask = td.apply(ids, rng=_gen(42))
    assert torch.equal(a_ids, b_ids)
    assert torch.equal(a_mask, b_mask)


def test_shape_preserved() -> None:
    td = TokenDropout(p=0.5, mask_token_id=0, mode="both")
    ids = torch.randint(1, 100, (3, 7, 11))
    new_ids, mask = td.apply(ids, rng=_gen(1))
    assert new_ids.shape == ids.shape
    assert mask.shape == ids.shape


def test_dtype_preserved() -> None:
    td = TokenDropout(p=0.5, mask_token_id=0, mode="replace")
    ids = torch.randint(1, 100, (4, 4), dtype=torch.int64)
    new_ids, _ = td.apply(ids, rng=_gen(2))
    assert new_ids.dtype == torch.int64

    ids32 = torch.randint(1, 100, (4, 4), dtype=torch.int32)
    new_ids32, _ = td.apply(ids32, rng=_gen(2))
    assert new_ids32.dtype == torch.int32


def test_batch_and_sequence_dims_respected() -> None:
    td = TokenDropout(p=1.0, mask_token_id=0, mode="replace")
    ids = torch.randint(1, 100, (2, 5))
    new_ids, mask = td.apply(ids, rng=_gen(3))
    assert new_ids.shape == (2, 5)
    assert mask.shape == (2, 5)
    # All replaced.
    assert torch.all(new_ids == 0)


def test_empty_tensor_input() -> None:
    td = TokenDropout(p=0.5, mask_token_id=0, mode="both")
    ids = torch.zeros((0,), dtype=torch.int64)
    new_ids, mask = td.apply(ids)
    assert new_ids.numel() == 0
    assert mask.numel() == 0
    assert new_ids.shape == ids.shape

    ids2 = torch.zeros((0, 5), dtype=torch.int64)
    new_ids2, mask2 = td.apply(ids2)
    assert new_ids2.shape == (0, 5)
    assert mask2.shape == (0, 5)


def test_approximate_drop_rate_statistical() -> None:
    td = TokenDropout(p=0.3, mask_token_id=0, mode="replace")
    ids = torch.ones((50, 200), dtype=torch.int64) * 5  # non-special, non-mask
    new_ids, _ = td.apply(ids, rng=_gen(123))
    rate = (new_ids == 0).float().mean().item()
    assert 0.27 < rate < 0.33, f"expected ~0.30, got {rate}"


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError):
        TokenDropout(p=0.1, mode="invalid_mode")


def test_invalid_p_raises() -> None:
    with pytest.raises(ValueError):
        TokenDropout(p=-0.1)
    with pytest.raises(ValueError):
        TokenDropout(p=1.5)


def test_different_seeds_produce_different_results() -> None:
    td = TokenDropout(p=0.5, mask_token_id=0, mode="replace")
    ids = torch.ones((8, 32), dtype=torch.int64) * 5
    a_ids, _ = td.apply(ids, rng=_gen(42))
    b_ids, _ = td.apply(ids, rng=_gen(43))
    assert not torch.equal(a_ids, b_ids)


def test_provided_loss_mask_is_respected() -> None:
    td = TokenDropout(p=1.0, mask_token_id=0, mode="loss_mask")
    ids = torch.tensor([[1, 2, 3, 4]])
    incoming = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
    _, mask = td.apply(ids, loss_mask=incoming, rng=_gen(0))
    # Everything zeroed by dropout, but the originally-zero position stays zero.
    assert torch.all(mask == 0.0)
