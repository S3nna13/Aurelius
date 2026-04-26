"""Unit tests for InstructionDatasetPacker."""

from __future__ import annotations

import pytest
import torch

from src.data.instruction_dataset_packer import (
    InstructionDatasetPacker,
    InstructionSample,
    PackedBatch,
)


def _mk(prompt_len: int, resp_len: int, base: int = 1) -> InstructionSample:
    p = list(range(base, base + prompt_len))
    r = list(range(base + prompt_len, base + prompt_len + resp_len))
    return InstructionSample(prompt_token_ids=p, response_token_ids=r)


def test_pack_two_small_samples_single_bin():
    packer = InstructionDatasetPacker(max_seq_len=32, pad_token_id=0)
    out = packer.pack([_mk(3, 4, base=1), _mk(2, 5, base=100)])
    assert len(out) == 1
    pb = out[0]
    assert pb.input_ids.shape == (1, 32)
    # Two samples present -> segment_ids hits both 1 and 2.
    segs = pb.segment_ids[0].tolist()
    assert 1 in segs and 2 in segs


def test_pack_oversized_raises():
    packer = InstructionDatasetPacker(max_seq_len=8)
    big = _mk(5, 5)  # len 10 > 8
    with pytest.raises(ValueError):
        packer.pack([big])


def test_ffd_larger_packed_first():
    packer = InstructionDatasetPacker(max_seq_len=16, pad_token_id=0)
    # Lengths 4, 10, 3, 8. FFD order: 10, 8, 4, 3.
    samples = [_mk(2, 2, base=10), _mk(5, 5, base=100), _mk(1, 2, base=200), _mk(4, 4, base=300)]
    out = packer.pack(samples)
    # bin1: 10 + 4 = 14; bin2: 8 + 3 = 11. Expect 2 bins.
    assert len(out) == 2
    # The largest sample (len 10, base=100) should land first in bin1.
    pb0 = out[0]
    # First token should come from the len-10 sample => base=100.
    assert pb0.input_ids[0, 0].item() == 100
    # Second sample appended (len 4, base=10) starts at offset 10.
    assert pb0.input_ids[0, 10].item() == 10


def test_loss_mask_only_response_tokens():
    packer = InstructionDatasetPacker(max_seq_len=32)
    out = packer.pack([_mk(4, 3, base=1), _mk(2, 5, base=50)])
    pb = out[0]
    lm = pb.loss_mask[0]
    seg = pb.segment_ids[0]
    # For every real (segment != 0) token, assert loss_mask correctness by
    # checking count equals total response tokens (3 + 5).
    assert lm.sum().item() == 8
    # Pad regions have loss_mask False.
    assert lm[seg == 0].sum().item() == 0


def test_segment_ids_increment():
    packer = InstructionDatasetPacker(max_seq_len=64)
    out = packer.pack([_mk(3, 3, base=1), _mk(2, 2, base=50), _mk(4, 4, base=100)])
    pb = out[0]
    seg = pb.segment_ids[0].tolist()
    # First sample seg=1 occupies positions 0..? Depending on FFD order by length desc:
    # lengths: 6, 4, 8 -> sorted: 8,6,4. So bin order: base=100 (seg1), base=1 (seg2), base=50 (seg3).  # noqa: E501
    unique_in_order = []
    for v in seg:
        if v != 0 and (not unique_in_order or unique_in_order[-1] != v):
            unique_in_order.append(v)
    assert unique_in_order == [1, 2, 3]


def test_attention_mask_block_diagonal():
    packer = InstructionDatasetPacker(max_seq_len=16, respect_sample_boundaries=True)
    out = packer.pack([_mk(2, 2, base=1), _mk(3, 3, base=50)])
    pb = out[0]
    attn = pb.attention_mask[0]  # [L,L]
    seg = pb.segment_ids[0]
    L = attn.shape[0]
    for i in range(L):
        for j in range(L):
            if attn[i, j]:
                # real tokens on both sides, same segment, causal.
                assert seg[i].item() != 0 and seg[j].item() != 0
                assert seg[i].item() == seg[j].item()
                assert j <= i


def test_batch_dim_is_one():
    packer = InstructionDatasetPacker(max_seq_len=32)
    out = packer.pack([_mk(2, 2)])
    pb = out[0]
    assert pb.input_ids.shape[0] == 1
    assert pb.attention_mask.shape[0] == 1
    assert pb.loss_mask.shape[0] == 1
    assert pb.segment_ids.shape[0] == 1


def test_empty_samples_returns_empty():
    packer = InstructionDatasetPacker(max_seq_len=16)
    assert packer.pack([]) == []


def test_pad_fills_remainder():
    packer = InstructionDatasetPacker(max_seq_len=16, pad_token_id=7)
    out = packer.pack([_mk(2, 3, base=1)])  # 5 real tokens, 11 pads.
    pb = out[0]
    seg = pb.segment_ids[0]
    # All positions where seg==0 must be pad_token_id.
    pad_positions = seg == 0
    assert pad_positions.sum().item() == 11
    assert torch.all(pb.input_ids[0][pad_positions] == 7).item()


def test_invalid_max_seq_len_raises():
    with pytest.raises(ValueError):
        InstructionDatasetPacker(max_seq_len=0)
    with pytest.raises(ValueError):
        InstructionDatasetPacker(max_seq_len=-5)
    with pytest.raises(ValueError):
        InstructionDatasetPacker(max_seq_len=3.5)  # type: ignore[arg-type]


def test_pack_iter_yields_batches():
    packer = InstructionDatasetPacker(max_seq_len=16)
    samples = [_mk(2, 2, base=b) for b in (1, 10, 20, 30, 40)]
    out = list(packer.pack_iter(iter(samples), batch_size=2))
    assert len(out) >= 1
    assert all(isinstance(pb, PackedBatch) for pb in out)
    # All yielded sequences have correct length.
    assert all(pb.input_ids.shape == (1, 16) for pb in out)


def test_determinism():
    packer = InstructionDatasetPacker(max_seq_len=32)
    samples = [_mk(3, 2, base=1), _mk(4, 4, base=20), _mk(1, 6, base=50)]
    a = packer.pack(samples)
    b = packer.pack(samples)
    assert len(a) == len(b)
    for pa, pb in zip(a, b):
        assert torch.equal(pa.input_ids, pb.input_ids)
        assert torch.equal(pa.attention_mask, pb.attention_mask)
        assert torch.equal(pa.loss_mask, pb.loss_mask)
        assert torch.equal(pa.segment_ids, pb.segment_ids)


def test_single_sample_pack():
    packer = InstructionDatasetPacker(max_seq_len=16)
    out = packer.pack([_mk(3, 4, base=1)])
    assert len(out) == 1
    pb = out[0]
    # 7 real tokens; seg=1 for them, seg=0 for pad.
    seg = pb.segment_ids[0]
    assert (seg == 1).sum().item() == 7
    assert (seg == 0).sum().item() == 9
    # Loss mask: 4 response tokens True.
    assert pb.loss_mask[0].sum().item() == 4


def test_cross_sample_attention_when_boundaries_disabled():
    packer = InstructionDatasetPacker(max_seq_len=16, respect_sample_boundaries=False)
    out = packer.pack([_mk(2, 2, base=1), _mk(2, 2, base=50)])
    pb = out[0]
    attn = pb.attention_mask[0]
    seg = pb.segment_ids[0]
    # At least one pair (i,j) where i>j, different segments, both real, is True.
    real_idx = (seg != 0).nonzero(as_tuple=True)[0].tolist()
    cross_found = False
    for i in real_idx:
        for j in real_idx:
            if j < i and seg[i].item() != seg[j].item():
                if attn[i, j]:
                    cross_found = True
                    break
        if cross_found:
            break
    assert cross_found, "expected cross-segment attention when boundaries disabled"
    # And it should remain causal: never attend to future real tokens.
    L = attn.shape[0]
    for i in range(L):
        for j in range(i + 1, L):
            assert not attn[i, j]
