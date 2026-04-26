"""Tests for sequence packing with intra-document masking."""

import torch

from src.data.packing import (
    PackedSequence,
    build_intra_doc_mask,
    collate_packed_batch,
    pack_sequences,
)


def test_pack_sequences_basic():
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    packed = pack_sequences(seqs, max_len=8)
    # All fit in one chunk: 3 + 2 + 4 = 9 > 8, so [1,2,3] + [4,5] = 5 tokens, then [6,7,8,9]
    assert len(packed) >= 1
    # Total tokens across all chunks == total input tokens
    total_out = sum(len(p.input_ids) for p in packed)
    total_in = sum(len(s) for s in seqs)
    assert total_out == total_in


def test_pack_sequences_respects_max_len():
    seqs = [[i] * 10 for i in range(5)]
    packed = pack_sequences(seqs, max_len=16)
    for p in packed:
        assert len(p.input_ids) <= 16


def test_pack_sequences_doc_ids_track_boundaries():
    seqs = [[1, 2], [3, 4, 5], [6]]
    packed = pack_sequences(seqs, max_len=20)
    assert len(packed) == 1
    p = packed[0]
    # doc_ids should be [0, 0, 1, 1, 1, 2]
    assert p.doc_ids == [0, 0, 1, 1, 1, 2]


def test_pack_sequences_oversized_doc_truncated():
    seqs = [[i for i in range(20)]]  # single doc longer than max_len
    packed = pack_sequences(seqs, max_len=8)
    assert len(packed) == 1
    assert len(packed[0].input_ids) == 8


def test_build_intra_doc_mask_shape():
    doc_ids = torch.tensor([0, 0, 1, 1, 1])
    mask = build_intra_doc_mask(doc_ids)
    assert mask.shape == (5, 5)
    assert mask.dtype == torch.bool


def test_build_intra_doc_mask_causal():
    """Upper triangle must be all False (causal)."""
    doc_ids = torch.tensor([0, 0, 0])
    mask = build_intra_doc_mask(doc_ids)
    # Upper triangle (excluding diagonal) must be False
    assert not mask[0, 1]  # future token
    assert not mask[0, 2]


def test_build_intra_doc_mask_no_cross_doc():
    """Tokens from different docs must not attend to each other."""
    doc_ids = torch.tensor([0, 0, 1, 1])
    mask = build_intra_doc_mask(doc_ids)
    # doc 1 tokens should NOT attend to doc 0 tokens
    assert not mask[2, 0]
    assert not mask[2, 1]
    assert not mask[3, 0]
    assert not mask[3, 1]


def test_build_intra_doc_mask_within_doc():
    """Within-doc causal attention must be allowed."""
    doc_ids = torch.tensor([0, 0, 0])
    mask = build_intra_doc_mask(doc_ids)
    # Each token attends to itself and previous same-doc tokens
    assert mask[0, 0]
    assert mask[1, 0]
    assert mask[1, 1]
    assert mask[2, 0]
    assert mask[2, 1]
    assert mask[2, 2]


def test_collate_packed_batch_shapes():
    ps1 = PackedSequence(input_ids=[1, 2, 3], doc_ids=[0, 0, 0], doc_lengths=[3])
    ps2 = PackedSequence(input_ids=[4, 5], doc_ids=[0, 0], doc_lengths=[2])
    batch = collate_packed_batch([ps1, ps2], max_len=8)
    assert batch["input_ids"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8, 8)


def test_collate_labels_padded_positions_are_minus100():
    ps = PackedSequence(input_ids=[1, 2], doc_ids=[0, 0], doc_lengths=[2])
    batch = collate_packed_batch([ps], max_len=6)
    # Padded positions should have label -100
    assert (batch["labels"][0, 2:] == -100).all()


def test_collate_mask_is_bool():
    ps = PackedSequence(input_ids=[1, 2, 3], doc_ids=[0, 0, 0], doc_lengths=[3])
    batch = collate_packed_batch([ps], max_len=4)
    assert batch["attention_mask"].dtype == torch.bool
