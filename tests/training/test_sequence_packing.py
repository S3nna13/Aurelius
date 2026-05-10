"""Tests for src/training/sequence_packing.py"""

import torch
from aurelius.training.sequence_packing import (
    PackedBatchCollator,
    PackedSequence,
    PackingStats,
    SequencePacker,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_seqs(*lengths, base=1):
    """Create 1-D LongTensors of given lengths with sequential token ids."""
    seqs = []
    tok = base
    for L in lengths:
        seqs.append(torch.arange(tok, tok + L, dtype=torch.long))
        tok += L
    return seqs


# ---------------------------------------------------------------------------
# PackedSequence dataclass
# ---------------------------------------------------------------------------


def test_packed_sequence_fields():
    ps = PackedSequence(
        token_ids=torch.zeros(8, dtype=torch.long),
        position_ids=torch.zeros(8, dtype=torch.long),
        seq_boundaries=[0, 4],
    )
    assert hasattr(ps, "token_ids")
    assert hasattr(ps, "position_ids")
    assert hasattr(ps, "seq_boundaries")
    assert hasattr(ps, "labels")
    assert ps.labels is None  # default


# ---------------------------------------------------------------------------
# SequencePacker.pack
# ---------------------------------------------------------------------------


def test_pack_returns_list_of_packed_sequences():
    packer = SequencePacker(max_length=10)
    seqs = _make_seqs(3, 4, 2)
    result = packer.pack(seqs)
    assert isinstance(result, list)
    assert all(isinstance(p, PackedSequence) for p in result)


def test_pack_chunks_at_most_max_length():
    packer = SequencePacker(max_length=8)
    seqs = _make_seqs(3, 4, 3, 2)
    result = packer.pack(seqs)
    for ps in result:
        assert len(ps.token_ids) == 8


def test_pack_no_tokens_dropped():
    packer = SequencePacker(max_length=10)
    seqs = _make_seqs(3, 4, 2, 1)
    sum(len(s) for s in seqs)
    result = packer.pack(seqs)
    # Count non-padding tokens across all bins
    for ps in result:
        for i in range(len(ps.token_ids)):
            if ps.position_ids[i] != 0 or i == 0 or (i in ps.seq_boundaries):
                pass
    # Simpler: count all distinct original token values
    all_packed_tokens = torch.cat([ps.token_ids for ps in result])
    # Original token values are 1..sum(lengths); count how many appear
    orig_tokens = set()
    for s in seqs:
        orig_tokens.update(s.tolist())
    packed_set = set(all_packed_tokens.tolist())
    assert orig_tokens.issubset(packed_set)


def test_position_ids_reset_at_boundaries():
    packer = SequencePacker(max_length=10)
    seqs = _make_seqs(3, 4)
    result = packer.pack(seqs)
    # At least one packed sequence should have position_ids starting at 0
    # and resetting to 0 at a boundary other than 0
    for ps in result:
        if len(ps.seq_boundaries) >= 2:
            boundary = ps.seq_boundaries[1]
            assert ps.position_ids[boundary].item() == 0


def test_position_ids_monotone_within_subsequence():
    packer = SequencePacker(max_length=10)
    seqs = _make_seqs(4, 3)
    result = packer.pack(seqs)
    for ps in result:
        # Check that within each sub-sequence position IDs go 0,1,2,...
        for i, start in enumerate(ps.seq_boundaries):
            # End of this sub-sequence = start of next boundary (or end of real tokens)
            if i + 1 < len(ps.seq_boundaries):
                end = ps.seq_boundaries[i + 1]
            else:
                # Last sub-sequence ends where real tokens end
                # Find end by looking for padding
                end = start
                while end < len(ps.position_ids) and not (
                    end > start
                    and ps.position_ids[end].item() == 0
                    and ps.token_ids[end].item() == packer.pad_token_id
                ):
                    end += 1
            seg = ps.position_ids[start:end]
            if len(seg) > 1:
                # Position IDs should be [0, 1, 2, ..., len-1]
                expected = torch.arange(len(seg), dtype=torch.long)
                assert torch.equal(seg, expected)


def test_pack_single_seq_larger_than_max_truncates():
    packer = SequencePacker(max_length=5)
    seqs = [torch.arange(10, dtype=torch.long)]
    result = packer.pack(seqs)
    assert len(result) == 1
    assert len(result[0].token_ids) == 5


def test_pack_empty_input_returns_empty():
    packer = SequencePacker(max_length=8)
    result = packer.pack([])
    assert result == []


# ---------------------------------------------------------------------------
# pack_with_labels
# ---------------------------------------------------------------------------


def test_pack_with_labels_alignment():
    packer = SequencePacker(max_length=10)
    seqs = _make_seqs(3, 4)
    labels = [torch.ones(3, dtype=torch.long), torch.zeros(4, dtype=torch.long)]
    result = packer.pack_with_labels(seqs, labels)
    for ps in result:
        assert ps.labels is not None
        assert len(ps.labels) == len(ps.token_ids)


def test_pack_with_labels_padding_is_minus100():
    packer = SequencePacker(max_length=10)
    seqs = [torch.tensor([1, 2, 3], dtype=torch.long)]
    labels = [torch.tensor([10, 20, 30], dtype=torch.long)]
    result = packer.pack_with_labels(seqs, labels)
    ps = result[0]
    pad_start = 3
    assert (ps.labels[pad_start:] == -100).all()


# ---------------------------------------------------------------------------
# PackingStats
# ---------------------------------------------------------------------------


def test_packing_stats_efficiency_in_range():
    packer = SequencePacker(max_length=8)
    seqs = _make_seqs(3, 4, 2)
    stats = PackingStats.from_packer(packer, seqs)
    assert 0.0 < stats.efficiency <= 1.0


def test_packing_stats_perfect_efficiency():
    # One sequence exactly fills the bin
    packer = SequencePacker(max_length=5)
    seqs = [torch.arange(5, dtype=torch.long)]
    stats = PackingStats.from_packer(packer, seqs)
    assert abs(stats.efficiency - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# PackedBatchCollator
# ---------------------------------------------------------------------------


def test_collator_output_shapes():
    packer = SequencePacker(max_length=8, pad_token_id=0)
    collator = PackedBatchCollator(packer)
    batch = [
        {"input_ids": [1, 2, 3]},
        {"input_ids": [4, 5, 6, 7]},
    ]
    out = collator(batch)
    B = out["input_ids"].shape[0]
    L = out["input_ids"].shape[1]
    assert L == 8
    assert out["position_ids"].shape == (B, L)
    assert out["attention_mask"].shape == (B, L)
    assert out["labels"].shape == (B, L)


def test_collator_attention_mask_real_tokens():
    packer = SequencePacker(max_length=8, pad_token_id=0)
    collator = PackedBatchCollator(packer)
    batch = [{"input_ids": [1, 2, 3]}]
    out = collator(batch)
    # First 3 tokens should be real (mask=True), rest padding
    assert out["attention_mask"][0, :3].all()
    assert not out["attention_mask"][0, 3:].any()


def test_collator_missing_labels_filled_with_minus100():
    packer = SequencePacker(max_length=8, pad_token_id=0)
    collator = PackedBatchCollator(packer)
    batch = [{"input_ids": [1, 2, 3]}]  # no 'labels' key
    out = collator(batch)
    assert (out["labels"] == -100).all()


# ---------------------------------------------------------------------------
# First-fit-decreasing: longer seqs go first
# ---------------------------------------------------------------------------


def test_ffd_longer_seqs_placed_first():
    packer = SequencePacker(max_length=6)
    seqs = [
        torch.ones(2, dtype=torch.long),  # short
        torch.ones(5, dtype=torch.long),  # long — should be placed first
        torch.ones(1, dtype=torch.long),  # shorter
    ]
    bins = packer._build_bins(seqs)
    # The bin containing index 1 (length 5) should come first
    first_bin = bins[0]
    assert 1 in first_bin
