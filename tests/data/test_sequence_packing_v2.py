"""Tests for sequence_packing_v2 — greedy sequence packing utilities.

All tests use tiny configs (max_seq_len <= 32) and short token lists to keep
execution fast and independent of any GPU / HuggingFace dependency.
"""

import torch

from src.data.sequence_packing_v2 import (
    PackConfig,
    SequencePacker,
    build_document_attention_mask,
    compute_packing_efficiency,
    create_position_ids,
    pack_sequences,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_seqs():
    """Three short sequences that easily fit in a single small chunk."""
    return [[1, 2, 3], [4, 5], [6, 7, 8, 9]]


def _tiny_cfg(**kw) -> PackConfig:
    defaults = dict(
        max_seq_len=16, pad_token_id=0, eos_token_id=2, add_eos=True, loss_mask_padding=True
    )
    defaults.update(kw)
    return PackConfig(**defaults)


# ---------------------------------------------------------------------------
# 1. PackConfig defaults
# ---------------------------------------------------------------------------


class TestPackConfigDefaults:
    def test_default_max_seq_len(self):
        cfg = PackConfig()
        assert cfg.max_seq_len == 2048

    def test_default_pad_token_id(self):
        assert PackConfig().pad_token_id == 0

    def test_default_eos_token_id(self):
        assert PackConfig().eos_token_id == 2

    def test_default_add_eos(self):
        assert PackConfig().add_eos is True

    def test_default_loss_mask_padding(self):
        assert PackConfig().loss_mask_padding is True


# ---------------------------------------------------------------------------
# 2. pack_sequences — return type
# ---------------------------------------------------------------------------


class TestPackSequencesReturnType:
    def test_returns_list(self):
        cfg = _tiny_cfg()
        result = pack_sequences(_short_seqs(), cfg)
        assert isinstance(result, list)

    def test_returns_list_of_dicts(self):
        cfg = _tiny_cfg()
        result = pack_sequences(_short_seqs(), cfg)
        assert all(isinstance(c, dict) for c in result)

    def test_empty_input_returns_empty_list(self):
        cfg = _tiny_cfg()
        assert pack_sequences([], cfg) == []


# ---------------------------------------------------------------------------
# 3. pack_sequences — chunk length constraint
# ---------------------------------------------------------------------------


class TestPackSequencesChunkLength:
    def test_each_chunk_length_le_max_seq_len(self):
        cfg = _tiny_cfg(max_seq_len=12)
        seqs = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13]]
        chunks = pack_sequences(seqs, cfg)
        for chunk in chunks:
            assert len(chunk["input_ids"]) <= cfg.max_seq_len

    def test_chunk_length_exactly_max_seq_len(self):
        """Every chunk must be padded to exactly max_seq_len."""
        cfg = _tiny_cfg(max_seq_len=16)
        chunks = pack_sequences(_short_seqs(), cfg)
        for chunk in chunks:
            assert len(chunk["input_ids"]) == cfg.max_seq_len

    def test_very_long_sequence_truncated_not_error(self):
        """A sequence longer than max_seq_len must not raise."""
        cfg = _tiny_cfg(max_seq_len=8)
        seqs = [list(range(20))]  # 20 tokens >> max_seq_len=8
        chunks = pack_sequences(seqs, cfg)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert len(chunk["input_ids"]) == cfg.max_seq_len


# ---------------------------------------------------------------------------
# 4. pack_sequences — required keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"input_ids", "attention_mask", "labels", "sequence_ids"}


class TestPackSequencesRequiredKeys:
    def test_all_required_keys_present(self):
        cfg = _tiny_cfg()
        chunks = pack_sequences(_short_seqs(), cfg)
        for chunk in chunks:
            assert REQUIRED_KEYS <= set(chunk.keys()), (
                f"Missing keys: {REQUIRED_KEYS - set(chunk.keys())}"
            )

    def test_all_lists_same_length(self):
        cfg = _tiny_cfg(max_seq_len=20)
        chunks = pack_sequences(_short_seqs(), cfg)
        for chunk in chunks:
            lengths = {k: len(chunk[k]) for k in REQUIRED_KEYS}
            assert len(set(lengths.values())) == 1, f"Key lengths differ: {lengths}"


# ---------------------------------------------------------------------------
# 5. pack_sequences — attention_mask: 1 for real tokens, 0 for padding
# ---------------------------------------------------------------------------


class TestPackSequencesAttentionMask:
    def test_attention_mask_ones_for_real_tokens(self):
        cfg = _tiny_cfg(max_seq_len=20, add_eos=False)
        seqs = [[10, 11, 12]]
        chunk = pack_sequences(seqs, cfg)[0]
        real_len = 3  # no EOS
        assert all(chunk["attention_mask"][i] == 1 for i in range(real_len))

    def test_attention_mask_zeros_for_padding(self):
        cfg = _tiny_cfg(max_seq_len=16, add_eos=False)
        seqs = [[10, 11, 12]]  # 3 real tokens, 13 padding
        chunk = pack_sequences(seqs, cfg)[0]
        assert all(chunk["attention_mask"][i] == 0 for i in range(3, 16))

    def test_attention_mask_values_binary(self):
        cfg = _tiny_cfg()
        for chunk in pack_sequences(_short_seqs(), cfg):
            assert all(v in (0, 1) for v in chunk["attention_mask"])


# ---------------------------------------------------------------------------
# 6. pack_sequences — labels -100 at padding positions
# ---------------------------------------------------------------------------


class TestPackSequencesLabels:
    def test_labels_minus100_at_padding(self):
        cfg = _tiny_cfg(max_seq_len=16, add_eos=False, loss_mask_padding=True)
        seqs = [[10, 11, 12]]  # 3 real tokens
        chunk = pack_sequences(seqs, cfg)[0]
        for i, (am, lbl) in enumerate(zip(chunk["attention_mask"], chunk["labels"])):
            if am == 0:  # padding position
                assert lbl == -100, f"Position {i}: expected -100, got {lbl}"

    def test_labels_not_minus100_for_real_tokens_when_masking(self):
        """Real-token labels should NOT be -100 when loss_mask_padding=True."""
        cfg = _tiny_cfg(max_seq_len=16, add_eos=False, loss_mask_padding=True)
        seqs = [[10, 11, 12]]
        chunk = pack_sequences(seqs, cfg)[0]
        for i in range(3):  # real tokens
            assert chunk["labels"][i] != -100, f"Real token at {i} incorrectly masked"

    def test_labels_no_masking_when_flag_off(self):
        """With loss_mask_padding=False, padding labels equal pad_token_id."""
        cfg = _tiny_cfg(max_seq_len=16, add_eos=False, loss_mask_padding=False)
        seqs = [[10, 11, 12]]
        chunk = pack_sequences(seqs, cfg)[0]
        # Padding positions should have labels == pad_token_id (not -100)
        for i in range(3, 16):
            assert chunk["labels"][i] == cfg.pad_token_id, (
                f"Position {i}: expected pad_token_id, got {chunk['labels'][i]}"
            )


# ---------------------------------------------------------------------------
# 7. compute_packing_efficiency — range check
# ---------------------------------------------------------------------------


class TestComputePackingEfficiency:
    def test_efficiency_in_valid_range(self):
        cfg = _tiny_cfg(max_seq_len=16)
        eff = compute_packing_efficiency(_short_seqs(), cfg)
        assert 0.0 < eff <= 1.0

    def test_efficiency_empty_input_returns_one(self):
        cfg = _tiny_cfg()
        assert compute_packing_efficiency([], cfg) == 1.0

    def test_efficiency_one_when_perfectly_packed(self):
        """Sequences that exactly fill each chunk → efficiency == 1.0."""
        cfg = PackConfig(
            max_seq_len=4, pad_token_id=0, eos_token_id=99, add_eos=False, loss_mask_padding=True
        )
        # One sequence of exactly 4 tokens → single full chunk, no padding
        seqs = [[1, 2, 3, 4]]
        eff = compute_packing_efficiency(seqs, cfg)
        assert eff == 1.0, f"Expected 1.0, got {eff}"

    def test_efficiency_decreases_with_more_padding(self):
        """Shorter sequences with large max_seq_len → lower efficiency."""
        cfg_small = PackConfig(
            max_seq_len=8, pad_token_id=0, eos_token_id=99, add_eos=False, loss_mask_padding=True
        )
        cfg_large = PackConfig(
            max_seq_len=32, pad_token_id=0, eos_token_id=99, add_eos=False, loss_mask_padding=True
        )
        seqs = [[1, 2, 3]]  # only 3 tokens
        eff_small = compute_packing_efficiency(seqs, cfg_small)
        eff_large = compute_packing_efficiency(seqs, cfg_large)
        assert eff_small >= eff_large


# ---------------------------------------------------------------------------
# 8. create_position_ids — reset at sequence boundary
# ---------------------------------------------------------------------------


class TestCreatePositionIds:
    def test_position_ids_reset_at_boundary(self):
        # seq 0: tokens at indices 0,1 ; seq 1: tokens at indices 2,3
        seq_ids = [0, 0, 1, 1]
        pos_ids = create_position_ids(seq_ids)
        # seq 0: 0,1 ; seq 1: 0,1 (reset)
        assert pos_ids == [0, 1, 0, 1], f"Got {pos_ids}"

    def test_first_token_of_each_seq_is_zero(self):
        seq_ids = [0, 0, 0, 1, 1, 2]
        pos_ids = create_position_ids(seq_ids)
        # First token of seq 0 → index 0, seq 1 → index 3, seq 2 → index 5
        assert pos_ids[0] == 0
        assert pos_ids[3] == 0
        assert pos_ids[5] == 0

    def test_padding_and_eos_positions_are_zero(self):
        seq_ids = [0, -1, 1, -1]
        pos_ids = create_position_ids(seq_ids)
        assert pos_ids[1] == 0  # EOS after seq 0
        assert pos_ids[3] == 0  # EOS/pad after seq 1

    def test_single_sequence_monotonic(self):
        seq_ids = [0, 0, 0, 0, 0]
        pos_ids = create_position_ids(seq_ids)
        assert pos_ids == [0, 1, 2, 3, 4]

    def test_all_padding(self):
        seq_ids = [-1, -1, -1]
        pos_ids = create_position_ids(seq_ids)
        assert pos_ids == [0, 0, 0]


# ---------------------------------------------------------------------------
# 9. build_document_attention_mask — shape
# ---------------------------------------------------------------------------


class TestBuildDocumentAttentionMask:
    def test_shape_is_T_by_T(self):
        seq_ids = [0, 0, 1, 1, -1]
        T = len(seq_ids)
        mask = build_document_attention_mask(seq_ids)
        assert mask.shape == (T, T), f"Expected ({T},{T}), got {mask.shape}"

    def test_dtype_is_bool(self):
        seq_ids = [0, 0, 1, -1]
        mask = build_document_attention_mask(seq_ids)
        assert mask.dtype == torch.bool

    def test_returns_tensor(self):
        mask = build_document_attention_mask([0, 0, 1])
        assert isinstance(mask, torch.Tensor)


# ---------------------------------------------------------------------------
# 10. build_document_attention_mask — cross-sequence blocking
# ---------------------------------------------------------------------------


class TestBuildDocumentAttentionMaskSemantics:
    def test_different_seqs_cannot_attend(self):
        # seq 0 at positions 0,1 ; seq 1 at positions 2,3
        seq_ids = [0, 0, 1, 1]
        mask = build_document_attention_mask(seq_ids)
        # seq 0 token attending to seq 1 token → False
        assert not mask[0, 2].item(), "seq0→seq1 should be blocked"
        assert not mask[0, 3].item()
        assert not mask[1, 2].item()
        assert not mask[1, 3].item()
        # seq 1 token attending to seq 0 token → False
        assert not mask[2, 0].item()
        assert not mask[3, 1].item()

    def test_same_seq_tokens_can_attend(self):
        seq_ids = [0, 0, 1, 1]
        mask = build_document_attention_mask(seq_ids)
        # Within seq 0
        assert mask[0, 0].item()
        assert mask[0, 1].item()
        assert mask[1, 0].item()
        assert mask[1, 1].item()
        # Within seq 1
        assert mask[2, 2].item()
        assert mask[2, 3].item()
        assert mask[3, 2].item()
        assert mask[3, 3].item()

    def test_padding_tokens_cannot_attend(self):
        # seq 0 at [0], EOS/pad at [1], seq 1 at [2]
        seq_ids = [0, -1, 1]
        mask = build_document_attention_mask(seq_ids)
        # Row 1 (padding) must be all False
        assert not mask[1, :].any().item(), "Padding row should be all False"
        # Col 1 (padding) must be all False
        assert not mask[:, 1].any().item(), "Padding col should be all False"


# ---------------------------------------------------------------------------
# 11. SequencePacker — pack returns list
# ---------------------------------------------------------------------------


class TestSequencePackerPack:
    def test_pack_returns_list(self):
        cfg = _tiny_cfg()
        packer = SequencePacker(cfg)
        result = packer.pack(_short_seqs())
        assert isinstance(result, list)

    def test_pack_chunks_have_required_keys(self):
        cfg = _tiny_cfg()
        packer = SequencePacker(cfg)
        for chunk in packer.pack(_short_seqs()):
            assert REQUIRED_KEYS <= set(chunk.keys())

    def test_efficiency_delegates_correctly(self):
        cfg = _tiny_cfg()
        packer = SequencePacker(cfg)
        eff = packer.efficiency(_short_seqs())
        direct = compute_packing_efficiency(_short_seqs(), cfg)
        assert eff == direct


# ---------------------------------------------------------------------------
# 12. SequencePacker.unpack — recovers correct number of sequences
# ---------------------------------------------------------------------------


class TestSequencePackerUnpack:
    def test_unpack_recovers_correct_number_of_seqs(self):
        """All source sequences that fit in a single chunk are recovered."""
        cfg = _tiny_cfg(max_seq_len=32)
        packer = SequencePacker(cfg)
        seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        chunks = packer.pack(seqs)
        # All seqs should fit in one chunk (total tokens + EOS = 3+1+2+1+4+1 = 12 <= 32)
        assert len(chunks) == 1
        recovered = packer.unpack(chunks[0])
        assert len(recovered) == len(seqs), f"Expected {len(seqs)} sequences, got {len(recovered)}"

    def test_unpack_recovers_correct_tokens(self):
        """Unpacked tokens must match original (excluding EOS/padding)."""
        cfg = _tiny_cfg(max_seq_len=32, add_eos=True)
        packer = SequencePacker(cfg)
        seqs = [[10, 20, 30], [40, 50]]
        chunks = packer.pack(seqs)
        assert len(chunks) == 1
        recovered = packer.unpack(chunks[0])
        assert recovered[0] == [10, 20, 30]
        assert recovered[1] == [40, 50]

    def test_unpack_no_eos_in_recovered(self):
        """EOS tokens (sequence_id == -1) must not appear in unpacked output."""
        cfg = _tiny_cfg(max_seq_len=32, add_eos=True, eos_token_id=2, pad_token_id=0)
        packer = SequencePacker(cfg)
        seqs = [[5, 6], [7, 8]]
        chunks = packer.pack(seqs)
        for chunk in chunks:
            recovered = packer.unpack(chunk)
            for seq in recovered:
                assert cfg.eos_token_id not in seq, f"EOS token found in recovered sequence: {seq}"
                assert cfg.pad_token_id not in seq, f"PAD token found in recovered sequence: {seq}"

    def test_unpack_multiple_chunks(self):
        """Unpack works correctly across multiple chunks."""
        cfg = PackConfig(
            max_seq_len=6, pad_token_id=0, eos_token_id=99, add_eos=False, loss_mask_padding=True
        )
        packer = SequencePacker(cfg)
        # Each seq of length 4 won't fit with another → separate chunks
        seqs = [[1, 2, 3, 4], [5, 6, 7, 8]]
        chunks = packer.pack(seqs)
        assert len(chunks) == 2
        rec0 = packer.unpack(chunks[0])
        rec1 = packer.unpack(chunks[1])
        assert len(rec0) == 1
        assert len(rec1) == 1
        assert rec0[0] == [1, 2, 3, 4]
        assert rec1[0] == [5, 6, 7, 8]
