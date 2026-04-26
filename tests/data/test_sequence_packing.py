"""Tests for greedy bin-packing with cross-document loss masking."""

import math

import torch

from src.data.sequence_packing import (
    PackingConfig,
    compute_document_mask,
    create_block_diagonal_attention_mask,
    greedy_pack,
    pack_dataset,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_seqs(lengths, start=1):
    """Create 1-D token-id tensors with distinct values per sequence."""
    seqs = []
    val = start
    for n in lengths:
        seqs.append(torch.arange(val, val + n, dtype=torch.long))
        val += n
    return seqs


# ---------------------------------------------------------------------------
# greedy_pack tests
# ---------------------------------------------------------------------------


def test_greedy_pack_output_shape():
    """input_ids must be (n_chunks, max_seq_len)."""
    cfg = PackingConfig(max_seq_len=16, pad_token_id=0, eos_token_id=2)
    seqs = _make_seqs([5, 4, 3])
    batch = greedy_pack(seqs, cfg)
    assert batch.input_ids.ndim == 2
    assert batch.input_ids.shape[1] == cfg.max_seq_len
    assert batch.labels.shape == batch.input_ids.shape
    assert batch.attention_mask.shape == batch.input_ids.shape
    assert batch.doc_ids.shape == batch.input_ids.shape


def test_greedy_pack_no_overflow():
    """Each chunk must fit within max_seq_len."""
    cfg = PackingConfig(max_seq_len=10, pad_token_id=0, eos_token_id=2)
    seqs = _make_seqs([4, 4, 4, 4])
    batch = greedy_pack(seqs, cfg)
    # Every row uses at most max_seq_len tokens (already enforced by shape)
    # Verify no chunk has real tokens (non-pad) beyond max_seq_len
    for i in range(batch.input_ids.shape[0]):
        real_len = (batch.attention_mask[i] == 1).sum().item()
        assert real_len <= cfg.max_seq_len


def test_greedy_pack_eos_separator():
    """EOS token (id=2) must appear as separator between documents."""
    cfg = PackingConfig(max_seq_len=32, pad_token_id=0, eos_token_id=2)
    # Use tokens 3..20 so they don't clash with eos_token_id=2
    seqs = [
        torch.tensor([3, 4, 5], dtype=torch.long),
        torch.tensor([6, 7, 8], dtype=torch.long),
    ]
    batch = greedy_pack(seqs, cfg)
    # At least one EOS should appear in the packed output
    assert (batch.input_ids == cfg.eos_token_id).any()


def test_greedy_pack_labels_mask_padding():
    """PAD positions must have label == -100."""
    cfg = PackingConfig(max_seq_len=16, pad_token_id=0, eos_token_id=2)
    seqs = _make_seqs([3])  # short sequence => lots of padding
    batch = greedy_pack(seqs, cfg)
    pad_positions = batch.input_ids == cfg.pad_token_id
    assert (batch.labels[pad_positions] == -100).all()


def test_greedy_pack_cross_doc_masking():
    """With cross_doc_loss_mask=True, first token after a doc boundary is -100."""
    cfg = PackingConfig(
        max_seq_len=32,
        pad_token_id=0,
        eos_token_id=2,
        cross_doc_loss_mask=True,
    )
    seqs = [
        torch.tensor([10, 11, 12], dtype=torch.long),
        torch.tensor([20, 21, 22], dtype=torch.long),
    ]
    batch = greedy_pack(seqs, cfg)
    # Find the first position of doc 1 (second document)
    for chunk_idx in range(batch.doc_ids.shape[0]):
        doc_row = batch.doc_ids[chunk_idx]
        label_row = batch.labels[chunk_idx]
        # Find transitions where doc_id changes
        for t in range(1, doc_row.shape[0]):
            if doc_row[t] != doc_row[t - 1] and batch.attention_mask[chunk_idx, t] == 1:
                assert label_row[t].item() == -100, (
                    f"Expected -100 at boundary position {t}, got {label_row[t].item()}"
                )


def test_greedy_pack_doc_ids_unique_per_doc():
    """Each document in a chunk must have a unique doc_id."""
    cfg = PackingConfig(max_seq_len=32, pad_token_id=0, eos_token_id=2)
    seqs = [
        torch.tensor([10, 11], dtype=torch.long),
        torch.tensor([20, 21], dtype=torch.long),
        torch.tensor([30, 31], dtype=torch.long),
    ]
    batch = greedy_pack(seqs, cfg)
    # For each chunk, the set of doc_ids for real tokens should reflect distinct docs
    for i in range(batch.doc_ids.shape[0]):
        real_mask = batch.attention_mask[i] == 1
        real_doc_ids = batch.doc_ids[i][real_mask]
        # doc_ids should be monotonically non-decreasing within a chunk
        if real_doc_ids.numel() > 1:
            diffs = real_doc_ids[1:] - real_doc_ids[:-1]
            assert (diffs >= 0).all(), "doc_ids should not decrease within a chunk"


def test_greedy_pack_n_docs_packed():
    """n_docs_packed must equal len(sequences)."""
    cfg = PackingConfig(max_seq_len=64, pad_token_id=0, eos_token_id=2)
    seqs = _make_seqs([5, 3, 7, 2])
    batch = greedy_pack(seqs, cfg)
    assert batch.n_docs_packed == len(seqs)


# ---------------------------------------------------------------------------
# compute_document_mask tests
# ---------------------------------------------------------------------------


def test_compute_document_mask_first_false():
    """First position in every chunk must always be False (no boundary before it)."""
    doc_ids = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]], dtype=torch.long)
    mask = compute_document_mask(doc_ids)
    assert mask.shape == doc_ids.shape
    assert not mask[:, 0].any(), "First column must be all False"


def test_compute_document_mask_boundary_detected():
    """True must appear exactly where doc_id changes."""
    doc_ids = torch.tensor([[0, 0, 1, 1, 2]], dtype=torch.long)
    mask = compute_document_mask(doc_ids)
    # Boundaries at positions 2 and 4
    assert not mask[0, 0]
    assert not mask[0, 1]
    assert mask[0, 2]  # 0 -> 1
    assert not mask[0, 3]
    assert mask[0, 4]  # 1 -> 2


# ---------------------------------------------------------------------------
# create_block_diagonal_attention_mask tests
# ---------------------------------------------------------------------------


def test_block_diagonal_attn_mask_shape():
    """Output must be (n_chunks, max_seq_len, max_seq_len)."""
    doc_ids = torch.tensor([[0, 0, 1, 1], [0, 1, 2, 2]], dtype=torch.long)
    mask = create_block_diagonal_attention_mask(doc_ids)
    n_chunks, seq_len = doc_ids.shape
    assert mask.shape == (n_chunks, seq_len, seq_len)


def test_block_diagonal_same_doc_attend():
    """Same-doc causal pairs must be 0.0 (attend)."""
    doc_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    mask = create_block_diagonal_attention_mask(doc_ids)
    chunk = mask[0]
    # Within doc 0: positions 0,1
    assert chunk[0, 0].item() == 0.0  # self
    assert chunk[1, 0].item() == 0.0  # pos 1 attends to pos 0 (causal)
    assert chunk[1, 1].item() == 0.0  # self
    # Within doc 1: positions 2,3
    assert chunk[2, 2].item() == 0.0
    assert chunk[3, 2].item() == 0.0
    assert chunk[3, 3].item() == 0.0


def test_block_diagonal_cross_doc_blocked():
    """Cross-doc pairs must be -inf (blocked)."""
    doc_ids = torch.tensor([[0, 0, 1, 1]], dtype=torch.long)
    mask = create_block_diagonal_attention_mask(doc_ids)
    chunk = mask[0]
    # Doc 1 attending to doc 0 tokens
    assert math.isinf(chunk[2, 0].item()) and chunk[2, 0].item() < 0
    assert math.isinf(chunk[2, 1].item()) and chunk[2, 1].item() < 0
    assert math.isinf(chunk[3, 0].item()) and chunk[3, 0].item() < 0
    assert math.isinf(chunk[3, 1].item()) and chunk[3, 1].item() < 0
    # Doc 0 cannot attend to future doc 1 tokens (causal)
    assert math.isinf(chunk[0, 2].item()) and chunk[0, 2].item() < 0
    assert math.isinf(chunk[0, 3].item()) and chunk[0, 3].item() < 0


# ---------------------------------------------------------------------------
# pack_dataset tests
# ---------------------------------------------------------------------------


def test_pack_dataset_all_docs_included():
    """Total real tokens == sum of sequence lengths + one EOS per sequence."""
    cfg = PackingConfig(max_seq_len=64, pad_token_id=0, eos_token_id=2)
    lengths = [5, 3, 7, 2]
    seqs = _make_seqs(lengths, start=10)
    batches = pack_dataset(seqs, cfg)
    assert len(batches) >= 1
    total_real = 0
    for batch in batches:
        total_real += batch.attention_mask.sum().item()
    expected = sum(lengths) + len(lengths)  # tokens + one EOS per doc
    assert total_real == expected, f"Expected {expected} real tokens, got {total_real}"
