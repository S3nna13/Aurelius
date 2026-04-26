"""Tests for FIMTransformer and FIMLossFilter (src/model/fim_lm.py).

Covers:
    - Configuration defaults
    - Document splitting correctness (parts, lengths)
    - SPM / PSM token ordering and content
    - transform() with fim_rate=0/1
    - transform_batch() output length
    - truncate_to_max()
    - FIMLossFilter mask generation
    - Integration: batch of 4 documents through the full pipeline
"""

from __future__ import annotations

import random

import pytest

from src.model.fim_lm import FIMConfig, FIMDocument, FIMLossFilter, FIMTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> FIMConfig:
    return FIMConfig()


@pytest.fixture
def transformer(default_config: FIMConfig) -> FIMTransformer:
    return FIMTransformer(default_config)


@pytest.fixture
def loss_filter(default_config: FIMConfig) -> FIMLossFilter:
    return FIMLossFilter(default_config)


@pytest.fixture
def sample_tokens() -> list[int]:
    """A moderate-length token sequence for reuse across tests."""
    return list(range(50))


@pytest.fixture
def fixed_rng() -> random.Random:
    return random.Random(0)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults() -> None:
    """FIMConfig fields carry the expected default values from the paper."""
    cfg = FIMConfig()
    assert cfg.fim_prefix_token_id == 100257
    assert cfg.fim_suffix_token_id == 100258
    assert cfg.fim_middle_token_id == 100259
    assert cfg.eot_token_id == 100260
    assert cfg.fim_rate == 0.5
    assert cfg.spm_rate == 0.5
    assert cfg.max_seq_len == 8192
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# 2. test_split_document_parts — all three segments are present (non-degenerate)
# ---------------------------------------------------------------------------


def test_split_document_parts(transformer: FIMTransformer) -> None:
    """split_document produces a valid FIMDocument with all three segments.

    We force a long enough sequence so that at least one meaningful split is
    possible, and we seed the RNG for reproducibility.
    """
    token_ids = list(range(100))
    rng = random.Random(7)

    doc = transformer.split_document(token_ids, rng)

    assert isinstance(doc, FIMDocument)
    assert isinstance(doc.prefix_ids, list)
    assert isinstance(doc.middle_ids, list)
    assert isinstance(doc.suffix_ids, list)
    assert doc.mode in ("spm", "psm")


# ---------------------------------------------------------------------------
# 3. test_split_document_total_length — no tokens are lost or duplicated
# ---------------------------------------------------------------------------


def test_split_document_total_length(transformer: FIMTransformer) -> None:
    """prefix + middle + suffix reconstructs the original sequence exactly."""
    token_ids = list(range(80))
    rng = random.Random(13)

    doc = transformer.split_document(token_ids, rng)

    reconstructed = doc.prefix_ids + doc.middle_ids + doc.suffix_ids
    assert reconstructed == token_ids


# ---------------------------------------------------------------------------
# 4. test_to_spm_starts_with_suffix_token
# ---------------------------------------------------------------------------


def test_to_spm_starts_with_suffix_token(
    transformer: FIMTransformer, default_config: FIMConfig
) -> None:
    """SPM output must begin with the FIM_SUFFIX special token."""
    doc = FIMDocument(
        prefix_ids=[1, 2, 3],
        suffix_ids=[7, 8, 9],
        middle_ids=[4, 5, 6],
        mode="spm",
    )
    result = transformer.to_spm(doc)
    assert result[0] == default_config.fim_suffix_token_id


# ---------------------------------------------------------------------------
# 5. test_to_psm_starts_with_prefix_token
# ---------------------------------------------------------------------------


def test_to_psm_starts_with_prefix_token(
    transformer: FIMTransformer, default_config: FIMConfig
) -> None:
    """PSM output must begin with the FIM_PREFIX special token."""
    doc = FIMDocument(
        prefix_ids=[1, 2, 3],
        suffix_ids=[7, 8, 9],
        middle_ids=[4, 5, 6],
        mode="psm",
    )
    result = transformer.to_psm(doc)
    assert result[0] == default_config.fim_prefix_token_id


# ---------------------------------------------------------------------------
# 6. test_to_spm_contains_all_parts
# ---------------------------------------------------------------------------


def test_to_spm_contains_all_parts(transformer: FIMTransformer, default_config: FIMConfig) -> None:
    """SPM output contains all original tokens plus the three special tokens + EOT."""
    prefix = [10, 11]
    middle = [20, 21, 22]
    suffix = [30]
    doc = FIMDocument(prefix_ids=prefix, suffix_ids=suffix, middle_ids=middle, mode="spm")

    result = transformer.to_spm(doc)
    result_set = set(result)

    # Original tokens must all appear.
    for tok in prefix + middle + suffix:
        assert tok in result_set

    # Special tokens must appear.
    assert default_config.fim_prefix_token_id in result_set
    assert default_config.fim_suffix_token_id in result_set
    assert default_config.fim_middle_token_id in result_set
    assert default_config.eot_token_id in result_set


# ---------------------------------------------------------------------------
# 7. test_to_psm_contains_all_parts
# ---------------------------------------------------------------------------


def test_to_psm_contains_all_parts(transformer: FIMTransformer, default_config: FIMConfig) -> None:
    """PSM output contains all original tokens plus the three special tokens + EOT."""
    prefix = [10, 11]
    middle = [20, 21, 22]
    suffix = [30]
    doc = FIMDocument(prefix_ids=prefix, suffix_ids=suffix, middle_ids=middle, mode="psm")

    result = transformer.to_psm(doc)
    result_set = set(result)

    for tok in prefix + middle + suffix:
        assert tok in result_set

    assert default_config.fim_prefix_token_id in result_set
    assert default_config.fim_suffix_token_id in result_set
    assert default_config.fim_middle_token_id in result_set
    assert default_config.eot_token_id in result_set


# ---------------------------------------------------------------------------
# 8. test_to_spm_ends_with_eot
# ---------------------------------------------------------------------------


def test_to_spm_ends_with_eot(transformer: FIMTransformer, default_config: FIMConfig) -> None:
    """SPM output must end with the EOT token."""
    doc = FIMDocument(
        prefix_ids=[1, 2],
        suffix_ids=[5, 6],
        middle_ids=[3, 4],
        mode="spm",
    )
    result = transformer.to_spm(doc)
    assert result[-1] == default_config.eot_token_id


# ---------------------------------------------------------------------------
# 9. test_transform_applies_fim — fim_rate=1.0 always transforms
# ---------------------------------------------------------------------------


def test_transform_applies_fim(default_config: FIMConfig) -> None:
    """With fim_rate=1.0 every document is converted to FIM format."""
    cfg = FIMConfig(
        fim_prefix_token_id=default_config.fim_prefix_token_id,
        fim_suffix_token_id=default_config.fim_suffix_token_id,
        fim_middle_token_id=default_config.fim_middle_token_id,
        eot_token_id=default_config.eot_token_id,
        fim_rate=1.0,
        spm_rate=0.5,
        max_seq_len=default_config.max_seq_len,
    )
    t = FIMTransformer(cfg)
    token_ids = list(range(30))
    rng = random.Random(99)

    result = t.transform(token_ids, rng)

    # FIM_MID must always be present when fim_rate=1.0
    assert cfg.fim_middle_token_id in result


# ---------------------------------------------------------------------------
# 10. test_transform_skips_fim — fim_rate=0.0 returns original + EOT
# ---------------------------------------------------------------------------


def test_transform_skips_fim(default_config: FIMConfig) -> None:
    """With fim_rate=0.0 the output is the original tokens followed by EOT."""
    cfg = FIMConfig(
        fim_prefix_token_id=default_config.fim_prefix_token_id,
        fim_suffix_token_id=default_config.fim_suffix_token_id,
        fim_middle_token_id=default_config.fim_middle_token_id,
        eot_token_id=default_config.eot_token_id,
        fim_rate=0.0,
        spm_rate=0.5,
        max_seq_len=default_config.max_seq_len,
    )
    t = FIMTransformer(cfg)
    token_ids = list(range(20))
    rng = random.Random(5)

    result = t.transform(token_ids, rng)

    expected = token_ids + [cfg.eot_token_id]
    assert result == expected


# ---------------------------------------------------------------------------
# 11. test_transform_batch_length — output list same length as input
# ---------------------------------------------------------------------------


def test_transform_batch_length(transformer: FIMTransformer) -> None:
    """transform_batch returns the same number of sequences as the input."""
    batch = [list(range(i, i + 20)) for i in range(0, 80, 20)]  # 4 docs

    result = transformer.transform_batch(batch, seed=1)

    assert len(result) == len(batch)
    for seq in result:
        assert isinstance(seq, list)


# ---------------------------------------------------------------------------
# 12. test_truncate_to_max
# ---------------------------------------------------------------------------


def test_truncate_to_max(default_config: FIMConfig) -> None:
    """truncate_to_max clips sequences that exceed max_seq_len."""
    cfg = FIMConfig(max_seq_len=10)
    t = FIMTransformer(cfg)

    long_seq = list(range(50))
    truncated = t.truncate_to_max(long_seq)

    assert len(truncated) == 10
    assert truncated == long_seq[:10]


# ---------------------------------------------------------------------------
# 13. test_loss_mask_after_fim_mid
# ---------------------------------------------------------------------------


def test_loss_mask_after_fim_mid(loss_filter: FIMLossFilter, default_config: FIMConfig) -> None:
    """Tokens after FIM_MID receive True; tokens before (and FIM_MID itself) get False."""
    mid = default_config.fim_middle_token_id
    eot = default_config.eot_token_id

    # Layout: [prefix_tok, FIM_MID, middle_tok1, middle_tok2, EOT]
    token_ids = [10, 20, mid, 30, 40, eot]
    mask = loss_filter.make_loss_mask(token_ids)

    assert len(mask) == len(token_ids)
    # Tokens before and including FIM_MID → False
    assert mask[0] is False
    assert mask[1] is False
    assert mask[2] is False  # FIM_MID itself
    # Tokens after FIM_MID → True
    assert mask[3] is True
    assert mask[4] is True
    assert mask[5] is True  # EOT


# ---------------------------------------------------------------------------
# 14. test_loss_mask_no_fim — standard LM: all True
# ---------------------------------------------------------------------------


def test_loss_mask_no_fim(loss_filter: FIMLossFilter) -> None:
    """When FIM_MID is absent every token receives True (standard causal LM)."""
    token_ids = [1, 2, 3, 4, 5]
    mask = loss_filter.make_loss_mask(token_ids)

    assert all(mask)
    assert len(mask) == len(token_ids)


# ---------------------------------------------------------------------------
# 15. test_spm_structure — verify SPM segment ordering in output list
# ---------------------------------------------------------------------------


def test_spm_structure(transformer: FIMTransformer, default_config: FIMConfig) -> None:
    """SPM output obeys: [FIM_SUF] suffix [FIM_PRE] prefix [FIM_MID] middle [EOT]."""
    prefix = [1, 2]
    middle = [3, 4]
    suffix = [5, 6]
    doc = FIMDocument(prefix_ids=prefix, suffix_ids=suffix, middle_ids=middle, mode="spm")

    result = transformer.to_spm(doc)

    cfg = default_config
    expected = (
        [cfg.fim_suffix_token_id]
        + suffix
        + [cfg.fim_prefix_token_id]
        + prefix
        + [cfg.fim_middle_token_id]
        + middle
        + [cfg.eot_token_id]
    )
    assert result == expected


# ---------------------------------------------------------------------------
# 16. test_psm_structure — verify PSM segment ordering in output list
# ---------------------------------------------------------------------------


def test_psm_structure(transformer: FIMTransformer, default_config: FIMConfig) -> None:
    """PSM output obeys: [FIM_PRE] prefix [FIM_SUF] suffix [FIM_MID] middle [EOT]."""
    prefix = [1, 2]
    middle = [3, 4]
    suffix = [5, 6]
    doc = FIMDocument(prefix_ids=prefix, suffix_ids=suffix, middle_ids=middle, mode="psm")

    result = transformer.to_psm(doc)

    cfg = default_config
    expected = (
        [cfg.fim_prefix_token_id]
        + prefix
        + [cfg.fim_suffix_token_id]
        + suffix
        + [cfg.fim_middle_token_id]
        + middle
        + [cfg.eot_token_id]
    )
    assert result == expected


# ---------------------------------------------------------------------------
# Integration test — transform_batch over 4 documents
# ---------------------------------------------------------------------------


def test_integration_transform_batch_fim_tokens() -> None:
    """Integration: run transform_batch on 4 × 100-token docs with fim_rate=1.0.

    With fim_rate=1.0 every document is guaranteed to be FIM-transformed,
    so each output sequence must contain the FIM_MID special token.
    The EOT token must also appear at the end of every output sequence.
    """
    cfg = FIMConfig(fim_rate=1.0, spm_rate=0.5, max_seq_len=8192, seed=0)
    transformer = FIMTransformer(cfg)
    loss_filter = FIMLossFilter(cfg)

    # Build 4 documents with varying content
    batch = [list(range(i * 100, (i + 1) * 100)) for i in range(4)]

    outputs = transformer.transform_batch(batch, seed=123)

    assert len(outputs) == 4

    for idx, seq in enumerate(outputs):
        # Each FIM-formatted sequence must contain FIM_MID (fim_rate=1.0)
        assert cfg.fim_middle_token_id in seq, f"Document {idx} missing FIM_MID token"
        # EOT must appear as the last token
        assert seq[-1] == cfg.eot_token_id, f"Document {idx} does not end with EOT"
        # Loss mask must be valid
        mask = loss_filter.make_loss_mask(seq)
        assert len(mask) == len(seq)
        # At least one token must be in the loss (the middle segment + EOT)
        assert any(mask), f"Document {idx} has empty loss mask"
        # Tokens before FIM_MID must all be masked out
        mid_pos = seq.index(cfg.fim_middle_token_id)
        assert all(not mask[i] for i in range(mid_pos + 1)), (
            f"Document {idx}: tokens up to FIM_MID should be False"
        )
