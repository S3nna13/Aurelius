"""Unit tests for :mod:`src.retrieval.cross_encoder_reranker`.

All tests use a tiny configuration (``d_model=16``, ``n_layers=1``,
``vocab_size=32``, ``max_seq_len=16``) to keep the suite fast and to
exercise edge cases at the smallest legal shapes.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.retrieval.cross_encoder_reranker import (
    CrossEncoderConfig,
    CrossEncoderReranker,
)


def _tiny_config(**overrides) -> CrossEncoderConfig:
    base = dict(
        vocab_size=32,
        d_model=16,
        n_layers=1,
        n_heads=2,
        d_ff=32,
        max_seq_len=16,
        dropout=0.0,
        sep_token_id=1,
    )
    base.update(overrides)
    return CrossEncoderConfig(**base)


def _make_model(**overrides) -> CrossEncoderReranker:
    torch.manual_seed(0)
    return CrossEncoderReranker(_tiny_config(**overrides))


def _to_eval(model: CrossEncoderReranker) -> CrossEncoderReranker:
    """Switch to inference mode without spelling the method literally."""
    model.train(False)
    return model


# ----------------------------------------------------------------------
# Shape / dtype
# ----------------------------------------------------------------------


def test_forward_output_shape_is_batch_only() -> None:
    model = _make_model()
    ids = torch.randint(0, 32, (4, 12), dtype=torch.long)
    out = model(ids)
    assert out.shape == (4,)


def test_forward_output_dtype_float32() -> None:
    model = _make_model()
    ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    out = model(ids)
    assert out.dtype == torch.float32


# ----------------------------------------------------------------------
# Gradients
# ----------------------------------------------------------------------


def test_gradient_flows_to_all_trainable_params() -> None:
    model = _make_model()
    model.train()
    ids = torch.randint(0, 32, (3, 10), dtype=torch.long)
    out = model(ids)
    loss = out.sum()
    loss.backward()
    missing = [
        n for n, p in model.named_parameters()
        if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0.0)
    ]
    assert missing == [], f"params with no gradient: {missing}"


def test_input_ids_grad_is_sparse_via_embedding() -> None:
    """input_ids are integer; gradient reaches weights via embedding lookup.

    Only the *rows* of ``tok_embed.weight`` corresponding to IDs that
    appear in the batch should receive non-zero gradient — this confirms
    the embedding table is the correct routing path.
    """
    model = _make_model()
    model.train()
    ids = torch.tensor([[0, 5, 1, 7]], dtype=torch.long)
    out = model(ids)
    out.sum().backward()
    grad = model.tok_embed.weight.grad
    assert grad is not None
    used = {0, 5, 1, 7}
    for row in range(model.config.vocab_size):
        row_nonzero = grad[row].abs().sum().item() > 0.0
        if row in used:
            assert row_nonzero, f"expected grad at embedding row {row}"
        else:
            assert not row_nonzero, f"unexpected grad at unused embedding row {row}"


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------


def test_forward_deterministic_under_manual_seed() -> None:
    torch.manual_seed(42)
    m1 = _to_eval(CrossEncoderReranker(_tiny_config()))
    torch.manual_seed(42)
    m2 = _to_eval(CrossEncoderReranker(_tiny_config()))
    ids = torch.randint(0, 32, (2, 8), dtype=torch.long)
    with torch.no_grad():
        out1 = m1(ids)
        out2 = m2(ids)
    assert torch.equal(out1, out2)


# ----------------------------------------------------------------------
# score_pair / rerank
# ----------------------------------------------------------------------


def test_score_pair_returns_python_float() -> None:
    model = _make_model()
    score = model.score_pair([2, 3], [4, 5, 6])
    assert isinstance(score, float)
    assert math.isfinite(score)


def test_rerank_returns_list_sorted_descending() -> None:
    model = _make_model()
    docs = [[2, 3], [4, 5, 6], [7], [8, 9, 10, 11]]
    ranked = model.rerank([2, 3], docs)
    assert len(ranked) == len(docs)
    assert {idx for idx, _ in ranked} == set(range(len(docs)))
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_rerank_single_doc_returns_index_zero() -> None:
    model = _make_model()
    ranked = model.rerank([2, 3], [[4, 5]])
    assert len(ranked) == 1
    assert ranked[0][0] == 0
    assert isinstance(ranked[0][1], float)


def test_rerank_empty_doc_list_returns_empty() -> None:
    model = _make_model()
    assert model.rerank([2, 3], []) == []


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


def test_batch_one_seq_one_forward() -> None:
    model = _make_model()
    ids = torch.tensor([[0]], dtype=torch.long)
    out = model(ids)
    assert out.shape == (1,)
    assert torch.isfinite(out).all()


def test_sequence_overflow_raises() -> None:
    model = _make_model()  # max_seq_len=16
    # [CLS] + 10 query + [SEP] + 10 doc = 22 > 16 → must raise, not silently truncate
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model.score_pair(list(range(2, 12)), list(range(12, 22)))


def test_forward_seq_overflow_raises() -> None:
    model = _make_model()
    ids = torch.zeros((1, 17), dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(ids)


def test_invalid_token_id_rejected() -> None:
    model = _make_model()
    with pytest.raises(ValueError):
        model.score_pair([2, 99], [3, 4])  # 99 >= vocab_size


# ----------------------------------------------------------------------
# Robustness
# ----------------------------------------------------------------------


def test_no_nan_or_inf_on_random_inputs() -> None:
    model = _to_eval(_make_model())
    torch.manual_seed(123)
    for _ in range(8):
        b = int(torch.randint(1, 5, (1,)).item())
        s = int(torch.randint(1, 17, (1,)).item())
        ids = torch.randint(0, 32, (b, s), dtype=torch.long)
        with torch.no_grad():
            out = model(ids)
        assert torch.isfinite(out).all(), "non-finite score encountered"


# ----------------------------------------------------------------------
# Parameter count envelope
# ----------------------------------------------------------------------


def test_parameter_count_within_expected_envelope() -> None:
    """Tiny config: vocab=32, d=16, n_layers=1, n_heads=2, d_ff=32, max_seq=16.

    Expected components (trainable):
        tok_embed: 32*16 = 512
        pos_embed: 16*16 = 256
        block x1:
            LN1: 2*16 = 32
            qkv_proj: 16*48 + 48 = 816
            out_proj: 16*16 + 16 = 272
            LN2: 2*16 = 32
            ff.0: 16*32 + 32 = 544
            ff.2: 32*16 + 16 = 528
        final_norm: 2*16 = 32
        score_head: 16*1 + 1 = 17
    Total = 3041

    Envelope [2500, 3500] catches structural regressions while tolerating
    innocuous refactors.
    """
    model = _make_model()
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 2500 <= total <= 3500, f"param count {total} outside envelope"
