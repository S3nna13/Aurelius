"""Tests for src/inference/medusa_heads.py"""

import torch
from aurelius.inference.medusa_heads import (
    MedusaDecoder,
    MedusaHead,
    MedusaHeads,
    MedusaVerifier,
)

D_MODEL = 32
VOCAB_SIZE = 64
N_HEADS = 4
B, T = 2, 8


# ---------------------------------------------------------------------------
# MedusaHead
# ---------------------------------------------------------------------------


def test_medusa_head_output_shape():
    head = MedusaHead(D_MODEL, VOCAB_SIZE)
    x = torch.randn(B, T, D_MODEL)
    out = head(x)
    assert out.shape == (B, T, VOCAB_SIZE)


def test_medusa_head_two_layer():
    head = MedusaHead(D_MODEL, VOCAB_SIZE, n_hidden_layers=2)
    x = torch.randn(B, T, D_MODEL)
    out = head(x)
    assert out.shape == (B, T, VOCAB_SIZE)


def test_medusa_head_gradient_flows():
    head = MedusaHead(D_MODEL, VOCAB_SIZE)
    x = torch.randn(B, T, D_MODEL, requires_grad=True)
    out = head(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# MedusaHeads
# ---------------------------------------------------------------------------


def test_medusa_heads_forward_returns_list():
    heads = MedusaHeads(D_MODEL, VOCAB_SIZE, n_heads=N_HEADS)
    x = torch.randn(B, T, D_MODEL)
    result = heads(x)
    assert isinstance(result, list)
    assert len(result) == N_HEADS
    for logits in result:
        assert logits.shape == (B, T, VOCAB_SIZE)


def test_medusa_heads_predict_next_tokens_shape():
    heads = MedusaHeads(D_MODEL, VOCAB_SIZE, n_heads=N_HEADS)
    x = torch.randn(B, T, D_MODEL)
    preds = heads.predict_next_tokens(x, greedy=True)
    assert preds.shape == (B, N_HEADS)
    assert preds.dtype == torch.long


def test_medusa_heads_predict_sample_shape():
    heads = MedusaHeads(D_MODEL, VOCAB_SIZE, n_heads=N_HEADS)
    x = torch.randn(1, T, D_MODEL)
    preds = heads.predict_next_tokens(x, greedy=False)
    assert preds.shape == (1, N_HEADS)


def test_medusa_heads_token_ids_in_vocab_range():
    heads = MedusaHeads(D_MODEL, VOCAB_SIZE, n_heads=N_HEADS)
    x = torch.randn(B, T, D_MODEL)
    preds = heads.predict_next_tokens(x)
    assert (preds >= 0).all()
    assert (preds < VOCAB_SIZE).all()


# ---------------------------------------------------------------------------
# MedusaVerifier
# ---------------------------------------------------------------------------


def test_verifier_all_match():
    verifier = MedusaVerifier(n_heads=N_HEADS)
    base_tokens = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    head_preds = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    accepted, n = verifier.verify(base_tokens, head_preds)
    assert n == 4
    assert list(accepted.tolist()) == [1, 2, 3, 4]


def test_verifier_no_match():
    verifier = MedusaVerifier(n_heads=N_HEADS)
    base_tokens = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    head_preds = torch.tensor([9, 9, 9, 9], dtype=torch.long)
    accepted, n = verifier.verify(base_tokens, head_preds)
    assert n == 0
    assert len(accepted) == 0


def test_verifier_partial_match():
    verifier = MedusaVerifier(n_heads=4)
    base_tokens = torch.tensor([5, 6, 7, 8], dtype=torch.long)
    head_preds = torch.tensor([5, 6, 99, 8], dtype=torch.long)
    accepted, n = verifier.verify(base_tokens, head_preds)
    assert n == 2
    assert list(accepted.tolist()) == [5, 6]


def test_verifier_2d_head_predictions():
    verifier = MedusaVerifier(n_heads=3)
    # 2D case: head_predictions[i, i] is diagonal
    base_tokens = torch.tensor([10, 20, 30], dtype=torch.long)
    head_preds_2d = torch.tensor([[10, 0, 0], [0, 20, 0], [0, 0, 30]], dtype=torch.long)
    accepted, n = verifier.verify(base_tokens, head_preds_2d)
    assert n == 3


# ---------------------------------------------------------------------------
# MedusaDecoder
# ---------------------------------------------------------------------------


def _make_base_model_fn(vocab_size: int, seq_tokens: list):
    """Return a callable that always returns the next token from seq_tokens."""
    call_count = [0]

    def fn(input_ids):
        idx = min(call_count[0], len(seq_tokens) - 1)
        call_count[0] += 1
        T = input_ids.shape[1]
        d = 16
        hidden = torch.randn(1, T, d)
        v = vocab_size
        logits = torch.full((1, T, v), -1e9)
        logits[0, -1, seq_tokens[idx]] = 10.0
        return hidden, logits

    return fn


def test_decoder_generate_length():
    heads = MedusaHeads(16, VOCAB_SIZE, n_heads=2)
    base_fn = _make_base_model_fn(VOCAB_SIZE, list(range(VOCAB_SIZE)))
    decoder = MedusaDecoder(base_fn, heads, n_steps=2)
    prompt = torch.zeros(1, 3, dtype=torch.long)
    out = decoder.generate(prompt, max_new_tokens=4)
    assert out.shape == (4,)
    assert out.dtype == torch.long


def test_decoder_tokens_in_vocab():
    heads = MedusaHeads(16, VOCAB_SIZE, n_heads=2)
    base_fn = _make_base_model_fn(VOCAB_SIZE, [5, 10, 15, 20] * 10)
    decoder = MedusaDecoder(base_fn, heads, n_steps=2)
    prompt = torch.zeros(1, 2, dtype=torch.long)
    out = decoder.generate(prompt, max_new_tokens=6)
    assert (out >= 0).all()
    assert (out < VOCAB_SIZE).all()
