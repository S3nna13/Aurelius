"""Tests for value-guided decoding (src/inference/value_guided_decoding.py)."""

from __future__ import annotations

import torch
import torch.nn as nn
from aurelius.inference.value_guided_decoding import (
    TokenValueFunction,
    ValueFunctionTrainer,
    ValueGuidedBeam,
    ValueGuidedDecoder,
)

# ── shared constants ────────────────────────────────────────────────────────
D_MODEL = 16
VOCAB_SIZE = 32
BEAM_WIDTH = 2
HIDDEN_SIZE = 32


# ── helpers ──────────────────────────────────────────────────────────────────


def make_value_fn() -> TokenValueFunction:
    return TokenValueFunction(d_model=D_MODEL, hidden_size=HIDDEN_SIZE)


def make_dummy_model_fn(vocab_size: int = VOCAB_SIZE, d_model: int = D_MODEL):
    """Returns a callable that produces (hidden, logits) from input_ids."""
    linear = nn.Linear(d_model, vocab_size)
    embed = nn.Embedding(vocab_size, d_model)

    def model_fn(input_ids: torch.Tensor):
        # input_ids: (1, T)
        h = embed(input_ids)  # (1, T, d_model)
        logits = linear(h)  # (1, T, vocab_size)
        return h, logits

    return model_fn


# ── TokenValueFunction ────────────────────────────────────────────────────────


def test_token_value_function_output_shape():
    """forward() should return (B, T)."""
    vf = make_value_fn()
    B, T = 3, 7
    h = torch.randn(B, T, D_MODEL)
    out = vf(h)
    assert out.shape == (B, T), f"expected ({B}, {T}), got {out.shape}"


def test_value_at_last_output_shape():
    """value_at_last() should return (B,)."""
    vf = make_value_fn()
    B, T = 4, 5
    h = torch.randn(B, T, D_MODEL)
    out = vf.value_at_last(h)
    assert out.shape == (B,), f"expected ({B},), got {out.shape}"


def test_token_value_function_finite():
    """Value outputs must be finite (no NaN / Inf)."""
    vf = make_value_fn()
    h = torch.randn(2, 6, D_MODEL)
    out = vf(h)
    assert torch.isfinite(out).all(), "value outputs contain NaN or Inf"


def test_token_value_function_gradients():
    """Gradients must flow back through the value function."""
    vf = make_value_fn()
    h = torch.randn(2, 4, D_MODEL, requires_grad=True)
    out = vf(h)
    loss = out.sum()
    loss.backward()
    assert h.grad is not None, "no gradient reached hidden_states"
    assert torch.isfinite(h.grad).all(), "gradient contains NaN/Inf"


# ── ValueGuidedBeam ───────────────────────────────────────────────────────────


def test_beam_extend_appends_token():
    """extend() must append next_token to token_ids."""
    beam = ValueGuidedBeam([1, 2, 3], score=0.0, value_score=0.0, lm_score=0.0)
    new_beam = beam.extend(next_token=7, next_lm_logprob=-0.5, next_value=0.8, alpha=0.5)
    assert new_beam.token_ids == [1, 2, 3, 7]


def test_beam_extend_lm_score_accumulates():
    """new lm_score = old lm_score + next_lm_logprob."""
    old_lm = -1.0
    beam = ValueGuidedBeam([0], score=0.0, value_score=0.0, lm_score=old_lm)
    next_lp = -0.3
    new_beam = beam.extend(next_token=5, next_lm_logprob=next_lp, next_value=0.0, alpha=0.0)
    assert abs(new_beam.lm_score - (old_lm + next_lp)) < 1e-6


def test_beam_extend_combined_score():
    """new score = alpha * value + (1 - alpha) * lm_score."""
    beam = ValueGuidedBeam([0], score=0.0, value_score=0.0, lm_score=-2.0)
    alpha = 0.4
    next_value = 0.9
    next_lp = -0.5
    new_beam = beam.extend(
        next_token=3, next_lm_logprob=next_lp, next_value=next_value, alpha=alpha
    )
    new_lm = -2.0 + next_lp
    expected = alpha * next_value + (1 - alpha) * new_lm
    assert abs(new_beam.score - expected) < 1e-6


# ── ValueGuidedDecoder ────────────────────────────────────────────────────────


def test_generate_output_shape():
    """generate() must return a tensor of shape (max_new_tokens,)."""
    vf = make_value_fn()
    model_fn = make_dummy_model_fn()
    decoder = ValueGuidedDecoder(model_fn, vf, beam_width=BEAM_WIDTH, alpha=0.5)
    prompt = torch.zeros(1, 4, dtype=torch.long)
    max_new = 6
    out = decoder.generate(prompt, max_new_tokens=max_new)
    assert out.shape == (max_new,), f"expected ({max_new},), got {out.shape}"


def test_generate_output_dtype():
    """generate() output must be a long (int64) tensor."""
    vf = make_value_fn()
    model_fn = make_dummy_model_fn()
    decoder = ValueGuidedDecoder(model_fn, vf, beam_width=BEAM_WIDTH, alpha=0.5)
    prompt = torch.zeros(1, 3, dtype=torch.long)
    out = decoder.generate(prompt, max_new_tokens=5)
    assert out.dtype == torch.long, f"expected torch.long, got {out.dtype}"


def test_generate_non_negative_token_ids():
    """All generated token ids must be non-negative."""
    vf = make_value_fn()
    model_fn = make_dummy_model_fn()
    decoder = ValueGuidedDecoder(model_fn, vf, beam_width=BEAM_WIDTH, alpha=0.5)
    prompt = torch.zeros(1, 3, dtype=torch.long)
    out = decoder.generate(prompt, max_new_tokens=8)
    assert (out >= 0).all(), "generated token ids contain negative values"


# ── ValueFunctionTrainer ──────────────────────────────────────────────────────


def test_compute_returns_shape():
    """compute_returns() must return (T,)."""
    vf = make_value_fn()
    opt = torch.optim.Adam(vf.parameters(), lr=1e-3)
    trainer = ValueFunctionTrainer(vf, opt, gamma=0.99)
    T = 10
    rewards = torch.rand(T)
    returns = trainer.compute_returns(rewards)
    assert returns.shape == (T,), f"expected ({T},), got {returns.shape}"


def test_compute_returns_last_element():
    """Last element of returns must equal the last reward."""
    vf = make_value_fn()
    opt = torch.optim.Adam(vf.parameters(), lr=1e-3)
    trainer = ValueFunctionTrainer(vf, opt, gamma=0.99)
    rewards = torch.tensor([0.1, 0.5, 0.9, 1.0])
    returns = trainer.compute_returns(rewards)
    assert abs(returns[-1].item() - rewards[-1].item()) < 1e-5


def test_compute_returns_discounted_properly():
    """G_0 should be > r_0 when future rewards are positive."""
    vf = make_value_fn()
    opt = torch.optim.Adam(vf.parameters(), lr=1e-3)
    trainer = ValueFunctionTrainer(vf, opt, gamma=0.99)
    # All positive rewards so G_0 accumulates discounted future rewards
    rewards = torch.ones(5) * 1.0
    returns = trainer.compute_returns(rewards)
    assert returns[0].item() > rewards[0].item(), (
        f"G_0={returns[0].item()} should exceed r_0={rewards[0].item()} "
        "when future rewards are positive"
    )


def test_train_step_returns_expected_keys():
    """train_step() must return dict with 'loss', 'mean_value', 'mean_return'."""
    vf = make_value_fn()
    opt = torch.optim.Adam(vf.parameters(), lr=1e-3)
    trainer = ValueFunctionTrainer(vf, opt, gamma=0.99)
    T = 5
    hidden = torch.randn(1, T, D_MODEL)
    rewards = torch.rand(T)
    info = trainer.train_step(hidden, rewards)
    assert set(info.keys()) == {"loss", "mean_value", "mean_return"}, (
        f"unexpected keys: {set(info.keys())}"
    )


def test_train_step_loss_is_finite():
    """train_step() loss must be a finite float."""
    vf = make_value_fn()
    opt = torch.optim.Adam(vf.parameters(), lr=1e-3)
    trainer = ValueFunctionTrainer(vf, opt, gamma=0.99)
    T = 8
    hidden = torch.randn(1, T, D_MODEL)
    rewards = torch.rand(T)
    info = trainer.train_step(hidden, rewards)
    assert torch.isfinite(torch.tensor(info["loss"])), f"loss is not finite: {info['loss']}"
