"""Tests for the Tuned Lens module (Belrose et al., arXiv:2303.08112).

~15 tests covering TunedLensTranslator, TunedLensEvaluator, TunedLensTrainer,
and LogitLens.
"""

from __future__ import annotations

import pytest
import torch
from aurelius.eval.tuned_lens import (
    LogitLens,
    TunedLensEvaluator,
    TunedLensTrainer,
    TunedLensTranslator,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B = 2  # batch size
T = 5  # sequence length
D = 16  # d_model
V = 32  # vocab size
N = 3  # n_layers


@pytest.fixture(scope="module")
def hidden_states() -> list[torch.Tensor]:
    """Fake per-layer hidden states."""
    torch.manual_seed(0)
    return [torch.randn(B, T, D) for _ in range(N)]


@pytest.fixture(scope="module")
def translator() -> TunedLensTranslator:
    return TunedLensTranslator(d_model=D, n_layers=N)


@pytest.fixture(scope="module")
def unembed_fn():
    """Simple linear unembed: d_model -> vocab_size."""
    torch.manual_seed(1)
    W = torch.randn(V, D)

    def _unembed(h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, D) -> (B, T, V)
        return h @ W.T

    return _unembed


@pytest.fixture(scope="module")
def evaluator(translator, unembed_fn) -> TunedLensEvaluator:
    return TunedLensEvaluator(translator=translator, unembed_fn=unembed_fn)


@pytest.fixture
def trainable_translator() -> TunedLensTranslator:
    """A fresh translator for training tests (not module-scoped to avoid mutation)."""
    return TunedLensTranslator(d_model=D, n_layers=N)


@pytest.fixture
def trainer(trainable_translator, unembed_fn):
    opt = torch.optim.SGD(trainable_translator.parameters(), lr=1e-3)
    return TunedLensTrainer(
        translator=trainable_translator,
        unembed_fn=unembed_fn,
        optimizer=opt,
    )


@pytest.fixture(scope="module")
def logit_lens_instance(unembed_fn) -> LogitLens:
    return LogitLens(unembed_fn=unembed_fn)


@pytest.fixture(scope="module")
def final_logits() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(B, T, V)


# ---------------------------------------------------------------------------
# TunedLensTranslator tests
# ---------------------------------------------------------------------------


def test_translator_forward_shape(translator, hidden_states):
    """1. forward() output shape is (B, T, d_model)."""
    out = translator.forward(hidden_states, layer_idx=0)
    assert out.shape == (B, T, D)


def test_translate_all_length(translator, hidden_states):
    """2. translate_all() returns a list of length n_layers."""
    outs = translator.translate_all(hidden_states)
    assert len(outs) == N


def test_translate_all_element_shapes(translator, hidden_states):
    """3. Each element from translate_all() has shape (B, T, d_model)."""
    outs = translator.translate_all(hidden_states)
    for out in outs:
        assert out.shape == (B, T, D)


def test_translator_gradient_flows(hidden_states):
    """4. Gradient flows through the translator (parameters receive grad)."""
    t = TunedLensTranslator(d_model=D, n_layers=N)
    hs = [h.detach().requires_grad_(False) for h in hidden_states]
    out = t.forward(hs, layer_idx=0)
    loss = out.sum()
    loss.backward()
    # At least the first translator's weight should have a gradient
    assert t.translators[0].weight.grad is not None
    assert t.translators[0].weight.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# TunedLensEvaluator tests
# ---------------------------------------------------------------------------


def test_evaluator_get_layer_logits_length(evaluator, hidden_states):
    """5. get_layer_logits() returns a list of length n_layers."""
    logits_list = evaluator.get_layer_logits(hidden_states)
    assert len(logits_list) == N


def test_evaluator_layer_logits_shapes(evaluator, hidden_states):
    """6. Each element from get_layer_logits() has shape (B, T, V)."""
    logits_list = evaluator.get_layer_logits(hidden_states)
    for logits in logits_list:
        assert logits.shape == (B, T, V)


def test_evaluator_layer_entropy_shape(evaluator, hidden_states):
    """7. layer_entropy() returns a tensor of shape (n_layers,)."""
    ent = evaluator.layer_entropy(hidden_states)
    assert ent.shape == (N,)


def test_evaluator_entropy_nonnegative(evaluator, hidden_states):
    """8. All entropy values are >= 0."""
    ent = evaluator.layer_entropy(hidden_states)
    assert (ent >= 0).all()


# ---------------------------------------------------------------------------
# TunedLensTrainer tests
# ---------------------------------------------------------------------------


def test_trainer_step_keys(trainer, hidden_states, final_logits):
    """9. train_step() returns a dict with 'loss' and 'mean_kl_per_layer' keys."""
    result = trainer.train_step(hidden_states, final_logits)
    assert "loss" in result
    assert "mean_kl_per_layer" in result


def test_trainer_step_loss_finite(trainer, hidden_states, final_logits):
    """10. The loss returned by train_step() is finite."""
    result = trainer.train_step(hidden_states, final_logits)
    assert torch.isfinite(torch.tensor(result["loss"]))


def test_trainer_gradient_flows(unembed_fn, hidden_states, final_logits):
    """11. Gradient flows through training step — translator params are updated."""
    t = TunedLensTranslator(d_model=D, n_layers=N)
    opt = torch.optim.SGD(t.parameters(), lr=1e-2)
    trainer_local = TunedLensTrainer(translator=t, unembed_fn=unembed_fn, optimizer=opt)

    # Record initial weight copy
    w_before = t.translators[0].weight.detach().clone()
    trainer_local.train_step(hidden_states, final_logits)
    w_after = t.translators[0].weight.detach().clone()

    assert not torch.allclose(w_before, w_after), "Parameters should change after training step"


# ---------------------------------------------------------------------------
# LogitLens tests
# ---------------------------------------------------------------------------


def test_logit_lens_forward_length(logit_lens_instance, hidden_states):
    """12. LogitLens.forward() returns a list of length n_layers."""
    outs = logit_lens_instance.forward(hidden_states)
    assert len(outs) == N


def test_logit_lens_forward_shapes(logit_lens_instance, hidden_states):
    """13. Each element from LogitLens.forward() has shape (B, T, V)."""
    outs = logit_lens_instance.forward(hidden_states)
    for out in outs:
        assert out.shape == (B, T, V)


def test_logit_lens_top_tokens_shape(logit_lens_instance, hidden_states):
    """14. top_tokens() returns a list of n_layers tensors, each (T, k)."""
    k = 5
    top = logit_lens_instance.top_tokens(hidden_states, k=k)
    # Stack to (n_layers, T, k)
    stacked = torch.stack(top)
    assert stacked.shape == (N, T, k)


def test_logit_lens_top_tokens_valid_indices(logit_lens_instance, hidden_states):
    """15. All top-k token indices are valid (in [0, V))."""
    k = 5
    top = logit_lens_instance.top_tokens(hidden_states, k=k)
    stacked = torch.stack(top)  # (N, T, k)
    assert (stacked >= 0).all()
    assert (stacked < V).all()
