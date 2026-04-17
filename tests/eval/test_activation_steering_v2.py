"""Tests for src/eval/activation_steering_v2.py.

Tiny config: d_model=16, n_layers=2, seq_len=8, batch=2, vocab=16.
All tests run actual forward (and where needed, backward) passes.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.eval.activation_steering_v2 import (
    ContrastivePairCollector,
    ConceptVectorExtractor,
    ActivationHook,
    ActivationSteerer,
    SteeringEffect,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
N_LAYERS = 2
SEQ_LEN = 8
BATCH = 2
VOCAB = 16

# ---------------------------------------------------------------------------
# Tiny transformer fixture
# ---------------------------------------------------------------------------


class _TinyTransformer(nn.Module):
    """Minimal transformer-like model that exposes `.layers` (nn.ModuleList).

    Architecture: Embedding -> 2x TransformerEncoderLayer -> Linear(vocab).
    """

    def __init__(self, d_model: int, n_layers: int, vocab: int, nhead: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab)

    def forward(self, input_ids: Tensor) -> Tensor:
        """Return logits (B, T, V)."""
        x = self.embed(input_ids)          # (B, T, D)
        for layer in self.layers:
            x = layer(x)                   # (B, T, D)
        return self.head(x)                # (B, T, V)


@pytest.fixture(scope="module")
def tiny_model() -> _TinyTransformer:
    torch.manual_seed(0)
    model = _TinyTransformer(D_MODEL, N_LAYERS, VOCAB)
    model.train(False)
    return model


@pytest.fixture(scope="module")
def concept_vec() -> Tensor:
    torch.manual_seed(1)
    v = torch.randn(D_MODEL)
    return F.normalize(v, dim=0)


@pytest.fixture(scope="module")
def input_ids() -> Tensor:
    torch.manual_seed(2)
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


# ---------------------------------------------------------------------------
# Helper: make (B, T, D) random activations
# ---------------------------------------------------------------------------

def _rand_acts(seed: int = 42) -> Tensor:
    torch.manual_seed(seed)
    return torch.randn(BATCH, SEQ_LEN, D_MODEL)


# ===========================================================================
# 1. ContrastivePairCollector.record: get_arrays returns correct shapes (N, D)
# ===========================================================================

def test_collector_record_shapes() -> None:
    collector = ContrastivePairCollector()
    n_records = 3
    for i in range(n_records):
        collector.record(_rand_acts(i), _rand_acts(i + 100))

    pos, neg = collector.get_arrays()
    assert pos.shape == (n_records * BATCH, D_MODEL), f"pos shape {pos.shape}"
    assert neg.shape == (n_records * BATCH, D_MODEL), f"neg shape {neg.shape}"


# ===========================================================================
# 2. ContrastivePairCollector.reset: empties lists
# ===========================================================================

def test_collector_reset() -> None:
    collector = ContrastivePairCollector()
    collector.record(_rand_acts(0), _rand_acts(1))
    assert len(collector.pos_list) == 1

    collector.reset()
    assert len(collector.pos_list) == 0
    assert len(collector.neg_list) == 0


# ===========================================================================
# 3. ConceptVectorExtractor.mean_diff: output shape (D,), unit norm
# ===========================================================================

def test_mean_diff_shape_and_norm() -> None:
    pos = _rand_acts(0).mean(dim=1)
    neg = _rand_acts(1).mean(dim=1)

    extractor = ConceptVectorExtractor(method="mean_diff")
    v = extractor.mean_diff(pos, neg)

    assert v.shape == (D_MODEL,), f"shape {v.shape}"
    assert math.isclose(v.norm().item(), 1.0, abs_tol=1e-5), f"norm {v.norm().item()}"


# ===========================================================================
# 4. ConceptVectorExtractor.pca_direction: output shape (D,), unit norm
# ===========================================================================

def test_pca_direction_shape_and_norm() -> None:
    pos = _rand_acts(0).mean(dim=1)
    neg = _rand_acts(1).mean(dim=1)

    extractor = ConceptVectorExtractor(method="pca")
    v = extractor.pca_direction(pos, neg)

    assert v.shape == (D_MODEL,), f"shape {v.shape}"
    assert math.isclose(v.norm().item(), 1.0, abs_tol=1e-5), f"norm {v.norm().item()}"


# ===========================================================================
# 5. ConceptVectorExtractor.logistic_direction: output shape (D,), unit norm
# ===========================================================================

def test_logistic_direction_shape_and_norm() -> None:
    pos = _rand_acts(0).mean(dim=1)
    neg = _rand_acts(1).mean(dim=1)

    extractor = ConceptVectorExtractor(method="logistic")
    v = extractor.logistic_direction(pos, neg, n_steps=20)

    assert v.shape == (D_MODEL,), f"shape {v.shape}"
    assert math.isclose(v.norm().item(), 1.0, abs_tol=1e-5), f"norm {v.norm().item()}"


# ===========================================================================
# 6. ActivationHook "capture": captures output, shape preserved
# ===========================================================================

def test_hook_capture(tiny_model: _TinyTransformer, input_ids: Tensor) -> None:
    hook = ActivationHook(layer_idx=0, mode="capture")
    handle = hook.register(tiny_model.layers[0])
    try:
        with torch.no_grad():
            logits_with_hook = tiny_model(input_ids)
    finally:
        handle.remove()

    assert hook.captured is not None, "captured should not be None after forward pass"
    assert hook.captured.shape == (BATCH, SEQ_LEN, D_MODEL), \
        f"captured shape {hook.captured.shape}"

    with torch.no_grad():
        logits_no_hook = tiny_model(input_ids)

    assert torch.allclose(logits_with_hook, logits_no_hook), \
        "capture hook must not alter output values"


# ===========================================================================
# 7. ActivationHook "add": output differs from original by alpha*vector
# ===========================================================================

def test_hook_add_modifies_output() -> None:
    torch.manual_seed(10)
    linear = nn.Linear(D_MODEL, D_MODEL, bias=False)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    with torch.no_grad():
        original_out = linear(x)

    alpha = 2.5
    vec = F.normalize(torch.randn(D_MODEL), dim=0)

    hook = ActivationHook(layer_idx=0, mode="add")
    hook.set_steering(vec, alpha)
    handle = hook.register(linear)
    try:
        with torch.no_grad():
            steered_out = linear(x)
    finally:
        handle.remove()

    diff = steered_out - original_out
    expected = alpha * vec
    for b in range(BATCH):
        for t in range(SEQ_LEN):
            assert torch.allclose(diff[b, t], expected, atol=1e-5), \
                f"diff at [{b},{t}] != alpha*vec"


# ===========================================================================
# 8. ActivationHook "project_out": component along vector removed (dot product ~0)
# ===========================================================================

def test_hook_project_out() -> None:
    torch.manual_seed(20)
    linear = nn.Linear(D_MODEL, D_MODEL, bias=False)
    x = torch.randn(BATCH, SEQ_LEN, D_MODEL)

    vec = F.normalize(torch.randn(D_MODEL), dim=0)

    hook = ActivationHook(layer_idx=0, mode="project_out")
    hook.set_steering(vec, alpha=1.0)
    handle = hook.register(linear)
    try:
        with torch.no_grad():
            projected_out = linear(x)
    finally:
        handle.remove()

    dots = (projected_out * vec).sum(dim=-1)  # (B, T)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-5), \
        f"max residual component: {dots.abs().max().item()}"


# ===========================================================================
# 9. ActivationSteerer.steered_forward: output shape (B, T, V)
# ===========================================================================

def test_steered_forward_shape(tiny_model: _TinyTransformer, concept_vec: Tensor,
                               input_ids: Tensor) -> None:
    steerer = ActivationSteerer(tiny_model, layers=[0])
    with torch.no_grad():
        logits = steerer.steered_forward(input_ids, concept_vec, alpha=1.0)

    assert logits.shape == (BATCH, SEQ_LEN, VOCAB), f"logits shape {logits.shape}"


# ===========================================================================
# 10. ActivationSteerer: alpha=0 -> steered output matches unsteered
# ===========================================================================

def test_steered_forward_alpha_zero(tiny_model: _TinyTransformer, concept_vec: Tensor,
                                    input_ids: Tensor) -> None:
    steerer = ActivationSteerer(tiny_model, layers=[0, 1])

    with torch.no_grad():
        unsteered = tiny_model(input_ids)
        steered = steerer.steered_forward(input_ids, concept_vec, alpha=0.0)

    assert torch.allclose(unsteered, steered, atol=1e-5), \
        "alpha=0 steering must not change logits"


# ===========================================================================
# 11. SteeringEffect.logit_diff: finite float, sign reflects direction of change
# ===========================================================================

def test_logit_diff_sign_and_finite(tiny_model: _TinyTransformer, concept_vec: Tensor,
                                    input_ids: Tensor) -> None:
    steerer = ActivationSteerer(tiny_model, layers=[0])
    effect = SteeringEffect()
    target = 0

    with torch.no_grad():
        original = tiny_model(input_ids)
        steered_pos = steerer.steered_forward(input_ids, concept_vec, alpha=10.0)
        steered_neg = steerer.steered_forward(input_ids, concept_vec, alpha=-10.0)

    diff_pos = effect.logit_diff(original, steered_pos, target)
    diff_neg = effect.logit_diff(original, steered_neg, target)

    assert math.isfinite(diff_pos), f"logit_diff not finite: {diff_pos}"
    assert math.isfinite(diff_neg), f"logit_diff not finite: {diff_neg}"
    # With +alpha the steered distribution moves in a direction relative to -alpha;
    # the two diffs must be different (sign of alpha controls the direction of shift).
    assert diff_pos != diff_neg, \
        "logit_diff with +alpha and -alpha should differ (opposite steering directions)"
    # +alpha should always produce a larger logit_diff than -alpha for the same target
    assert diff_pos > diff_neg, \
        f"expected diff_pos({diff_pos:.4f}) > diff_neg({diff_neg:.4f})"


# ===========================================================================
# 12. SteeringEffect.kl_divergence: >= 0, 0.0 for identical logits
# ===========================================================================

def test_kl_divergence_non_negative_and_zero() -> None:
    effect = SteeringEffect()
    torch.manual_seed(99)
    logits = torch.randn(BATCH, SEQ_LEN, VOCAB)

    kl_same = effect.kl_divergence(logits, logits)
    assert math.isclose(kl_same, 0.0, abs_tol=1e-6), f"KL for identical logits = {kl_same}"

    logits2 = logits + torch.randn_like(logits) * 2.0
    kl_diff = effect.kl_divergence(logits, logits2)
    assert kl_diff >= 0.0, f"KL divergence should be non-negative, got {kl_diff}"


# ===========================================================================
# 13. SteeringEffect.top_k_shift: in [0,1], 1.0 for identical logits
# ===========================================================================

def test_top_k_shift_range_and_identity() -> None:
    effect = SteeringEffect()
    torch.manual_seed(7)
    logits = torch.randn(BATCH, SEQ_LEN, VOCAB)

    sim_same = effect.top_k_shift(logits, logits, k=5)
    assert math.isclose(sim_same, 1.0, abs_tol=1e-6), f"top_k_shift for same logits = {sim_same}"

    logits2 = -logits
    sim_diff = effect.top_k_shift(logits, logits2, k=5)
    assert 0.0 <= sim_diff <= 1.0, f"top_k_shift out of range: {sim_diff}"


# ===========================================================================
# 14. Hooks are removed after steered_forward (no lingering state)
# ===========================================================================

def test_hooks_removed_after_steered_forward(tiny_model: _TinyTransformer,
                                             concept_vec: Tensor,
                                             input_ids: Tensor) -> None:
    steerer = ActivationSteerer(tiny_model, layers=[0, 1])

    with torch.no_grad():
        _ = steerer.steered_forward(input_ids, concept_vec, alpha=5.0)

    assert len(steerer._hooks) == 0, "hooks not cleared after steered_forward"

    with torch.no_grad():
        logits_a = tiny_model(input_ids)
        logits_b = tiny_model(input_ids)

    assert torch.allclose(logits_a, logits_b), \
        "lingering hooks affected subsequent forward pass"


# ===========================================================================
# 15. Concept extraction from orthogonal pos/neg clusters:
#     concept vector is orthogonal to grand mean (captures contrast, not centre)
# ===========================================================================

def test_concept_from_orthogonal_clusters_orthogonal_to_mean() -> None:
    D = D_MODEL
    e0 = torch.zeros(D)
    e0[0] = 1.0
    e1 = torch.zeros(D)
    e1[1] = 1.0

    torch.manual_seed(123)
    N = 8
    noise_scale = 0.05
    pos = e0.unsqueeze(0).expand(N, D) + torch.randn(N, D) * noise_scale
    neg = e1.unsqueeze(0).expand(N, D) + torch.randn(N, D) * noise_scale

    extractor = ConceptVectorExtractor(method="mean_diff")
    v = extractor.extract(pos, neg)

    assert math.isclose(v.norm().item(), 1.0, abs_tol=1e-5), f"concept vector norm {v.norm()}"
    assert v.norm().item() > 0.5, "concept vector should not be near zero"

    grand_mean = torch.cat([pos, neg], dim=0).mean(dim=0)
    grand_mean_norm = F.normalize(grand_mean, dim=0)
    dot_with_mean = (v * grand_mean_norm).sum().abs().item()

    assert dot_with_mean < 0.2, \
        f"concept vector should be nearly orthogonal to grand mean, dot={dot_with_mean:.4f}"
