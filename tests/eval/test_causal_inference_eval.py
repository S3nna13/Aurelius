"""Tests for src/eval/causal_inference_eval.py.

Uses a self-contained tiny transformer (no external model imports) to keep
tests fast and dependency-free.  The model exposes attention weights via
an ``attn_weights`` attribute on each attention sub-module so that
AttentionPatternAnalyzer can capture them.

Configuration: vocab_size=16, d_model=16, n_layers=2, n_heads=4, B=2, T=8.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.causal_inference_eval import (
    AttentionPatternAnalyzer,
    CausalEffectEstimator,
    CausalEvalConfig,
    CounterfactualGenerator,
    LogitLensAnalyzer,
)

# ---------------------------------------------------------------------------
# Tiny transformer for tests
# ---------------------------------------------------------------------------

VOCAB = 16
D_MODEL = 16
N_LAYERS = 2
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 4
B = 2
T = 8


class TinyAttention(nn.Module):
    """Scaled dot-product attention that stores weights in self.attn_weights."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_weights: torch.Tensor | None = None  # [B, H, T, T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_local, T_local, D = x.shape
        H = self.n_heads
        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.split(D, dim=-1)  # each [B, T, D]

        # Reshape to [B, H, T, head_dim]
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B_local, T_local, H, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, T, T]

        # Causal mask
        mask = torch.triu(torch.ones(T_local, T_local, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)  # [B, H, T, T]
        self.attn_weights = weights.detach().clone()

        out = torch.matmul(weights, v)  # [B, H, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B_local, T_local, D)
        return self.proj(out)


class TinyBlock(nn.Module):
    """Single transformer block with pre-norm."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attn = TinyAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class TinyTransformer(nn.Module):
    """Minimal decoder-only transformer compatible with Causal Inference API."""

    def __init__(
        self,
        vocab_size: int = VOCAB,
        d_model: int = D_MODEL,
        n_layers: int = N_LAYERS,
        n_heads: int = N_HEADS,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TinyBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)  # [B, T, vocab_size]


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> TinyTransformer:
    torch.manual_seed(42)
    m = TinyTransformer(VOCAB, D_MODEL, N_LAYERS, N_HEADS)
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, VOCAB, (B, T))


@pytest.fixture(scope="module")
def cfg_gen(model) -> CounterfactualGenerator:
    return CounterfactualGenerator(model, VOCAB)


@pytest.fixture(scope="module")
def cfg_est(model) -> CausalEffectEstimator:
    return CausalEffectEstimator(model)


@pytest.fixture(scope="module")
def attn_analyzer(model) -> AttentionPatternAnalyzer:
    return AttentionPatternAnalyzer(model)


@pytest.fixture(scope="module")
def logit_lens(model) -> LogitLensAnalyzer:
    return LogitLensAnalyzer(model, VOCAB)


# ---------------------------------------------------------------------------
# 1. CounterfactualGenerator — intervene_token
# ---------------------------------------------------------------------------


def test_intervene_token_changes_only_target_position(cfg_gen, input_ids):
    """Replacing position 3 must change only column 3."""
    new_tok = (input_ids[0, 3].item() + 1) % VOCAB
    result = cfg_gen.intervene_token(input_ids, pos=3, new_token_id=new_tok)
    assert (result[:, :3] == input_ids[:, :3]).all()
    assert (result[:, 4:] == input_ids[:, 4:]).all()
    assert (result[:, 3] == new_tok).all()


def test_intervene_token_does_not_modify_original(cfg_gen, input_ids):
    """The original tensor must be unchanged (clone semantics)."""
    original_val = input_ids[0, 0].item()
    new_tok = (original_val + 1) % VOCAB
    _ = cfg_gen.intervene_token(input_ids, pos=0, new_token_id=new_tok)
    assert input_ids[0, 0].item() == original_val


# ---------------------------------------------------------------------------
# 2. CounterfactualGenerator — intervene_span
# ---------------------------------------------------------------------------


def test_intervene_span_zeros_the_span(cfg_gen, input_ids):
    """Tokens in [2, 5) must all become fill_id=0."""
    result = cfg_gen.intervene_span(input_ids, start=2, end=5, fill_id=0)
    assert (result[:, 2:5] == 0).all()


def test_intervene_span_leaves_outside_unchanged(cfg_gen, input_ids):
    """Positions outside the span are preserved."""
    result = cfg_gen.intervene_span(input_ids, start=2, end=5, fill_id=0)
    assert (result[:, :2] == input_ids[:, :2]).all()
    assert (result[:, 5:] == input_ids[:, 5:]).all()


# ---------------------------------------------------------------------------
# 3. CounterfactualGenerator — generate_minimal_pair
# ---------------------------------------------------------------------------


def test_generate_minimal_pair_differs_at_exactly_one_position(cfg_gen, input_ids):
    """The two sequences must differ at exactly position *pos*."""
    pos = 4
    token_a, token_b = 1, 7
    seq_a, seq_b = cfg_gen.generate_minimal_pair(input_ids, pos, token_a, token_b)
    diff_mask = seq_a != seq_b  # [B, T]
    for b in range(B):
        diff_positions = diff_mask[b].nonzero(as_tuple=False).squeeze(-1).tolist()
        assert diff_positions == [pos], f"Batch {b}: expected diff at [{pos}], got {diff_positions}"


def test_generate_minimal_pair_correct_tokens(cfg_gen, input_ids):
    """seq_a[:, pos] == token_a and seq_b[:, pos] == token_b."""
    pos = 2
    token_a, token_b = 3, 9
    seq_a, seq_b = cfg_gen.generate_minimal_pair(input_ids, pos, token_a, token_b)
    assert (seq_a[:, pos] == token_a).all()
    assert (seq_b[:, pos] == token_b).all()


# ---------------------------------------------------------------------------
# 4. CounterfactualGenerator — logit_diff
# ---------------------------------------------------------------------------


def test_logit_diff_output_shape(cfg_gen, input_ids):
    """logit_diff must return a [B] tensor."""
    seq_a = cfg_gen.intervene_token(input_ids, 0, 5)
    seq_b = cfg_gen.intervene_token(input_ids, 0, 10)
    diff = cfg_gen.logit_diff(seq_a, seq_b, target_pos=T - 1)
    assert diff.shape == (B,), f"Expected ({B},), got {diff.shape}"


def test_logit_diff_is_zero_for_identical_inputs(cfg_gen, input_ids):
    """Logit diff of identical sequences must be zero."""
    diff = cfg_gen.logit_diff(input_ids, input_ids.clone(), target_pos=0)
    assert torch.allclose(diff, torch.zeros(B)), f"Expected zeros, got {diff}"


# ---------------------------------------------------------------------------
# 5. CausalEffectEstimator — total_effect
# ---------------------------------------------------------------------------


def test_total_effect_returns_finite_float(cfg_est, input_ids):
    """total_effect must return a Python float and be finite."""
    effect = cfg_est.total_effect(
        input_ids,
        intervention_pos=1,
        token_a=2,
        token_b=5,
        outcome_pos=T - 1,
    )
    assert isinstance(effect, float)
    assert math.isfinite(effect)


def test_total_effect_is_zero_for_identical_tokens(cfg_est, input_ids):
    """Intervening with the same token (a == b) should give 0 total effect."""
    effect = cfg_est.total_effect(
        input_ids,
        intervention_pos=1,
        token_a=3,
        token_b=3,
        outcome_pos=T - 1,
    )
    assert math.isclose(effect, 0.0, abs_tol=1e-6), f"Expected 0.0, got {effect}"


# ---------------------------------------------------------------------------
# 6. CausalEffectEstimator — indirect_effect
# ---------------------------------------------------------------------------


def test_indirect_effect_equals_total_minus_direct(cfg_est, input_ids):
    """indirect_effect(total, direct) == total - direct."""
    total = cfg_est.total_effect(
        input_ids,
        intervention_pos=1,
        token_a=2,
        token_b=6,
        outcome_pos=T - 1,
    )
    direct = cfg_est.direct_effect(
        input_ids,
        intervention_pos=1,
        token_a=2,
        token_b=6,
        outcome_pos=T - 1,
        mediator_pos=4,
    )
    indirect = cfg_est.indirect_effect(total, direct)
    assert math.isclose(indirect, total - direct, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 7. AttentionPatternAnalyzer — extract_attention_weights
# ---------------------------------------------------------------------------


def test_extract_attention_weights_returns_list(attn_analyzer, input_ids):
    """extract_attention_weights should return a list."""
    weights = attn_analyzer.extract_attention_weights(input_ids)
    assert isinstance(weights, list)


def test_extract_attention_weights_length(attn_analyzer, input_ids):
    """Should return one tensor per layer."""
    weights = attn_analyzer.extract_attention_weights(input_ids)
    assert len(weights) == N_LAYERS


def test_extract_attention_weights_tensor_shape(attn_analyzer, input_ids):
    """Each weight tensor must be [B, n_heads, T, T]."""
    weights = attn_analyzer.extract_attention_weights(input_ids)
    for i, w in enumerate(weights):
        assert isinstance(w, torch.Tensor), f"Layer {i}: not a Tensor"
        assert w.shape == (B, N_HEADS, T, T), (
            f"Layer {i}: expected ({B}, {N_HEADS}, {T}, {T}), got {w.shape}"
        )


# ---------------------------------------------------------------------------
# 8. AttentionPatternAnalyzer — attention_rollout
# ---------------------------------------------------------------------------


def test_attention_rollout_output_shape(attn_analyzer, input_ids):
    """attention_rollout must return [B, T, T]."""
    weights = attn_analyzer.extract_attention_weights(input_ids)
    rollout = attn_analyzer.attention_rollout(weights)
    assert rollout.shape == (B, T, T), f"Expected ({B}, {T}, {T}), got {rollout.shape}"


def test_attention_rollout_rows_sum_to_one(attn_analyzer, input_ids):
    """Each row of the rollout matrix must sum to ~1 (row-stochastic)."""
    weights = attn_analyzer.extract_attention_weights(input_ids)
    rollout = attn_analyzer.attention_rollout(weights)
    row_sums = rollout.sum(dim=-1)  # [B, T]
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        f"Row sums deviate from 1: min={row_sums.min().item():.6f}, max={row_sums.max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# 9. AttentionPatternAnalyzer — attention_flow
# ---------------------------------------------------------------------------


def test_attention_flow_output_shape(attn_analyzer, input_ids):
    """attention_flow must return [B, T]."""
    weights = attn_analyzer.extract_attention_weights(input_ids)
    flow = attn_analyzer.attention_flow(weights, target_pos=T - 1)
    assert flow.shape == (B, T), f"Expected ({B}, {T}), got {flow.shape}"


def test_attention_flow_non_negative(attn_analyzer, input_ids):
    """All flow values must be non-negative (they come from softmax)."""
    weights = attn_analyzer.extract_attention_weights(input_ids)
    flow = attn_analyzer.attention_flow(weights, target_pos=0)
    assert (flow >= 0).all(), "Some flow values are negative"


# ---------------------------------------------------------------------------
# 10. LogitLensAnalyzer — extract_intermediate_logits
# ---------------------------------------------------------------------------


def test_extract_intermediate_logits_length(logit_lens, input_ids):
    """Should return one logit tensor per layer."""
    inter = logit_lens.extract_intermediate_logits(input_ids)
    assert len(inter) == N_LAYERS


def test_extract_intermediate_logits_shape(logit_lens, input_ids):
    """Each logit tensor must be [B, T, vocab_size]."""
    inter = logit_lens.extract_intermediate_logits(input_ids)
    for i, logits in enumerate(inter):
        assert logits.shape == (B, T, VOCAB), (
            f"Layer {i}: expected ({B}, {T}, {VOCAB}), got {logits.shape}"
        )


# ---------------------------------------------------------------------------
# 11. LogitLensAnalyzer — top_k_predictions
# ---------------------------------------------------------------------------


def test_top_k_predictions_returns_k_tokens_per_position(logit_lens, input_ids):
    """top_k_predictions must return k indices per (batch, position)."""
    k = 5
    inter = logit_lens.extract_intermediate_logits(input_ids)
    preds = logit_lens.top_k_predictions(inter[0], k=k)
    assert len(preds) == B * T
    for pos_preds in preds:
        assert len(pos_preds) == k, f"Expected {k} predictions, got {len(pos_preds)}"


def test_top_k_predictions_valid_indices(logit_lens, input_ids):
    """All returned indices must be valid vocab indices."""
    inter = logit_lens.extract_intermediate_logits(input_ids)
    preds = logit_lens.top_k_predictions(inter[0], k=3)
    for pos_preds in preds:
        for idx in pos_preds:
            assert 0 <= idx < VOCAB, f"Invalid token index: {idx}"


# ---------------------------------------------------------------------------
# 12. LogitLensAnalyzer — prediction_confidence_by_layer
# ---------------------------------------------------------------------------


def test_prediction_confidence_by_layer_returns_float_list(logit_lens, input_ids):
    """Should return a list of floats in [0, 1], one per layer."""
    inter = logit_lens.extract_intermediate_logits(input_ids)
    with torch.no_grad():
        final_logits = logit_lens.model(input_ids)  # [B, T, vocab]
    if isinstance(final_logits, (tuple, list)):
        candidate = final_logits[0]
        if candidate is None or (isinstance(candidate, torch.Tensor) and candidate.dim() == 0):
            final_logits = final_logits[1]
        else:
            final_logits = candidate
    final_tokens = final_logits.argmax(dim=-1)  # [B, T]

    confs = logit_lens.prediction_confidence_by_layer(inter, final_tokens)
    assert isinstance(confs, list)
    assert len(confs) == N_LAYERS
    for c in confs:
        assert isinstance(c, float), f"Expected float, got {type(c)}"
        assert 0.0 <= c <= 1.0, f"Confidence {c} out of [0, 1]"


# ---------------------------------------------------------------------------
# 13. CausalEvalConfig — defaults
# ---------------------------------------------------------------------------


def test_causal_eval_config_defaults():
    """CausalEvalConfig must have correct default values."""
    cfg = CausalEvalConfig()
    assert cfg.vocab_size == 64
    assert cfg.d_model == 32
    assert cfg.n_layers == 2
    assert cfg.n_attention_heads == 4
    assert cfg.logit_lens_k == 5


def test_causal_eval_config_is_mutable():
    """Dataclass fields should be overridable."""
    cfg = CausalEvalConfig(vocab_size=128, d_model=64)
    assert cfg.vocab_size == 128
    assert cfg.d_model == 64
    assert cfg.n_layers == 2
