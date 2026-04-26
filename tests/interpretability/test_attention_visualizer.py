"""
Tests for src/interpretability/attention_visualizer.py

Covers: AttentionVisualizer construction, compute_attention_map,
highlight_top_tokens, rollout_attention, normalize_for_display,
shape/dtype validation, and ATTENTION_VISUALIZER_REGISTRY.
"""

from __future__ import annotations

import pytest
import torch

from src.interpretability.attention_visualizer import (
    ATTENTION_VISUALIZER_REGISTRY,
    AttentionVisualizer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

H = 4  # number of heads
T = 6  # sequence length


def _make_visualizer(max_seq_len: int = 128) -> AttentionVisualizer:
    return AttentionVisualizer(max_seq_len=max_seq_len)


def _random_attn_weights(n_heads: int = H, seq_len: int = T) -> torch.Tensor:
    """Random softmax-normalized attention weights, shape (H, T, T)."""
    return torch.softmax(torch.randn(n_heads, seq_len, seq_len), dim=-1)


def _uniform_attn_weights(n_heads: int = H, seq_len: int = T) -> torch.Tensor:
    """Uniform attention weights, shape (H, T, T)."""
    return torch.full((n_heads, seq_len, seq_len), 1.0 / seq_len)


def _identity_attn_map(seq_len: int = T) -> torch.Tensor:
    """Identity attention map, shape (T, T)."""
    return torch.eye(seq_len, dtype=torch.float32)


def _random_attn_map(seq_len: int = T) -> torch.Tensor:
    """Random valid attention map, shape (T, T)."""
    return torch.softmax(torch.randn(seq_len, seq_len), dim=-1)


# ---------------------------------------------------------------------------
# 1. Constructor
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_default_max_seq_len(self):
        v = AttentionVisualizer()
        assert v.max_seq_len == 128

    def test_custom_max_seq_len(self):
        v = AttentionVisualizer(max_seq_len=256)
        assert v.max_seq_len == 256

    def test_invalid_max_seq_len_type(self):
        with pytest.raises(TypeError):
            AttentionVisualizer(max_seq_len="128")  # type: ignore[arg-type]

    def test_non_positive_max_seq_len(self):
        with pytest.raises(ValueError):
            AttentionVisualizer(max_seq_len=0)

    def test_negative_max_seq_len(self):
        with pytest.raises(ValueError):
            AttentionVisualizer(max_seq_len=-1)


# ---------------------------------------------------------------------------
# 2. compute_attention_map
# ---------------------------------------------------------------------------


class TestComputeAttentionMap:
    def test_output_shape(self):
        v = _make_visualizer()
        attn = _random_attn_weights()
        result = v.compute_attention_map(attn)
        assert result.shape == (T, T)

    def test_uniform_weights_mean(self):
        v = _make_visualizer()
        attn = _uniform_attn_weights()
        result = v.compute_attention_map(attn)
        expected = torch.full((T, T), 1.0 / T)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_identity_heads_mean(self):
        v = _make_visualizer()
        eye = torch.eye(T).unsqueeze(0).expand(H, T, T)
        result = v.compute_attention_map(eye)
        expected = torch.eye(T)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_invalid_not_tensor(self):
        v = _make_visualizer()
        with pytest.raises(TypeError):
            v.compute_attention_map([[0.5, 0.5], [0.5, 0.5]])  # type: ignore[arg-type]

    def test_invalid_wrong_dim(self):
        v = _make_visualizer()
        with pytest.raises(ValueError):
            v.compute_attention_map(torch.randn(T, T))

    def test_invalid_not_square(self):
        v = _make_visualizer()
        with pytest.raises(ValueError):
            v.compute_attention_map(torch.randn(H, T, T + 1))

    def test_invalid_integer_dtype(self):
        v = _make_visualizer()
        with pytest.raises(TypeError):
            v.compute_attention_map(torch.randint(0, 10, (H, T, T)))

    def test_exceeds_max_seq_len(self):
        v = AttentionVisualizer(max_seq_len=4)
        with pytest.raises(ValueError):
            v.compute_attention_map(torch.randn(H, T, T))


# ---------------------------------------------------------------------------
# 3. highlight_top_tokens
# ---------------------------------------------------------------------------


class TestHighlightTopTokens:
    def test_output_type(self):
        v = _make_visualizer()
        attn_map = _random_attn_map()
        token_ids = list(range(T))
        result = v.highlight_top_tokens(attn_map, token_ids, top_k=3)
        assert isinstance(result, dict)

    def test_keys_match_positions(self):
        v = _make_visualizer()
        attn_map = _random_attn_map()
        token_ids = list(range(T))
        result = v.highlight_top_tokens(attn_map, token_ids, top_k=3)
        assert set(result.keys()) == set(range(T))

    def test_top_k_length(self):
        v = _make_visualizer()
        attn_map = _random_attn_map()
        token_ids = list(range(T))
        result = v.highlight_top_tokens(attn_map, token_ids, top_k=3)
        for pos, top_list in result.items():
            assert len(top_list) == 3

    def test_top_k_sorted_descending(self):
        v = _make_visualizer()
        attn_map = _random_attn_map()
        token_ids = list(range(T))
        result = v.highlight_top_tokens(attn_map, token_ids, top_k=3)
        for pos, top_list in result.items():
            scores = [score for _, score in top_list]
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1] - 1e-6

    def test_top_k_values_are_scores(self):
        v = _make_visualizer()
        attn_map = _identity_attn_map()
        token_ids = list(range(T))
        result = v.highlight_top_tokens(attn_map, token_ids, top_k=1)
        # Identity: position i attends only to itself with score 1.0
        for i in range(T):
            assert result[i] == [(i, 1.0)]

    def test_top_k_equals_seq_len(self):
        v = _make_visualizer()
        attn_map = _random_attn_map()
        token_ids = list(range(T))
        result = v.highlight_top_tokens(attn_map, token_ids, top_k=T)
        for pos, top_list in result.items():
            assert len(top_list) == T

    def test_invalid_attn_map_not_tensor(self):
        v = _make_visualizer()
        with pytest.raises(TypeError):
            v.highlight_top_tokens([[0.5, 0.5], [0.5, 0.5]], [0, 1], top_k=1)  # type: ignore[arg-type]

    def test_invalid_token_ids_type(self):
        v = _make_visualizer()
        with pytest.raises(TypeError):
            v.highlight_top_tokens(_identity_attn_map(), (0, 1, 2, 3, 4, 5), top_k=1)  # type: ignore[arg-type]

    def test_invalid_token_ids_length(self):
        v = _make_visualizer()
        with pytest.raises(ValueError):
            v.highlight_top_tokens(_identity_attn_map(), [0, 1, 2], top_k=1)

    def test_invalid_top_k_type(self):
        v = _make_visualizer()
        with pytest.raises(TypeError):
            v.highlight_top_tokens(_identity_attn_map(), list(range(T)), top_k="3")  # type: ignore[arg-type]

    def test_invalid_top_k_non_positive(self):
        v = _make_visualizer()
        with pytest.raises(ValueError):
            v.highlight_top_tokens(_identity_attn_map(), list(range(T)), top_k=0)

    def test_invalid_top_k_too_large(self):
        v = _make_visualizer()
        with pytest.raises(ValueError):
            v.highlight_top_tokens(_identity_attn_map(), list(range(T)), top_k=T + 1)

    def test_exceeds_max_seq_len(self):
        v = AttentionVisualizer(max_seq_len=4)
        with pytest.raises(ValueError):
            v.highlight_top_tokens(_identity_attn_map(), list(range(T)), top_k=1)


# ---------------------------------------------------------------------------
# 4. rollout_attention
# ---------------------------------------------------------------------------


class TestRolloutAttention:
    def test_single_layer_shape(self):
        v = _make_visualizer()
        attn = _random_attn_map()
        result = v.rollout_attention([attn])
        assert result.shape == (T, T)

    def test_single_layer_equals_input(self):
        v = _make_visualizer()
        attn = _random_attn_map()
        result = v.rollout_attention([attn])
        assert torch.allclose(result, attn, atol=1e-6)

    def test_multi_layer_shape(self):
        v = _make_visualizer()
        maps = [_random_attn_map() for _ in range(3)]
        result = v.rollout_attention(maps)
        assert result.shape == (T, T)

    def test_identity_rollout_is_identity(self):
        v = _make_visualizer()
        eye = _identity_attn_map()
        result = v.rollout_attention([eye, eye, eye])
        assert torch.allclose(result, eye, atol=1e-6)

    def test_two_layer_matrix_multiplication(self):
        v = _make_visualizer()
        a1 = torch.tensor([[0.5, 0.5], [0.0, 1.0]], dtype=torch.float32)
        a2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        result = v.rollout_attention([a1, a2])
        expected = a2 @ a1
        assert torch.allclose(result, expected, atol=1e-6)

    def test_three_layer_matrix_multiplication(self):
        v = _make_visualizer()
        a1 = torch.tensor([[0.5, 0.5], [0.0, 1.0]], dtype=torch.float32)
        a2 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        a3 = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
        result = v.rollout_attention([a1, a2, a3])
        expected = a3 @ a2 @ a1
        assert torch.allclose(result, expected, atol=1e-6)

    def test_invalid_not_list(self):
        v = _make_visualizer()
        with pytest.raises(TypeError):
            v.rollout_attention(_identity_attn_map())  # type: ignore[arg-type]

    def test_invalid_empty_list(self):
        v = _make_visualizer()
        with pytest.raises(ValueError):
            v.rollout_attention([])

    def test_invalid_mismatched_shapes(self):
        v = _make_visualizer()
        with pytest.raises(ValueError):
            v.rollout_attention(
                [
                    torch.randn(3, 3),
                    torch.randn(4, 4),
                ]
            )

    def test_invalid_not_tensor_in_list(self):
        v = _make_visualizer()
        with pytest.raises(TypeError):
            v.rollout_attention([_identity_attn_map(), [[0.5, 0.5], [0.5, 0.5]]])  # type: ignore[list-item]

    def test_exceeds_max_seq_len(self):
        v = AttentionVisualizer(max_seq_len=4)
        with pytest.raises(ValueError):
            v.rollout_attention([_identity_attn_map()])


# ---------------------------------------------------------------------------
# 5. normalize_for_display
# ---------------------------------------------------------------------------


class TestNormalizeForDisplay:
    def test_output_shape(self):
        v = _make_visualizer()
        attn = _random_attn_map()
        result = v.normalize_for_display(attn)
        assert result.shape == (T, T)

    def test_range_zero_to_one(self):
        v = _make_visualizer()
        attn = _random_attn_map()
        result = v.normalize_for_display(attn)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_min_is_zero(self):
        v = _make_visualizer()
        attn = torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float32)
        result = v.normalize_for_display(attn)
        assert abs(result.min().item()) < 1e-6

    def test_max_is_one(self):
        v = _make_visualizer()
        attn = torch.tensor([[0.2, 0.4], [0.6, 0.8]], dtype=torch.float32)
        result = v.normalize_for_display(attn)
        assert abs(result.max().item() - 1.0) < 1e-6

    def test_constant_map_returns_zeros(self):
        v = _make_visualizer()
        attn = torch.ones(T, T) * 0.5
        result = v.normalize_for_display(attn)
        assert torch.allclose(result, torch.zeros(T, T), atol=1e-6)

    def test_invalid_not_tensor(self):
        v = _make_visualizer()
        with pytest.raises(TypeError):
            v.normalize_for_display([[0.5, 0.5], [0.5, 0.5]])  # type: ignore[arg-type]

    def test_invalid_not_square(self):
        v = _make_visualizer()
        with pytest.raises(ValueError):
            v.normalize_for_display(torch.randn(T, T + 1))

    def test_invalid_integer_dtype(self):
        v = _make_visualizer()
        with pytest.raises(TypeError):
            v.normalize_for_display(torch.randint(0, 10, (T, T)))

    def test_exceeds_max_seq_len(self):
        v = AttentionVisualizer(max_seq_len=4)
        with pytest.raises(ValueError):
            v.normalize_for_display(_identity_attn_map())


# ---------------------------------------------------------------------------
# 6. ATTENTION_VISUALIZER_REGISTRY
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registry_has_default(self):
        assert "default" in ATTENTION_VISUALIZER_REGISTRY

    def test_default_is_class(self):
        assert ATTENTION_VISUALIZER_REGISTRY["default"] is AttentionVisualizer

    def test_default_is_callable(self):
        cls = ATTENTION_VISUALIZER_REGISTRY["default"]
        instance = cls()
        assert isinstance(instance, AttentionVisualizer)
