"""Tests for attention-sink prefill."""

from __future__ import annotations

import pytest
import torch

from src.inference.attention_sink_prefill import (
    AttentionSinkPrefill,
    attention_sink_prefill,
    build_attention_sink_mask,
    repeat_kv_heads,
)

TINY = {
    "n_layers": 2,
    "d_model": 64,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 16,
    "d_ff": 128,
    "vocab_size": 256,
    "max_seq_len": 64,
}


def make_module(S: int = 2, W: int = 4) -> AttentionSinkPrefill:
    return AttentionSinkPrefill(
        d_model=TINY["d_model"],
        n_heads=TINY["n_heads"],
        n_kv_heads=TINY["n_kv_heads"],
        head_dim=TINY["head_dim"],
        S=S,
        W=W,
    )


def reference_attention_sink_prefill(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    S: int,
    W: int,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, d = Q.shape
    K = repeat_kv_heads(K, H)
    V = repeat_kv_heads(V, H)
    Y = torch.zeros_like(Q)
    P = torch.zeros(B, H, T, T, dtype=Q.dtype, device=Q.device)
    scale = d**-0.5

    if attention_mask is None:
        attention_mask = torch.ones(B, T, dtype=torch.bool, device=Q.device)
    else:
        attention_mask = attention_mask.to(dtype=torch.bool, device=Q.device)

    for b in range(B):
        for t in range(T):
            if not attention_mask[b, t]:
                continue
            keep = []
            for j in range(t + 1):
                if not attention_mask[b, j]:
                    continue
                if j < S or j >= max(0, t - W + 1):
                    keep.append(j)
            scores = torch.einsum("hd,shd->hs", Q[b, t] * scale, K[b, keep])
            weights = torch.softmax(scores, dim=-1)
            P[b, :, t, keep] = weights
            Y[b, t] = torch.einsum("hs,shd->hd", weights, V[b, keep])
    return Y, P


def test_build_attention_sink_mask_shape_and_dtype():
    mask = build_attention_sink_mask(seq_len=6, S=2, W=3)
    assert mask.shape == (6, 6)
    assert mask.dtype == torch.bool


def test_build_attention_sink_mask_matches_sink_and_window_pattern():
    mask = build_attention_sink_mask(seq_len=6, S=2, W=3)
    expected = torch.tensor(
        [
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [True, True, True, False, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
            [True, True, False, True, True, True],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)


def test_attention_sink_prefill_shape_and_dtype_tiny_config():
    Q = torch.randn(2, 5, 4, 16)
    K = torch.randn(2, 5, 2, 16)
    V = torch.randn(2, 5, 2, 16)
    Y, P = attention_sink_prefill(Q, K, V, S=2, W=3)
    assert Y.shape == (2, 5, 4, 16)
    assert P.shape == (2, 4, 5, 5)
    assert Y.dtype == Q.dtype
    assert P.dtype == Q.dtype


def test_module_forward_shape_and_dtype_tiny_config():
    module = make_module(S=2, W=4)
    X = torch.randn(2, 7, TINY["d_model"])
    Y, P = module(X, return_attention=True)
    assert Y.shape == (2, 7, TINY["d_model"])
    assert P.shape == (2, TINY["n_heads"], 7, 7)
    assert Y.dtype == X.dtype


def test_loss_backward_produces_finite_grads_on_all_trainable_params():
    torch.manual_seed(0)
    module = make_module(S=2, W=4)
    X = torch.randn(2, 6, TINY["d_model"], requires_grad=True)
    Y = module(X)
    loss = Y.square().mean()
    loss.backward()

    assert X.grad is not None
    assert torch.isfinite(X.grad).all()
    for param in module.parameters():
        assert param.grad is not None
        assert torch.isfinite(param.grad).all()


def test_determinism_under_torch_manual_seed():
    torch.manual_seed(123)
    module_a = make_module(S=2, W=4)
    X_a = torch.randn(2, 6, TINY["d_model"])
    Y_a, P_a = module_a(X_a, return_attention=True)

    torch.manual_seed(123)
    module_b = make_module(S=2, W=4)
    X_b = torch.randn(2, 6, TINY["d_model"])
    Y_b, P_b = module_b(X_b, return_attention=True)

    assert torch.equal(X_a, X_b)
    assert torch.allclose(Y_a, Y_b)
    assert torch.allclose(P_a, P_b)


def test_batch_one_seq_len_one_edge_case():
    module = make_module(S=4, W=4)
    X = torch.randn(1, 1, TINY["d_model"])
    Y, P = module(X, return_attention=True)
    assert Y.shape == (1, 1, TINY["d_model"])
    assert P.shape == (1, TINY["n_heads"], 1, 1)
    assert torch.allclose(P, torch.ones_like(P))


def test_padded_inputs_zero_out_invalid_positions_and_stay_finite():
    module = make_module(S=2, W=3)
    X = torch.randn(2, 5, TINY["d_model"])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])
    Y, P = module(X, attention_mask=attention_mask, return_attention=True)
    invalid = ~attention_mask.to(torch.bool)
    assert torch.isfinite(Y).all()
    assert torch.isfinite(P).all()
    assert torch.allclose(Y[invalid], torch.zeros_like(Y[invalid]))
    invalid_queries = invalid.unsqueeze(1).unsqueeze(-1)
    assert torch.allclose(P * invalid_queries, torch.zeros_like(P))


def test_extreme_inputs_do_not_produce_nan_or_inf():
    module = make_module(S=2, W=4)
    X = torch.full((2, 8, TINY["d_model"]), 1.0e4)
    Y, P = module(X, return_attention=True)
    assert torch.isfinite(Y).all()
    assert torch.isfinite(P).all()


def test_attention_sink_prefill_matches_reference_formulation():
    torch.manual_seed(7)
    Q = torch.randn(2, 6, 4, 16)
    K = torch.randn(2, 6, 2, 16)
    V = torch.randn(2, 6, 2, 16)
    actual_Y, actual_P = attention_sink_prefill(Q, K, V, S=2, W=3)
    ref_Y, ref_P = reference_attention_sink_prefill(Q, K, V, S=2, W=3)
    assert torch.allclose(actual_Y, ref_Y, atol=1e-5)
    assert torch.allclose(actual_P, ref_P, atol=1e-5)


def test_attention_sink_prefill_matches_reference_with_padding_mask():
    torch.manual_seed(9)
    Q = torch.randn(2, 6, 4, 16)
    K = torch.randn(2, 6, 2, 16)
    V = torch.randn(2, 6, 2, 16)
    attention_mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
    actual_Y, actual_P = attention_sink_prefill(Q, K, V, S=2, W=3, attention_mask=attention_mask)
    ref_Y, ref_P = reference_attention_sink_prefill(
        Q, K, V, S=2, W=3, attention_mask=attention_mask
    )
    assert torch.allclose(actual_Y, ref_Y, atol=1e-5)
    assert torch.allclose(actual_P, ref_P, atol=1e-5)


def test_attention_weights_sum_to_one_for_valid_queries():
    module = make_module(S=2, W=3)
    X = torch.randn(2, 5, TINY["d_model"])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    _, P = module(X, attention_mask=attention_mask, return_attention=True)
    row_sums = P.sum(dim=-1)
    expected = attention_mask.unsqueeze(1).to(dtype=row_sums.dtype)
    assert torch.allclose(row_sums, expected, atol=1e-6)


def test_invalid_attention_mask_shape_raises():
    module = make_module(S=2, W=3)
    X = torch.randn(2, 5, TINY["d_model"])
    with pytest.raises(ValueError, match="attention_mask"):
        module(X, attention_mask=torch.ones(2, 4))
