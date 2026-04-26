"""Tests for best-of-N reranking with verifier-ranker scoring."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.inference.best_of_n_reranking import (
    BestOfNReranker,
    BestOfNRerankingConfig,
    best_of_n_reranking_loss,
    select_best_of_n,
)

TINY_CFG = BestOfNRerankingConfig(
    vocab_size=256,
    d_model=64,
    d_hidden=128,
    pad_token_id=0,
)


def make_module() -> BestOfNReranker:
    return BestOfNReranker(TINY_CFG)


def test_config_defaults():
    cfg = BestOfNRerankingConfig(vocab_size=256)
    assert cfg.d_model == 64
    assert cfg.d_hidden == 128
    assert cfg.pad_token_id == 0
    assert cfg.eps == pytest.approx(1e-8)


def test_select_best_of_n_matches_reference_formulation():
    Y = torch.tensor(
        [
            [[1, 2], [3, 4], [5, 6]],
            [[7, 8], [9, 10], [11, 12]],
        ],
        dtype=torch.long,
    )
    r = torch.tensor([[0.1, 0.9, 0.4], [1.2, -0.5, 1.1]], dtype=torch.float32)

    y_star, n_star = select_best_of_n(Y, r)

    ref_n_star = torch.tensor([1, 0], dtype=torch.long)
    ref_y_star = torch.stack([Y[0, 1], Y[1, 0]], dim=0)
    assert torch.equal(n_star, ref_n_star)
    assert torch.equal(y_star, ref_y_star)


def test_loss_matches_reference_cross_entropy():
    r = torch.tensor([[0.2, 0.9, -0.3], [1.0, 0.5, 0.8]], dtype=torch.float32)
    u = torch.tensor([[0.1, 0.7, 0.2], [0.4, 0.3, 0.9]], dtype=torch.float32)

    loss = best_of_n_reranking_loss(r, u)
    ref = F.cross_entropy(r, u.argmax(dim=1))
    assert torch.allclose(loss, ref, atol=1e-5)


def test_forward_shape_and_dtype_on_tiny_config():
    torch.manual_seed(0)
    module = make_module()
    x = torch.randint(1, TINY_CFG.vocab_size, (2, 5), dtype=torch.long)
    Y = torch.randint(1, TINY_CFG.vocab_size, (2, 4, 6), dtype=torch.long)

    r, y_star, n_star = module(x, Y)

    assert r.shape == (2, 4)
    assert r.dtype == torch.float32
    assert y_star.shape == (2, 6)
    assert y_star.dtype == torch.long
    assert n_star.shape == (2,)
    assert n_star.dtype == torch.long


def test_backward_produces_finite_gradients_for_all_params():
    torch.manual_seed(1)
    module = make_module()
    x = torch.randint(1, TINY_CFG.vocab_size, (2, 4), dtype=torch.long)
    Y = torch.randint(1, TINY_CFG.vocab_size, (2, 3, 5), dtype=torch.long)
    u = torch.tensor([[0.0, 1.0, 0.5], [0.2, 0.1, 0.8]], dtype=torch.float32)

    r = module.score(x, Y)
    loss = best_of_n_reranking_loss(r, u)
    loss.backward()

    for name, param in module.named_parameters():
        assert param.grad is not None, f"missing grad for {name}"
        assert torch.isfinite(param.grad).all(), f"non-finite grad for {name}"


def test_deterministic_under_manual_seed():
    torch.manual_seed(123)
    module_a = make_module()
    x_a = torch.randint(1, TINY_CFG.vocab_size, (2, 4), dtype=torch.long)
    Y_a = torch.randint(1, TINY_CFG.vocab_size, (2, 3, 5), dtype=torch.long)
    out_a = module_a(x_a, Y_a)

    torch.manual_seed(123)
    module_b = make_module()
    x_b = torch.randint(1, TINY_CFG.vocab_size, (2, 4), dtype=torch.long)
    Y_b = torch.randint(1, TINY_CFG.vocab_size, (2, 3, 5), dtype=torch.long)
    out_b = module_b(x_b, Y_b)

    assert torch.equal(x_a, x_b)
    assert torch.equal(Y_a, Y_b)
    for a, b in zip(out_a, out_b):
        assert torch.allclose(a, b)


def test_batch_one_and_seq_len_one():
    torch.manual_seed(2)
    module = make_module()
    x = torch.tensor([[7]], dtype=torch.long)
    Y = torch.tensor([[[3], [9], [1]]], dtype=torch.long)

    r, y_star, n_star = module(x, Y)

    assert r.shape == (1, 3)
    assert y_star.shape == (1, 1)
    assert n_star.shape == (1,)
    assert torch.isfinite(r).all()


def test_padded_tokens_are_ignored_with_explicit_masks():
    torch.manual_seed(3)
    module = make_module()
    x = torch.tensor([[5, 6, 0, 0]], dtype=torch.long)
    Y_a = torch.tensor([[[8, 9, 0, 0], [4, 5, 0, 0]]], dtype=torch.long)
    Y_b = torch.tensor([[[8, 9, 13, 14], [4, 5, 21, 22]]], dtype=torch.long)
    x_mask = torch.tensor([[True, True, False, False]])
    Y_mask = torch.tensor([[[True, True, False, False], [True, True, False, False]]])

    r_a = module.score(x, Y_a, x_mask=x_mask, Y_mask=Y_mask)
    r_b = module.score(x, Y_b, x_mask=x_mask, Y_mask=Y_mask)
    assert torch.allclose(r_a, r_b, atol=1e-5)


def test_implicit_pad_mask_matches_explicit_masks():
    torch.manual_seed(4)
    module = make_module()
    x = torch.tensor([[3, 4, 0, 0]], dtype=torch.long)
    Y = torch.tensor([[[1, 2, 0], [7, 0, 0]]], dtype=torch.long)
    x_mask = x.ne(TINY_CFG.pad_token_id)
    Y_mask = Y.ne(TINY_CFG.pad_token_id)

    r_implicit = module.score(x, Y)
    r_explicit = module.score(x, Y, x_mask=x_mask, Y_mask=Y_mask)
    assert torch.allclose(r_implicit, r_explicit, atol=1e-5)


def test_numerical_stability_on_extreme_inputs():
    torch.manual_seed(5)
    module = make_module()
    with torch.no_grad():
        module.embed.weight.fill_(1.0e3)
        for param in module.proj.parameters():
            param.fill_(1.0e2)

    x = torch.zeros((2, 4), dtype=torch.long)
    Y = torch.full((2, 3, 5), fill_value=255, dtype=torch.long)
    x_mask = torch.zeros_like(x, dtype=torch.bool)
    Y_mask = torch.ones_like(Y, dtype=torch.bool)

    r, y_star, n_star = module(x, Y, x_mask=x_mask, Y_mask=Y_mask)
    assert torch.isfinite(r).all()
    assert torch.isfinite(y_star.to(torch.float32)).all()
    assert torch.isfinite(n_star.to(torch.float32)).all()


def test_selected_candidate_matches_argmax_score_from_forward():
    torch.manual_seed(6)
    module = make_module()
    x = torch.randint(1, TINY_CFG.vocab_size, (3, 4), dtype=torch.long)
    Y = torch.randint(1, TINY_CFG.vocab_size, (3, 5, 6), dtype=torch.long)

    r, y_star, n_star = module(x, Y)
    ref_y_star, ref_n_star = select_best_of_n(Y, r)
    assert torch.equal(n_star, ref_n_star)
    assert torch.equal(y_star, ref_y_star)


def test_all_masked_candidates_remain_finite():
    torch.manual_seed(7)
    module = make_module()
    x = torch.tensor([[0, 0, 0]], dtype=torch.long)
    Y = torch.tensor([[[0, 0], [1, 2], [0, 0]]], dtype=torch.long)
    x_mask = torch.zeros_like(x, dtype=torch.bool)
    Y_mask = torch.tensor([[[False, False], [True, True], [False, False]]])

    r = module.score(x, Y, x_mask=x_mask, Y_mask=Y_mask)
    assert r.shape == (1, 3)
    assert torch.isfinite(r).all()
