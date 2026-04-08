"""Tests for H3-style long convolution block."""

import pytest
import torch

from src.model.h3 import H3Block


def test_h3_output_shape_matches_input():
    block = H3Block(d_model=16, kernel_size=3)
    x = torch.randn(2, 8, 16)
    y = block(x)
    assert y.shape == x.shape


def test_h3_backward_produces_gradients():
    block = H3Block(d_model=16, kernel_size=3)
    x = torch.randn(2, 8, 16, requires_grad=True)
    loss = block(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert block.out_proj.weight.grad is not None


def test_h3_respects_sequence_length_after_padding_trim():
    block = H3Block(d_model=8, kernel_size=5)
    x = torch.randn(1, 4, 8)
    y = block(x)
    assert y.shape[1] == 4


def test_h3_different_kernel_sizes_change_output():
    x = torch.randn(1, 6, 8)
    block_a = H3Block(d_model=8, kernel_size=3)
    block_b = H3Block(d_model=8, kernel_size=5)
    assert not torch.allclose(block_a(x), block_b(x))


def test_h3_rejects_bad_rank():
    block = H3Block(d_model=8)
    with pytest.raises(ValueError):
        block(torch.randn(2, 8))


def test_h3_rejects_invalid_hparams():
    with pytest.raises(ValueError):
        H3Block(d_model=0)
    with pytest.raises(ValueError):
        H3Block(d_model=8, kernel_size=0)


def test_h3_handles_single_token_sequences():
    block = H3Block(d_model=8, kernel_size=3)
    y = block(torch.randn(2, 1, 8))
    assert y.shape == (2, 1, 8)
