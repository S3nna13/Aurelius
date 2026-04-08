"""Tests for diffusion language-model head."""

import pytest
import torch

from src.model.diffusion_lm_head import DiffusionLMHead, sinusoidal_timestep_embedding


def test_sinusoidal_timestep_embedding_shape():
    emb = sinusoidal_timestep_embedding(torch.tensor([0, 1, 2]), dim=8)
    assert emb.shape == (3, 8)


def test_sinusoidal_timestep_embedding_rejects_bad_dim():
    with pytest.raises(ValueError):
        sinusoidal_timestep_embedding(torch.tensor([0]), dim=0)


def test_diffusion_lm_head_output_shape():
    head = DiffusionLMHead(d_model=16, vocab_size=32)
    hidden = torch.randn(2, 5, 16)
    timesteps = torch.tensor([1, 2])
    logits = head(hidden, timesteps)
    assert logits.shape == (2, 5, 32)


def test_diffusion_lm_head_timestep_conditioning_changes_output():
    head = DiffusionLMHead(d_model=16, vocab_size=32)
    hidden = torch.randn(2, 5, 16)
    logits_a = head(hidden, torch.tensor([1, 1]))
    logits_b = head(hidden, torch.tensor([2, 2]))
    assert not torch.allclose(logits_a, logits_b)


def test_diffusion_lm_head_backward_produces_gradients():
    head = DiffusionLMHead(d_model=16, vocab_size=32)
    hidden = torch.randn(2, 5, 16, requires_grad=True)
    logits = head(hidden, torch.tensor([1, 2]))
    loss = logits.pow(2).mean()
    loss.backward()
    assert hidden.grad is not None
    assert head.out_proj.weight.grad is not None


def test_diffusion_lm_head_rejects_bad_hidden_rank():
    head = DiffusionLMHead(d_model=16, vocab_size=32)
    with pytest.raises(ValueError):
        head(torch.randn(2, 16), torch.tensor([1, 2]))


def test_diffusion_lm_head_rejects_bad_timestep_shape():
    head = DiffusionLMHead(d_model=16, vocab_size=32)
    with pytest.raises(ValueError):
        head(torch.randn(2, 5, 16), torch.tensor([[1, 2]]))
