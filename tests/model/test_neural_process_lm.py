"""Tests for src/model/neural_process_lm.py."""

import pytest
import torch

from src.model.neural_process_lm import (
    ContextEncoder,
    LatentEncoder,
    NPDecoder,
    NeuralProcessLM,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

D_MODEL = 16
D_LATENT = 8
K = 4   # context size
B = 2   # batch size


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def context_inputs() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(K, D_MODEL)


@pytest.fixture()
def context_outputs() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(K, D_MODEL)


@pytest.fixture()
def query_embs() -> torch.Tensor:
    torch.manual_seed(2)
    return torch.randn(B, D_MODEL)


@pytest.fixture()
def query_targets() -> torch.Tensor:
    torch.manual_seed(3)
    return torch.randn(B, D_MODEL)


@pytest.fixture()
def ctx_encoder() -> ContextEncoder:
    return ContextEncoder(D_MODEL, D_MODEL, d_repr=32)


@pytest.fixture()
def lat_encoder() -> LatentEncoder:
    return LatentEncoder(32, D_LATENT)


@pytest.fixture()
def np_decoder() -> NPDecoder:
    return NPDecoder(D_MODEL, D_LATENT, D_MODEL)


@pytest.fixture()
def np_lm() -> NeuralProcessLM:
    return NeuralProcessLM(D_MODEL, D_LATENT)


# ---------------------------------------------------------------------------
# 1. ContextEncoder output shape is (d_repr,)
# ---------------------------------------------------------------------------

def test_context_encoder_output_shape(
    ctx_encoder: ContextEncoder,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
) -> None:
    r = ctx_encoder(context_inputs, context_outputs)
    assert r.shape == (32,), f"Expected (32,), got {r.shape}"


# ---------------------------------------------------------------------------
# 2. ContextEncoder output changes with different context
# ---------------------------------------------------------------------------

def test_context_encoder_varies_with_context(
    ctx_encoder: ContextEncoder,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
) -> None:
    r1 = ctx_encoder(context_inputs, context_outputs)
    # Use different context
    torch.manual_seed(99)
    other_inputs = torch.randn(K, D_MODEL)
    other_outputs = torch.randn(K, D_MODEL)
    r2 = ctx_encoder(other_inputs, other_outputs)
    assert not torch.allclose(r1, r2), "Different context should produce different representations"


# ---------------------------------------------------------------------------
# 3. LatentEncoder.forward returns (mu, log_sigma) tuple
# ---------------------------------------------------------------------------

def test_latent_encoder_returns_tuple(lat_encoder: LatentEncoder) -> None:
    r = torch.randn(32)
    result = lat_encoder(r)
    assert isinstance(result, tuple) and len(result) == 2, (
        "LatentEncoder.forward should return a tuple of length 2"
    )


# ---------------------------------------------------------------------------
# 4. mu shape is (d_latent,)
# ---------------------------------------------------------------------------

def test_latent_encoder_mu_shape(lat_encoder: LatentEncoder) -> None:
    r = torch.randn(32)
    mu, log_sigma = lat_encoder(r)
    assert mu.shape == (D_LATENT,), f"Expected ({D_LATENT},), got {mu.shape}"


# ---------------------------------------------------------------------------
# 5. sample output shape is (d_latent,)
# ---------------------------------------------------------------------------

def test_latent_encoder_sample_shape(lat_encoder: LatentEncoder) -> None:
    r = torch.randn(32)
    mu, log_sigma = lat_encoder(r)
    z = lat_encoder.sample(mu, log_sigma)
    assert z.shape == (D_LATENT,), f"Expected ({D_LATENT},), got {z.shape}"


# ---------------------------------------------------------------------------
# 6. Two samples differ (stochastic)
# ---------------------------------------------------------------------------

def test_latent_encoder_sample_stochastic(lat_encoder: LatentEncoder) -> None:
    r = torch.randn(32)
    mu, log_sigma = lat_encoder(r)
    z1 = lat_encoder.sample(mu, log_sigma)
    z2 = lat_encoder.sample(mu, log_sigma)
    assert not torch.allclose(z1, z2), "Two latent samples should differ (stochastic)"


# ---------------------------------------------------------------------------
# 7. NPDecoder output shape is (B, d_output)
# ---------------------------------------------------------------------------

def test_np_decoder_output_shape(
    np_decoder: NPDecoder, query_embs: torch.Tensor
) -> None:
    z = torch.randn(B, D_LATENT)
    out = np_decoder(query_embs, z)
    assert out.shape == (B, D_MODEL), f"Expected ({B}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 8. NPDecoder gradient flows
# ---------------------------------------------------------------------------

def test_np_decoder_gradient_flows(
    np_decoder: NPDecoder, query_embs: torch.Tensor
) -> None:
    z = torch.randn(B, D_LATENT)
    out = np_decoder(query_embs, z)
    loss = out.sum()
    loss.backward()
    # Check that at least one parameter has a gradient
    has_grad = any(p.grad is not None for p in np_decoder.parameters())
    assert has_grad, "NPDecoder should have gradients after backward pass"


# ---------------------------------------------------------------------------
# 9. NeuralProcessLM.encode_context returns (mu, log_sigma)
# ---------------------------------------------------------------------------

def test_np_lm_encode_context_returns_tuple(
    np_lm: NeuralProcessLM,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
) -> None:
    result = np_lm.encode_context(context_inputs, context_outputs)
    assert isinstance(result, tuple) and len(result) == 2, (
        "encode_context should return a tuple of (mu, log_sigma)"
    )
    mu, log_sigma = result
    assert mu.shape == (D_LATENT,)
    assert log_sigma.shape == (D_LATENT,)


# ---------------------------------------------------------------------------
# 10. forward output shape is (B, d_model)
# ---------------------------------------------------------------------------

def test_np_lm_forward_output_shape(
    np_lm: NeuralProcessLM,
    query_embs: torch.Tensor,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
) -> None:
    out = np_lm(query_embs, context_inputs, context_outputs)
    assert out.shape == (B, D_MODEL), f"Expected ({B}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. forward with n_samples=3 still returns (B, d_model)
# ---------------------------------------------------------------------------

def test_np_lm_forward_n_samples(
    np_lm: NeuralProcessLM,
    query_embs: torch.Tensor,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
) -> None:
    out = np_lm(query_embs, context_inputs, context_outputs, n_samples=3)
    assert out.shape == (B, D_MODEL), f"Expected ({B}, {D_MODEL}), got {out.shape}"


# ---------------------------------------------------------------------------
# 12. forward output is finite
# ---------------------------------------------------------------------------

def test_np_lm_forward_is_finite(
    np_lm: NeuralProcessLM,
    query_embs: torch.Tensor,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
) -> None:
    out = np_lm(query_embs, context_inputs, context_outputs)
    assert torch.isfinite(out).all(), "forward output should be finite"


# ---------------------------------------------------------------------------
# 13. elbo_loss returns expected keys
# ---------------------------------------------------------------------------

def test_np_lm_elbo_loss_keys(
    np_lm: NeuralProcessLM,
    query_embs: torch.Tensor,
    query_targets: torch.Tensor,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
) -> None:
    losses = np_lm.elbo_loss(query_embs, query_targets, context_inputs, context_outputs)
    assert set(losses.keys()) == {"recon_loss", "kl_loss", "total_loss"}, (
        f"Expected keys {{'recon_loss', 'kl_loss', 'total_loss'}}, got {set(losses.keys())}"
    )


# ---------------------------------------------------------------------------
# 14. elbo_loss total_loss is finite
# ---------------------------------------------------------------------------

def test_np_lm_elbo_loss_total_finite(
    np_lm: NeuralProcessLM,
    query_embs: torch.Tensor,
    query_targets: torch.Tensor,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
) -> None:
    losses = np_lm.elbo_loss(query_embs, query_targets, context_inputs, context_outputs)
    assert torch.isfinite(losses["total_loss"]), "total_loss should be finite"


# ---------------------------------------------------------------------------
# 15. KL loss is non-negative
# ---------------------------------------------------------------------------

def test_np_lm_kl_loss_non_negative(
    np_lm: NeuralProcessLM,
    query_embs: torch.Tensor,
    query_targets: torch.Tensor,
    context_inputs: torch.Tensor,
    context_outputs: torch.Tensor,
) -> None:
    losses = np_lm.elbo_loss(query_embs, query_targets, context_inputs, context_outputs)
    # KL divergence should be >= 0 (it's a sum, so the formula should yield >= 0)
    assert losses["kl_loss"].item() >= 0.0, (
        f"KL loss should be non-negative, got {losses['kl_loss'].item()}"
    )
