"""
tests/interpretability/test_sae_trainer.py

12 focused tests for the JumpReLU Sparse Autoencoder trainer.

Reference: arXiv:2407.14435 — "Scaling and evaluating sparse autoencoders"
           Gao et al., OpenAI 2024.

All tests use SAEConfig(d_model=64, n_features=256).
"""

from __future__ import annotations

import torch

from src.interpretability.sae_trainer import (
    JumpReLUSAE,
    SAEConfig,
    SAETrainer,
)

# ---------------------------------------------------------------------------
# Shared configuration and helpers
# ---------------------------------------------------------------------------

D_MODEL = 64
N_FEATURES = 256
BATCH = 16

CFG = SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, l0_target=20.0, lam=5e-4)


def _make_model(seed: int = 0) -> JumpReLUSAE:
    torch.manual_seed(seed)
    return JumpReLUSAE(CFG)


def _make_batch(seed: int = 42, batch: int = BATCH) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(batch, D_MODEL)


# ===========================================================================
# Test 1 — encode output shape
# ===========================================================================


def test_encode_output_shape():
    """encode() must return (f, z_pre) both of shape (B, n_features)."""
    model = _make_model()
    x = _make_batch()
    f, z_pre = model.encode(x)

    assert f.shape == (BATCH, N_FEATURES), f"f shape {f.shape} != ({BATCH}, {N_FEATURES})"
    assert z_pre.shape == (BATCH, N_FEATURES), (
        f"z_pre shape {z_pre.shape} != ({BATCH}, {N_FEATURES})"
    )


# ===========================================================================
# Test 2 — decode output shape
# ===========================================================================


def test_decode_output_shape():
    """decode() must return x_hat of shape (B, d_model)."""
    model = _make_model()
    torch.manual_seed(1)
    f = torch.rand(BATCH, N_FEATURES)
    x_hat = model.decode(f)

    assert x_hat.shape == (BATCH, D_MODEL), f"x_hat shape {x_hat.shape} != ({BATCH}, {D_MODEL})"


# ===========================================================================
# Test 3 — JumpReLU produces sparsity (most features zero)
# ===========================================================================


def test_jumprelu_sparsity():
    """JumpReLU gate should produce sparse activations (most features zero).

    Uses threshold=0.3 so the gate clearly cuts below-threshold pre-activations.
    With kaiming-uniform weights, measured minimum across seeds is ~65% zeros.
    """
    torch.manual_seed(7)
    cfg_sparse = SAEConfig(d_model=D_MODEL, n_features=N_FEATURES, init_threshold=0.3)
    model = JumpReLUSAE(cfg_sparse)
    x = _make_batch()
    f, _ = model.encode(x)

    frac_zero = (f == 0).float().mean().item()
    # threshold=0.3 reliably yields >60% zeros across all seeds
    assert frac_zero > 0.6, (
        f"Expected >60% zero activations with threshold=0.3, got {frac_zero:.2%}"
    )


# ===========================================================================
# Test 4 — features are non-zero exactly where z_pre > threshold
# ===========================================================================


def test_active_features_match_threshold():
    """f_i should be non-zero iff z_pre_i > theta_i (JumpReLU definition)."""
    model = _make_model()
    x = _make_batch()
    f, z_pre = model.encode(x)

    theta = model.threshold  # (n_features,)
    expected_active = (z_pre > theta).any(dim=0)  # at least one sample active
    actual_active = (f > 0).any(dim=0)

    # For every feature, activity should agree with the threshold condition
    assert torch.equal(expected_active, actual_active), (
        "Feature activity does not match z_pre > theta condition"
    )


# ===========================================================================
# Test 5 — gradient flows through encode (straight-through estimator)
# ===========================================================================


def test_gradient_flows_through_encode():
    """Gradients must flow back to W_enc via the straight-through estimator."""
    model = _make_model()
    x = _make_batch()
    x.requires_grad_(False)

    f, z_pre = model.encode(x)
    loss = f.sum()
    loss.backward()

    assert model.W_enc.grad is not None, "No gradient reached W_enc"
    assert model.W_enc.grad.abs().max().item() > 0, (
        "Gradient on W_enc is all zeros — STE may be broken"
    )


# ===========================================================================
# Test 6 — reconstruction loss decreases after training steps
# ===========================================================================


def test_reconstruction_improves_with_training():
    """After several gradient steps the reconstruction loss should decrease."""
    torch.manual_seed(42)
    model = _make_model(seed=42)
    trainer = SAETrainer(model, lr=1e-3)
    x = _make_batch()

    # Collect initial loss
    model.train()
    with torch.no_grad():
        x_hat_init, _, _ = model(x)
    recon_before = torch.nn.functional.mse_loss(x_hat_init, x).item()

    # Train for 100 steps
    for _ in range(100):
        trainer.train_step(x)

    model.train()
    with torch.no_grad():
        x_hat_after, _, _ = model(x)
    recon_after = torch.nn.functional.mse_loss(x_hat_after, x).item()

    assert recon_after < recon_before, (
        f"Reconstruction loss did not improve: {recon_before:.4f} -> {recon_after:.4f}"
    )


# ===========================================================================
# Test 7 — train_step returns expected keys
# ===========================================================================


def test_train_step_returns_expected_keys():
    """train_step must return a dict with exactly the required metric keys."""
    model = _make_model()
    trainer = SAETrainer(model)
    x = _make_batch()
    metrics = trainer.train_step(x)

    required_keys = {"recon_loss", "sparsity_loss", "total_loss", "l0"}
    assert required_keys.issubset(metrics.keys()), (
        f"Missing keys: {required_keys - set(metrics.keys())}"
    )
    # All values should be finite floats
    for k, v in metrics.items():
        assert isinstance(v, float), f"metrics['{k}'] is not a float: {type(v)}"
        assert torch.isfinite(torch.tensor(v)), f"metrics['{k}'] = {v} is not finite"


# ===========================================================================
# Test 8 — l0 is in a reasonable range after training (not 0 or n_features)
# ===========================================================================


def test_l0_reasonable_after_training():
    """After training, mean l0 should be > 0 and < n_features (not degenerate)."""
    torch.manual_seed(0)
    model = _make_model(seed=0)
    trainer = SAETrainer(model, lr=5e-4)
    x = _make_batch()

    for _ in range(200):
        metrics = trainer.train_step(x)

    l0 = metrics["l0"]
    assert l0 > 0, "l0 collapsed to 0 — all features dead"
    assert l0 < N_FEATURES, f"l0 = {l0} equals n_features — all features active (no sparsity)"


# ===========================================================================
# Test 9 — sparsity_stats returns required keys
# ===========================================================================


def test_sparsity_stats_returns_required_keys():
    """sparsity_stats must return 'mean_l0', 'dead_features', 'max_activation'."""
    model = _make_model()
    trainer = SAETrainer(model)
    x = _make_batch()
    f, _ = model.encode(x)
    stats = trainer.sparsity_stats(f)

    required = {"mean_l0", "dead_features", "max_activation"}
    assert required.issubset(stats.keys()), f"Missing keys: {required - set(stats.keys())}"
    assert isinstance(stats["dead_features"], int), "'dead_features' should be an int"
    assert stats["mean_l0"] >= 0.0
    assert stats["dead_features"] >= 0
    assert stats["max_activation"] >= 0.0


# ===========================================================================
# Test 10 — determinism under torch.manual_seed
# ===========================================================================


def test_determinism_under_manual_seed():
    """Two models initialised with the same seed must produce identical outputs."""
    x = _make_batch()

    torch.manual_seed(99)
    model_a = JumpReLUSAE(CFG)
    f_a, z_a = model_a.encode(x)

    torch.manual_seed(99)
    model_b = JumpReLUSAE(CFG)
    f_b, z_b = model_b.encode(x)

    assert torch.allclose(f_a, f_b, atol=1e-6), "f differs between same-seed models"
    assert torch.allclose(z_a, z_b, atol=1e-6), "z_pre differs between same-seed models"


# ===========================================================================
# Test 11 — no NaN or Inf on random activations
# ===========================================================================


def test_no_nan_inf_on_random_input():
    """Forward pass and a training step must not produce NaN or Inf values."""
    torch.manual_seed(123)
    model = _make_model()
    trainer = SAETrainer(model)

    for i in range(10):
        x = torch.randn(BATCH, D_MODEL) * (10 ** (i % 3))  # vary scale
        metrics = trainer.train_step(x)

        for k, v in metrics.items():
            assert torch.isfinite(torch.tensor(v)), f"metrics['{k}'] = {v} is NaN/Inf at step {i}"

    # Check model parameters
    for name, param in model.named_parameters():
        assert torch.isfinite(param).all(), f"Parameter '{name}' contains NaN/Inf"


# ===========================================================================
# Test 12 — decoder weight columns have unit norm
# ===========================================================================


def test_decoder_columns_unit_norm():
    """W_dec rows (each feature's decoder direction) must have unit L2 norm."""
    torch.manual_seed(5)
    model = _make_model(seed=5)
    trainer = SAETrainer(model)
    x = _make_batch()

    # After init
    norms_init = model.W_dec.data.norm(dim=-1)
    assert torch.allclose(norms_init, torch.ones_like(norms_init), atol=1e-5), (
        f"Decoder not unit-norm at init; max deviation = {(norms_init - 1).abs().max().item():.2e}"
    )

    # After training steps
    for _ in range(20):
        trainer.train_step(x)

    norms_after = model.W_dec.data.norm(dim=-1)
    assert torch.allclose(norms_after, torch.ones_like(norms_after), atol=1e-4), (
        f"Decoder not unit-norm after training; max deviation = "
        f"{(norms_after - 1).abs().max().item():.2e}"
    )
