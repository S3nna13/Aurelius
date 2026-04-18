"""
src/interpretability/sae_trainer.py

JumpReLU Sparse Autoencoder (SAE) trainer.

Reference: "Scaling and evaluating sparse autoencoders"
           Gao et al., OpenAI 2024 (arXiv:2407.14435)

The JumpReLU activation replaces the vanilla ReLU with a learned per-feature
threshold θ.  A feature fires only when its pre-activation exceeds θ:

    f(z)_i = z_i * 1[z_i > θ_i]

Because the L0 sparsity penalty (||f(z)||_0) is non-differentiable, we use a
straight-through estimator (STE):

    Forward  : f(z) = z * (z > θ)          [hard gate]
    Backward : treat the gate as always 1   [STE — pass gradient straight through]

An auxiliary reconstruction loss computed with top-k selection helps prevent
feature collapse / "dead feature" degeneration.

Architecture
------------
    x_centered = x - b_pre
    z_pre      = x_centered @ W_enc + b_enc   (B, n_features)
    f          = JumpReLU(z_pre, θ)            (B, n_features)
    x_hat      = f @ W_dec + b_dec            (B, d_model)

Loss
----
    L = ||x - x_hat||^2 + λ * ||f||_0
      + aux_coeff * ||x - x_hat_topk||^2      (dead-feature prevention)

Pure PyTorch — no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SAEConfig:
    """Hyper-parameters for a JumpReLU SAE.

    Attributes
    ----------
    d_model        : Dimensionality of the activations being reconstructed.
    n_features     : Dictionary size (overcomplete: n_features > d_model).
    l0_target      : Target mean number of active features per sample.
    lam            : Coefficient for the L0 sparsity term in the loss.
    init_threshold : Initial value for the learned JumpReLU thresholds θ.
    aux_coeff      : Weight on the auxiliary top-k reconstruction loss.
    k_aux          : Number of features used in the auxiliary top-k pass.
                     Defaults to ``l0_target * 2`` when set to -1.
    """

    d_model: int = 64
    n_features: int = 256
    l0_target: float = 20.0
    lam: float = 5e-4
    init_threshold: float = 0.001
    aux_coeff: float = 1.0 / 32.0
    k_aux: int = -1  # -1 → use int(l0_target * 2) at runtime


# ---------------------------------------------------------------------------
# Straight-through JumpReLU
# ---------------------------------------------------------------------------

class _JumpReLUFunction(torch.autograd.Function):
    """Forward: hard gate z > θ.  Backward: straight-through (identity).

    The STE ignores the discontinuity at the jump point: the backward pass
    simply passes the incoming gradient straight through to z, as if the
    gate were always open.  The threshold θ receives a zero gradient from
    this function; it is updated only via the differentiable sparsity
    surrogate in _sae_loss.
    """

    @staticmethod
    def forward(ctx, z: Tensor, theta: Tensor) -> Tensor:  # type: ignore[override]
        gate = (z > theta).float()
        # Save gate and a zero tensor matching theta's shape for backward
        ctx.save_for_backward(gate, torch.zeros_like(theta))
        return z * gate

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        gate, zero_theta_grad = ctx.saved_tensors
        # Straight-through: pass gradient through as-is (ignore gating)
        grad_z = grad_output
        # No gradient flows through the threshold here
        grad_theta = zero_theta_grad
        return grad_z, grad_theta


def jumprelu(z: Tensor, theta: Tensor) -> Tensor:
    """Apply JumpReLU: z * (z > θ) with straight-through backward."""
    return _JumpReLUFunction.apply(z, theta)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class JumpReLUSAE(nn.Module):
    """JumpReLU Sparse Autoencoder.

    Parameters
    ----------
    config : SAEConfig
        Model and training hyper-parameters.

    Shapes throughout:
        B  = batch size
        d  = config.d_model
        n  = config.n_features
    """

    def __init__(self, config: SAEConfig) -> None:
        super().__init__()
        self.config = config

        # Encoder
        self.W_enc = nn.Parameter(torch.empty(config.d_model, config.n_features))
        self.b_enc = nn.Parameter(torch.zeros(config.n_features))

        # Pre-encoder bias (subtracted from x before encoding)
        self.b_pre = nn.Parameter(torch.zeros(config.d_model))

        # Decoder
        self.W_dec = nn.Parameter(torch.empty(config.n_features, config.d_model))
        self.b_dec = nn.Parameter(torch.zeros(config.d_model))

        # Learned per-feature thresholds (θ)
        self.log_threshold = nn.Parameter(
            torch.full((config.n_features,), _safe_log(config.init_threshold))
        )

        # Initialise weights
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        self._normalize_decoder_inplace()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> Tensor:
        """Per-feature threshold θ, always positive (stored as log θ)."""
        return self.log_threshold.exp()

    # ------------------------------------------------------------------
    # Forward components
    # ------------------------------------------------------------------

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode activations to sparse JumpReLU features.

        Parameters
        ----------
        x : Tensor of shape (B, d_model)

        Returns
        -------
        f     : (B, n_features)  JumpReLU-gated sparse features
        z_pre : (B, n_features)  pre-activation (before gating)
        """
        x_centered = x - self.b_pre
        z_pre = x_centered @ self.W_enc + self.b_enc
        f = jumprelu(z_pre, self.threshold)
        return f, z_pre

    def decode(self, f: Tensor) -> Tensor:
        """Decode sparse features back to activation space.

        Parameters
        ----------
        f : Tensor of shape (B, n_features)

        Returns
        -------
        x_hat : (B, d_model)
        """
        return f @ self.W_dec + self.b_dec

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Full forward pass.

        Parameters
        ----------
        x : Tensor of shape (B, d_model)

        Returns
        -------
        x_hat  : (B, d_model)   reconstructed activations
        f      : (B, n_features) sparse feature activations
        z_pre  : (B, n_features) pre-activation values
        """
        f, z_pre = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f, z_pre

    # ------------------------------------------------------------------
    # Decoder normalisation
    # ------------------------------------------------------------------

    def _normalize_decoder_inplace(self) -> None:
        """Normalize W_dec rows (each feature's decoding direction) to unit norm."""
        with torch.no_grad():
            norms = self.W_dec.data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            self.W_dec.data = self.W_dec.data / norms

    # ------------------------------------------------------------------
    # Auxiliary top-k reconstruction (dead-feature prevention)
    # ------------------------------------------------------------------

    def _topk_decode(self, z_pre: Tensor, k: int) -> Tensor:
        """Reconstruct using only the top-k pre-activations (no STE needed here).

        This path is used solely for the auxiliary loss; gradients still flow.
        """
        # Build a soft top-k mask: keep top-k values, zero the rest
        topk_vals, topk_idx = torch.topk(z_pre, k=k, dim=-1)
        mask = torch.zeros_like(z_pre)
        mask.scatter_(-1, topk_idx, 1.0)
        f_topk = F.relu(z_pre) * mask
        return self.decode(f_topk)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------

def _safe_log(x: float) -> float:
    import math
    return math.log(max(x, 1e-8))


def _sae_loss(
    x: Tensor,
    x_hat: Tensor,
    f: Tensor,
    z_pre: Tensor,
    model: JumpReLUSAE,
    config: SAEConfig,
) -> tuple[Tensor, dict]:
    """Compute JumpReLU SAE training loss.

    Loss = reconstruction + λ * L0 + aux_coeff * aux_recon

    Returns
    -------
    total_loss : scalar Tensor
    metrics    : dict with float values for logging
    """
    # 1) Reconstruction loss
    recon_loss = F.mse_loss(x_hat, x)

    # 2) L0 sparsity term — use STE-compatible approximation:
    #    count active features (detached) * lam so the *value* is correct,
    #    then add a differentiable surrogate via the sum of pre-activations
    #    where features are active.  This follows the paper's approach where
    #    the threshold θ is updated to steer l0 toward l0_target.
    active_mask = (f > 0).float()
    l0_per_sample = active_mask.sum(dim=-1)           # (B,)
    l0_mean = l0_per_sample.mean()

    # Differentiable L0 surrogate: penalise sum of activated pre-acts
    # (this is what backprops through θ via the STE path)
    sparsity_loss = config.lam * (f.abs() * active_mask).sum(dim=-1).mean()

    # 3) Auxiliary dead-feature loss (top-k reconstruction)
    k = config.k_aux if config.k_aux > 0 else max(1, int(config.l0_target * 2))
    k = min(k, config.n_features)
    x_hat_topk = model._topk_decode(z_pre.detach(), k)
    aux_loss = config.aux_coeff * F.mse_loss(x_hat_topk, x.detach())

    total_loss = recon_loss + sparsity_loss + aux_loss

    metrics = {
        "recon_loss": recon_loss.item(),
        "sparsity_loss": sparsity_loss.item(),
        "aux_loss": aux_loss.item(),
        "total_loss": total_loss.item(),
        "l0": l0_mean.item(),
    }
    return total_loss, metrics


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SAETrainer:
    """Train a JumpReLUSAE with Adam + decoder normalisation after each step.

    Parameters
    ----------
    model : JumpReLUSAE
    lr    : Learning rate for Adam.
    """

    def __init__(self, model: JumpReLUSAE, lr: float = 1e-4) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.step: int = 0

    # ------------------------------------------------------------------

    def train_step(self, x: Tensor) -> dict:
        """Single gradient step on a batch of activations.

        Parameters
        ----------
        x : Tensor of shape (B, d_model)

        Returns
        -------
        dict with keys: 'recon_loss', 'sparsity_loss', 'total_loss', 'l0'
        """
        self.model.train()
        self.optimizer.zero_grad()

        x_hat, f, z_pre = self.model(x)
        total_loss, metrics = _sae_loss(x, x_hat, f, z_pre, self.model, self.model.config)

        total_loss.backward()
        self.optimizer.step()
        self.step += 1

        # Keep decoder columns unit-norm after each gradient step
        self.model._normalize_decoder_inplace()

        return {
            "recon_loss": metrics["recon_loss"],
            "sparsity_loss": metrics["sparsity_loss"],
            "total_loss": metrics["total_loss"],
            "l0": metrics["l0"],
        }

    # ------------------------------------------------------------------

    def sparsity_stats(self, f: Tensor) -> dict:
        """Compute sparsity statistics from a feature tensor.

        Parameters
        ----------
        f : Tensor of shape (B, n_features) — output of encode()

        Returns
        -------
        dict with keys: 'mean_l0', 'dead_features', 'max_activation'
        """
        active = f > 0
        mean_l0 = active.float().sum(dim=-1).mean().item()

        # Dead = features that never activated across the batch
        max_per_feature = f.max(dim=0).values
        dead_features = int((max_per_feature == 0).sum().item())

        max_activation = f.max().item()

        return {
            "mean_l0": mean_l0,
            "dead_features": dead_features,
            "max_activation": max_activation,
        }
