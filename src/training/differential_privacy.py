"""
Differentially Private SGD (DP-SGD) with per-sample gradient clipping and Gaussian noise.

Implements:
- PrivacyAccountant: RDP / moments accountant for (epsilon, delta)-DP tracking
- PerSampleGradientClipper: clip per-sample gradients by L2 norm
- GaussianMechanism: add calibrated Gaussian noise
- DPSGDOptimizer: DP-SGD optimizer combining clipping + noise
- DPTrainer: high-level trainer with privacy budget tracking
- DPConfig: dataclass of hyperparameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# PrivacyAccountant
# ---------------------------------------------------------------------------


class PrivacyAccountant:
    """
    Tracks privacy budget using Rényi Differential Privacy (RDP) moments
    accountant, then converts to (epsilon, delta)-DP.

    Reference approach:
      - Per step RDP at order alpha: eps_rdp(alpha) = alpha / (2 * sigma^2)
        (Gaussian mechanism, subsampled with rate q via simplified bound)
      - Composition: eps_total_rdp = T * eps_rdp(alpha)
      - Conversion: eps = eps_total_rdp - log(delta) / (alpha - 1)
    """

    def __init__(
        self,
        noise_multiplier: float,
        max_grad_norm: float,
        delta: float = 1e-5,
    ) -> None:
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta
        # sigma = noise_multiplier (sensitivity is normalised to max_grad_norm
        # which cancels in the ratio)
        self._sigma = noise_multiplier

    def epsilon_from_steps(self, n_steps: int, sample_rate: float) -> float:
        """
        Closed-form (approximate) epsilon using the standard CLT-style bound:

            eps ≈ sqrt(2 * T * log(1/delta)) / (sigma * sqrt(T))
                = sqrt(2 * log(1/delta) / T) / sigma    [per-step noise scale]

        Simpler practical formula used in Abadi et al. (2016):

            eps ≈ (q * sqrt(T * log(1/delta))) / sigma

        where q = sample_rate, T = n_steps, sigma = noise_multiplier.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if not (0 < sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")
        q = sample_rate
        T = n_steps
        sigma = self._sigma
        delta = self.delta
        # Standard approximate DP-SGD bound
        eps = q * math.sqrt(T * 2.0 * math.log(1.25 / delta)) / sigma
        return float(eps)

    def moments_accountant_epsilon(
        self,
        n_steps: int,
        sample_rate: float,
        alpha: int = 10,
    ) -> float:
        """
        RDP composition at order *alpha*, then convert to (epsilon, delta)-DP.

        Per-step RDP bound for Gaussian mechanism with subsampling (q << 1):
            eps_rdp(alpha) ≈ alpha * q^2 / (2 * sigma^2)

        After T steps (composition):
            eps_rdp_total = T * eps_rdp(alpha)

        Conversion (Balle et al. 2020 / Mironov 2017):
            eps(delta) = eps_rdp_total + log(1/delta) / (alpha - 1)
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if not (0 < sample_rate <= 1):
            raise ValueError("sample_rate must be in (0, 1]")
        if alpha <= 1:
            raise ValueError("alpha must be > 1")
        q = sample_rate
        T = n_steps
        sigma = self._sigma
        delta = self.delta
        # Per-step RDP (Gaussian mechanism, amplified by subsampling)
        rdp_per_step = alpha * (q**2) / (2.0 * sigma**2)
        rdp_total = T * rdp_per_step
        # Convert RDP → (eps, delta)-DP
        eps = rdp_total + math.log(1.0 / delta) / (alpha - 1)
        return float(eps)

    def total_privacy_spent(
        self,
        n_steps: int,
        dataset_size: int,
        batch_size: int,
    ) -> dict[str, float]:
        """
        Return {"epsilon": float, "delta": float, "sample_rate": float}.
        Uses moments_accountant_epsilon for the epsilon estimate.
        """
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        sample_rate = batch_size / dataset_size
        eps = self.moments_accountant_epsilon(n_steps, sample_rate)
        return {
            "epsilon": eps,
            "delta": self.delta,
            "sample_rate": sample_rate,
        }


# ---------------------------------------------------------------------------
# PerSampleGradientClipper
# ---------------------------------------------------------------------------


class PerSampleGradientClipper:
    """
    Clips per-sample gradients by their L2 norm, then aggregates (sums) into
    param.grad buffers ready for noise addition.
    """

    def __init__(self, model: nn.Module, max_norm: float) -> None:
        if max_norm <= 0:
            raise ValueError("max_norm must be positive")
        self.model = model
        self.max_norm = max_norm
        # Build name → param mapping for convenience
        self._named_params: dict[str, nn.Parameter] = dict(model.named_parameters())

    def clip_gradients(self, per_sample_grads: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Clip per-sample gradients.

        Args:
            per_sample_grads: mapping name → Tensor of shape [B, *param_shape]

        Returns:
            Same mapping with clipped tensors of the same shape.
        """
        if not per_sample_grads:
            return {}

        # Determine batch size from the first entry
        first = next(iter(per_sample_grads.values()))
        B = first.shape[0]

        # Compute per-sample global L2 norm across all parameters
        # Shape: [B]
        squared_norms = torch.zeros(B, device=first.device, dtype=first.dtype)
        for g in per_sample_grads.values():
            # g: [B, *shape] → flatten spatial dims → [B, -1]
            flat = g.reshape(B, -1)
            squared_norms = squared_norms + flat.pow(2).sum(dim=1)
        norms = squared_norms.sqrt()  # [B]

        # Clip factor: max(1, norm / max_norm)^{-1}
        # Equivalent to min(1, max_norm / norm)
        clip_factor = torch.clamp(self.max_norm / (norms + 1e-6), max=1.0)  # [B]

        # Apply clipping
        clipped: dict[str, Tensor] = {}
        for name, g in per_sample_grads.items():
            # Broadcast clip_factor over parameter dimensions
            shape = (B,) + (1,) * (g.dim() - 1)
            clipped[name] = g * clip_factor.view(shape)

        return clipped

    def aggregate(self, clipped_grads: dict[str, Tensor]) -> None:
        """
        Sum clipped per-sample grads over the batch dimension and store in
        param.grad. This *replaces* any existing .grad on each parameter.

        Args:
            clipped_grads: mapping name → Tensor[B, *param_shape]
        """
        for name, g in clipped_grads.items():
            # Sum over batch dimension [B, *shape] → [*shape]
            summed = g.sum(dim=0)
            param = self._named_params[name]
            if param.grad is None:
                param.grad = summed.clone()
            else:
                param.grad.copy_(summed)


# ---------------------------------------------------------------------------
# GaussianMechanism
# ---------------------------------------------------------------------------


class GaussianMechanism:
    """
    Adds calibrated Gaussian noise to clipped, aggregated gradients to
    achieve (epsilon, delta)-DP via the Gaussian mechanism.
    """

    def __init__(self, noise_multiplier: float, max_grad_norm: float) -> None:
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive")
        if max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive")
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def add_noise(self, grad: Tensor) -> Tensor:
        """
        Return grad + N(0, (noise_multiplier * max_grad_norm)^2 * I).
        """
        std = self.noise_multiplier * self.max_grad_norm
        noise = torch.randn_like(grad) * std
        return grad + noise

    def sensitivity(self, max_grad_norm: float) -> float:
        """
        L2 sensitivity of the clipped gradient sum = max_grad_norm.
        """
        return float(max_grad_norm)


# ---------------------------------------------------------------------------
# DPSGDOptimizer
# ---------------------------------------------------------------------------


class DPSGDOptimizer:
    """
    DP-SGD optimizer that:
      1. Computes per-sample gradients
      2. Clips per-sample gradients
      3. Aggregates (sums) into param.grad
      4. Adds Gaussian noise to param.grad
      5. Calls standard SGD step
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        max_grad_norm: float,
        noise_multiplier: float,
    ) -> None:
        self.model = model
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier

        self.clipper = PerSampleGradientClipper(model, max_grad_norm)
        self.mechanism = GaussianMechanism(noise_multiplier, max_grad_norm)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Per-sample Jacobian computation
    # ------------------------------------------------------------------

    def compute_per_sample_grads(
        self,
        loss_per_sample: Tensor,
        params: list[nn.Parameter],
    ) -> dict[str, Tensor]:
        """
        Compute per-sample gradients using autograd.grad (one call per sample).

        Args:
            loss_per_sample: [B] individual losses (must have grad_fn)
            params: list of parameters to differentiate w.r.t.

        Returns:
            Dict mapping param name → Tensor[B, *param_shape]
        """
        B = loss_per_sample.shape[0]
        # Build name list aligned with params
        param_to_name: dict[int, str] = {id(p): n for n, p in self.model.named_parameters()}
        param_names = [param_to_name[id(p)] for p in params]

        per_sample_grads: dict[str, list[Tensor]] = {n: [] for n in param_names}

        for i in range(B):
            grads = torch.autograd.grad(
                loss_per_sample[i],
                params,
                retain_graph=(i < B - 1),
                create_graph=False,
                allow_unused=True,
            )
            for name, g in zip(param_names, grads):
                if g is None:
                    # Parameter not used; create zero grad
                    p = dict(self.model.named_parameters())[name]
                    per_sample_grads[name].append(torch.zeros_like(p))
                else:
                    per_sample_grads[name].append(g)

        # Stack into [B, *shape]
        return {n: torch.stack(gs, dim=0) for n, gs in per_sample_grads.items()}

    # ------------------------------------------------------------------
    # Full DP step
    # ------------------------------------------------------------------

    def step(self, per_sample_losses: Tensor) -> None:
        """
        Execute one DP-SGD step given per-sample losses.

        Args:
            per_sample_losses: [B] tensor of scalar losses, each with grad_fn
        """
        params = [p for p in self.model.parameters() if p.requires_grad]

        # 1. Compute per-sample gradients
        per_sample_grads = self.compute_per_sample_grads(per_sample_losses, params)

        # 2. Clip
        clipped = self.clipper.clip_gradients(per_sample_grads)

        # 3. Aggregate into param.grad
        self.clipper.aggregate(clipped)

        # 4. Add Gaussian noise to each param.grad
        for param in params:
            if param.grad is not None:
                param.grad = self.mechanism.add_noise(param.grad)

        # 5. SGD step
        self.optimizer.step()
        self.optimizer.zero_grad()


# ---------------------------------------------------------------------------
# DPTrainer
# ---------------------------------------------------------------------------


class DPTrainer:
    """
    High-level trainer wrapping DPSGDOptimizer with PrivacyAccountant.
    Expects a model with a forward method returning per-token logits.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float,
        max_grad_norm: float,
        noise_multiplier: float,
        delta: float = 1e-5,
    ) -> None:
        self.model = model
        self.optimizer = DPSGDOptimizer(model, lr, max_grad_norm, noise_multiplier)
        self.accountant = PrivacyAccountant(noise_multiplier, max_grad_norm, delta)
        self.n_steps: int = 0
        self._sample_rate: float = 1.0  # updated on first train_step call

    def train_step(
        self,
        input_ids: Tensor,
        labels: Tensor,
    ) -> tuple[float, float]:
        """
        Perform one DP training step.

        Args:
            input_ids: [B, T] token ids
            labels:    [B, T] target token ids (-100 positions are ignored)

        Returns:
            (mean_loss: float, epsilon: float)
        """
        B, T = input_ids.shape
        self.model.train()

        # Compute per-sample losses
        # Forward pass per sample to get independent computation graphs
        per_sample_losses = []
        for b in range(B):
            inp = input_ids[b : b + 1]  # [1, T]
            lbl = labels[b : b + 1]  # [1, T]

            logits = self.model(inp)  # [1, T, vocab]
            logits_flat = logits.view(-1, logits.size(-1))  # [T, vocab]
            lbl_flat = lbl.view(-1)  # [T]

            loss = nn.functional.cross_entropy(logits_flat, lbl_flat, ignore_index=-100)
            per_sample_losses.append(loss)

        per_sample_tensor = torch.stack(per_sample_losses)  # [B]
        mean_loss = per_sample_tensor.detach().mean().item()

        # DP-SGD step
        self.optimizer.step(per_sample_tensor)

        self.n_steps += 1

        # Compute epsilon (use sample_rate=1/B as default proxy)
        sample_rate = 1.0 / max(B, 1)
        eps = self.accountant.epsilon_from_steps(self.n_steps, sample_rate)

        return mean_loss, eps

    def privacy_report(self) -> dict[str, object]:
        """
        Return a dictionary summarising the current privacy budget.
        """
        if self.n_steps == 0:
            return {
                "n_steps": 0,
                "epsilon": 0.0,
                "delta": self.accountant.delta,
                "noise_multiplier": self.accountant.noise_multiplier,
                "max_grad_norm": self.accountant.max_grad_norm,
            }
        sample_rate = 1.0 / 4  # reasonable default; caller can override
        eps = self.accountant.epsilon_from_steps(self.n_steps, sample_rate)
        return {
            "n_steps": self.n_steps,
            "epsilon": eps,
            "delta": self.accountant.delta,
            "noise_multiplier": self.accountant.noise_multiplier,
            "max_grad_norm": self.accountant.max_grad_norm,
        }


# ---------------------------------------------------------------------------
# DPConfig
# ---------------------------------------------------------------------------


@dataclass
class DPConfig:
    """Hyperparameter dataclass for DP-SGD training."""

    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    delta: float = 1e-5
    lr: float = 1e-4
    target_epsilon: float = 8.0
