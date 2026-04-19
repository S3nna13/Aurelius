"""KTO v2 — refined Kahneman-Tversky Optimization.

Reference: Ethayarajh et al. 2024, arXiv:2402.01306 ("KTO: Model Alignment as
Prospect Theoretic Optimization").

V2 refinements over the base KTO loss in ``src/alignment/kto.py``:
    * Asymmetric ``beta`` for desirable vs. undesirable samples.
    * Reference-point ``z_ref`` estimated online via EMA of the batch
      mean log-ratio ``log pi_theta - log pi_ref``.
    * Supports batches that are all-chosen, all-rejected, or mixed
      (does not require paired examples).
    * Functional variant exposed for stateless use.

Per-sample loss:
    desirable   : -log sigma( beta_d * (log pi_theta - log pi_ref - z_ref) )
    undesirable : -log sigma( beta_u * (z_ref - log pi_theta + log pi_ref) )

Pure PyTorch; no foreign dependencies.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _validate_shapes(
    policy_logprobs: Tensor,
    ref_logprobs: Tensor,
    is_desirable: Tensor,
) -> None:
    if policy_logprobs.dim() != 1:
        raise ValueError(
            f"policy_logprobs must be 1-D [B], got shape {tuple(policy_logprobs.shape)}"
        )
    if ref_logprobs.shape != policy_logprobs.shape:
        raise ValueError(
            "ref_logprobs shape mismatch: "
            f"{tuple(ref_logprobs.shape)} vs {tuple(policy_logprobs.shape)}"
        )
    if is_desirable.shape != policy_logprobs.shape:
        raise ValueError(
            "is_desirable shape mismatch: "
            f"{tuple(is_desirable.shape)} vs {tuple(policy_logprobs.shape)}"
        )
    if is_desirable.dtype != torch.bool:
        raise ValueError(
            f"is_desirable must be a bool tensor, got dtype {is_desirable.dtype}"
        )


def kto_v2_loss_functional(
    policy_logprobs: Tensor,
    ref_logprobs: Tensor,
    is_desirable: Tensor,
    beta_d: float = 0.1,
    beta_u: float = 0.1,
    z_ref: float | Tensor = 0.0,
    lambda_d: float = 1.0,
    lambda_u: float = 1.0,
) -> Tensor:
    """Stateless KTO v2 loss.

    Args:
        policy_logprobs: 1-D tensor [B] of policy log-probs ``log pi_theta(y|x)``.
        ref_logprobs:    1-D tensor [B] of reference log-probs ``log pi_ref(y|x)``.
        is_desirable:    1-D bool tensor [B]; True -> chosen/desirable example.
        beta_d, beta_u:  positive scalar weights for desirable / undesirable.
        z_ref:           scalar reference point (float or 0-D tensor).
        lambda_d, lambda_u: scalar mixing weights (default 1.0).

    Returns:
        Scalar loss (mean over the batch).
    """
    _validate_shapes(policy_logprobs, ref_logprobs, is_desirable)

    if isinstance(z_ref, Tensor):
        z = z_ref.detach().to(policy_logprobs.dtype).to(policy_logprobs.device)
    else:
        z = torch.tensor(
            float(z_ref), dtype=policy_logprobs.dtype, device=policy_logprobs.device
        )

    log_ratio = policy_logprobs - ref_logprobs  # [B]

    # per-sample "value" under prospect theory: positive when argument is favourable.
    desirable_arg = beta_d * (log_ratio - z)
    undesirable_arg = beta_u * (z - log_ratio)

    # -log sigma(x) == softplus(-x), numerically stable.
    desirable_loss = F.softplus(-desirable_arg)
    undesirable_loss = F.softplus(-undesirable_arg)

    mask_d = is_desirable.to(policy_logprobs.dtype)
    mask_u = 1.0 - mask_d

    per_sample = lambda_d * mask_d * desirable_loss + lambda_u * mask_u * undesirable_loss
    return per_sample.mean()


class KTOv2Loss(nn.Module):
    """Stateful KTO v2 loss module with online reference-point EMA.

    The reference point ``z_ref`` is maintained as a non-persistent buffer and
    updated in-place during ``forward`` using an EMA over the *batch mean* of
    the log-ratio ``log pi_theta - log pi_ref``. The EMA update is fully
    detached, so no gradients flow back through ``z_ref``.
    """

    def __init__(
        self,
        beta_desirable: float = 0.1,
        beta_undesirable: float = 0.1,
        z_ref_ema: float = 0.9,
        lambda_d: float = 1.0,
        lambda_u: float = 1.0,
    ) -> None:
        super().__init__()
        if beta_desirable <= 0.0:
            raise ValueError("beta_desirable must be positive")
        if beta_undesirable <= 0.0:
            raise ValueError("beta_undesirable must be positive")
        if not 0.0 <= z_ref_ema <= 1.0:
            raise ValueError("z_ref_ema must lie in [0, 1]")
        self.beta_desirable = float(beta_desirable)
        self.beta_undesirable = float(beta_undesirable)
        self.z_ref_ema = float(z_ref_ema)
        self.lambda_d = float(lambda_d)
        self.lambda_u = float(lambda_u)

        # Running reference point. Scalar buffer.
        self.register_buffer("z_ref", torch.zeros((), dtype=torch.float32))
        # Tracks whether the buffer has ever been updated (for cold-start init).
        self.register_buffer(
            "_z_ref_initialized", torch.zeros((), dtype=torch.bool)
        )

    def reset_z_ref(self) -> None:
        """Reset the running reference point to zero (uninitialized)."""
        with torch.no_grad():
            self.z_ref.zero_()
            self._z_ref_initialized.zero_()

    @torch.no_grad()
    def _update_z_ref(self, batch_mean_log_ratio: Tensor) -> None:
        val = batch_mean_log_ratio.detach().to(self.z_ref.dtype).to(self.z_ref.device)
        if not bool(self._z_ref_initialized):
            self.z_ref.copy_(val)
            self._z_ref_initialized.fill_(True)
        else:
            new = self.z_ref_ema * self.z_ref + (1.0 - self.z_ref_ema) * val
            self.z_ref.copy_(new)

    def forward(
        self,
        policy_logprobs: Tensor,
        ref_logprobs: Tensor,
        is_desirable: Tensor,
    ) -> Tensor:
        _validate_shapes(policy_logprobs, ref_logprobs, is_desirable)

        # NaN-safe log ratio for EMA update: skip update if any non-finite.
        log_ratio = policy_logprobs - ref_logprobs
        finite = torch.isfinite(log_ratio)
        if bool(finite.all()):
            batch_mean = log_ratio.detach().mean()
            self._update_z_ref(batch_mean)

        # Use the *current* z_ref (already updated) as the detached scalar.
        z_scalar = self.z_ref.detach()

        return kto_v2_loss_functional(
            policy_logprobs=policy_logprobs,
            ref_logprobs=ref_logprobs,
            is_desirable=is_desirable,
            beta_d=self.beta_desirable,
            beta_u=self.beta_undesirable,
            z_ref=z_scalar,
            lambda_d=self.lambda_d,
            lambda_u=self.lambda_u,
        )

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"beta_desirable={self.beta_desirable}, "
            f"beta_undesirable={self.beta_undesirable}, "
            f"z_ref_ema={self.z_ref_ema}, "
            f"lambda_d={self.lambda_d}, lambda_u={self.lambda_u}"
        )


__all__ = ["KTOv2Loss", "kto_v2_loss_functional"]
