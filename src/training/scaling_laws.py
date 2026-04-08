"""
Neural scaling law utilities for the Aurelius LLM project.

Implements Chinchilla scaling laws (Hoffmann et al. 2022) and related utilities
for predicting optimal model size and training tokens given a compute budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ScalingConfig:
    """Chinchilla scaling law parameters (Hoffmann et al. 2022 fitted constants)."""

    E: float = 1.61      # irreducible entropy
    A: float = 406.4     # parameter scaling coefficient
    B: float = 410.7     # data scaling coefficient
    alpha: float = 0.34  # parameter scaling exponent
    beta: float = 0.28   # data scaling exponent


class ChinchillaPredictor:
    """
    Predict optimal training configurations using Chinchilla scaling laws.

    Args:
        config: ScalingConfig (defaults to Chinchilla fitted constants)
    """

    def __init__(self, config: ScalingConfig | None = None):
        self.config = config or ScalingConfig()

    def loss(self, n_params: float, n_tokens: float) -> float:
        """
        Predict cross-entropy loss given model size and training tokens.

        L(N, D) = E + A/N^alpha + B/D^beta
        """
        cfg = self.config
        return cfg.E + cfg.A / math.pow(n_params, cfg.alpha) + cfg.B / math.pow(n_tokens, cfg.beta)

    def optimal_allocation(self, flop_budget: float) -> dict:
        """
        Find optimal (N*, D*) for a given FLOPs budget C.

        C ≈ 6 * N * D  (standard transformer FLOPs estimate)

        Derivation via Lagrange multipliers under C = 6*N*D:
            N* = (C / 6) * (B * beta) / (A * alpha + B * beta) ... but the
        simplest closed-form from equal-scaling rule N* ≈ D*/20 gives:
            N* = sqrt(C / 120),  D* = C / (6 * N*)

        Returns: {'n_params': float, 'n_tokens': float,
                  'predicted_loss': float, 'flops': float}
        """
        cfg = self.config
        # Analytically optimal allocation under C = 6*N*D
        # From dL/dN = 0 and dL/dD = 0 subject to 6ND = C:
        #   A*alpha / N^(alpha+1) = lambda * 6*D  and  B*beta / D^(beta+1) = lambda * 6*N
        # Dividing:  (A*alpha/N^(alpha+1)) / (B*beta/D^(beta+1)) = D/N
        # => (A*alpha)/(B*beta) * (D/N)^(beta) * (N/D)^alpha = D/N ... rearranges to:
        #   a_ratio = (A*alpha) / (B*beta)
        #   N* = (C/6)^(beta/(alpha+beta)) * a_ratio^(-beta/(alpha+beta)) / something
        # Simplified exact form used in Hoffmann et al. appendix:
        #   N* = (C / 6)^(beta / (alpha + beta)) * (A * alpha / (B * beta))^(-beta / (alpha + beta))
        # but with the Chinchilla constants, this closely matches N* = sqrt(C/120).
        # We use the exact analytic form here for correctness.

        a_ratio = (cfg.A * cfg.alpha) / (cfg.B * cfg.beta)
        exp_n = cfg.beta / (cfg.alpha + cfg.beta)
        exp_d = cfg.alpha / (cfg.alpha + cfg.beta)

        n_star = math.pow(flop_budget / 6.0, exp_n) * math.pow(a_ratio, -cfg.beta / (cfg.alpha + cfg.beta))
        d_star = (flop_budget / 6.0) / n_star

        predicted = self.loss(n_star, d_star)
        return {
            "n_params": n_star,
            "n_tokens": d_star,
            "predicted_loss": predicted,
            "flops": flop_budget,
        }

    def isoflop_curve(
        self,
        flop_budget: float,
        n_points: int = 20,
    ) -> list[dict]:
        """
        Sample (N, D) pairs along the isoFLOP curve C = 6*N*D,
        varying N from C/(6*1e12) to C/(6*1e6) (log-spaced).

        Returns list of dicts with 'n_params', 'n_tokens', 'loss', 'flops'.
        """
        n_min = flop_budget / (6.0 * 1e12)
        n_max = flop_budget / (6.0 * 1e6)

        log_min = math.log(n_min)
        log_max = math.log(n_max)

        points = []
        for i in range(n_points):
            t = i / (n_points - 1) if n_points > 1 else 0.0
            n = math.exp(log_min + t * (log_max - log_min))
            d = flop_budget / (6.0 * n)
            l = self.loss(n, d)
            points.append({
                "n_params": n,
                "n_tokens": d,
                "loss": l,
                "flops": flop_budget,
            })
        return points

    def extrapolate_loss(
        self,
        observed_n: float,
        observed_d: float,
        observed_loss: float,
        target_n: float,
        target_d: float,
    ) -> float:
        """
        Given an observed loss point, estimate loss at a new (N, D) by fitting
        the scaling law constants to the observation and extrapolating.

        Fits E to match the observed loss (keeping A, B, alpha, beta from config),
        then predicts at target (N, D).

        Fitted E = observed_loss - A/N^alpha - B/D^beta
        Extrapolated = fitted_E + A/target_N^alpha + B/target_D^beta
        """
        cfg = self.config
        fitted_e = (
            observed_loss
            - cfg.A / math.pow(observed_n, cfg.alpha)
            - cfg.B / math.pow(observed_d, cfg.beta)
        )
        return (
            fitted_e
            + cfg.A / math.pow(target_n, cfg.alpha)
            + cfg.B / math.pow(target_d, cfg.beta)
        )


def compute_training_budget(
    gpu_flops: float,
    n_gpus: int,
    training_days: float,
    efficiency: float = 0.4,
) -> float:
    """
    Compute total training FLOPs from hardware specs.

    total_flops = gpu_flops * n_gpus * training_days * 86400 * efficiency

    Args:
        gpu_flops: FLOPs per second per GPU (e.g. A100 = 312e12)
        n_gpus: number of GPUs
        training_days: number of training days
        efficiency: model FLOPs utilization (MFU), default 0.4

    Returns:
        float: total training FLOPs
    """
    return gpu_flops * n_gpus * training_days * 86400.0 * efficiency


def kaplan_scaling_law(n_params: float, n_tokens: float) -> float:
    """
    Kaplan et al. 2020 scaling law (earlier, less accurate than Chinchilla).

    L(N, D) = (N_c/N)^alpha_N + (D_c/D)^alpha_D
    where N_c=8.8e13, D_c=5.4e13, alpha_N=0.076, alpha_D=0.095

    Returns predicted loss.
    """
    N_c = 8.8e13
    D_c = 5.4e13
    alpha_N = 0.076
    alpha_D = 0.095
    return math.pow(N_c / n_params, alpha_N) + math.pow(D_c / n_tokens, alpha_D)


def recommended_model_size(target_compute_budget_flops: float) -> dict:
    """
    Given a compute budget, recommend optimal N, D and approximate architecture.

    Architecture rule of thumb:
        d_model ≈ sqrt(N / (12 * n_layers))  with n_layers = 32
        n_heads = max(1, d_model // 64)

    Returns:
        {
            'n_params': float, 'n_tokens': float, 'predicted_loss': float,
            'n_layers': int, 'd_model': int, 'n_heads': int
        }
    """
    predictor = ChinchillaPredictor()
    allocation = predictor.optimal_allocation(target_compute_budget_flops)

    n_params = allocation["n_params"]
    n_tokens = allocation["n_tokens"]
    predicted_loss = allocation["predicted_loss"]

    n_layers = 32
    # Round d_model to nearest multiple of 64 (head_dim) so divisibility holds
    d_model_raw = int(math.sqrt(n_params / (12.0 * n_layers)))
    head_dim = 64
    d_model = max(head_dim, round(d_model_raw / head_dim) * head_dim)
    n_heads = max(1, d_model // head_dim)

    return {
        "n_params": n_params,
        "n_tokens": n_tokens,
        "predicted_loss": predicted_loss,
        "n_layers": n_layers,
        "d_model": d_model,
        "n_heads": n_heads,
    }


class ComputeOptimalScheduler:
    """
    Plan training schedule to follow compute-optimal scaling throughout training.

    At each checkpoint, evaluate whether it's better to:
    (a) continue training with current model
    (b) scale up to a larger model for remaining compute

    Args:
        initial_n_params: float
        total_flop_budget: float
        predictor: ChinchillaPredictor
    """

    def __init__(
        self,
        initial_n_params: float,
        total_flop_budget: float,
        predictor: ChinchillaPredictor | None = None,
    ):
        self.initial_n_params = initial_n_params
        self.total_flop_budget = total_flop_budget
        self.predictor = predictor or ChinchillaPredictor()

    def should_scale_up(
        self,
        current_n_params: float,
        flops_used_so_far: float,
        current_loss: float,
    ) -> dict:
        """
        Return {'scale_up': bool, 'optimal_n': float, 'expected_gain': float}

        Scale up if optimal_n > current_n by more than 20%.
        expected_gain = current_loss - loss(optimal_n, remaining_tokens)
        """
        remaining_flops = self.total_flop_budget - flops_used_so_far
        if remaining_flops <= 0:
            return {
                "scale_up": False,
                "optimal_n": current_n_params,
                "expected_gain": 0.0,
            }

        allocation = self.predictor.optimal_allocation(remaining_flops)
        optimal_n = allocation["n_params"]
        remaining_tokens = allocation["n_tokens"]

        scale_up = optimal_n > current_n_params * 1.2

        future_loss = self.predictor.loss(optimal_n, remaining_tokens)
        expected_gain = current_loss - future_loss

        return {
            "scale_up": scale_up,
            "optimal_n": optimal_n,
            "expected_gain": expected_gain,
        }
