from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PRAXISConfig:
    # Sampling
    n_group: int = 8
    temperature: float = 0.9
    max_new_tokens: int = 128
    d_model: int = 2048
    # DAPO clip bounds
    eps_low: float = 0.20
    eps_high: float = 0.28
    # KL + entropy
    beta_kl: float = 0.04
    lambda_ent: float = 0.001
    # Constitutional
    n_principles: int = 8
    tau_gate: float = 0.4
    n_criteria: int = 4
    # SRC (Steering-Reward Correspondence)
    steer_layers: list[int] = field(default_factory=lambda: [12, 16, 20])
    steer_alpha: float = 0.3
    lambda_src: float = 0.1
    # ESA (Expert Safety Affinity)
    safety_experts: list[int] = field(default_factory=lambda: [0, 1])
    alpha_esa: float = 0.01
    tau_safety: float = 0.5
    # MTAH (Multi-Token Alignment Horizon)
    gamma_mtah: float = 0.95
    k_mtah: int = 2
    # Thinking tokens
    think_weight: float = 0.5
    answer_weight: float = 1.0
    # WARP
    warp_interval: int = 50
    warp_anchor_mu: float = 0.05
    # Uncertainty
    mc_dropout_n: int = 20
