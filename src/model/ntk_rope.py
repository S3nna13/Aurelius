import torch
import math
from dataclasses import dataclass


@dataclass
class NTKRoPEConfig:
    head_dim: int = 128
    max_seq_len: int = 32768        # target extended context length
    theta: float = 500_000.0        # original RoPE theta
    scale_factor: float = 4.0       # context extension factor
    original_max_seq_len: int = 8192


def ntk_scaled_theta(theta: float, scale_factor: float, head_dim: int) -> float:
    """Compute the NTK-scaled theta for context extension.

    Formula: theta_new = theta * scale_factor^(head_dim / (head_dim - 2))

    This scales the base frequency such that lower-frequency dimensions
    (which encode long-range position) are smoothly extended.
    """
    return theta * (scale_factor ** (head_dim / (head_dim - 2)))


def ntk_rope_frequencies(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500_000.0,
    scale_factor: float = 4.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute RoPE frequency tensor with NTK-aware theta scaling.

    Unlike YaRN which applies per-dimension ramp functions, NTK simply
    uses a globally scaled theta and otherwise computes standard RoPE.

    Returns:
        Complex tensor of shape (max_seq_len, head_dim // 2).
    """
    new_theta = ntk_scaled_theta(theta, scale_factor, head_dim)
    freqs = 1.0 / (new_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(max_seq_len, device=device).float()
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def dynamic_ntk_frequencies(
    head_dim: int,
    current_seq_len: int,
    original_max_seq_len: int,
    theta: float = 500_000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Dynamic NTK: compute scale_factor from actual sequence length at runtime.

    scale_factor = current_seq_len / original_max_seq_len
    No scaling if current_seq_len <= original_max_seq_len.

    Returns:
        Complex tensor of shape (current_seq_len, head_dim // 2).
    """
    if current_seq_len <= original_max_seq_len:
        scale_factor = 1.0
    else:
        scale_factor = current_seq_len / original_max_seq_len
    return ntk_rope_frequencies(head_dim, current_seq_len, theta, scale_factor, device)
