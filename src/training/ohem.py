"""Online Hard Example Mining (OHEM) for language model training.

Selects the hardest examples (tokens or sequences) by loss magnitude
and returns a weighted loss over only those examples, focusing training
on the most informative samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class OHEMMode(Enum):
    TOKEN = "token"  # mine hardest individual tokens
    SEQUENCE = "sequence"  # mine hardest sequences (by mean loss), then use all their tokens


@dataclass
class OHEMConfig:
    keep_fraction: float = 0.7  # fraction of hardest examples to keep (0 < f <= 1.0)
    mode: OHEMMode = OHEMMode.TOKEN
    min_keep: int = 1  # always keep at least this many examples


def ohem_mask(
    per_token_loss: torch.Tensor,
    cfg: OHEMConfig,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a (B, S) bool mask indicating which tokens are selected by OHEM.

    Args:
        per_token_loss: (B, S) per-token loss values (NOT reduced).
        cfg: OHEMConfig controlling selection behaviour.
        padding_mask: (B, S) bool tensor, True = valid token, False = padding.
                      If None, all tokens are considered valid.

    Returns:
        (B, S) bool tensor where True marks tokens selected for the loss.
    """
    B, S = per_token_loss.shape

    if padding_mask is None:
        padding_mask = torch.ones(B, S, dtype=torch.bool, device=per_token_loss.device)

    if cfg.mode == OHEMMode.TOKEN:
        return _ohem_mask_token(per_token_loss, cfg, padding_mask)
    else:
        return _ohem_mask_sequence(per_token_loss, cfg, padding_mask)


def _ohem_mask_token(
    per_token_loss: torch.Tensor,
    cfg: OHEMConfig,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    B, S = per_token_loss.shape

    # Set padded positions to -inf so they are never selected by topk
    masked_loss = per_token_loss.clone()
    masked_loss[~padding_mask] = float("-inf")

    flat = masked_loss.reshape(-1)
    valid_count = int(padding_mask.sum().item())
    n_keep = max(cfg.min_keep, int(valid_count * cfg.keep_fraction))
    # Clamp to valid_count so we don't request more than available
    n_keep = min(n_keep, valid_count)

    topk_indices = torch.topk(flat, n_keep).indices

    flat_mask = torch.zeros(B * S, dtype=torch.bool, device=per_token_loss.device)
    flat_mask[topk_indices] = True

    return flat_mask.reshape(B, S)


def _ohem_mask_sequence(
    per_token_loss: torch.Tensor,
    cfg: OHEMConfig,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    B, S = per_token_loss.shape

    # Compute per-sequence mean loss over valid tokens only
    valid_counts = padding_mask.sum(dim=1).float().clamp(min=1.0)  # (B,)
    masked_loss = per_token_loss.clone()
    masked_loss[~padding_mask] = 0.0
    seq_means = masked_loss.sum(dim=1) / valid_counts  # (B,)

    n_keep = max(cfg.min_keep, int(B * cfg.keep_fraction))
    n_keep = min(n_keep, B)

    topk_indices = torch.topk(seq_means, n_keep).indices  # indices of selected sequences

    # Build mask: True for all valid tokens in selected sequences
    mask = torch.zeros(B, S, dtype=torch.bool, device=per_token_loss.device)
    mask[topk_indices] = padding_mask[topk_indices]

    return mask


def ohem_loss(
    per_token_loss: torch.Tensor,
    cfg: OHEMConfig,
    padding_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply OHEM selection and return scalar mean loss over kept examples.

    Args:
        per_token_loss: (B, S) per-token loss values (NOT reduced).
        cfg: OHEMConfig controlling selection behaviour.
        padding_mask: (B, S) bool tensor, True = valid token, False = padding.
                      If None, all tokens are considered valid.

    Returns:
        Scalar loss (mean over selected tokens/sequences).
    """
    mask = ohem_mask(per_token_loss, cfg, padding_mask=padding_mask)
    return per_token_loss[mask].mean()
