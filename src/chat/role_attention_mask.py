"""Role-aware attention mask builder for Aurelius.

Given a sequence of ``RoleSpan`` ranges produced by a chat template
(ChatML / Llama-3 / Harmony), build an additive attention mask that
enforces:

    1. Standard causal masking.
    2. System priority: system tokens are attended from every position
       (including positions that are before the system span, should any
       exist -- in practice system spans are first).
    3. User/tool barrier: user tokens cannot attend to later tool
       outputs. A user utterance was "already said" and should not be
       retroactively conditioned on tool calls that came after it.
    4. An optional "loss mask" helper selecting positions that
       contribute to the next-token loss (assistant tokens by default).

The returned attention mask uses additive convention:
    0.0  -> attend
    -1e9 -> masked

This matches the standard ``scaled_dot_product_attention`` additive
bias convention used elsewhere in Aurelius.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch

MASK_VALUE: float = -1e9

VALID_ROLES = frozenset({"system", "user", "assistant", "tool"})


class RoleSpanError(ValueError):
    """Raised when a sequence of RoleSpans is malformed."""


@dataclass(frozen=True)
class RoleSpan:
    """Token range [start, end) associated with a chat role.

    Attributes:
        role: Chat role (system/user/assistant/tool).
        start: Inclusive start token index.
        end: Exclusive end token index. Must satisfy end > start.
    """

    role: str
    start: int
    end: int


def validate_spans(spans: Sequence[RoleSpan], seq_len: int) -> None:
    """Validate that spans tile [0, seq_len) with no gaps or overlaps.

    Raises:
        RoleSpanError: On empty spans, unknown roles, out-of-range
            indices, gaps, or overlaps.
    """
    if seq_len <= 0:
        raise RoleSpanError(f"seq_len must be positive, got {seq_len}")
    if not spans:
        raise RoleSpanError("spans must be non-empty")

    ordered = sorted(spans, key=lambda s: s.start)
    cursor = 0
    for span in ordered:
        if span.role not in VALID_ROLES:
            raise RoleSpanError(
                f"unknown role {span.role!r}; expected one of {sorted(VALID_ROLES)}"
            )
        if span.start < 0 or span.end > seq_len:
            raise RoleSpanError(
                f"span {span} out of range for seq_len={seq_len}"
            )
        if span.end <= span.start:
            raise RoleSpanError(f"span {span} has non-positive length")
        if span.start < cursor:
            raise RoleSpanError(f"span {span} overlaps previous span at {cursor}")
        if span.start > cursor:
            raise RoleSpanError(
                f"gap between {cursor} and {span.start} -- spans must tile [0, seq_len)"
            )
        cursor = span.end
    if cursor != seq_len:
        raise RoleSpanError(
            f"spans end at {cursor} but seq_len={seq_len} -- must cover entire sequence"
        )


def _role_indices(
    spans: Sequence[RoleSpan], seq_len: int
) -> dict:
    """Return a dict mapping role -> list of (start, end) ranges."""
    buckets: dict = {}
    for s in spans:
        buckets.setdefault(s.role, []).append((s.start, s.end))
    return buckets


def build_role_mask(
    spans: Sequence[RoleSpan],
    seq_len: int,
    system_priority: bool = True,
    causal: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build an additive [S, S] attention mask from role spans.

    Args:
        spans: RoleSpans tiling [0, seq_len).
        seq_len: Total sequence length.
        system_priority: If True, every query position may attend to
            every system token (even outside the causal window).
        causal: If True, apply standard causal masking first.
        device: Target torch device.
        dtype: Target torch dtype. MASK_VALUE is cast to this dtype.

    Returns:
        Tensor of shape [seq_len, seq_len] where 0.0 means "attend" and
        MASK_VALUE means "masked".
    """
    validate_spans(spans, seq_len)

    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    neg = torch.tensor(MASK_VALUE, device=device, dtype=dtype)

    if causal:
        # Mask strict upper triangle (j > i).
        idx = torch.arange(seq_len, device=device)
        causal_block = idx.unsqueeze(0) > idx.unsqueeze(1)  # [S,S] bool
        mask = torch.where(causal_block, neg, mask)

    buckets = _role_indices(spans, seq_len)

    # System priority: unmask system-token *columns* for all query rows.
    if system_priority:
        for s0, s1 in buckets.get("system", []):
            mask[:, s0:s1] = 0.0

    # User -> later-tool barrier: user query rows cannot attend to any
    # tool-token columns that begin at/after the user span start.
    # (Tool outputs that physically came before a user turn can still be
    # attended; those are historical context the user already saw.)
    tool_ranges = buckets.get("tool", [])
    for u0, u1 in buckets.get("user", []):
        for t0, t1 in tool_ranges:
            if t0 >= u0:
                mask[u0:u1, t0:t1] = MASK_VALUE

    return mask


def build_loss_mask(
    spans: Sequence[RoleSpan],
    seq_len: int,
    loss_roles: Tuple[str, ...] = ("assistant",),
) -> torch.Tensor:
    """Build a [seq_len] boolean vector selecting loss-contributing positions.

    A position i is True iff the token at position i belongs to one of
    the configured ``loss_roles``. Caller is responsible for any
    shift-by-one when computing next-token CE loss.
    """
    validate_spans(spans, seq_len)
    for role in loss_roles:
        if role not in VALID_ROLES:
            raise RoleSpanError(
                f"unknown loss role {role!r}; expected subset of {sorted(VALID_ROLES)}"
            )

    mask = torch.zeros(seq_len, dtype=torch.bool)
    loss_set = set(loss_roles)
    for s in spans:
        if s.role in loss_set:
            mask[s.start:s.end] = True
    return mask


__all__ = [
    "MASK_VALUE",
    "RoleSpan",
    "RoleSpanError",
    "build_loss_mask",
    "build_role_mask",
    "validate_spans",
]
