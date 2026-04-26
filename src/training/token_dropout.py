"""Token-level dropout regularization for LM training.

Distinct from standard torch.nn.Dropout (which acts on activations). This
operates on token ids, either replacing them with a [MASK]-style token or
zeroing their loss contribution via a loss mask.
"""

from __future__ import annotations

import torch

_VALID_MODES = ("replace", "loss_mask", "both")


class TokenDropout:
    """Randomly drops or masks input tokens during LM training.

    Args:
        p: drop probability in [0, 1].
        mask_token_id: token id substituted in "replace" / "both" modes.
        mode: one of "replace", "loss_mask", "both".
        exclude_special_ids: token ids never considered for dropout.
    """

    def __init__(
        self,
        p: float = 0.1,
        mask_token_id: int = 0,
        mode: str = "replace",
        exclude_special_ids: tuple[int, ...] = (),
    ) -> None:
        if not isinstance(p, (int, float)) or p < 0.0 or p > 1.0:
            raise ValueError(f"p must be in [0, 1], got {p!r}")
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        self.p = float(p)
        self.mask_token_id = int(mask_token_id)
        self.mode = mode
        self.exclude_special_ids = tuple(int(i) for i in exclude_special_ids)

    def apply(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply token dropout.

        Returns a tuple `(new_input_ids, new_loss_mask)`. The returned tensors
        are fresh allocations; the inputs are not mutated.
        """
        if not torch.is_tensor(input_ids):
            raise TypeError("input_ids must be a torch.Tensor")

        device = input_ids.device
        shape = input_ids.shape

        # Default loss mask: all positions contribute.
        if loss_mask is None:
            new_loss_mask = torch.ones(shape, dtype=torch.float32, device=device)
        else:
            new_loss_mask = loss_mask.clone()

        new_input_ids = input_ids.clone()

        # Short-circuit: nothing to do.
        if input_ids.numel() == 0 or self.p == 0.0:
            return new_input_ids, new_loss_mask

        # Sample Bernoulli(p) per position using the provided RNG (if any)
        # so that seeded runs are deterministic.
        probs = torch.full(shape, self.p, dtype=torch.float32, device=device)
        if rng is not None:
            # torch.bernoulli respects the generator kwarg.
            drop = torch.bernoulli(probs, generator=rng).to(torch.bool)
        else:
            drop = torch.bernoulli(probs).to(torch.bool)

        # Exclude specials from dropping.
        if self.exclude_special_ids:
            special = torch.zeros(shape, dtype=torch.bool, device=device)
            for sid in self.exclude_special_ids:
                special = special | (input_ids == sid)
            drop = drop & (~special)

        if self.mode in ("replace", "both"):
            new_input_ids = torch.where(
                drop,
                torch.full_like(input_ids, self.mask_token_id),
                new_input_ids,
            )
        if self.mode in ("loss_mask", "both"):
            new_loss_mask = torch.where(
                drop,
                torch.zeros_like(new_loss_mask),
                new_loss_mask,
            )

        return new_input_ids, new_loss_mask
