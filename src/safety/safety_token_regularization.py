"""Safety Token Regularization (STR) for Aurelius.

STR (2026) identifies salient tokens from safety refusal templates and
constrains their logits during fine-tuning.  Prevents loss of safety
behaviour with minimal overhead.  Restores pretrained safety to 0% HRR on
Alpaca fine-tuning.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

DEFAULT_SAFETY_TEMPLATES: tuple[str, ...] = (
    "I cannot",
    "I'm sorry",
    "I apologize",
    "That request is harmful",
    "I can't assist",
    "I am not able to",
    "I cannot comply",
    "I'm unable to",
    "I must refuse",
    "This is inappropriate",
)


class SafetyTokenRegularizer:
    """Training-time regularizer that anchors safety-critical token logits.

    During initialization the model is run in eval mode (no gradients) over a
    set of refusal templates.  The most salient tokens (top-K by logit at each
    template position, unioned with the template's own token ids) are stored as
    the *safety token set*.  Their mean logits across all template positions
    are cached as *reference logits*.

    At training time :meth:`compute_str_loss` penalises the MSE between the
    current model's safety-token logits and the cached reference values:

        ``loss_str = lambda_str * mean((logits[..., safety_ids] - ref_logits)^2)``

    Args:
        model: The pretrained model (frozen reference).  Must expose
            ``forward(input_ids, labels=None) -> (loss, logits, present_key_values)``.
        tokenizer: Any object with an ``encode(text, *, add_bos=False, add_eos=False) -> list[int]``
            method.
        templates: Iterable of refusal template strings.  Defaults to
            :data:`DEFAULT_SAFETY_TEMPLATES`.
        lambda_str: Weight of the quadratic penalty term.
        top_k_per_template: Number of highest-logit tokens to extract from
            each template forward pass.  If ``None`` defaults to ``5``.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        templates: Iterable[str] | None = None,
        lambda_str: float = 0.01,
        top_k_per_template: int | None = None,
    ) -> None:
        self.lambda_str = float(lambda_str)
        self.top_k_per_template = top_k_per_template if top_k_per_template is not None else 5
        self.templates = tuple(templates) if templates is not None else DEFAULT_SAFETY_TEMPLATES
        self._mode_lock = threading.Lock()

        # Discover device from model parameters
        params = list(model.parameters())
        if not params:
            raise ValueError("Model has no parameters to discover device from")
        self._device = params[0].device

        # Extract safety token ids and reference logits (no grad, eval mode)
        self.safety_token_ids: Tensor
        self.ref_logits: Tensor
        self.safety_token_ids, self.ref_logits = self._extract_safety_tokens(model, tokenizer)

    @torch.no_grad()
    def _extract_safety_tokens(
        self,
        model: nn.Module,
        tokenizer: Any,
    ) -> tuple[Tensor, Tensor]:
        """Build safety token set and cached reference logits."""
        was_training = model.training
        with self._mode_lock:
            model.eval()

        try:
            # ------------------------------------------------------------------
            # Pass 1 — discover salient token ids
            # ------------------------------------------------------------------
            salient_ids: set[int] = set()
            for template in self.templates:
                ids = tokenizer.encode(template, add_bos=False, add_eos=False)
                if not ids:
                    continue
                salient_ids.update(ids)
                input_ids = torch.tensor([ids], device=self._device)
                # Forward returns (loss, logits, present_key_values)
                out = model(input_ids=input_ids, labels=None)
                logits = out[1] if isinstance(out, tuple) else out
                if logits is None:
                    continue
                # Flatten over positions and take global top-k
                flat = logits[0].view(-1)
                k = min(self.top_k_per_template, flat.numel())
                if k > 0:
                    _, topk_flat = torch.topk(flat, k=k)
                    token_ids = (topk_flat % logits.size(-1)).cpu().tolist()
                    salient_ids.update(token_ids)

            if not salient_ids:
                # Fallback so the regularizer is a well-behaved no-op
                salient_ids = {0}

            safety_token_ids = torch.tensor(
                sorted(salient_ids),
                device=self._device,
                dtype=torch.long,
            )

            # ------------------------------------------------------------------
            # Pass 2 — cache mean reference logits for each safety token
            # ------------------------------------------------------------------
            ref_sum = torch.zeros(
                safety_token_ids.size(0), device=self._device, dtype=torch.float32
            )
            ref_count = torch.zeros_like(ref_sum)

            for template in self.templates:
                ids = tokenizer.encode(template, add_bos=False, add_eos=False)
                if not ids:
                    continue
                input_ids = torch.tensor([ids], device=self._device)
                out = model(input_ids=input_ids, labels=None)
                logits = out[1] if isinstance(out, tuple) else out
                if logits is None:
                    continue
                # Gather logits for safety tokens at every position
                safety_logits = logits[0, :, safety_token_ids]  # [T, K]
                ref_sum += safety_logits.sum(dim=0)  # [K]
                ref_count += logits.size(1)

            ref_logits = ref_sum / ref_count.clamp_min(1.0)
            return safety_token_ids, ref_logits

        finally:
            if was_training:
                with self._mode_lock:
                    model.train()

    def compute_str_loss(self, logits: Tensor) -> Tensor:
        """Compute STR penalty for the given logits.

        Args:
            logits: Raw model logits of shape ``[..., vocab_size]``.  In
                typical LM training this is ``[B, T, V]``.

        Returns:
            Scalar tensor — the weighted MSE between the safety-token logits
            and their pretrained reference values.
        """
        if logits.dim() < 2:
            raise ValueError(f"logits must have at least 2 dimensions, got {logits.dim()}")

        # logits[..., safety_token_ids] -> [..., K]
        safety_logits = logits.index_select(-1, self.safety_token_ids)

        # Broadcast reference logits to match safety_logits shape
        ref_shape = [1] * (safety_logits.dim() - 1) + [-1]
        ref = self.ref_logits.view(*ref_shape)

        mse = ((safety_logits - ref) ** 2).mean()
        return self.lambda_str * mse
