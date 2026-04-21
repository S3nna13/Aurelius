"""DoLa Decoding — Decoding by Contrasting Layers (Chuang et al., 2023).

DoLa improves factual accuracy by contrasting the output distribution of a late
(final) transformer layer against an earlier intermediate layer.  The intuition
is that factual knowledge is encoded more strongly in deeper layers, so
subtracting the earlier-layer distribution sharpens factually-relevant tokens.

Two contrast modes are provided:

* ``subtract`` — log_p_late - α·log_p_early  (direct log-probability difference)
* ``jsd``      — log_p_late + α·(log_p_late − log_m)  where m = 0.5·(p_late + p_early)

Reference
---------
Chuang, Y.-S., Xie, Y., Luo, H., Kim, Y., Glass, J., & He, P. (2023).
DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language
Models.  arXiv:2309.03883.

Public API
----------
DoLaConfig         — configuration dataclass
DoLaLayerOutput    — thin wrapper around a layer's logits + layer index
DoLaDecoder        — contrast, sample, and decode step
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DoLaConfig:
    """Configuration for DoLa layer-contrastive decoding."""

    early_exit_layer: int = 12
    """Index of the intermediate layer used as the 'early' / amateur side."""

    alpha: float = 0.5
    """Interpolation weight.

    * alpha=0  → pure late-layer logits (no contrast)
    * alpha=1  → maximum contrast
    """

    contrast_mode: str = "subtract"
    """Either ``"subtract"`` or ``"jsd"``."""

    temperature: float = 1.0
    """Softmax temperature applied before sampling."""

    top_k: int = 0
    """Top-k filtering; 0 disables it."""

    top_p: float = 1.0
    """Nucleus (top-p) filtering; 1.0 disables it."""

    vocab_size: int = 32000
    """Vocabulary size — used only for shape validation."""


# ---------------------------------------------------------------------------
# Layer output container
# ---------------------------------------------------------------------------

@dataclass
class DoLaLayerOutput:
    """Logits emitted by a single transformer layer."""

    logits: Tensor
    """Raw (un-normalised) logits; shape ``[vocab]`` or ``[B, vocab]``."""

    layer_idx: int
    """Zero-based layer index (for bookkeeping / assertions)."""


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class DoLaDecoder:
    """Contrasts final-layer and early-layer logits for factuality-aware decoding.

    Parameters
    ----------
    config:
        :class:`DoLaConfig` instance controlling contrast mode, alpha, and
        sampling hyper-parameters.
    """

    def __init__(self, config: DoLaConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Core contrast methods
    # ------------------------------------------------------------------

    def _subtract_contrast(
        self,
        late_logits: Tensor,
        early_logits: Tensor,
    ) -> Tensor:
        """Subtract mode: log_p_late − α·log_p_early.

        The result is an *unnormalised* log-probability tensor.  The caller is
        responsible for converting to a probability distribution before sampling
        (e.g. via softmax).

        Parameters
        ----------
        late_logits:
            Raw logits from the final layer; shape ``[V]`` or ``[B, V]``.
        early_logits:
            Raw logits from the early layer; same shape as *late_logits*.

        Returns
        -------
        Tensor
            Contrast logits of the same shape.
        """
        T = self.config.temperature
        log_p_late = F.log_softmax(late_logits / T, dim=-1)
        log_p_early = F.log_softmax(early_logits / T, dim=-1)
        return log_p_late - self.config.alpha * log_p_early

    def _jsd_contrast(
        self,
        late_logits: Tensor,
        early_logits: Tensor,
    ) -> Tensor:
        """JSD mode: log_p_late + α·log(p_late / m)  where m = 0.5·(p_late+p_early).

        The JSD-derived term upweights tokens that are relatively more probable
        under the late layer than under the mixture, amplifying late-layer
        preference.

        Parameters
        ----------
        late_logits:
            Raw logits from the final layer; shape ``[V]`` or ``[B, V]``.
        early_logits:
            Raw logits from the early layer; same shape as *late_logits*.

        Returns
        -------
        Tensor
            Contrast logits of the same shape.
        """
        T = self.config.temperature
        p_late = F.softmax(late_logits / T, dim=-1)
        p_early = F.softmax(early_logits / T, dim=-1)
        m = 0.5 * (p_late + p_early)
        log_p_late = torch.log(p_late.clamp(min=1e-10))
        log_m = torch.log(m.clamp(min=1e-10))
        # Simplified JSD-inspired score: log_p_late + α·(log_p_late − log_m)
        return log_p_late + self.config.alpha * (log_p_late - log_m)

    def contrast_logits(
        self,
        late_logits: Tensor,
        early_logits: Tensor,
    ) -> Tensor:
        """Dispatch to the configured contrast mode.

        Parameters
        ----------
        late_logits:
            Raw logits from the final (late) layer.
        early_logits:
            Raw logits from the early / intermediate layer.

        Returns
        -------
        Tensor
            Contrast logits; same shape as the inputs.

        Raises
        ------
        ValueError
            If :attr:`DoLaConfig.contrast_mode` is not recognised.
        """
        if self.config.contrast_mode == "subtract":
            return self._subtract_contrast(late_logits, early_logits)
        elif self.config.contrast_mode == "jsd":
            return self._jsd_contrast(late_logits, early_logits)
        else:
            raise ValueError(
                f"Unknown contrast_mode '{self.config.contrast_mode}'. "
                "Expected 'subtract' or 'jsd'."
            )

    # ------------------------------------------------------------------
    # Jensen-Shannon Divergence (utility, also used in tests)
    # ------------------------------------------------------------------

    def jsd(self, p: Tensor, q: Tensor) -> Tensor:
        """Jensen-Shannon Divergence between two probability distributions.

        JSD(p, q) = 0.5·KL(p ‖ m) + 0.5·KL(q ‖ m)   where m = 0.5·(p + q).

        Parameters
        ----------
        p, q:
            Probability tensors of the same shape (last dimension is the
            event dimension).

        Returns
        -------
        Tensor
            Scalar JSD value (non-negative).
        """
        m = 0.5 * (p + q)
        eps = 1e-10

        def kl(a: Tensor, b: Tensor) -> Tensor:
            # KL(a || b) = sum a * log(a/b)
            ratio = a / b.clamp(min=eps)
            return (a * torch.log(ratio.clamp(min=eps))).sum(dim=-1)

        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, logits: Tensor) -> Tensor:
        """Sample token ids from contrast logits.

        Applies temperature scaling, top-k, and top-p filtering in sequence,
        then draws a single categorical sample per batch element.

        Parameters
        ----------
        logits:
            Contrast logits; shape ``[V]`` or ``[B, V]``.

        Returns
        -------
        Tensor
            Token ids; shape ``[B]`` for batched input, scalar for unbatched.
        """
        squeeze = logits.dim() == 1
        if squeeze:
            logits = logits.unsqueeze(0)  # [1, V]

        T = self.config.temperature
        scaled = logits / T if T != 1.0 else logits

        # Top-k filtering
        if self.config.top_k > 0:
            k = min(self.config.top_k, scaled.size(-1))
            top_k_vals = scaled.topk(k, dim=-1).values
            threshold = top_k_vals[..., -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

        # Top-p (nucleus) filtering
        if self.config.top_p < 1.0:
            sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
            cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Remove tokens with cumulative probability above the threshold
            # (shift by one to include the token that crosses the threshold)
            remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > self.config.top_p
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            # Scatter back to original ordering
            scaled = torch.zeros_like(scaled).scatter_(-1, sorted_idx, sorted_logits)

        probs = F.softmax(scaled, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]

        return token_ids.squeeze(0) if squeeze else token_ids

    # ------------------------------------------------------------------
    # Decode step
    # ------------------------------------------------------------------

    def decode_step(
        self,
        late_layer_output: DoLaLayerOutput,
        early_layer_output: DoLaLayerOutput,
    ) -> dict[str, Tensor]:
        """Perform one DoLa decoding step.

        Computes contrast logits from the two layer outputs and samples a token.

        Parameters
        ----------
        late_layer_output:
            Output from the final transformer layer.
        early_layer_output:
            Output from the early intermediate layer.

        Returns
        -------
        dict with keys:

        * ``"token_ids"``       — sampled token id(s)
        * ``"contrast_logits"`` — raw contrast logits before sampling
        * ``"late_logits"``     — original late-layer logits
        * ``"early_logits"``    — original early-layer logits
        """
        late = late_layer_output.logits
        early = early_layer_output.logits

        contrast = self.contrast_logits(late, early)
        token_ids = self.sample(contrast)

        return {
            "token_ids": token_ids,
            "contrast_logits": contrast,
            "late_logits": late,
            "early_logits": early,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.inference import DECODER_REGISTRY  # noqa: E402

DECODER_REGISTRY["dola"] = DoLaDecoder
