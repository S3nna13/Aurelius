"""Context Window Extension via dynamic NTK-aware RoPE scaling.

Implements strategies from:
- "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens"
  (arXiv:2402.13753)
- "YaRN: Efficient Context Window Extension of Large Language Models"
  (arXiv:2309.00071)

Provides dynamic NTK-aware RoPE scaling that allows a model trained at a
short context (e.g. 8K) to do inference at 128K+ without catastrophic
positional degradation.

Public surface:
    ContextWindowExtension   -- static methods for each scaling strategy
    DynamicContextScaler     -- auto-selects strategy based on seq_len
    CONTEXT_EXTENSION_REGISTRY -- maps strategy name -> callable/class

Dependencies: pure PyTorch only.
"""

from __future__ import annotations

import math

import torch


class ContextWindowExtension:
    """Static scaling strategies for extending RoPE-based context windows.

    Four strategies are provided:

    1. LinearScaling: uniformly compress position indices by (train/target).
    2. NTKAwareScaling: scale the RoPE base θ so high-frequency dimensions
       retain precision at the target length. Formula from NTK-aware paper:
           new_base = base * (target/train)^(dim/(dim-2))
    3. YaRNScaling: non-uniform per-dimension scaling using a ramp function.
       High-frequency dims (short wavelength) are left unscaled; low-frequency
       dims (long wavelength) are linearly scaled; a ramp interpolates between.
       Attention magnitude is corrected by mscale = 0.1*ln(scale)+1.0.
    4. LongRoPEScaling: apply learned per-dimension rescale_factors to the
       inverse frequencies, as described in arXiv:2402.13753.
    """

    @staticmethod
    def _standard_cos_sin(
        dim: int,
        base: float,
        seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute standard RoPE cos/sin cache, shape [seq_len, dim]."""
        half = dim // 2
        idx = torch.arange(0, half, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (idx * 2.0 / dim))  # [D/2]
        t = torch.arange(seq_len, dtype=torch.float32, device=device)  # [S]
        freqs = torch.outer(t, inv_freq)  # [S, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, D]
        return emb.cos(), emb.sin()

    @staticmethod
    def linear_scale(
        cos: torch.Tensor,
        sin: torch.Tensor,
        train_len: int,
        target_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Linear (positional interpolation) scaling.

        Compresses position indices uniformly: pos -> pos * (train_len /
        target_len). Equivalent to interpolating cos/sin at fractional
        positions.

        Parameters
        ----------
        cos, sin:
            Pre-computed RoPE cache of shape [seq_len, dim] at the
            *training* positions. The caller is expected to pass caches
            built with positions 0..seq_len-1.
        train_len:
            Maximum context length the model was trained on.
        target_len:
            Target (extended) context length.

        Returns
        -------
        (cos_scaled, sin_scaled), each of shape [seq_len, dim].
        """
        if target_len <= train_len:
            return cos, sin

        scale = train_len / target_len  # < 1 when extending
        seq_len = cos.shape[0]
        dim = cos.shape[1]
        # Recompute at compressed positions.
        half = dim // 2
        # We need the base & actual positions; extract from the passed tensors
        # would require inversion. Instead, recompute from scratch at scaled
        # positions (this is what PI / linear scaling does).
        t_orig = torch.arange(seq_len, dtype=torch.float32, device=cos.device)
        t_scaled = t_orig * scale  # fractional positions in [0, train_len)

        # Extract inv_freq from the first row of cos (position 0 gives 1s —
        # use position 1 instead, which encodes inv_freq directly in freqs).
        # More robust: recompute using default base 10000 via the dim only.
        # Since we don't have the original base here, we reconstruct the
        # cos/sin using the scaled position tensor against the passed cos/sin.
        # For correctness without base knowledge, we operate on the angle
        # domain: cos[t_i, :] = cos(t_i * inv_freq_concat).
        # Since the caller pre-built the cache at integer positions, we
        # must re-derive inv_freq from the cache.
        #
        # Practical approach used by most codebases: rebuild at scaled t.
        # We recover inv_freq from row 1 of the *angle* (before cos/sin).
        # angle[1, :half] = 1 * inv_freq  => inv_freq = arccos-unsafe.
        # Safest: treat the cache as opaque and just re-interpolate the angles.
        #
        # We recompute the angle tensor for the first half of dims (the second
        # half is a copy), then re-cos/sin at the scaled positions.
        # angle[s, i] = s * inv_freq[i]; for s=1: angle[1, i] = inv_freq[i].
        # Extract angles from the embedding before cos/sin using:
        #   freq[1, :half] = arccos(cos[1, :half]) — valid only for small vals.
        # More robust: pass-through approach — use scaled t and rebuild.
        # Because this method signature doesn't provide base, we operate on
        # the angle array recovered from the cache.
        #
        # Recover inv_freq robustly: use torch.atan2(sin, cos) which gives
        # the angle at each position, then take position 1 (angle = 1*inv_freq).
        if seq_len < 2:
            # Can't recover inv_freq from a single-position cache; return as-is.
            return cos, sin

        angles = torch.atan2(sin[:, :half], cos[:, :half])  # [S, D/2]
        # angles[s, i] = s * inv_freq[i]  (modulo 2pi wrapping)
        # For most contexts s=1 won't wrap (inv_freq < 1 always for base>1).
        inv_freq = angles[1, :]  # [D/2]

        freqs = torch.outer(t_scaled, inv_freq)  # [S, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, D]
        return emb.cos(), emb.sin()

    @staticmethod
    def ntk_aware_scale(
        dim: int,
        base: float,
        train_len: int,
        target_len: int,
        seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """NTK-aware RoPE scaling.

        Adjusts the RoPE base θ so that the effective NTK wavelengths cover
        the target context length. Formula:

            new_base = base * (target_len / train_len)^(dim / (dim - 2))

        This preserves relative positional information at all frequencies.

        Parameters
        ----------
        dim:
            Head embedding dimension (must be even).
        base:
            Original RoPE base θ (e.g. 10000.0).
        train_len:
            Original maximum training sequence length.
        target_len:
            Target (extended) sequence length.
        seq_len:
            Number of positions to generate cos/sin for.
        device:
            Torch device for output tensors.

        Returns
        -------
        (cos, sin), each of shape [seq_len, dim].
        """
        if target_len <= train_len:
            new_base = base
        else:
            scale = target_len / train_len
            new_base = base * (scale ** (dim / (dim - 2)))

        return ContextWindowExtension._standard_cos_sin(dim, new_base, seq_len, device)

    @staticmethod
    def yarn_scale(
        dim: int,
        base: float,
        train_len: int,
        target_len: int,
        seq_len: int,
        alpha: float = 1.0,
        beta: float = 32.0,
        mscale: float = 0.1,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """YaRN: non-uniform per-dimension RoPE scaling.

        Applies a ramp function r(i) to interpolate between:
        - High-frequency dimensions (wavelength < α·train_len):
          no scaling, attend locally.
        - Low-frequency dimensions (wavelength > β·train_len):
          linear scaling (divide inv_freq by scale factor).
        - In-between: ramp r(i) = clamp((λ_i/α - 1) / (β/α - 1), 0, 1)
          blends between the two regimes.

        Attention magnitude correction: mscale_val = 0.1·ln(scale)+1.0 is
        multiplied into both cos and sin.

        Parameters
        ----------
        dim:
            Head embedding dimension (even).
        base:
            Original RoPE base θ.
        train_len:
            Original training context length.
        target_len:
            Target extended context length.
        seq_len:
            Number of positions to generate.
        alpha:
            Low-freq threshold (wavelengths > alpha * 2π·base^(2i/dim)
            get full linear scaling). Paper default: 1.0.
        beta:
            High-freq threshold. Paper default: 32.0.
        mscale:
            Mscale coefficient. mscale_val = mscale*ln(scale)+1.0.
        device:
            Output device.

        Returns
        -------
        (cos, sin), each of shape [seq_len, dim].
        """
        half = dim // 2
        scale = target_len / train_len if target_len > train_len else 1.0

        idx = torch.arange(0, half, dtype=torch.float32, device=device)
        # Standard RoPE inverse frequencies: θ_i = base^(-2i/dim)
        inv_freq_base = 1.0 / (base ** (idx * 2.0 / dim))  # [D/2]

        if scale <= 1.0:
            # No extension needed — standard RoPE.
            freqs = torch.outer(
                torch.arange(seq_len, dtype=torch.float32, device=device),
                inv_freq_base,
            )
            emb = torch.cat((freqs, freqs), dim=-1)
            return emb.cos(), emb.sin()

        # Wavelength of each frequency band: λ_i = 2π / θ_i = 2π / inv_freq_base
        wavelength = 2.0 * math.pi / inv_freq_base  # [D/2]

        # Ramp function r(i) in [0, 1]:
        #   r=0 -> high freq (no scaling)
        #   r=1 -> low freq (linear scaling, divide by scale)
        # Condition: wavelength < alpha => high freq; wavelength > beta => low freq
        # From the paper: r(i) = clamp((λ_i/α - 1) / (β/α - 1), 0, 1)
        alpha_t = float(alpha)
        beta_t = float(beta)
        if beta_t <= alpha_t:
            beta_t = alpha_t + 1.0  # guard against degenerate range

        ramp = (wavelength / alpha_t - 1.0) / (beta_t / alpha_t - 1.0)
        ramp = torch.clamp(ramp, 0.0, 1.0)  # [D/2]

        # Blended inverse frequencies:
        #   low-freq  (r=1): inv_freq / scale  (linear interpolation)
        #   high-freq (r=0): inv_freq           (extrapolation, no change)
        inv_freq_interp = inv_freq_base / scale
        inv_freq_yarn = inv_freq_base * (1.0 - ramp) + inv_freq_interp * ramp

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq_yarn)  # [S, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, D]

        # Attention magnitude scale correction.
        mscale_val = mscale * math.log(scale) + 1.0

        cos_out = emb.cos() * mscale_val
        sin_out = emb.sin() * mscale_val
        return cos_out, sin_out

    @staticmethod
    def longrope_scale(
        dim: int,
        base: float,
        seq_len: int,
        rescale_factors: torch.Tensor,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LongRoPE: per-dimension learned rescale factors.

        Applies per-dimension rescale factors to the standard RoPE inverse
        frequencies:

            inv_freq_scaled[i] = inv_freq[i] / rescale_factors[i]

        The rescale_factors tensor holds learned (or heuristic) values that
        encode how much each frequency band should be stretched, allowing the
        model to handle extremely long sequences (arXiv:2402.13753).

        Parameters
        ----------
        dim:
            Head embedding dimension (even).
        base:
            Original RoPE base θ.
        seq_len:
            Number of positions to generate.
        rescale_factors:
            Per-dimension scale factors, shape [dim//2]. Values should be
            >= 1.0 (1.0 means no rescaling for that dimension).
        device:
            Output device. If None, uses rescale_factors.device.

        Returns
        -------
        (cos, sin), each of shape [seq_len, dim].
        """
        if device is None:
            device = rescale_factors.device

        half = dim // 2
        if rescale_factors.shape[0] != half:
            raise ValueError(
                f"rescale_factors must have shape [{half}] for dim={dim}, "
                f"got {list(rescale_factors.shape)}"
            )

        idx = torch.arange(0, half, dtype=torch.float32, device=device)
        inv_freq_base = 1.0 / (base ** (idx * 2.0 / dim))  # [D/2]

        factors = rescale_factors.to(dtype=torch.float32, device=device)
        # Avoid division by zero.
        factors = torch.clamp(factors, min=1e-6)
        inv_freq_scaled = inv_freq_base / factors  # [D/2]

        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq_scaled)  # [S, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, D]
        return emb.cos(), emb.sin()


# ---------------------------------------------------------------------------
# DynamicContextScaler
# ---------------------------------------------------------------------------


class DynamicContextScaler:
    """Automatically select a context-extension strategy based on seq_len.

    Selection policy:
    - seq_len <= train_len:                 standard RoPE (no scaling)
    - train_len < seq_len <= 4*train_len:   YaRN scaling
    - seq_len > 4*train_len:               LongRoPE or NTK-aware scaling
                                            (NTK-aware when no rescale_factors)

    Parameters
    ----------
    strategy:
        One of "auto", "linear", "ntk", "yarn", "longrope".
        "auto" applies the policy above; the others force a specific strategy.
    dim:
        Head embedding dimension.
    base:
        RoPE base θ.
    train_len:
        Training context length (positions the model was trained on).
    rescale_factors:
        Optional per-dimension LongRoPE factors, shape [dim//2]. Required
        when strategy=="longrope"; ignored otherwise.
    yarn_alpha:
        YaRN low-freq threshold (default 1.0).
    yarn_beta:
        YaRN high-freq threshold (default 32.0).
    yarn_mscale:
        YaRN mscale coefficient (default 0.1).
    """

    def __init__(
        self,
        strategy: str,
        dim: int,
        base: float,
        train_len: int,
        rescale_factors: torch.Tensor | None = None,
        yarn_alpha: float = 1.0,
        yarn_beta: float = 32.0,
        yarn_mscale: float = 0.1,
    ) -> None:
        if strategy not in ("auto", "linear", "ntk", "yarn", "longrope"):
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                "Choose from: 'auto', 'linear', 'ntk', 'yarn', 'longrope'."
            )
        self.strategy = strategy
        self.dim = dim
        self.base = base
        self.train_len = train_len
        self.rescale_factors = rescale_factors
        self.yarn_alpha = yarn_alpha
        self.yarn_beta = yarn_beta
        self.yarn_mscale = yarn_mscale

    def _resolve_strategy(self, seq_len: int) -> str:
        """Resolve 'auto' to a concrete strategy based on seq_len."""
        if self.strategy != "auto":
            return self.strategy
        if seq_len <= self.train_len:
            return "standard"
        if seq_len <= 4 * self.train_len:
            return "yarn"
        # Beyond 4x: prefer LongRoPE if factors available, else NTK.
        return "longrope" if self.rescale_factors is not None else "ntk"

    def get_cos_sin(
        self,
        seq_len: int,
        device: torch.device | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (cos, sin) cache for the given sequence length.

        Parameters
        ----------
        seq_len:
            Number of positions to compute embeddings for.
        device:
            Target device.

        Returns
        -------
        (cos, sin), each of shape [seq_len, dim], all values finite.
        """
        resolved = self._resolve_strategy(seq_len)

        if resolved == "standard":
            return ContextWindowExtension._standard_cos_sin(self.dim, self.base, seq_len, device)
        elif resolved == "linear":
            # Build standard cache at train positions, then apply linear scale.
            cos_base, sin_base = ContextWindowExtension._standard_cos_sin(
                self.dim, self.base, seq_len, device
            )
            return ContextWindowExtension.linear_scale(
                cos_base, sin_base, self.train_len, max(seq_len, self.train_len)
            )
        elif resolved == "ntk":
            return ContextWindowExtension.ntk_aware_scale(
                self.dim, self.base, self.train_len, max(seq_len, self.train_len), seq_len, device
            )
        elif resolved == "yarn":
            return ContextWindowExtension.yarn_scale(
                self.dim,
                self.base,
                self.train_len,
                max(seq_len, self.train_len),
                seq_len,
                alpha=self.yarn_alpha,
                beta=self.yarn_beta,
                mscale=self.yarn_mscale,
                device=device,
            )
        elif resolved == "longrope":
            if self.rescale_factors is None:
                # Fall back to NTK-aware when no factors are provided.
                return ContextWindowExtension.ntk_aware_scale(
                    self.dim,
                    self.base,
                    self.train_len,
                    max(seq_len, self.train_len),
                    seq_len,
                    device,
                )
            return ContextWindowExtension.longrope_scale(
                self.dim, self.base, seq_len, self.rescale_factors, device
            )
        else:
            raise RuntimeError(f"Unhandled resolved strategy: {resolved!r}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

CONTEXT_EXTENSION_REGISTRY: dict = {
    "linear": ContextWindowExtension.linear_scale,
    "ntk": ContextWindowExtension.ntk_aware_scale,
    "yarn": ContextWindowExtension.yarn_scale,
    "longrope": ContextWindowExtension.longrope_scale,
}

__all__ = [
    "ContextWindowExtension",
    "DynamicContextScaler",
    "CONTEXT_EXTENSION_REGISTRY",
]
