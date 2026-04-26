"""Spectral Gradient Filtering: FFT-based low-frequency gradient amplification.

Instead of EMA, this maintains a fixed-length history of past gradients and uses
the Fast Fourier Transform to separate low-frequency (slow, generalizing) components
from high-frequency (fast, memorizing) components, then amplifies the slow part.

This is a complementary approach to GrokFast (grokfast.py), which uses EMA.
Where EMA approximates a low-pass filter exponentially, this implementation applies
an exact brickwall low-pass filter in the frequency domain via rfft/irfft.

Reference inspiration: Lee et al. 2024, "GrokFast: Accelerated Grokking by
Amplifying Slow Gradients" arXiv:2405.20233
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SpectralGradConfig:
    """Configuration for spectral gradient filtering.

    Attributes:
        window: Number of gradient history steps to retain (W). The FFT
            is computed over this window; larger windows give finer frequency
            resolution but use more memory.
        cutoff_freq: Fraction of frequency bins to keep (0–1).  A value of
            0.1 keeps only the lowest 10 % of frequency bins (slow gradients).
            1.0 keeps all bins (no filtering). 0.0 zeros everything out.
        alpha: Amplification factor for the slow (low-frequency) component.
            Amplified gradient = g_t + alpha * slow_component.
        param_filter: Which parameters to apply filtering to.
            "all"             – every parameter with a gradient.
            "weights_only"    – parameters whose name ends with ".weight".
            "embeddings_only" – parameters whose name contains "embed".
    """

    window: int = 32
    cutoff_freq: float = 0.1
    alpha: float = 2.0
    param_filter: str = "all"  # "all" | "weights_only" | "embeddings_only"


class GradientHistory:
    """Per-parameter rolling gradient history for spectral filtering.

    Stores up to ``maxlen`` consecutive gradient snapshots and exposes a
    method to extract the slow (low-frequency) component via FFT.
    """

    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self._maxlen = maxlen
        self._buf: deque[torch.Tensor] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update(self, grad: torch.Tensor) -> None:
        """Append a gradient snapshot to the history (CPU clone to save VRAM)."""
        self._buf.append(grad.detach().cpu().clone())

    def get_slow_component(self, cutoff_freq: float) -> torch.Tensor | None:
        """Return the slow (low-frequency) component of the gradient history.

        Steps:
          1. Stack the W stored gradients along a new leading time axis.
          2. Apply torch.fft.rfft along the time axis.
          3. Zero out all frequency bins above ``cutoff_freq * W``.
          4. Reconstruct via torch.fft.irfft and return the *last* time step.

        Returns:
            A tensor with the same shape as the stored gradients, on the same
            device as the stored gradients (CPU), or ``None`` if the history
            is not yet full.
        """
        if len(self._buf) < self._maxlen:
            return None

        # G shape: (W, *param_shape)
        G = torch.stack(list(self._buf), dim=0).float()
        W = G.shape[0]

        # FFT along the time axis (dim=0) → (W//2+1, *param_shape)
        G_fft = torch.fft.rfft(G, dim=0)

        # Determine the number of frequency bins to zero out.
        # rfft of length W produces W//2+1 bins (index 0 … W//2).
        n_bins = G_fft.shape[0]  # W//2 + 1
        cutoff_idx = max(1, int(round(cutoff_freq * n_bins)))
        # Zero out everything above cutoff_idx
        if cutoff_idx < n_bins:
            G_fft[cutoff_idx:] = 0.0

        # Reconstruct; specify n=W so irfft returns exactly W time steps
        slow_G = torch.fft.irfft(G_fft, n=W, dim=0)  # (W, *param_shape)

        # Return the reconstructed gradient at the latest time step
        return slow_G[-1]

    def __len__(self) -> int:
        return len(self._buf)


def _param_is_selected(name: str, param_filter: str) -> bool:
    """Return True when *name* should be filtered under *param_filter*."""
    if param_filter == "all":
        return True
    if param_filter == "weights_only":
        return name.endswith(".weight")
    if param_filter == "embeddings_only":
        return "embed" in name
    raise ValueError(
        f"Unknown param_filter '{param_filter}'. "
        "Choose from 'all', 'weights_only', 'embeddings_only'."
    )


class SpectralGradFilter:
    """Optimizer wrapper that amplifies low-frequency (slow) gradient components.

    Wrap any ``torch.optim.Optimizer`` with this class to apply spectral
    gradient filtering before each optimizer step.

    Example::

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = SpectralGradConfig(window=32, cutoff_freq=0.1, alpha=2.0)
        filtered_opt = SpectralGradFilter(optimizer, config)

        # Training loop
        loss.backward()
        filtered_opt.step()   # amplifies slow grads, then calls optimizer.step
        filtered_opt.zero_grad()
    """

    def __init__(self, optimizer: torch.optim.Optimizer, config: SpectralGradConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        # Map from parameter id → GradientHistory
        self._histories: dict[int, GradientHistory] = {}
        # Map from parameter id → parameter name (best-effort, filled in step)
        self._param_names: dict[int, str] = {}
        # Collect all parameters across all param groups
        self._all_params: list[tuple[int, torch.nn.Parameter]] = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                pid = id(p)
                self._all_params.append((pid, p))
                self._histories[pid] = GradientHistory(maxlen=config.window)
                # Name is unknown at construction; will be inferred or left empty
                self._param_names[pid] = ""

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self) -> None:
        """Amplify slow gradient components, then call optimizer.step()."""
        cfg = self.config
        for pid, p in self._all_params:
            if p.grad is None:
                continue
            name = self._param_names.get(pid, "")
            if not _param_is_selected(name, cfg.param_filter):
                continue

            g = p.grad.data
            hist = self._histories[pid]
            hist.update(g)

            slow = hist.get_slow_component(cfg.cutoff_freq)
            if slow is not None:
                # Move slow component to the same device/dtype as the gradient
                slow = slow.to(device=g.device, dtype=g.dtype)
                p.grad.data = g + cfg.alpha * slow

        self.optimizer.step()

    # ------------------------------------------------------------------
    # Delegation helpers
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Delegate zero_grad to the wrapped optimizer."""
        self.optimizer.zero_grad(set_to_none=set_to_none)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serializable state including optimizer state and gradient histories."""
        histories_state = {}
        for pid, hist in self._histories.items():
            histories_state[pid] = {
                "maxlen": hist._maxlen,
                # Store as list of tensors (CPU, float32) for portability
                "buf": [t.clone() for t in hist._buf],
            }
        return {
            "optimizer": self.optimizer.state_dict(),
            "config": {
                "window": self.config.window,
                "cutoff_freq": self.config.cutoff_freq,
                "alpha": self.config.alpha,
                "param_filter": self.config.param_filter,
            },
            "histories": histories_state,
            # Preserve id→name mapping (useful for debugging)
            "param_names": dict(self._param_names),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from a dict produced by :meth:`state_dict`."""
        self.optimizer.load_state_dict(state["optimizer"])

        cfg_d = state.get("config", {})
        self.config = SpectralGradConfig(
            window=cfg_d.get("window", self.config.window),
            cutoff_freq=cfg_d.get("cutoff_freq", self.config.cutoff_freq),
            alpha=cfg_d.get("alpha", self.config.alpha),
            param_filter=cfg_d.get("param_filter", self.config.param_filter),
        )

        self._param_names.update(state.get("param_names", {}))

        histories_state = state.get("histories", {})
        for pid_key, h_state in histories_state.items():
            pid = int(pid_key)
            if pid not in self._histories:
                # Parameter not in current model; skip
                continue
            hist = GradientHistory(maxlen=h_state["maxlen"])
            for t in h_state["buf"]:
                hist._buf.append(t.clone())
            self._histories[pid] = hist

    # ------------------------------------------------------------------
    # Convenience: register param names from a model
    # ------------------------------------------------------------------

    def register_model(self, model: nn.Module) -> SpectralGradFilter:
        """Associate parameter names from *model* with tracked parameters.

        Call this after construction to enable ``param_filter`` modes other
        than ``"all"``.  Returns self for chaining.
        """
        name_map = {id(p): name for name, p in model.named_parameters()}
        for pid in list(self._param_names.keys()):
            if pid in name_map:
                self._param_names[pid] = name_map[pid]
        return self


# ---------------------------------------------------------------------------
# Stateless functional API
# ---------------------------------------------------------------------------


def apply_spectral_filter(
    model: nn.Module,
    grads: dict[str, torch.Tensor],
    histories: dict[str, GradientHistory],
    config: SpectralGradConfig,
) -> dict[str, torch.Tensor]:
    """Stateless spectral gradient filter operating on explicit grad dicts.

    This does NOT modify ``model`` or ``grads`` in-place; it returns a new
    dict with amplified gradients.

    Args:
        model: The model (used only for ``param_filter`` name matching).
        grads: Mapping from parameter name → gradient tensor.
        histories: Mapping from parameter name → :class:`GradientHistory`.
            Histories are updated in-place as a side effect.
        config: Filter configuration.

    Returns:
        A dict with the same keys as *grads*, where each gradient has been
        (potentially) amplified by its slow component.
    """
    amplified: dict[str, torch.Tensor] = {}

    for name, g in grads.items():
        if not _param_is_selected(name, config.param_filter):
            amplified[name] = g
            continue

        if name not in histories:
            histories[name] = GradientHistory(maxlen=config.window)

        hist = histories[name]
        hist.update(g)

        slow = hist.get_slow_component(config.cutoff_freq)
        if slow is not None:
            slow = slow.to(device=g.device, dtype=g.dtype)
            amplified[name] = g + config.alpha * slow
        else:
            amplified[name] = g

    return amplified
