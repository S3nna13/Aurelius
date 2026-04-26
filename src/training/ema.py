"""Exponential Moving Average (EMA) model averaging for Aurelius LLM.

Implements:
- ModelEMA: EMA with warmup bias correction, copy_to/store/restore, foreach updates,
  multi-GPU (DDP) support, and state_dict checkpointing.
- StochasticWeightAveraging: Uniform averaging over periodic checkpoints (SWA).
"""

from __future__ import annotations

import copy
from collections.abc import Generator, Iterator
from contextlib import contextmanager

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_params(model: nn.Module) -> list[torch.Tensor]:
    """Return parameters from the underlying module (unwrap DDP if needed)."""
    m = getattr(model, "module", model)  # handle DDP wrapper
    return list(m.parameters())


def _get_named_params(model: nn.Module) -> Iterator:
    m = getattr(model, "module", model)
    return m.named_parameters()


# ---------------------------------------------------------------------------
# ModelEMA
# ---------------------------------------------------------------------------


class ModelEMA:
    """Exponential Moving Average of model parameters.

    Shadow parameters are updated after each optimizer step:
        θ_ema = d * θ_ema + (1 - d) * θ_current

    where d = effective_decay incorporates a warmup correction
    (analogous to AdamW bias correction) that prevents the EMA from
    being dragged toward zero during the first few steps:
        effective_decay = min(decay, (1 + step) / (10 + step))

    Usage::

        ema = ModelEMA(model, decay=0.9999)
        # inside training loop, after optimizer.step():
        ema.update(model)
        # for validation:
        with ema.average_parameters(model):
            val_loss = evaluate(model)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
        device: torch.device | None = None,
        foreach: bool = True,
    ) -> None:
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.foreach = foreach
        self._step: int = 0

        # Determine device: use model's first param device if not specified
        if device is None:
            try:
                device = next(_get_params(model)[0:1].__iter__()).device
            except (StopIteration, IndexError):
                device = torch.device("cpu")
        self.device = device

        # Build shadow parameter list (float32 clones on target device)
        m = getattr(model, "module", model)
        self.shadow_params: list[torch.Tensor] = [
            p.detach().clone().float().to(self.device) for p in m.parameters()
        ]

        # Backup for store/restore
        self._backup: list[torch.Tensor] | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def effective_decay(self) -> float:
        """Decay after warmup correction: min(decay, (1+step)/(10+step))."""
        warmup_decay = (1.0 + self._step) / (10.0 + self._step)
        return min(self.decay, warmup_decay)

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, model: nn.Module) -> None:
        """Update shadow params from current model params.

        Call this *after* optimizer.step() each training step.
        """
        d = self.effective_decay
        one_minus_d = 1.0 - d

        model_params = _get_params(model)

        if self.foreach:
            # Collect float versions of current model params on the shadow device
            current = [p.detach().float().to(self.device) for p in model_params]
            # In-place: shadow = shadow * d + current * (1 - d)
            torch._foreach_mul_(self.shadow_params, d)
            torch._foreach_add_(self.shadow_params, current, alpha=one_minus_d)
        else:
            with torch.no_grad():
                for shadow, param in zip(self.shadow_params, model_params):
                    cur = param.detach().float().to(self.device)
                    shadow.mul_(d).add_(cur, alpha=one_minus_d)

        self._step += 1

    # ------------------------------------------------------------------
    # copy_to / store / restore
    # ------------------------------------------------------------------

    def copy_to(self, model: nn.Module) -> None:
        """Copy EMA shadow params INTO the model (for evaluation)."""
        m = getattr(model, "module", model)
        with torch.no_grad():
            for shadow, param in zip(self.shadow_params, m.parameters()):
                param.copy_(shadow.to(param.dtype).to(param.device))

    def store(self, model: nn.Module) -> None:
        """Save current model params so they can be restored after copy_to()."""
        m = getattr(model, "module", model)
        self._backup = [p.detach().clone() for p in m.parameters()]

    def restore(self, model: nn.Module) -> None:
        """Restore previously stored model params (undo copy_to)."""
        if self._backup is None:
            raise RuntimeError("store() must be called before restore().")
        m = getattr(model, "module", model)
        with torch.no_grad():
            for param, backup in zip(m.parameters(), self._backup):
                param.copy_(backup)
        self._backup = None

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Generator[None, None, None]:
        """Context manager: temporarily swap in EMA weights, then restore."""
        self.store(model)
        self.copy_to(model)
        try:
            yield
        finally:
            self.restore(model)

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return serializable state dict suitable for torch.save."""
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "foreach": self.foreach,
            "step": self._step,
            "device": str(self.device),
            "shadow_params": [p.cpu() for p in self.shadow_params],
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load from a state dict (as returned by state_dict())."""
        self.decay = state_dict["decay"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.foreach = state_dict["foreach"]
        self._step = state_dict["step"]
        self.device = torch.device(state_dict["device"])
        self.shadow_params = [p.to(self.device) for p in state_dict["shadow_params"]]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_model(self) -> nn.Module:
        """Return a *copy* of the model with EMA weights loaded.

        The original model is NOT modified.
        """
        raise NotImplementedError(
            "get_model() requires the original model architecture to copy. "
            "Pass the model: use ema.copy_to(copy.deepcopy(model)) instead."
        )

    def get_ema_model(self, model: nn.Module) -> nn.Module:
        """Return a deep-copy of model with EMA weights loaded.

        The original model is NOT modified.
        """
        ema_model = copy.deepcopy(model)
        self.copy_to(ema_model)
        return ema_model


# ---------------------------------------------------------------------------
# StochasticWeightAveraging
# ---------------------------------------------------------------------------


class StochasticWeightAveraging:
    """SWA: uniform average of model weights taken at regular intervals.

    Unlike EMA (which uses exponential decay), SWA maintains a simple
    running mean of N uniformly-weighted checkpoints.

    Averaging starts at *swa_start_step* and a snapshot is taken every
    *swa_freq* steps thereafter.

    Usage::

        swa = StochasticWeightAveraging(model, swa_start_step=1000, swa_freq=100)
        # inside training loop:
        if swa.update(model, step):
            # a snapshot was taken this step
            pass
        # after training:
        swa_model = swa.get_swa_model()
    """

    def __init__(
        self,
        model: nn.Module,
        swa_start_step: int = 1000,
        swa_freq: int = 100,
    ) -> None:
        self.swa_start_step = swa_start_step
        self.swa_freq = swa_freq
        self._n_averaged: int = 0

        # Running average stored as float32 tensors (CPU)
        m = getattr(model, "module", model)
        self._avg_params: list[torch.Tensor] | None = None
        self._model_ref = m  # keep arch reference for get_swa_model()

    @property
    def n_averaged(self) -> int:
        """Number of checkpoints averaged so far."""
        return self._n_averaged

    def update(self, model: nn.Module, step: int) -> bool:
        """Conditionally average in current weights.

        Returns True if a SWA step was taken (snapshot captured), False otherwise.
        """
        if step < self.swa_start_step:
            return False

        steps_since_start = step - self.swa_start_step
        if steps_since_start % self.swa_freq != 0:
            return False

        # Capture snapshot
        m = getattr(model, "module", model)
        n = self._n_averaged

        if self._avg_params is None:
            # First checkpoint: initialize the running mean
            self._avg_params = [p.detach().clone().float().cpu() for p in m.parameters()]
        else:
            # Running mean: avg = (avg * n + p) / (n + 1)
            with torch.no_grad():
                for avg_p, param in zip(self._avg_params, m.parameters()):
                    cur = param.detach().float().cpu()
                    avg_p.mul_(n).add_(cur).div_(n + 1)

        self._n_averaged += 1
        return True

    def get_swa_model(self) -> nn.Module:
        """Return a new model instance with SWA-averaged weights."""
        if self._avg_params is None:
            raise RuntimeError("No SWA updates have been performed yet.")

        swa_model = copy.deepcopy(self._model_ref)
        with torch.no_grad():
            for param, avg_p in zip(swa_model.parameters(), self._avg_params):
                param.copy_(avg_p.to(param.dtype).to(param.device))
        return swa_model
