"""Loss spike detection and automatic checkpoint-rollback recovery.

Usage::

    config = SpikeConfig()
    recovery = LossSpikeRecovery(model, optimizer, config)

    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        optimizer.step()

        info = recovery.step(loss.item(), grad_norm, step)
        if info["spike"]:
            # optimizer.step() already happened — recovery rolls back weights
            print(f"Spike at step {step}, rolled back to step {info['step_restored_to']}")

        if recovery.should_save_checkpoint(step, loss.item()):
            recovery.checkpoint_buffer.save(model, optimizer, step)
"""

from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpikeConfig:
    window_size: int = 20              # rolling window for baseline loss
    spike_threshold: float = 2.5      # z-score threshold to declare spike
    grad_norm_limit: float = 100.0    # hard gradient norm limit (also a spike signal)
    min_steps_before_check: int = 50  # don't check before warmup
    cooldown_steps: int = 10          # steps to skip after recovery
    max_recoveries: int = 5           # abort training after this many recoveries


# ---------------------------------------------------------------------------
# LossHistory
# ---------------------------------------------------------------------------


class LossHistory:
    """Rolling window statistics for z-score-based spike detection."""

    def __init__(self, window: int) -> None:
        self._window: deque[float] = deque(maxlen=window)

    def update(self, loss: float) -> None:
        self._window.append(float(loss))

    @property
    def mean(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    @property
    def std(self) -> float:
        if len(self._window) < 2:
            return 0.0
        m = self.mean
        variance = sum((x - m) ** 2 for x in self._window) / len(self._window)
        return math.sqrt(variance)

    def z_score(self, loss: float) -> float:
        """Return z-score of *loss* relative to the rolling window.

        When std is zero (all history values are identical), returns a very
        large positive value if the candidate loss is meaningfully larger than
        the mean, and 0.0 otherwise.  This ensures that a sudden spike into
        a completely flat loss baseline is still detected.
        """
        s = self.std
        if s == 0.0:
            m = self.mean
            if m == 0.0:
                return 0.0
            # Use relative deviation so a 10× jump always signals a spike
            rel_dev = (float(loss) - m) / abs(m)
            # Map relative deviation into a z-score-like scale
            return rel_dev * 10.0 if rel_dev > 0 else 0.0
        return (float(loss) - self.mean) / s

    def is_spike(self, loss: float, threshold: float) -> bool:
        """Return True if *loss* exceeds *threshold* standard deviations above mean."""
        return self.z_score(loss) > threshold

    def __len__(self) -> int:
        return len(self._window)


# ---------------------------------------------------------------------------
# CheckpointBuffer
# ---------------------------------------------------------------------------


class CheckpointBuffer:
    """In-memory ring-buffer of the last *capacity* good checkpoints.

    Uses ``copy.deepcopy`` on ``state_dict`` snapshots — no file I/O.
    """

    def __init__(self, capacity: int = 3) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity = capacity
        # Each entry: (step, model_state, optimizer_state)
        self._buffer: deque[tuple[int, dict, dict]] = deque(maxlen=capacity)

    def save(self, model: nn.Module, optimizer: Any, step: int) -> None:
        """Snapshot model and optimizer state at *step*."""
        model_state = copy.deepcopy(model.state_dict())
        opt_state = copy.deepcopy(optimizer.state_dict())
        self._buffer.append((step, model_state, opt_state))

    def restore_latest(self, model: nn.Module, optimizer: Any) -> int:
        """Restore the most recent checkpoint.

        Returns
        -------
        int
            The training step that was restored.

        Raises
        ------
        RuntimeError
            If the buffer is empty.
        """
        if not self._buffer:
            raise RuntimeError("CheckpointBuffer is empty — cannot restore.")
        step, model_state, opt_state = self._buffer[-1]
        model.load_state_dict(model_state)
        optimizer.load_state_dict(opt_state)
        return step

    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# LossSpikeRecovery
# ---------------------------------------------------------------------------


class LossSpikeRecovery:
    """Integrates with a training loop to detect loss spikes and auto-recover.

    The caller is responsible for calling ``optimizer.step()``; this class
    must NOT call it internally.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Any,
        config: SpikeConfig,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config

        self._loss_history = LossHistory(config.window_size)
        self.checkpoint_buffer = CheckpointBuffer(capacity=3)

        self._recovery_count: int = 0
        self._cooldown_remaining: int = 0
        self._aborted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        loss: float,
        grad_norm: float,
        step: int,
    ) -> dict:
        """Call after each optimizer step.

        Parameters
        ----------
        loss:
            Scalar loss value for this step.
        grad_norm:
            Global gradient norm for this step (computed before clipping).
        step:
            Current training step (0-indexed or 1-indexed — caller's choice,
            used only for reporting).

        Returns
        -------
        dict with keys:
            ``spike``          – bool, whether a spike was detected
            ``recovered``      – bool, whether a rollback was performed
            ``n_recoveries``   – int, cumulative recovery count
            ``z_score``        – float, z-score of this loss
            ``step_restored_to`` – int or None, step number of restored ckpt
        """
        if self._aborted:
            raise RuntimeError(
                f"Training aborted: max_recoveries ({self.config.max_recoveries}) exceeded."
            )

        result: dict = {
            "spike": False,
            "recovered": False,
            "n_recoveries": self._recovery_count,
            "z_score": 0.0,
            "step_restored_to": None,
        }

        # Decrement cooldown counter
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            # Still update history with current loss so baseline tracks reality
            self._loss_history.update(loss)
            return result

        # Compute z-score before adding current loss to history
        z = self._loss_history.z_score(loss)
        result["z_score"] = z

        # Determine if this is a spike
        warmup_done = step >= self.config.min_steps_before_check
        loss_spike = (
            warmup_done
            and len(self._loss_history) >= 2
            and self._loss_history.is_spike(loss, self.config.spike_threshold)
        )
        grad_spike = grad_norm > self.config.grad_norm_limit

        is_spike = loss_spike or grad_spike
        result["spike"] = is_spike

        if is_spike:
            self._recovery_count += 1
            result["n_recoveries"] = self._recovery_count

            if self._recovery_count > self.config.max_recoveries:
                self._aborted = True
                raise RuntimeError(
                    f"Training aborted: exceeded max_recoveries "
                    f"({self.config.max_recoveries})."
                )

            # Attempt rollback
            if len(self.checkpoint_buffer) > 0:
                restored_step = self.checkpoint_buffer.restore_latest(
                    self.model, self.optimizer
                )
                result["recovered"] = True
                result["step_restored_to"] = restored_step
            # Even without a checkpoint we count it as a recovery event

            self._cooldown_remaining = self.config.cooldown_steps
            # Don't add the spike loss to history
        else:
            self._loss_history.update(loss)

        return result

    def should_save_checkpoint(self, step: int, loss: float) -> bool:
        """Return True if this step is a good candidate for snapshotting.

        Conditions:
        - Past the warmup period.
        - Not in cooldown after a recovery.
        - Current loss is not a spike (below threshold).
        """
        if step < self.config.min_steps_before_check:
            return False
        if self._cooldown_remaining > 0:
            return False
        # If we have enough history, reject spike losses
        if (
            len(self._loss_history) >= 2
            and self._loss_history.is_spike(loss, self.config.spike_threshold)
        ):
            return False
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def recovery_count(self) -> int:
        """Total number of recoveries performed so far."""
        return self._recovery_count

    @property
    def is_stable(self) -> bool:
        """True if no spikes have occurred and training is not aborted."""
        return self._recovery_count == 0 and not self._aborted

    @property
    def loss_history(self) -> LossHistory:
        """Expose the internal LossHistory (useful for adaptive clipping)."""
        return self._loss_history


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------


def adaptive_grad_clip(
    grad_norm: float,
    history: LossHistory,
    multiplier: float = 3.0,
) -> float:
    """Compute a dynamic gradient clip value based on recent loss-window statistics.

    The clip value is ``history.mean * multiplier`` when the mean is positive,
    otherwise it falls back to ``grad_norm`` (no clipping needed).

    Parameters
    ----------
    grad_norm:
        The current raw gradient norm.
    history:
        A ``LossHistory`` instance tracking recent loss values (used as a proxy
        for the expected gradient scale).
    multiplier:
        Scale factor applied to the history mean to produce the clip threshold.

    Returns
    -------
    float
        A clip value.  When ``grad_norm`` is abnormally large relative to
        ``history.mean``, the returned value will be smaller than ``grad_norm``.
    """
    if len(history) < 2 or history.mean <= 0.0:
        return float(grad_norm)
    clip_value = history.mean * multiplier
    return float(clip_value)
