"""Adaptive KL controller and PPO-specific LR scheduler.

Classes
-------
AdaptiveKLConfig    — dataclass for AdaptiveKLController settings
AdaptiveKLController — InstructGPT-style adaptive β for KL penalty
FixedKLController   — no-op controller that keeps β constant
PPOWarmupScheduler  — linear LR warmup then cosine decay for PPO
"""

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# KL controllers
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveKLConfig:
    target_kl: float = 0.01         # desired KL divergence per step
    initial_kl_coef: float = 0.2    # starting β
    kl_coef_min: float = 0.0        # floor for β
    kl_coef_max: float = 1.0        # ceiling for β
    adaptation_horizon: int = 10    # steps between β updates
    # Multiplier logic (InstructGPT style):
    # if mean_kl > 2 * target_kl:   kl_coef *= 1.5
    # if mean_kl < 0.5 * target_kl: kl_coef *= 0.5
    # else: no change


class AdaptiveKLController:
    """Adjusts the KL penalty coefficient β to track a target KL divergence.

    Every `adaptation_horizon` calls to :meth:`update`, the running mean of
    the supplied KL values is compared against ``cfg.target_kl`` and β is
    adjusted using the InstructGPT multiplier rule.

    Parameters
    ----------
    cfg : AdaptiveKLConfig
        Configuration dataclass.
    """

    def __init__(self, cfg: AdaptiveKLConfig) -> None:
        self.cfg = cfg
        self._kl_coef: float = cfg.initial_kl_coef
        self._step: int = 0
        self._n_updates: int = 0
        self._kl_history: list[float] = []

    @property
    def kl_coef(self) -> float:
        """Current KL penalty coefficient β."""
        return self._kl_coef

    def update(self, kl_value: float) -> float:
        """Record a KL sample; adapt β every `adaptation_horizon` steps.

        Parameters
        ----------
        kl_value : float
            Observed KL divergence for the current step.

        Returns
        -------
        float
            Current ``kl_coef`` after any adaptation.
        """
        self._kl_history.append(kl_value)
        self._step += 1

        if self._step % self.cfg.adaptation_horizon == 0:
            mean_kl = sum(self._kl_history) / len(self._kl_history)
            self._kl_history = []  # reset window

            target = self.cfg.target_kl
            if mean_kl > 2.0 * target:
                new_coef = self._kl_coef * 1.5
            elif mean_kl < 0.5 * target:
                new_coef = self._kl_coef * 0.5
            else:
                new_coef = self._kl_coef

            self._kl_coef = float(
                max(self.cfg.kl_coef_min, min(self.cfg.kl_coef_max, new_coef))
            )
            self._n_updates += 1

        return self._kl_coef

    def get_stats(self) -> dict[str, float]:
        """Return diagnostic statistics.

        Returns
        -------
        dict with keys ``kl_coef``, ``mean_kl``, ``n_updates``, ``step``.
        ``mean_kl`` is the mean of the current (incomplete) window, or 0.0
        if the window is empty.
        """
        mean_kl = (
            sum(self._kl_history) / len(self._kl_history)
            if self._kl_history
            else 0.0
        )
        return {
            "kl_coef": float(self._kl_coef),
            "mean_kl": float(mean_kl),
            "n_updates": int(self._n_updates),
            "step": int(self._step),
        }


class FixedKLController:
    """No adaptation — kl_coef stays constant throughout training.

    Parameters
    ----------
    kl_coef : float
        The fixed KL penalty coefficient.
    """

    def __init__(self, kl_coef: float) -> None:
        self._kl_coef = float(kl_coef)
        self._step: int = 0

    @property
    def kl_coef(self) -> float:
        return self._kl_coef

    def update(self, kl_value: float) -> float:  # noqa: ARG002
        """Accept a KL sample and return the (unchanged) coefficient."""
        self._step += 1
        return self._kl_coef

    def get_stats(self) -> dict[str, float]:
        return {
            "kl_coef": float(self._kl_coef),
            "mean_kl": 0.0,
            "n_updates": 0,
            "step": int(self._step),
        }


# ---------------------------------------------------------------------------
# PPO LR scheduler
# ---------------------------------------------------------------------------

class PPOWarmupScheduler:
    """Linear LR warmup then cosine decay, specialised for PPO's update cadence.

    Each call to :meth:`step` advances the internal counter by one and sets
    the learning rate on every param group of the wrapped optimizer.

    Warmup phase  (step < n_warmup_steps):
        lr = base_lr * step / n_warmup_steps

    Cosine decay  (step >= n_warmup_steps):
        progress = (step - n_warmup) / (n_total - n_warmup)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose param groups will be updated.
    n_warmup_steps : int
        Number of linear warmup steps.
    n_total_steps : int
        Total training steps (warmup + cosine decay).
    min_lr_ratio : float
        ``min_lr = base_lr * min_lr_ratio``.  Defaults to 0.1.
    """

    def __init__(
        self,
        optimizer,
        n_warmup_steps: int,
        n_total_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        if n_total_steps <= n_warmup_steps:
            raise ValueError(
                f"n_total_steps ({n_total_steps}) must be greater than "
                f"n_warmup_steps ({n_warmup_steps})"
            )
        self.optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_total_steps = n_total_steps
        self.min_lr_ratio = min_lr_ratio

        # Record the base LR from each param group at construction time
        self._base_lrs: list[float] = [
            float(pg["lr"]) for pg in optimizer.param_groups
        ]
        self._step: int = 0

    def _compute_lr(self, step: int, base_lr: float) -> float:
        min_lr = base_lr * self.min_lr_ratio
        if step < self.n_warmup_steps:
            # Linear warmup — avoid division by zero when n_warmup_steps == 0
            if self.n_warmup_steps == 0:
                return base_lr
            return base_lr * step / self.n_warmup_steps
        # Cosine decay
        decay_steps = self.n_total_steps - self.n_warmup_steps
        progress = (step - self.n_warmup_steps) / decay_steps
        progress = min(progress, 1.0)
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> float:
        """Advance the step counter, update optimizer LRs, return current LR.

        Returns the LR of the **first** param group (representative value).
        """
        self._step += 1
        lrs = []
        for pg, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            lr = self._compute_lr(self._step, base_lr)
            pg["lr"] = lr
            lrs.append(lr)
        return lrs[0]

    def get_lr(self) -> float:
        """Return the current LR of the first param group without stepping."""
        if not self._base_lrs:
            return 0.0
        return self._compute_lr(self._step, self._base_lrs[0])
