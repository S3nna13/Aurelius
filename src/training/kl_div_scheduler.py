"""KL Divergence schedulers for RLHF/DPO training.

Classes
-------
KLSchedulerConfig       — dataclass for create_kl_scheduler factory
AdaptiveKLScheduler     — adaptively adjusts beta based on observed KL
CyclicKLScheduler       — cycles beta between base_beta and max_beta
WarmupKLScheduler       — warms up beta from start_beta to end_beta, then decays

Functions
---------
create_kl_scheduler     — factory that builds the right scheduler from a config
"""

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class KLSchedulerConfig:
    """Configuration for :func:`create_kl_scheduler`.

    Parameters
    ----------
    scheduler_type : str
        One of ``'adaptive'``, ``'cyclic'``, or ``'warmup'``.
    target_kl : float
        Desired KL divergence (used by adaptive scheduler).
    initial_beta : float
        Starting beta value.
    warmup_steps : int
        Warmup steps for warmup scheduler; also used as ``kl_horizon`` for
        adaptive scheduler when not specified separately.
    min_beta : float
        Floor for beta (adaptive scheduler).
    max_beta : float
        Ceiling for beta (adaptive/cyclic schedulers).
    """

    scheduler_type: str = "adaptive"
    target_kl: float = 0.02
    initial_beta: float = 0.2
    warmup_steps: int = 100
    min_beta: float = 0.001
    max_beta: float = 10.0


# ---------------------------------------------------------------------------
# AdaptiveKLScheduler
# ---------------------------------------------------------------------------


class AdaptiveKLScheduler:
    """Adjusts the KL penalty coefficient (beta) to track a target KL value.

    Every ``kl_horizon`` calls to :meth:`update`, the running mean of the
    supplied KL values is compared against ``target_kl`` and beta is
    multiplied or divided by ``adjustment_factor``.

    Rules (Ziegler et al. 2019 / InstructGPT style):
    - mean_kl > target_kl * 1.5 → beta *= adjustment_factor
    - mean_kl < target_kl / 1.5 → beta /= adjustment_factor
    - otherwise                  → beta unchanged
    Beta is clipped to ``[min_beta, max_beta]`` after every update.

    Parameters
    ----------
    target_kl : float
        Desired KL divergence.
    initial_beta : float
        Starting beta.
    kl_horizon : int
        Number of KL samples collected before each adaptation.
    adjustment_factor : float
        Multiplicative factor applied when beta needs changing.
    min_beta : float
        Lower bound for beta.
    max_beta : float
        Upper bound for beta.
    """

    def __init__(
        self,
        target_kl: float = 0.02,
        initial_beta: float = 0.2,
        kl_horizon: int = 10,
        adjustment_factor: float = 2.0,
        min_beta: float = 0.001,
        max_beta: float = 10.0,
    ) -> None:
        self.target_kl = target_kl
        self.initial_beta = initial_beta
        self.kl_horizon = kl_horizon
        self.adjustment_factor = adjustment_factor
        self.min_beta = min_beta
        self.max_beta = max_beta

        self._beta: float = initial_beta
        self._kl_history: list[float] = []
        self._step: int = 0
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, kl_divergence: float) -> float:
        """Record a new KL sample; adapt beta every ``kl_horizon`` steps.

        Parameters
        ----------
        kl_divergence : float
            Observed KL divergence for the current step.

        Returns
        -------
        float
            Current beta after any adaptation.
        """
        self._kl_history.append(float(kl_divergence))
        self._step += 1

        if self._step % self.kl_horizon == 0:
            mean_kl = sum(self._kl_history) / len(self._kl_history)
            self._kl_history = []  # reset window

            if mean_kl > self.target_kl * 1.5:
                new_beta = self._beta * self.adjustment_factor
            elif mean_kl < self.target_kl / 1.5:
                new_beta = self._beta / self.adjustment_factor
            else:
                new_beta = self._beta

            self._beta = float(max(self.min_beta, min(self.max_beta, new_beta)))
            self._n_updates += 1

        return self._beta

    def get_beta(self) -> float:
        """Return the current beta without advancing the step counter."""
        return self._beta

    def get_stats(self) -> dict:
        """Return a diagnostic snapshot.

        Returns
        -------
        dict with keys ``'beta'``, ``'mean_kl'``, ``'target_kl'``,
        ``'n_updates'``.
        """
        mean_kl = sum(self._kl_history) / len(self._kl_history) if self._kl_history else 0.0
        return {
            "beta": float(self._beta),
            "mean_kl": float(mean_kl),
            "target_kl": float(self.target_kl),
            "n_updates": int(self._n_updates),
        }

    def reset(self) -> None:
        """Restore all state to initial values."""
        self._beta = self.initial_beta
        self._kl_history = []
        self._step = 0
        self._n_updates = 0


# ---------------------------------------------------------------------------
# CyclicKLScheduler
# ---------------------------------------------------------------------------


class CyclicKLScheduler:
    """Cycles the KL penalty coefficient between ``base_beta`` and ``max_beta``.

    Parameters
    ----------
    base_beta : float
        Minimum beta value (bottom of the cycle).
    max_beta : float
        Maximum beta value (top of the cycle).
    cycle_steps : int
        Total number of steps in one full cycle.
    mode : str
        Shape of the cycle.  One of ``'triangular'``, ``'cosine'``, or
        ``'step'``.
    """

    _VALID_MODES = {"triangular", "cosine", "step"}

    def __init__(
        self,
        base_beta: float = 0.1,
        max_beta: float = 1.0,
        cycle_steps: int = 100,
        mode: str = "triangular",
    ) -> None:
        if mode not in self._VALID_MODES:
            raise ValueError(f"mode must be one of {self._VALID_MODES}, got '{mode}'")
        self.base_beta = base_beta
        self.max_beta = max_beta
        self.cycle_steps = cycle_steps
        self.mode = mode

        self._step: int = 0
        self._beta: float = base_beta

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_beta(self, step: int) -> float:
        cycle_pos = step % self.cycle_steps  # position within cycle [0, cycle_steps)
        ratio = cycle_pos / self.cycle_steps  # ∈ [0, 1)
        beta_range = self.max_beta - self.base_beta

        if self.mode == "triangular":
            # Ramp up for first half, ramp down for second half
            if ratio < 0.5:
                scale = ratio * 2.0
            else:
                scale = (1.0 - ratio) * 2.0
            return self.base_beta + beta_range * scale

        elif self.mode == "cosine":
            # Cosine shape: starts at base, peaks at half cycle, returns to base
            scale = 0.5 * (1.0 - math.cos(2.0 * math.pi * ratio))
            return self.base_beta + beta_range * scale

        else:  # 'step'
            # First half at max_beta, second half at base_beta
            if ratio < 0.5:
                return self.max_beta
            return self.base_beta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> float:
        """Advance one step and return the current beta."""
        self._beta = self._compute_beta(self._step)
        self._step += 1
        return self._beta

    def get_beta(self) -> float:
        """Return the current beta without advancing the step counter."""
        return self._beta


# ---------------------------------------------------------------------------
# WarmupKLScheduler
# ---------------------------------------------------------------------------


class WarmupKLScheduler:
    """Warms beta up from ``start_beta`` to ``end_beta``, then optionally decays.

    Warmup phase  (step < warmup_steps):
        beta = start_beta + (end_beta - start_beta) * step / warmup_steps

    Post-warmup decay (step >= warmup_steps):
        - ``'constant'``: beta stays at end_beta
        - ``'linear'``: linearly decays to 0 over ``decay_steps``
        - ``'cosine'``: cosine decay to 0 over ``decay_steps``

    Parameters
    ----------
    start_beta : float
        Beta at step 0.
    end_beta : float
        Beta at the end of warmup.
    warmup_steps : int
        Number of linear warmup steps.
    decay_steps : int
        Number of decay steps after warmup.
    decay_type : str
        One of ``'linear'``, ``'cosine'``, ``'constant'``.
    """

    _VALID_DECAY_TYPES = {"linear", "cosine", "constant"}

    def __init__(
        self,
        start_beta: float = 0.0,
        end_beta: float = 0.1,
        warmup_steps: int = 100,
        decay_steps: int = 1000,
        decay_type: str = "linear",
    ) -> None:
        if decay_type not in self._VALID_DECAY_TYPES:
            raise ValueError(
                f"decay_type must be one of {self._VALID_DECAY_TYPES}, got '{decay_type}'"
            )
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.decay_type = decay_type

        self._step: int = 0
        self._beta: float = start_beta

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_beta(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            # Linear warmup: step 0 → start_beta, step (warmup_steps-1) → end_beta
            if self.warmup_steps == 1:
                return self.end_beta
            return self.start_beta + (self.end_beta - self.start_beta) * (
                step / (self.warmup_steps - 1)
            )

        # Post-warmup phase
        decay_step = step - self.warmup_steps

        if self.decay_type == "constant" or self.decay_steps <= 0:
            return self.end_beta

        progress = min(decay_step / self.decay_steps, 1.0)

        if self.decay_type == "linear":
            return self.end_beta * (1.0 - progress)

        else:  # 'cosine'
            return self.end_beta * 0.5 * (1.0 + math.cos(math.pi * progress))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> float:
        """Advance one step and return the current beta."""
        self._beta = self._compute_beta(self._step)
        self._step += 1
        return self._beta

    def get_beta(self) -> float:
        """Return the current beta without advancing the step counter."""
        return self._beta

    def get_schedule(self, n_steps: int) -> list[float]:
        """Return the full beta schedule for ``n_steps`` steps without
        modifying the scheduler's internal state.

        Parameters
        ----------
        n_steps : int
            Number of steps to project.

        Returns
        -------
        List[float]
            Beta values for steps 0, 1, ..., n_steps-1.
        """
        return [self._compute_beta(s) for s in range(n_steps)]


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_kl_scheduler(
    config: KLSchedulerConfig,
) -> AdaptiveKLScheduler | CyclicKLScheduler | WarmupKLScheduler:
    """Instantiate the appropriate KL scheduler from a :class:`KLSchedulerConfig`.

    Parameters
    ----------
    config : KLSchedulerConfig
        Configuration object specifying ``scheduler_type`` and parameters.

    Returns
    -------
    AdaptiveKLScheduler | CyclicKLScheduler | WarmupKLScheduler

    Raises
    ------
    ValueError
        If ``config.scheduler_type`` is not recognised.
    """
    stype = config.scheduler_type.lower()

    if stype == "adaptive":
        return AdaptiveKLScheduler(
            target_kl=config.target_kl,
            initial_beta=config.initial_beta,
            kl_horizon=config.warmup_steps,
            min_beta=config.min_beta,
            max_beta=config.max_beta,
        )
    elif stype == "cyclic":
        return CyclicKLScheduler(
            base_beta=config.min_beta,
            max_beta=config.max_beta,
        )
    elif stype == "warmup":
        return WarmupKLScheduler(
            start_beta=0.0,
            end_beta=config.initial_beta,
            warmup_steps=config.warmup_steps,
        )
    else:
        raise ValueError(
            f"Unknown scheduler_type '{config.scheduler_type}'. "
            "Expected one of 'adaptive', 'cyclic', 'warmup'."
        )
