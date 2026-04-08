import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class GradAccumConfig:
    n_accum: int = 8           # number of micro-steps per optimizer step
    clip_grad_norm: float | None = 1.0  # gradient clipping (None = no clip)


class GradAccumManager:
    """Manages gradient accumulation over multiple micro-batches.

    Correctly scales loss, accumulates gradients, and only calls
    optimizer.step() every n_accum steps.

    Usage:
        mgr = GradAccumManager(model, optimizer, cfg)

        for i, batch in enumerate(loader):
            loss = model(**batch)[0]

            should_step = mgr.step(loss)
            # mgr.step() handles: backward, grad clipping, optimizer step, zero_grad

            if should_step:
                print(f"Optimizer step at micro-step {i}")

        # Flush remaining accumulated gradients
        mgr.flush()
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cfg: GradAccumConfig = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.cfg = cfg or GradAccumConfig()
        self._accum_steps = 0
        self._accumulated_loss = 0.0
        self.n_optimizer_steps = 0

    def step(self, loss: torch.Tensor) -> bool:
        """Process one micro-batch loss.

        Scales loss, runs backward, accumulates gradients.
        If n_accum steps reached: clip grads, optimizer.step(), zero_grad().

        Returns:
            True if optimizer.step() was called this micro-step.
        """
        # Scale loss by 1/n_accum to get the correct mean over the full batch
        scaled_loss = loss / self.cfg.n_accum
        scaled_loss.backward()

        self._accum_steps += 1
        self._accumulated_loss += scaled_loss.item()

        if self._accum_steps >= self.cfg.n_accum:
            self._do_optimizer_step()
            return True
        return False

    def flush(self) -> bool:
        """Force an optimizer step if any gradients are accumulated.

        Call at end of epoch/dataset to handle partial last accumulation.

        Returns:
            True if optimizer.step() was called.
        """
        if self._accum_steps > 0:
            self._do_optimizer_step()
            return True
        return False

    def _do_optimizer_step(self) -> None:
        if self.cfg.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.clip_grad_norm
            )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._accum_steps = 0
        self._accumulated_loss = 0.0
        self.n_optimizer_steps += 1

    @property
    def current_accum_steps(self) -> int:
        return self._accum_steps

    @property
    def accumulated_loss(self) -> float:
        return self._accumulated_loss
