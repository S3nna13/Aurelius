from dataclasses import dataclass, field
import math
import torch
import torch.nn as nn
from collections import defaultdict


@dataclass
class GradSnapshot:
    step: int
    layer_norms: dict[str, float]   # layer_prefix -> L2 norm of all grads in that layer
    param_norms: dict[str, float]   # param_name -> individual grad L2 norm
    global_norm: float              # sqrt(sum of all squared grad norms)


class GradTracker:
    """Records gradient norms after each backward pass.

    Usage:
        tracker = GradTracker(model)
        loss.backward()
        tracker.snapshot(step=0)          # call after backward, before optimizer.step()

        print(tracker.history)            # list[GradSnapshot]
        print(tracker.summary())          # text report flagging anomalies
    """

    VANISHING_THRESHOLD = 1e-7
    EXPLODING_THRESHOLD = 10.0

    def __init__(self, model: nn.Module):
        self.model = model
        self._history: list[GradSnapshot] = []

    def snapshot(self, step: int = 0) -> GradSnapshot:
        """Capture current .grad norms for all parameters that have gradients.

        Layer prefix is derived by taking everything before the last dot in param name.
        e.g. 'layers.0.attn.q_proj.weight' -> layer='layers.0.attn.q_proj'

        Parameters with .grad is None are skipped.
        """
        param_norms: dict[str, float] = {}
        layer_squared: dict[str, float] = defaultdict(float)
        total_squared = 0.0

        for name, p in self.model.named_parameters():
            if p.grad is None:
                continue

            norm = p.grad.detach().float().norm(2).item()
            param_norms[name] = norm

            parts = name.split(".")
            layer_prefix = ".".join(parts[:-1]) if len(parts) > 1 else "root"

            layer_squared[layer_prefix] += norm * norm
            total_squared += norm * norm

        layer_norms: dict[str, float] = {
            k: math.sqrt(v) for k, v in layer_squared.items()
        }
        global_norm = math.sqrt(total_squared)

        snap = GradSnapshot(
            step=step,
            layer_norms=layer_norms,
            param_norms=param_norms,
            global_norm=global_norm,
        )
        self._history.append(snap)
        return snap

    @property
    def history(self) -> list[GradSnapshot]:
        return self._history

    def summary(self, last_n: int = 1) -> str:
        """Return human-readable report for the last_n snapshots.

        Flag layers where norm < VANISHING_THRESHOLD as [VANISHING]
        Flag layers where norm > EXPLODING_THRESHOLD as [EXPLODING]
        Sort layers by norm descending.
        """
        if not self._history:
            return "No snapshots recorded."

        snapshots = self._history[-last_n:]
        lines: list[str] = []

        for snap in snapshots:
            lines.append(f"=== Step {snap.step} | Global norm: {snap.global_norm:.6f} ===")

            sorted_layers = sorted(
                snap.layer_norms.items(), key=lambda x: x[1], reverse=True
            )

            for layer, norm in sorted_layers:
                flag = ""
                if norm < self.VANISHING_THRESHOLD:
                    flag = " [VANISHING]"
                elif norm > self.EXPLODING_THRESHOLD:
                    flag = " [EXPLODING]"
                lines.append(f"  {layer:60s}  {norm:.6e}{flag}")

            lines.append("")

        return "\n".join(lines)
