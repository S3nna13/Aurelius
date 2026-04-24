import math
import torch
import numpy as np
from dataclasses import dataclass
from typing import Callable


@dataclass
class LossEstimate:
    split: str
    mean_loss: float
    std_loss: float
    n_batches: int


class LossEstimator:
    """Estimates train/val loss over multiple batches with no_grad context."""

    def estimate(
        self,
        model: torch.nn.Module,
        get_batch_fn: Callable[[str], tuple[torch.Tensor, torch.Tensor]],
        splits: list[str],
        eval_iters: int = 200,
        device: str = "cpu",
    ) -> dict[str, LossEstimate]:
        model.eval()
        results = {}
        with torch.no_grad():
            for split in splits:
                losses = []
                for _ in range(eval_iters):
                    x, y = get_batch_fn(split)
                    x, y = x.to(device), y.to(device)
                    try:
                        output = model(x)
                        if isinstance(output, tuple):
                            loss = output[0]
                        else:
                            loss = output
                        val = loss.item() if hasattr(loss, "item") else float(loss)
                        losses.append(val)
                    except Exception:
                        losses.append(float("nan"))
                # filter out NaN values before computing stats
                valid = [v for v in losses if not math.isnan(v)]
                results[split] = LossEstimate(
                    split=split,
                    mean_loss=float(np.mean(valid)) if valid else float("nan"),
                    std_loss=float(np.std(valid)) if valid else 0.0,
                    n_batches=len(valid),
                )
        model.train()
        return results

    def estimate_perplexity(self, mean_loss: float) -> float:
        """Compute perplexity from mean cross-entropy loss, capped at exp(20)."""
        return math.exp(min(mean_loss, 20.0))


LOSS_ESTIMATOR = LossEstimator()
