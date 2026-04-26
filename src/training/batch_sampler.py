from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class BatchSamplerConfig:
    block_size: int = 1024
    batch_size: int = 12
    device: str = "cpu"


class RandomOffsetBatchSampler:
    """Samples (x, y) batches from token arrays via random offsets."""

    def __init__(
        self,
        data: np.ndarray | None = None,
        config: BatchSamplerConfig | None = None,
    ):
        self.data = data
        self.config = config or BatchSamplerConfig()

    def from_array(
        self, data: np.ndarray, config: BatchSamplerConfig | None = None
    ) -> "RandomOffsetBatchSampler":
        return RandomOffsetBatchSampler(data=data, config=config or self.config)

    def get_batch(self, data: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        d = data if data is not None else self.data
        if d is None:
            raise ValueError("No data provided")
        cfg = self.config
        max_offset = max(1, len(d) - cfg.block_size)
        ix = torch.randint(max_offset, (cfg.batch_size,))
        x = torch.stack([torch.from_numpy(d[i : i + cfg.block_size].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(d[i + 1 : i + 1 + cfg.block_size].astype(np.int64)) for i in ix]
        )
        return x.to(cfg.device), y.to(cfg.device)

    def estimate_steps_to_tokens(self, total_tokens: int, epochs: float = 1.0) -> int:
        """Chinchilla-style: how many steps to process N tokens."""
        tokens_per_step = self.config.batch_size * self.config.block_size
        return int(total_tokens * epochs / tokens_per_step)


BATCH_SAMPLER = RandomOffsetBatchSampler()
