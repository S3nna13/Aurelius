from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass

import torch


@dataclass
class StreamConfig:
    batch_size: int = 8
    shuffle_buffer: int = 1000
    prefetch: int = 2
    drop_last: bool = True


class StreamingDataloader:
    """Streaming dataloader with reservoir shuffle buffer and prefetch."""

    def __init__(self, dataset, config: StreamConfig | None = None) -> None:
        self.dataset = dataset
        self.config = config or StreamConfig()

    def _reservoir_sample(self, indices: list[int], k: int) -> list[int]:
        reservoir: list[int] = []
        for i, val in enumerate(indices):
            if i < k:
                reservoir.append(val)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = val
        return reservoir

    def __iter__(self) -> Iterator[dict]:
        n = len(self.dataset)
        all_indices = list(range(n))
        random.shuffle(all_indices)

        pos = 0
        buffer: list[dict] = []

        def _fill_buffer() -> None:
            nonlocal pos
            target = self.config.shuffle_buffer
            while len(buffer) < target and pos < n:
                end = min(pos + (target - len(buffer)), n)
                chunk_indices = all_indices[pos:end]
                for idx in chunk_indices:
                    buffer.append(self.dataset[idx])
                pos = end

        _fill_buffer()

        while True:
            if len(buffer) < self.config.batch_size:
                _fill_buffer()
            if len(buffer) < self.config.batch_size:
                if not self.config.drop_last and buffer:
                    yield self.collate(buffer)
                break

            batch_items = buffer[: self.config.batch_size]
            del buffer[: self.config.batch_size]
            yield self.collate(batch_items)

            if len(buffer) < self.config.batch_size:
                _fill_buffer()

    def __len__(self) -> int:
        return len(self.dataset) // self.config.batch_size

    def collate(self, samples: list[dict]) -> dict:
        input_ids = torch.stack([s["input_ids"] for s in samples])
        labels = torch.stack([s["labels"] for s in samples])
        return {"input_ids": input_ids, "labels": labels}
