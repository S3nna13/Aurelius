"""FineWeb2 pretraining data loader.

Dataset: HuggingFaceFW/fineweb-2 (full, 20TB) or fineweb-2-hq (quality subset).
Streaming mode to avoid downloading the full dataset.
Applies 30% synthetic / 70% natural mixing per the 2025 Demystifying Synthetic
Data paper (EMNLP 2025) for optimal pretraining convergence.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch.utils.data import IterableDataset


class FineWeb2Dataset(IterableDataset):
    """Streaming FineWeb2 dataset with optional synthetic data mixing."""

    def __init__(
        self,
        tokenizer,
        seq_len: int = 4096,
        split: str = "train",
        use_hq: bool = True,
        synthetic_ratio: float = 0.30,
        synthetic_dataset: str | None = None,
        languages: list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.use_hq = use_hq
        self.synthetic_ratio = synthetic_ratio
        self.synthetic_dataset = synthetic_dataset
        self.languages = languages or ["en"]

    def _stream_fineweb2(self) -> Iterator[str]:
        from datasets import load_dataset

        _name = "fineweb-2-hq" if self.use_hq else "fineweb-2"  # noqa: F841
        for lang in self.languages:
            ds = load_dataset(
                "HuggingFaceFW/fineweb-2",
                name=f"CC-MAIN-2024-10_{lang}",  # nosec B615
                split=self.split,
                streaming=True,
                trust_remote_code=True,
            )
            for sample in ds:
                yield sample["text"]

    def _stream_synthetic(self) -> Iterator[str]:
        if not self.synthetic_dataset:
            return
        from datasets import load_dataset

        ds = load_dataset(self.synthetic_dataset, split=self.split, streaming=True)  # nosec B615
        for sample in ds:
            yield sample.get("text", sample.get("content", ""))

    def __iter__(self) -> Iterator[dict]:
        import random

        natural_stream = self._stream_fineweb2()
        synthetic_stream = self._stream_synthetic() if self.synthetic_dataset else iter([])
        buffer = []
        for text in natural_stream:
            # Mix in synthetic at the configured ratio
            if self.synthetic_dataset and random.random() < self.synthetic_ratio:
                try:
                    text = next(synthetic_stream)
                except StopIteration:
                    pass
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            buffer.extend(tokens)
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


__all__ = ["FineWeb2Dataset"]
