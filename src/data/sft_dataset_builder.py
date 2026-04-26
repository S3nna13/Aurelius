from __future__ import annotations

import json
import random
from dataclasses import dataclass, field


@dataclass
class SFTExample:
    prompt: str
    response: str
    system_prompt: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class SFTDatasetConfig:
    max_length: int = 2048
    dedup: bool = True
    shuffle: bool = True
    seed: int = 42
    split_ratio: float = 0.9


class SFTDatasetBuilder:
    def __init__(self, config: SFTDatasetConfig | None = None) -> None:
        self.config = config if config is not None else SFTDatasetConfig()
        self._examples: list[SFTExample] = []
        self._after_filter: int = 0
        self._after_dedup: int = 0

    def add(self, example: SFTExample) -> None:
        self._examples.append(example)

    def add_batch(self, examples: list[SFTExample]) -> None:
        self._examples.extend(examples)

    def filter_by_length(self, examples: list[SFTExample] | None = None) -> list[SFTExample]:
        src = examples if examples is not None else self._examples
        max_len = self.config.max_length
        return [e for e in src if len(e.prompt) + len(e.response) <= max_len]

    def deduplicate(self, examples: list[SFTExample] | None = None) -> list[SFTExample]:
        src = examples if examples is not None else self._examples
        seen: set = set()
        result: list[SFTExample] = []
        for e in src:
            if e.prompt not in seen:
                seen.add(e.prompt)
                result.append(e)
        return result

    def build(self) -> dict[str, list[SFTExample]]:
        filtered = self.filter_by_length(self._examples)
        self._after_filter = len(filtered)
        deduped = self.deduplicate(filtered)
        self._after_dedup = len(deduped)
        data = list(deduped)
        if self.config.shuffle:
            rng = random.Random(self.config.seed)
            rng.shuffle(data)
        split = int(len(data) * self.config.split_ratio)
        return {"train": data[:split], "val": data[split:]}

    def to_jsonl_lines(self, examples: list[SFTExample]) -> list[str]:
        lines = []
        for e in examples:
            record = {
                "prompt": e.prompt,
                "response": e.response,
                "system_prompt": e.system_prompt,
                "metadata": e.metadata,
            }
            lines.append(json.dumps(record))
        return lines

    def stats(self) -> dict:
        splits = self.build()
        return {
            "total": len(self._examples),
            "after_filter": self._after_filter,
            "after_dedup": self._after_dedup,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
        }


SFT_BUILDER_REGISTRY: dict = {"default": SFTDatasetBuilder}
