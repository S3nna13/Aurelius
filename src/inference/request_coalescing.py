"""Request coalescing for inference.

Coalesces concurrent identical requests (by prompt hash) into a single
inference call, then fans out the result to all waiters.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CoalescingConfig:
    timeout: float = 30.0


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


@dataclass
class CoalescingSlot:
    prompt: str
    prompt_hash: str
    event: threading.Event = field(default_factory=threading.Event)
    result: Any = None
    error: BaseException | None = None
    created_at: float = field(default_factory=time.time)


class RequestCoalescer:
    """Thread-safe coalescer that deduplicates concurrent requests by prompt hash."""

    def __init__(self, config: CoalescingConfig | None = None) -> None:
        self._config = config if config is not None else CoalescingConfig()
        self._lock = threading.Lock()
        self._slots: dict[str, list[CoalescingSlot]] = {}
        self._total_deduplicated: int = 0
        self._total_unique: int = 0

    @property
    def config(self) -> CoalescingConfig:
        return self._config

    def submit(self, prompt: str) -> CoalescingSlot:
        ph = _prompt_hash(prompt)
        with self._lock:
            if ph in self._slots:
                slot = CoalescingSlot(prompt=prompt, prompt_hash=ph)
                self._slots[ph].append(slot)
                self._total_deduplicated += 1
                return slot

            slot = CoalescingSlot(prompt=prompt, prompt_hash=ph)
            self._slots[ph] = [slot]
            self._total_unique += 1
            return slot

    def resolve(self, prompt: str, result: Any) -> int:
        ph = _prompt_hash(prompt)
        with self._lock:
            slots = self._slots.pop(ph, [])
        for slot in slots:
            slot.result = result
            slot.event.set()
        return len(slots)

    def reject(self, prompt: str, error: BaseException) -> int:
        ph = _prompt_hash(prompt)
        with self._lock:
            slots = self._slots.pop(ph, [])
        for slot in slots:
            slot.error = error
            slot.event.set()
        return len(slots)

    def wait(self, slot: CoalescingSlot, timeout: float | None = None) -> Any:
        t = timeout if timeout is not None else self._config.timeout
        if not slot.event.wait(timeout=t):
            raise TimeoutError(
                f"Coalesced request timed out after {t}s for prompt hash {slot.prompt_hash}"
            )
        if slot.error is not None:
            raise slot.error
        return slot.result

    def clear(self) -> int:
        with self._lock:
            remaining = sum(len(v) for v in self._slots.values())
            self._slots.clear()
            self._total_deduplicated = 0
            self._total_unique = 0
        return remaining

    def stats(self) -> dict[str, float]:
        with self._lock:
            pending = sum(len(v) for v in self._slots.values())
        return {
            "pending": float(pending),
            "unique": float(self._total_unique),
            "deduplicated": float(self._total_deduplicated),
            "savings_ratio": float(self._total_deduplicated) / float(self._total_unique + self._total_deduplicated)
            if (self._total_unique + self._total_deduplicated) > 0
            else 0.0,
        }

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._slots.values())
