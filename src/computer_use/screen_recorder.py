"""
screen_recorder.py
Records screen state snapshots over time.
"""

from __future__ import annotations

import time
import collections
from dataclasses import dataclass


@dataclass(frozen=True)
class ScreenSnapshot:
    timestamp: float
    width: int
    height: int
    content: str
    frame_id: int


class ScreenRecorder:
    def __init__(self, max_frames: int = 100) -> None:
        self._max_frames = max_frames
        self._frames: collections.deque[ScreenSnapshot] = collections.deque(maxlen=max_frames)
        self._next_frame_id: int = 0

    def capture(self, width: int, height: int, content: str) -> ScreenSnapshot:
        snapshot = ScreenSnapshot(
            timestamp=time.monotonic(),
            width=width,
            height=height,
            content=content,
            frame_id=self._next_frame_id,
        )
        self._next_frame_id += 1
        return snapshot

    def record(self, snapshot: ScreenSnapshot) -> None:
        self._frames.append(snapshot)

    def capture_and_record(self, width: int, height: int, content: str) -> ScreenSnapshot:
        snapshot = self.capture(width, height, content)
        self.record(snapshot)
        return snapshot

    def frames(self) -> list[ScreenSnapshot]:
        return list(self._frames)

    def latest(self) -> ScreenSnapshot | None:
        if not self._frames:
            return None
        return self._frames[-1]

    def diff(self, frame_a: ScreenSnapshot, frame_b: ScreenSnapshot) -> dict:
        content_changed = frame_a.content != frame_b.content
        size_changed = (frame_a.width != frame_b.width) or (frame_a.height != frame_b.height)
        changed = content_changed or size_changed
        return {
            "changed": changed,
            "content_changed": content_changed,
            "size_changed": size_changed,
        }

    def export_log(self) -> list[dict]:
        return [
            {
                "frame_id": snap.frame_id,
                "timestamp": snap.timestamp,
                "width": snap.width,
                "height": snap.height,
                "content_len": len(snap.content),
            }
            for snap in self._frames
        ]


SCREEN_RECORDER_REGISTRY: dict = {"default": ScreenRecorder}
