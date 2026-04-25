"""Video frame sampler — uniform and scene-change strategies without external codecs.

Supports:
    * A single ``.npy`` file with shape ``(N, H, W, C)`` or ``(H, W, C)``.
    * A directory of ``.npy`` frame files (each ``(H, W, C)``).

Only numpy and the standard library are used; opencv and ffmpeg are **not**
required.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import numpy as np


class VideoFrameSampler:
    """Extract and downsample frames from a numpy-serialised video source.

    Args:
        scene_threshold: MSE difference above which two consecutive frames are
            considered a scene change. Default is tuned for 0-255 uint8 data.
    """

    DEFAULT_SCENE_THRESHOLD: ClassVar[float] = 100.0

    def __init__(self, scene_threshold: float = DEFAULT_SCENE_THRESHOLD) -> None:
        self.scene_threshold = scene_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(
        self,
        video_path: str,
        max_frames: int = 8,
        strategy: str = "uniform",
    ) -> list[np.ndarray]:
        """Sample up to *max_frames* frames from *video_path*.

        Args:
            video_path: Path to a ``.npy`` file or a directory of ``.npy``
                frame files.
            max_frames: Maximum number of frames to return.
            strategy: ``"uniform"`` (fixed interval) or ``"scene_change"``
                (MSE-based scene detection).

        Returns:
            List of frame arrays, each with shape ``(H, W, C)``.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            ValueError: On unsupported strategy or empty source.
        """
        if strategy not in {"uniform", "scene_change"}:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        frames = self._load_frames(video_path)
        if not frames:
            raise ValueError(f"No frames could be loaded from: {video_path}")

        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {max_frames}")

        if strategy == "uniform":
            return self._sample_uniform(frames, max_frames)
        return self._sample_scene_change(frames, max_frames)

    @staticmethod
    def frame_count(video_path: str) -> int:
        """Return the number of frames available at *video_path*.

        Args:
            video_path: Path to a ``.npy`` file or directory of ``.npy`` frames.

        Returns:
            Frame count.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            ValueError: If the source contains no loadable frames.
        """
        p = Path(video_path)
        if not p.exists():
            raise FileNotFoundError(f"Video path not found: {video_path}")

        if p.is_file():
            arr = np.load(str(p), mmap_mode="r")
            if arr.ndim == 3:
                return 1
            if arr.ndim == 4:
                return int(arr.shape[0])
            raise ValueError(
                f"Expected a 3D or 4D array, got {arr.ndim}D in {video_path}"
            )

        if p.is_dir():
            count = sum(
                1 for f in p.iterdir() if f.is_file() and f.suffix == ".npy"
            )
            if count == 0:
                raise ValueError(
                    f"No .npy frame files found in directory: {video_path}"
                )
            return count

        raise FileNotFoundError(f"Video path not found: {video_path}")

    @staticmethod
    def mse_diff(a: np.ndarray, b: np.ndarray) -> float:
        """Compute mean-squared error between two identically-shaped frames.

        Args:
            a: Frame array of shape ``(H, W, C)``.
            b: Frame array of shape ``(H, W, C)``.

        Returns:
            Scalar MSE value.

        Raises:
            ValueError: If shapes mismatch.
        """
        if a.shape != b.shape:
            raise ValueError(
                f"Shape mismatch: {a.shape} vs {b.shape}"
            )
        diff = a.astype(np.float64) - b.astype(np.float64)
        return float(np.mean(diff * diff))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_frames(video_path: str) -> list[np.ndarray]:
        """Load all frames from *video_path* into a list."""
        p = Path(video_path)
        if not p.exists():
            raise FileNotFoundError(f"Video path not found: {video_path}")

        if p.is_file():
            arr = np.load(str(p))
            if arr.ndim == 3:
                return [arr]
            if arr.ndim == 4:
                return [arr[i] for i in range(arr.shape[0])]
            raise ValueError(
                f"Expected a 3D or 4D array, got {arr.ndim}D in {video_path}"
            )

        if p.is_dir():
            files = sorted(
                f for f in p.iterdir() if f.is_file() and f.suffix == ".npy"
            )
            if not files:
                raise ValueError(
                    f"No .npy frame files found in directory: {video_path}"
                )
            frames: list[np.ndarray] = []
            for f in files:
                arr = np.load(str(f))
                if arr.ndim == 3:
                    frames.append(arr)
                elif arr.ndim == 4:
                    frames.extend([arr[i] for i in range(arr.shape[0])])
                else:
                    raise ValueError(
                        f"Expected a 3D or 4D array, got {arr.ndim}D in {f}"
                    )
            return frames

        raise FileNotFoundError(f"Video path not found: {video_path}")

    @staticmethod
    def _sample_uniform(frames: list[np.ndarray], max_frames: int) -> list[np.ndarray]:
        n = len(frames)
        if max_frames >= n:
            return frames[:]
        indices = np.linspace(0, n - 1, max_frames, dtype=int)
        return [frames[i] for i in indices]

    def _sample_scene_change(
        self, frames: list[np.ndarray], max_frames: int
    ) -> list[np.ndarray]:
        n = len(frames)
        if max_frames >= n:
            return frames[:]

        scene_indices = [0]
        for i in range(1, n):
            if self.mse_diff(frames[i - 1], frames[i]) > self.scene_threshold:
                scene_indices.append(i)

        if len(scene_indices) > max_frames:
            selected = np.linspace(0, len(scene_indices) - 1, max_frames, dtype=int)
            scene_indices = [scene_indices[i] for i in selected]
        elif len(scene_indices) < max_frames:
            existing = set(scene_indices)
            remaining = [i for i in range(n) if i not in existing]
            needed = max_frames - len(scene_indices)
            if needed >= len(remaining):
                scene_indices.extend(remaining)
            else:
                extra = np.linspace(0, len(remaining) - 1, needed, dtype=int)
                scene_indices.extend([remaining[i] for i in extra])
            scene_indices.sort()

        return [frames[i] for i in scene_indices]


__all__ = ["VideoFrameSampler"]
