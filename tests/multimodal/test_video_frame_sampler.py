"""Tests for src/multimodal/video_frame_sampler.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.multimodal.video_frame_sampler import VideoFrameSampler


class TestVideoFrameSampler:
    # ------------------------------------------------------------------
    # Uniform sampling
    # ------------------------------------------------------------------

    def test_uniform_sampling_synthetic(self, tmp_path: Path):
        """Uniform strategy returns exactly max_frames from a 10-frame source."""
        frames = np.stack([np.ones((4, 4, 3), dtype=np.uint8) * i for i in range(10)])
        video_path = tmp_path / "video.npy"
        np.save(video_path, frames)

        sampler = VideoFrameSampler()
        result = sampler.sample(str(video_path), max_frames=5, strategy="uniform")
        assert len(result) == 5
        assert all(isinstance(f, np.ndarray) for f in result)
        assert result[0].shape == (4, 4, 3)
        # Lin-space indices for 10 frames / 5 samples → [0, 2, 4, 6, 9]
        assert np.array_equal(result[0], frames[0])
        assert np.array_equal(result[-1], frames[9])

    def test_uniform_returns_all_when_fewer_than_max(self, tmp_path: Path):
        """If the source has fewer frames than max_frames, return all of them."""
        frames = np.random.randint(0, 256, (3, 4, 4, 3), dtype=np.uint8)
        video_path = tmp_path / "video.npy"
        np.save(video_path, frames)

        result = VideoFrameSampler().sample(str(video_path), max_frames=8, strategy="uniform")
        assert len(result) == 3

    def test_uniform_from_directory(self, tmp_path: Path):
        """Uniform sampling works over a directory of individual .npy frames."""
        for i in range(10):
            fpath = tmp_path / f"frame_{i:03d}.npy"
            np.save(fpath, np.ones((4, 4, 3), dtype=np.uint8) * i)

        result = VideoFrameSampler().sample(str(tmp_path), max_frames=4, strategy="uniform")
        assert len(result) == 4

    # ------------------------------------------------------------------
    # Scene-change sampling
    # ------------------------------------------------------------------

    def test_scene_change_detects_changes(self, tmp_path: Path):
        """Alternating black/white frames should produce many scene changes."""
        frames = []
        for i in range(10):
            if i % 2 == 0:
                frames.append(np.zeros((4, 4, 3), dtype=np.uint8))
            else:
                frames.append(np.ones((4, 4, 3), dtype=np.uint8) * 255)
        video_path = tmp_path / "video.npy"
        np.save(video_path, np.stack(frames))

        sampler = VideoFrameSampler()
        result = sampler.sample(str(video_path), max_frames=8, strategy="scene_change")
        assert len(result) <= 8
        # First frame is always included; with alternating extremes nearly
        # every boundary exceeds the default threshold.
        assert len(result) >= 2

    def test_scene_change_fills_uniform_when_no_changes(self, tmp_path: Path):
        """Identical frames produce zero scene changes; sampler falls back."""
        frames = np.stack([np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(10)])
        video_path = tmp_path / "video.npy"
        np.save(video_path, frames)

        result = VideoFrameSampler().sample(str(video_path), max_frames=5, strategy="scene_change")
        assert len(result) == 5

    # ------------------------------------------------------------------
    # Errors and edge cases
    # ------------------------------------------------------------------

    def test_file_not_found(self):
        """Non-existent path must raise FileNotFoundError."""
        sampler = VideoFrameSampler()
        with pytest.raises(FileNotFoundError):
            sampler.sample("/nonexistent/path/video.npy", max_frames=8)

    def test_empty_directory_raises(self, tmp_path: Path):
        """An empty directory has no loadable frames."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError):
            VideoFrameSampler().sample(str(empty_dir), max_frames=8)

    def test_invalid_strategy(self, tmp_path: Path):
        """Only 'uniform' and 'scene_change' are accepted."""
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        video_path = tmp_path / "video.npy"
        np.save(video_path, frames)
        with pytest.raises(ValueError):
            VideoFrameSampler().sample(str(video_path), strategy="random")

    def test_non_positive_max_frames(self, tmp_path: Path):
        """max_frames must be a positive integer."""
        frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
        video_path = tmp_path / "video.npy"
        np.save(video_path, frames)
        with pytest.raises(ValueError):
            VideoFrameSampler().sample(str(video_path), max_frames=0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def test_frame_count_file(self, tmp_path: Path):
        """frame_count on a 4D .npy file returns the leading dimension."""
        frames = np.random.randint(0, 256, (7, 4, 4, 3), dtype=np.uint8)
        video_path = tmp_path / "video.npy"
        np.save(video_path, frames)
        assert VideoFrameSampler.frame_count(str(video_path)) == 7

    def test_frame_count_directory(self, tmp_path: Path):
        """frame_count on a directory of .npy files returns the file count."""
        for i in range(5):
            fpath = tmp_path / f"frame_{i:03d}.npy"
            np.save(fpath, np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8))
        assert VideoFrameSampler.frame_count(str(tmp_path)) == 5

    def test_mse_diff(self):
        """mse_diff between black and white frames equals 65025 for uint8."""
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.ones((4, 4, 3), dtype=np.uint8) * 255
        mse = VideoFrameSampler.mse_diff(a, b)
        expected = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
        assert mse == pytest.approx(expected)
        assert mse == pytest.approx(65025.0)

    def test_mse_diff_shape_mismatch(self):
        """mse_diff raises ValueError when shapes differ."""
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.zeros((4, 3, 3), dtype=np.uint8)
        with pytest.raises(ValueError):
            VideoFrameSampler.mse_diff(a, b)

    def test_mse_diff_identical_frames(self):
        """mse_diff of identical frames is zero."""
        a = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        assert VideoFrameSampler.mse_diff(a, a) == 0.0
