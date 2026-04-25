"""Tests for audio feature extractor."""
from __future__ import annotations

import pytest

from src.multimodal.audio_features import AudioFeatureExtractor


class TestAudioFeatureExtractor:
    def test_extracts_features(self):
        ext = AudioFeatureExtractor(sample_rate=16000)
        samples = [0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5, 0.0]
        feats = ext.extract(samples)
        assert "rms" in feats
        assert "peak" in feats
        assert "zero_crossings" in feats
        assert "duration_s" in feats

    def test_peak_detection(self):
        ext = AudioFeatureExtractor()
        feats = ext.extract([0.0, 0.8, -0.6])
        assert feats["peak"] == 0.8

    def test_empty(self):
        ext = AudioFeatureExtractor()
        feats = ext.extract([])
        assert feats["duration_s"] == 0.0

    def test_duration(self):
        ext = AudioFeatureExtractor(sample_rate=1000)
        feats = ext.extract([0.0] * 1000)
        assert feats["duration_s"] == 1.0

    def test_zero_crossings(self):
        ext = AudioFeatureExtractor()
        feats = ext.extract([1.0, -1.0, 1.0, -1.0])
        assert feats["zero_crossings"] == 3.0