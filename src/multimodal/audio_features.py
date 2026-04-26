"""Multimodal audio feature extractor for classification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AudioFeatureExtractor:
    """Extract basic audio features from raw sample data."""

    sample_rate: int = 16000

    def extract(self, samples: list[float]) -> dict[str, float]:
        if not samples:
            return {"rms": 0.0, "peak": 0.0, "zero_crossings": 0.0, "duration_s": 0.0}
        peak = max(abs(s) for s in samples)
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        zc = sum(1 for i in range(1, len(samples)) if samples[i] * samples[i - 1] < 0)
        return {
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "zero_crossings": float(zc),
            "duration_s": round(len(samples) / self.sample_rate, 3),
        }


AUDIO_FEATURES = AudioFeatureExtractor()
