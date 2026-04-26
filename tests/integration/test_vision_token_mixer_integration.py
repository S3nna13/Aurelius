"""Integration test for VisionTokenMixer — early-fusion 10% vision token interleaving.

Verifies end-to-end mix_batch behaviour on 3 samples:
  - Output dict keys present
  - Sequence lengths consistent
  - vision_ratio constraint holds within tolerance
  - vision_mask sums match n_vision_inserted for every sample
"""

from __future__ import annotations

import pytest

from src.data.vision_token_mixer import VisionTokenMixer, VisionTokenMixerConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mixer() -> VisionTokenMixer:
    cfg = VisionTokenMixerConfig(vision_ratio=0.1, pad_token_id=0, vision_token_id=1)
    return VisionTokenMixer(cfg)


@pytest.fixture
def three_samples() -> list[dict]:
    """Three samples with varying text and vision token lengths."""
    return [
        {
            "text_ids": list(range(500, 590)),  # 90 text tokens
            "vision_ids": list(range(2, 17)),  # 15 vision tokens — some will be truncated
        },
        {
            "text_ids": list(range(600, 660)),  # 60 text tokens
            "vision_ids": list(range(20, 28)),  # 8 vision tokens
        },
        {
            "text_ids": list(range(700, 820)),  # 120 text tokens
            "vision_ids": [],  # no vision tokens
        },
    ]


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_mix_batch_integration(mixer, three_samples):
    """Full mix_batch pipeline: shapes, ratio, and mask-sum consistency."""
    results = mixer.mix_batch(three_samples)

    assert len(results) == 3, "mix_batch should return one dict per input sample"

    for idx, (sample, result) in enumerate(zip(three_samples, results)):
        mixed = result["mixed_ids"]
        mask = result["vision_mask"]

        # --- Key presence ---
        assert "mixed_ids" in result, f"Sample {idx}: missing 'mixed_ids'"
        assert "vision_mask" in result, f"Sample {idx}: missing 'vision_mask'"

        # --- Shape consistency ---
        assert len(mixed) == len(mask), (
            f"Sample {idx}: mixed_ids length {len(mixed)} != vision_mask length {len(mask)}"
        )
        assert len(mixed) >= len(sample["text_ids"]), (
            f"Sample {idx}: output shorter than original text"
        )

        # --- Ratio constraint (skip sample with no vision tokens) ---
        vision_set = set(sample["vision_ids"])
        if vision_set:
            actual_ratio = mixer.compute_ratio(mixed, vision_set)
            assert abs(actual_ratio - 0.1) <= 0.02, (
                f"Sample {idx}: vision_ratio expected ~0.1, got {actual_ratio:.4f}"
            )
        else:
            # No vision tokens inserted — mixed should equal text
            assert mixed == sample["text_ids"], (
                f"Sample {idx}: empty vision_ids but mixed != text_ids"
            )

        # --- vision_mask sums match n_vision_inserted ---
        n_vision_in_mask = sum(mask)
        n_vision_in_mixed = sum(1 for t in mixed if t in vision_set)
        assert n_vision_in_mask == n_vision_in_mixed, (
            f"Sample {idx}: mask sum {n_vision_in_mask} != "
            f"actual vision count in mixed {n_vision_in_mixed}"
        )
