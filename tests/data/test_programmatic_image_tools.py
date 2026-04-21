"""Unit tests for src/data/programmatic_image_tools.py.

Pure PyTorch only — no PIL, cv2, torchvision, scipy, sklearn, or numpy.
Images are torch.Tensor[C, H, W] with values in [0, 1].
"""

import pytest
import torch

from src.data.programmatic_image_tools import (
    blob_count,
    crop_region,
    detect_objects,
    pixel_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solid(value: float, C: int = 3, H: int = 8, W: int = 8) -> torch.Tensor:
    """Return a [C, H, W] tensor filled with *value*."""
    return torch.full((C, H, W), value, dtype=torch.float32)


def _bright_patch(canvas: torch.Tensor, r1: int, r2: int, c1: int, c2: int) -> torch.Tensor:
    """Set rows r1:r2, cols c1:c2 to 1.0 on all channels (in-place clone returned)."""
    t = canvas.clone()
    t[:, r1:r2, c1:c2] = 1.0
    return t


# ---------------------------------------------------------------------------
# crop_region — 3 tests
# ---------------------------------------------------------------------------


def test_crop_basic():
    """Cropping a [3, 8, 8] image with box (2, 2, 6, 6) yields shape [3, 4, 4]."""
    image = torch.rand(3, 8, 8)
    out = crop_region(image, (2, 2, 6, 6))
    assert out.shape == (3, 4, 4)


def test_crop_clamp():
    """A box extending beyond the image boundary clamps silently — no exception."""
    image = torch.rand(3, 8, 8)
    # box extends well beyond width and height
    out = crop_region(image, (5, 5, 20, 20))
    # Should produce a 3×3 patch (rows 5-7, cols 5-7)
    assert out.shape[0] == 3
    assert out.shape[1] > 0
    assert out.shape[2] > 0


def test_crop_single_pixel():
    """A 1×1 box returns a [C, 1, 1] tensor."""
    image = torch.rand(3, 8, 8)
    out = crop_region(image, (3, 3, 4, 4))
    assert out.shape == (3, 1, 1)


# ---------------------------------------------------------------------------
# pixel_distance — 3 tests
# ---------------------------------------------------------------------------


def test_pixel_distance_zero():
    """Same point → distance 0.0."""
    image = torch.zeros(3, 8, 8)
    d = pixel_distance(image, (3, 4), (3, 4))
    assert d == 0.0


def test_pixel_distance_known():
    """(0, 0) to (3, 4) is a 3-4-5 right triangle → distance 5.0."""
    image = torch.zeros(3, 8, 8)
    d = pixel_distance(image, (0, 0), (3, 4))
    assert abs(d - 5.0) < 1e-6


def test_pixel_distance_float():
    """Return type must be a Python float."""
    image = torch.zeros(3, 8, 8)
    d = pixel_distance(image, (1, 1), (2, 2))
    assert isinstance(d, float)


# ---------------------------------------------------------------------------
# blob_count — 4 tests
# ---------------------------------------------------------------------------


def test_blob_count_empty():
    """All-zero image → 0 blobs."""
    image = _solid(0.0)
    assert blob_count(image) == 0


def test_blob_count_single():
    """One bright pixel above threshold → 1 blob."""
    image = _solid(0.0)
    image[:, 4, 4] = 1.0
    assert blob_count(image, threshold=0.5) == 1


def test_blob_count_two_separate():
    """Two isolated bright pixels → 2 blobs."""
    image = _solid(0.0)
    image[:, 1, 1] = 1.0
    image[:, 6, 6] = 1.0
    assert blob_count(image, threshold=0.5) == 2


def test_blob_count_min_size():
    """A single-pixel blob is excluded when min_size=2."""
    image = _solid(0.0)
    image[:, 4, 4] = 1.0
    assert blob_count(image, threshold=0.5, min_size=2) == 0


# ---------------------------------------------------------------------------
# detect_objects — 5 tests
# ---------------------------------------------------------------------------


def test_detect_objects_empty():
    """All-zero image → empty list."""
    image = _solid(0.0)
    objs = detect_objects(image)
    assert objs == []


def test_detect_objects_one_blob():
    """One blob → list of 1 dict with the required keys."""
    image = _solid(0.0)
    image = _bright_patch(image, 2, 6, 2, 6)  # 4×4 bright square
    objs = detect_objects(image, threshold=0.5, min_size=1)
    assert len(objs) == 1
    obj = objs[0]
    assert "box" in obj
    assert "area" in obj
    assert "centroid" in obj


def test_detect_objects_two_blobs():
    """Two isolated 2×2 blobs → list of 2 dicts."""
    image = _solid(0.0, H=16, W=16)
    image = _bright_patch(image, 1, 3, 1, 3)   # top-left 2×2
    image = _bright_patch(image, 10, 12, 10, 12)  # bottom-right 2×2
    objs = detect_objects(image, threshold=0.5, min_size=1)
    assert len(objs) == 2


def test_detect_centroid_correct():
    """A 4×4 bright square at rows 2:6, cols 2:6 → centroid (3.5, 3.5)."""
    image = _solid(0.0)
    image = _bright_patch(image, 2, 6, 2, 6)
    objs = detect_objects(image, threshold=0.5, min_size=1)
    assert len(objs) == 1
    cx, cy = objs[0]["centroid"]
    assert abs(cx - 3.5) < 1e-4
    assert abs(cy - 3.5) < 1e-4


def test_detect_area_correct():
    """A 4×4 bright square → area 16."""
    image = _solid(0.0)
    image = _bright_patch(image, 2, 6, 2, 6)
    objs = detect_objects(image, threshold=0.5, min_size=1)
    assert len(objs) == 1
    assert objs[0]["area"] == 16


# ---------------------------------------------------------------------------
# Determinism — 1 test
# ---------------------------------------------------------------------------


def test_determinism():
    """All four functions return the same result on repeated calls with the same input."""
    torch.manual_seed(42)
    image = torch.rand(3, 16, 16)
    box = (2, 2, 10, 10)

    crop1 = crop_region(image, box)
    crop2 = crop_region(image, box)
    assert torch.equal(crop1, crop2)

    d1 = pixel_distance(image, (1, 1), (5, 4))
    d2 = pixel_distance(image, (1, 1), (5, 4))
    assert d1 == d2

    bc1 = blob_count(image, threshold=0.5)
    bc2 = blob_count(image, threshold=0.5)
    assert bc1 == bc2

    objs1 = detect_objects(image, threshold=0.5)
    objs2 = detect_objects(image, threshold=0.5)
    assert len(objs1) == len(objs2)
    for o1, o2 in zip(objs1, objs2):
        assert o1["box"] == o2["box"]
        assert o1["area"] == o2["area"]
        assert o1["centroid"] == o2["centroid"]
