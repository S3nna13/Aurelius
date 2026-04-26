"""Integration test for programmatic_image_tools.

Creates a synthetic 3×32×32 tensor with two distinct bright blobs and calls
all four functions in sequence, asserting that counts match, areas sum
correctly, and no exceptions are raised.
"""

import torch

from src.data.programmatic_image_tools import (
    blob_count,
    crop_region,
    detect_objects,
    pixel_distance,
)


def test_full_pipeline_two_blobs():
    """End-to-end pipeline: crop → pixel_distance → blob_count → detect_objects."""
    # ------------------------------------------------------------------
    # 1.  Build a synthetic 3×32×32 image with two isolated bright blobs.
    # ------------------------------------------------------------------
    image = torch.zeros(3, 32, 32, dtype=torch.float32)

    # Blob A: rows 4:10, cols 4:10  → 6×6 = 36 pixels
    image[:, 4:10, 4:10] = 1.0

    # Blob B: rows 20:26, cols 20:26 → 6×6 = 36 pixels
    image[:, 20:26, 20:26] = 1.0

    # ------------------------------------------------------------------
    # 2.  crop_region — extract a region containing only Blob A.
    # ------------------------------------------------------------------
    crop_a = crop_region(image, (2, 2, 12, 12))
    assert crop_a.shape == (3, 10, 10), f"Unexpected crop shape: {crop_a.shape}"

    # Blob A should be fully contained; its pixels should be all 1.0 within
    # rows 2:8, cols 2:8 of the crop (relative coords: orig rows 4:10, cols 4:10
    # minus the crop origin at row 2, col 2).
    assert crop_a[:, 2:8, 2:8].min().item() == 1.0

    # ------------------------------------------------------------------
    # 3.  pixel_distance — distance between the two blob centroids.
    #     Blob A centroid ≈ (6.5, 6.5), Blob B centroid ≈ (22.5, 22.5).
    # ------------------------------------------------------------------
    centroid_a = (6.5, 6.5)  # (x, y) = (col, row)
    centroid_b = (22.5, 22.5)

    dist = pixel_distance(
        image, (int(centroid_a[0]), int(centroid_a[1])), (int(centroid_b[0]), int(centroid_b[1]))
    )
    assert isinstance(dist, float)
    # Distance between (6,6) and (22,22) = sqrt(16²+16²) ≈ 22.627
    assert dist > 20.0, f"Distance unexpectedly small: {dist}"

    # ------------------------------------------------------------------
    # 4.  blob_count — should find exactly 2 blobs.
    # ------------------------------------------------------------------
    n_blobs = blob_count(image, threshold=0.5, min_size=1)
    assert n_blobs == 2, f"Expected 2 blobs, got {n_blobs}"

    # With min_size larger than a single pixel but smaller than a blob:
    n_blobs_filtered = blob_count(image, threshold=0.5, min_size=10)
    assert n_blobs_filtered == 2, "Both large blobs should survive min_size=10"

    # With min_size larger than any blob, nothing survives:
    n_blobs_none = blob_count(image, threshold=0.5, min_size=100)
    assert n_blobs_none == 0, "No blob should survive min_size=100"

    # ------------------------------------------------------------------
    # 5.  detect_objects — detailed per-blob stats.
    # ------------------------------------------------------------------
    objects = detect_objects(image, threshold=0.5, min_size=1)
    assert len(objects) == 2, f"Expected 2 detected objects, got {len(objects)}"

    # Sort by top-left y-coordinate so Blob A comes first.
    objects_sorted = sorted(objects, key=lambda o: (o["box"][1], o["box"][0]))

    # Both blobs have the same area (6×6 = 36 pixels).
    for obj in objects_sorted:
        assert obj["area"] == 36, f"Expected area 36, got {obj['area']}"
        assert "box" in obj
        assert "centroid" in obj
        x1, y1, x2, y2 = obj["box"]
        assert x2 > x1 and y2 > y1

    # Combined area equals sum of individual areas.
    total_area = sum(o["area"] for o in objects_sorted)
    assert total_area == 72, f"Total area should be 72, got {total_area}"

    # Blob A centroid should be near (6.5, 6.5).
    cx_a, cy_a = objects_sorted[0]["centroid"]
    assert abs(cx_a - 6.5) < 0.1, f"Blob A cx wrong: {cx_a}"
    assert abs(cy_a - 6.5) < 0.1, f"Blob A cy wrong: {cy_a}"

    # Blob B centroid should be near (22.5, 22.5).
    cx_b, cy_b = objects_sorted[1]["centroid"]
    assert abs(cx_b - 22.5) < 0.1, f"Blob B cx wrong: {cx_b}"
    assert abs(cy_b - 22.5) < 0.1, f"Blob B cy wrong: {cy_b}"

    # ------------------------------------------------------------------
    # 6.  Consistency: blob_count should agree with detect_objects count.
    # ------------------------------------------------------------------
    assert n_blobs == len(objects), (
        f"blob_count ({n_blobs}) disagrees with detect_objects ({len(objects)})"
    )
