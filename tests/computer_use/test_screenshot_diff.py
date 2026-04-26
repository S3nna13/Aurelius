"""Tests for src.computer_use.screenshot_diff."""

from __future__ import annotations

import pytest

from src.computer_use.screenshot_diff import (
    ChangedRegion,
    DiffResult,
    ScreenshotDiff,
    ScreenshotDiffError,
    load_grayscale_pixels,
    load_rgb_pixels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

GRID_10X10_BLACK: list[list[int]] = [[0] * 10 for _ in range(10)]

GRID_10X10_WHITE: list[list[int]] = [[255] * 10 for _ in range(10)]


def _single_pixel_changed() -> tuple[list[list[int]], list[list[int]]]:
    before = [[0] * 10 for _ in range(10)]
    after = [[0] * 10 for _ in range(10)]
    after[5][5] = 255
    return before, after


def _top_left_quadrant() -> tuple[list[list[int]], list[list[int]]]:
    before = [[0] * 20 for _ in range(20)]
    after = [[0] * 20 for _ in range(20)]
    for y in range(10):
        for x in range(10):
            after[y][x] = 255
    return before, after


# ---------------------------------------------------------------------------
# ScreenshotDiff
# ---------------------------------------------------------------------------

class TestScreenshotDiffConstruction:
    def test_default_construction(self):
        sd = ScreenshotDiff()
        assert sd is not None

    def test_custom_noise_threshold(self):
        sd = ScreenshotDiff(noise_threshold=10)
        assert sd is not None

    def test_negative_threshold_raises(self):
        with pytest.raises(ScreenshotDiffError):
            ScreenshotDiff(noise_threshold=-1)


class TestScreenshotDiffIdentical:
    def test_identical_images_no_regions(self):
        sd = ScreenshotDiff()
        result = sd.compare(GRID_10X10_BLACK, GRID_10X10_BLACK)
        assert len(result.changed_regions) == 0
        assert result.total_changed_pixels == 0

    def test_identical_images_changed_ratio_zero(self):
        sd = ScreenshotDiff()
        result = sd.compare(GRID_10X10_BLACK, GRID_10X10_BLACK)
        assert result.changed_ratio == 0.0

    def test_identical_images_dimensions_set(self):
        sd = ScreenshotDiff()
        result = sd.compare(GRID_10X10_BLACK, GRID_10X10_BLACK)
        assert result.width == 10
        assert result.height == 10


class TestScreenshotDiffCompletelyDifferent:
    def test_all_pixels_changed_one_region(self):
        sd = ScreenshotDiff()
        result = sd.compare(GRID_10X10_BLACK, GRID_10X10_WHITE)
        assert len(result.changed_regions) == 1

    def test_all_changed_pixel_count(self):
        sd = ScreenshotDiff()
        result = sd.compare(GRID_10X10_BLACK, GRID_10X10_WHITE)
        assert result.total_changed_pixels == 100

    def test_all_changed_changed_ratio_one(self):
        sd = ScreenshotDiff()
        result = sd.compare(GRID_10X10_BLACK, GRID_10X10_WHITE)
        assert result.changed_ratio == 1.0

    def test_all_changed_bounding_box_full(self):
        sd = ScreenshotDiff()
        result = sd.compare(GRID_10X10_BLACK, GRID_10X10_WHITE)
        region = result.changed_regions[0]
        assert region.x == 0
        assert region.y == 0
        assert region.width == 10
        assert region.height == 10


class TestScreenshotDiffSinglePixel:
    def test_single_pixel_detected(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before, after = _single_pixel_changed()
        result = sd.compare(before, after)
        assert len(result.changed_regions) == 1

    def test_single_pixel_count(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before, after = _single_pixel_changed()
        result = sd.compare(before, after)
        assert result.total_changed_pixels == 1

    def test_single_pixel_bounding_box(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before, after = _single_pixel_changed()
        result = sd.compare(before, after)
        region = result.changed_regions[0]
        assert region.x == 5
        assert region.y == 5
        assert region.width == 1
        assert region.height == 1

    def test_single_pixel_filtered_by_noise_threshold(self):
        sd = ScreenshotDiff(noise_threshold=100)
        before, after = _single_pixel_changed()
        result = sd.compare(before, after)
        assert len(result.changed_regions) == 0


class TestScreenshotDiffQuadrant:
    def test_quadrant_detected(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before, after = _top_left_quadrant()
        result = sd.compare(before, after)
        assert len(result.changed_regions) == 1

    def test_quadrant_pixel_count(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before, after = _top_left_quadrant()
        result = sd.compare(before, after)
        assert result.total_changed_pixels == 100

    def test_quadrant_bounding_box(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before, after = _top_left_quadrant()
        result = sd.compare(before, after)
        region = result.changed_regions[0]
        assert region.x == 0
        assert region.y == 0
        assert region.width == 10
        assert region.height == 10


class TestScreenshotDiffMultipleRegions:
    def test_two_separate_regions(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before = [[0] * 20 for _ in range(20)]
        after = [[0] * 20 for _ in range(20)]
        # top-left block
        for y in range(5):
            for x in range(5):
                after[y][x] = 255
        # bottom-right block
        for y in range(15, 20):
            for x in range(15, 20):
                after[y][x] = 255
        result = sd.compare(before, after)
        assert len(result.changed_regions) == 2

    def test_two_regions_combined_pixels(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before = [[0] * 20 for _ in range(20)]
        after = [[0] * 20 for _ in range(20)]
        for y in range(5):
            for x in range(5):
                after[y][x] = 255
        for y in range(15, 20):
            for x in range(15, 20):
                after[y][x] = 255
        result = sd.compare(before, after)
        assert result.total_changed_pixels == 50


class TestScreenshotDiffMask:
    def test_mask_excludes_pixels(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before = [[0] * 10 for _ in range(10)]
        after = [[0] * 10 for _ in range(10)]
        mask = [[False] * 10 for _ in range(10)]
        # change pixel at (5,5) but mask it
        after[5][5] = 255
        mask[5][5] = True
        result = sd.compare(before, after, mask=mask)
        assert len(result.changed_regions) == 0

    def test_mask_partial_exclusion(self):
        sd = ScreenshotDiff(noise_threshold=1)
        before = [[0] * 10 for _ in range(10)]
        after = [[0] * 10 for _ in range(10)]
        mask = [[False] * 10 for _ in range(10)]
        # change two pixels, mask one
        after[5][5] = 255
        after[3][3] = 255
        mask[5][5] = True
        result = sd.compare(before, after, mask=mask)
        assert len(result.changed_regions) == 1
        assert result.total_changed_pixels == 1

    def test_mask_wrong_dimensions_raises(self):
        sd = ScreenshotDiff()
        bad_mask = [[False] * 5 for _ in range(5)]
        with pytest.raises(ScreenshotDiffError):
            sd.compare(GRID_10X10_BLACK, GRID_10X10_BLACK, mask=bad_mask)


class TestScreenshotDiffValidation:
    def test_empty_before_raises(self):
        sd = ScreenshotDiff()
        with pytest.raises(ScreenshotDiffError):
            sd.compare([], GRID_10X10_BLACK)

    def test_empty_after_raises(self):
        sd = ScreenshotDiff()
        with pytest.raises(ScreenshotDiffError):
            sd.compare(GRID_10X10_BLACK, [])

    def test_height_mismatch_raises(self):
        sd = ScreenshotDiff()
        taller = [[0] * 10 for _ in range(20)]
        with pytest.raises(ScreenshotDiffError):
            sd.compare(GRID_10X10_BLACK, taller)

    def test_width_mismatch_raises(self):
        sd = ScreenshotDiff()
        wider = [[0] * 20 for _ in range(10)]
        with pytest.raises(ScreenshotDiffError):
            sd.compare(GRID_10X10_BLACK, wider)

    def test_inconsistent_row_width_raises(self):
        sd = ScreenshotDiff()
        ragged = [[0] * 10 for _ in range(10)]
        ragged[5] = [0] * 8
        with pytest.raises(ScreenshotDiffError):
            sd.compare(ragged, GRID_10X10_BLACK)


class TestChangedRegionDataclass:
    def test_area_property(self):
        region = ChangedRegion(
            x=10, y=20, width=100, height=50,
            pixel_count=500, total_pixels=5000, ratio=0.1,
        )
        assert region.area == 5000

    def test_area_of_single_pixel(self):
        region = ChangedRegion(
            x=0, y=0, width=1, height=1,
            pixel_count=1, total_pixels=1, ratio=1.0,
        )
        assert region.area == 1


class TestDiffResultDataclass:
    def test_default_empty(self):
        result = DiffResult()
        assert result.changed_regions == []
        assert result.total_changed_pixels == 0
        assert result.changed_ratio == 0.0

    def test_custom_values(self):
        region = ChangedRegion(
            x=0, y=0, width=10, height=10,
            pixel_count=100, total_pixels=100, ratio=1.0,
        )
        result = DiffResult(
            changed_regions=[region],
            total_changed_pixels=100,
            total_pixels=400,
            width=20,
            height=20,
            changed_ratio=0.25,
        )
        assert result.total_changed_pixels == 100
        assert result.changed_ratio == 0.25


class TestLoadFunctions:
    def test_load_grayscale_pixels_type(self, tmp_path):
        try:
            from PIL import Image
            path = str(tmp_path / "test.png")
            img = Image.new("L", (5, 5), color=128)
            img.save(path)
            pixels = load_grayscale_pixels(path)
            assert len(pixels) == 5
            assert len(pixels[0]) == 5
            assert pixels[2][2] == 128
        except ImportError:
            pytest.skip("PIL not available")

    def test_load_rgb_pixels_type(self, tmp_path):
        try:
            from PIL import Image
            path = str(tmp_path / "test_rgb.png")
            img = Image.new("RGB", (4, 4), color=(255, 0, 0))
            img.save(path)
            pixels = load_rgb_pixels(path)
            assert len(pixels) == 4
            assert len(pixels[0]) == 4
            # red = (255 << 16) = 16711680
            assert pixels[0][0] == (255 << 16)
        except ImportError:
            pytest.skip("PIL not available")
