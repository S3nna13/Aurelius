"""Tests for src.computer_use.screen_analyzer."""

from __future__ import annotations

from src.computer_use.screen_analyzer import (
    SCREEN_ANALYZER_REGISTRY,
    ScreenAnalyzer,
)


class TestDetectElements:
    def test_detect_single_element(self):
        analyzer = ScreenAnalyzer()
        red = (255, 0, 0)
        black = (0, 0, 0)
        pixel_data = [
            [black, black, black],
            [black, red, red],
            [black, red, red],
        ]
        elements = analyzer.detect_elements(pixel_data)
        assert len(elements) == 2
        black_elem = next(e for e in elements if e["dominant_color"] == black)
        red_elem = next(e for e in elements if e["dominant_color"] == red)
        assert black_elem["pixel_count"] == 5
        assert red_elem["pixel_count"] == 4
        assert red_elem["x"] == 1
        assert red_elem["y"] == 1
        assert red_elem["width"] == 2
        assert red_elem["height"] == 2

    def test_detect_multiple_elements(self):
        analyzer = ScreenAnalyzer()
        blue = (0, 0, 255)
        green = (0, 255, 0)
        bg = (255, 255, 255)
        pixel_data = [
            [bg, blue, blue, bg, green, green],
            [bg, blue, blue, bg, green, green],
            [bg, bg, bg, bg, bg, bg],
        ]
        elements = analyzer.detect_elements(pixel_data)
        assert len(elements) == 3
        colors = {e["dominant_color"] for e in elements}
        assert colors == {bg, blue, green}
        blue_elem = next(e for e in elements if e["dominant_color"] == blue)
        assert blue_elem["width"] == 2
        assert blue_elem["height"] == 2

    def test_detect_elements_empty(self):
        analyzer = ScreenAnalyzer()
        assert analyzer.detect_elements([]) == []
        assert analyzer.detect_elements([[]]) == []


class TestFindTextRegions:
    def test_find_text_region_uniform_band(self):
        analyzer = ScreenAnalyzer()
        white = (255, 255, 255)
        black = (0, 0, 0)
        gray = (128, 128, 128)
        pixel_data = [
            [white, white, white, white],  # 0 changes
            [white, white, black, black],  # 1 change
            [white, white, white, white],  # 0 changes
            [gray, white, gray, white],  # 3 changes
            [gray, white, gray, white],  # 3 changes
            [white, white, white, white],  # 0 changes
        ]
        regions = analyzer.find_text_regions(pixel_data)
        assert len(regions) == 1
        region = regions[0]
        assert region["y"] == 0
        assert region["height"] == 3
        assert region["width"] == 4

    def test_find_text_regions_empty(self):
        analyzer = ScreenAnalyzer()
        assert analyzer.find_text_regions([]) == []


class TestCountColorChanges:
    def test_no_changes(self):
        analyzer = ScreenAnalyzer()
        row = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]
        assert analyzer.count_color_changes(row) == 0

    def test_single_change(self):
        analyzer = ScreenAnalyzer()
        row = [(0, 0, 0), (255, 255, 255), (255, 255, 255)]
        assert analyzer.count_color_changes(row) == 1

    def test_multiple_changes(self):
        analyzer = ScreenAnalyzer()
        row = [(0, 0, 0), (255, 255, 255), (0, 0, 0), (255, 255, 255)]
        assert analyzer.count_color_changes(row) == 3

    def test_empty_and_single(self):
        analyzer = ScreenAnalyzer()
        assert analyzer.count_color_changes([]) == 0
        assert analyzer.count_color_changes([(1, 2, 3)]) == 0


class TestExtractDominantColor:
    def test_single_color(self):
        analyzer = ScreenAnalyzer()
        region = [
            [(10, 20, 30), (10, 20, 30)],
            [(10, 20, 30), (10, 20, 30)],
        ]
        assert analyzer.extract_dominant_color(region) == (10, 20, 30)

    def test_mixed_colors(self):
        analyzer = ScreenAnalyzer()
        region = [
            [(0, 0, 0), (255, 255, 255), (255, 255, 255)],
            [(0, 0, 0), (255, 255, 255), (255, 255, 255)],
        ]
        assert analyzer.extract_dominant_color(region) == (255, 255, 255)

    def test_empty_region(self):
        analyzer = ScreenAnalyzer()
        assert analyzer.extract_dominant_color([]) == (0, 0, 0)
        assert analyzer.extract_dominant_color([[]]) == (0, 0, 0)


class TestScreenAnalyzerRegistry:
    def test_registry_contains_default(self):
        assert "default" in SCREEN_ANALYZER_REGISTRY

    def test_registry_maps_to_correct_class(self):
        assert SCREEN_ANALYZER_REGISTRY["default"] is ScreenAnalyzer
