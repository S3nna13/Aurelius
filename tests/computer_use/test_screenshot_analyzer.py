"""Tests for screenshot_analyzer — screen content analysis."""

from __future__ import annotations

from src.computer_use.screenshot_analyzer import (
    ScreenRegion,
    ScreenshotAnalyzer,
    TextElement,
    UIElement,
    find_text_in_regions,
)


class TestScreenRegion:
    def test_contains_point(self):
        r = ScreenRegion(x=10, y=20, width=100, height=200)
        assert r.contains(50, 100)
        assert not r.contains(0, 0)

    def test_center(self):
        r = ScreenRegion(x=10, y=20, width=100, height=200)
        cx, cy = r.center()
        assert cx == 60
        assert cy == 120

    def test_intersection(self):
        a = ScreenRegion(0, 0, 100, 100)
        b = ScreenRegion(50, 50, 100, 100)
        inter = a.intersection(b)
        assert inter is not None
        assert inter.x == 50
        assert inter.y == 50

    def test_no_intersection(self):
        a = ScreenRegion(0, 0, 10, 10)
        b = ScreenRegion(100, 100, 10, 10)
        assert a.intersection(b) is None


class TestTextElement:
    def test_text_element_creation(self):
        t = TextElement(text="Submit", region=ScreenRegion(10, 10, 50, 20), confidence=0.95)
        assert t.text == "Submit"
        assert t.confidence == 0.95


class TestUIElement:
    def test_ui_element_clicks_center(self):
        e = UIElement(
            element_type="button",
            label="OK",
            region=ScreenRegion(100, 100, 50, 20),
        )
        cx, cy = e.click_point()
        assert cx == 125
        assert cy == 110


class TestScreenshotAnalyzer:
    def test_analyze_empty(self):
        analyzer = ScreenshotAnalyzer()
        result = analyzer.analyze(width=1920, height=1080)
        assert result.width == 1920
        assert result.height == 1080
        assert len(result.text_elements) == 0
        assert len(result.ui_elements) == 0

    def test_register_and_detect_text(self):
        analyzer = ScreenshotAnalyzer()
        analyzer.register_text_element(TextElement("hello", ScreenRegion(0, 0, 50, 20)))
        analyzer.register_text_element(TextElement("world", ScreenRegion(100, 0, 50, 20)))
        result = analyzer.analyze(width=1920, height=1080)
        assert len(result.text_elements) == 2

    def test_find_text_by_content(self):
        analyzer = ScreenshotAnalyzer()
        analyzer.register_text_element(TextElement("Submit", ScreenRegion(10, 10, 50, 20)))
        found = analyzer.find_text("Submit")
        assert len(found) == 1
        assert found[0].text == "Submit"

    def test_find_text_case_insensitive(self):
        analyzer = ScreenshotAnalyzer()
        analyzer.register_text_element(TextElement("Cancel", ScreenRegion(10, 10, 50, 20)))
        found = analyzer.find_text("cancel")
        assert len(found) == 1

    def test_clear_elements(self):
        analyzer = ScreenshotAnalyzer()
        analyzer.register_text_element(TextElement("x", ScreenRegion(0, 0, 10, 10)))
        analyzer.clear()
        result = analyzer.analyze(width=100, height=100)
        assert len(result.text_elements) == 0


class TestFindTextInRegions:
    def test_find_text_in_multiple_regions(self):
        regions = [
            ScreenRegion(0, 0, 100, 50),
            ScreenRegion(0, 50, 100, 50),
        ]
        elements = [
            TextElement("hello", ScreenRegion(10, 10, 30, 10)),
            TextElement("world", ScreenRegion(10, 60, 30, 10)),
            TextElement("other", ScreenRegion(200, 200, 30, 10)),
        ]
        found = find_text_in_regions(elements, regions)
        assert len(found) == 2
        assert all(e.text in ("hello", "world") for e in found)
