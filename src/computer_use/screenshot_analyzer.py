from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScreenRegion:
    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0

    def contains(self, px: float, py: float) -> bool:
        return self.x <= px <= self.x + self.width and self.y <= py <= self.y + self.height

    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def intersection(self, other: ScreenRegion) -> ScreenRegion | None:
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        if x1 < x2 and y1 < y2:
            return ScreenRegion(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
        return None


@dataclass
class TextElement:
    text: str
    region: ScreenRegion
    confidence: float = 1.0


@dataclass
class UIElement:
    element_type: str
    label: str
    region: ScreenRegion

    def click_point(self) -> tuple[float, float]:
        return self.region.center()


@dataclass
class ScreenshotAnalysis:
    width: int
    height: int
    text_elements: list[TextElement] = field(default_factory=list)
    ui_elements: list[UIElement] = field(default_factory=list)


class ScreenshotAnalyzer:
    def __init__(self) -> None:
        self._text_elements: list[TextElement] = []

    def register_text_element(self, element: TextElement) -> None:
        self._text_elements.append(element)

    def analyze(self, width: int, height: int) -> ScreenshotAnalysis:
        return ScreenshotAnalysis(
            width=width,
            height=height,
            text_elements=list(self._text_elements),
        )

    def find_text(self, text: str, case_sensitive: bool = False) -> list[TextElement]:
        if case_sensitive:
            return [e for e in self._text_elements if text in e.text]
        text_lower = text.lower()
        return [e for e in self._text_elements if text_lower in e.text.lower()]

    def clear(self) -> None:
        self._text_elements.clear()


def find_text_in_regions(elements: list[TextElement], regions: list[ScreenRegion]) -> list[TextElement]:
    return [e for e in elements if any(r.contains(e.region.x, e.region.y) for r in regions)]


SCREENSHOT_ANALYZER = ScreenshotAnalyzer()
