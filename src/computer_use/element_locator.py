"""
element_locator.py
Locates UI elements by selector or spatial position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ElementQuery:
    selector: str
    selector_type: str = "css"
    region: Optional[tuple[int, int, int, int]] = None  # (x, y, w, h)


@dataclass(frozen=True)
class LocatedElement:
    element_id: str
    selector: str
    x: int
    y: int
    width: int
    height: int
    text: str = ""
    visible: bool = True

    def center(self) -> tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def area(self) -> int:
        return self.width * self.height


class ElementLocator:
    def __init__(self) -> None:
        self._elements: list[LocatedElement] = []

    def register(self, element: LocatedElement) -> None:
        self._elements.append(element)

    def find(self, query: ElementQuery) -> list[LocatedElement]:
        needle = query.selector.lower()
        results: list[LocatedElement] = []
        for el in self._elements:
            if needle not in el.selector.lower():
                continue
            if query.region is not None:
                rx, ry, rw, rh = query.region
                cx, cy = el.center()
                if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
                    continue
            results.append(el)
        return results

    def find_at(self, x: int, y: int) -> Optional[LocatedElement]:
        for el in self._elements:
            if el.x <= x <= el.x + el.width and el.y <= y <= el.y + el.height:
                return el
        return None

    def visible_elements(self) -> list[LocatedElement]:
        return [el for el in self._elements if el.visible]

    def clear(self) -> None:
        self._elements.clear()


ELEMENT_LOCATOR_REGISTRY: dict[str, type] = {"default": ElementLocator}

REGISTRY = ELEMENT_LOCATOR_REGISTRY
