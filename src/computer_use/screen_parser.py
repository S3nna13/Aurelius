"""Screen parsing abstraction for the Aurelius computer_use surface.

Inspired by OpenDevin/OpenDevin (browser tool), MoonshotAI/Kimi-Dev (coding agent loop),
Apache-2.0, clean-room reimplementation.

No playwright, pyautogui, or OS accessibility API imports. Pure stdlib JSON parsing only.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AccessibilityNode:
    """A node in an accessibility tree."""

    role: str
    name: str
    bbox: tuple[int, int, int, int] | None = None
    children: list[AccessibilityNode] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreenSnapshot:
    """A snapshot of the screen state."""

    width: int
    height: int
    root_node: AccessibilityNode
    ocr_text: str | None = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ScreenParser(ABC):
    """Abstract interface for screen parsers."""

    @abstractmethod
    def parse(self, raw_data: dict) -> ScreenSnapshot:
        """Parse raw data into a ScreenSnapshot.

        Parameters
        ----------
        raw_data:
            Raw dictionary representation of screen state.

        Returns
        -------
        ScreenSnapshot
        """
        ...


# ---------------------------------------------------------------------------
# JSON tree parser (pure stdlib — no OS calls)
# ---------------------------------------------------------------------------


class JSONTreeParser(ScreenParser):
    """Parses an accessibility tree from a JSON dict representation.

    Expected dict schema::

        {
            "width": int,
            "height": int,
            "root": {
                "role": str,
                "name": str,
                "bbox": [x, y, w, h] | null,          # optional
                "attributes": {...},                    # optional
                "children": [<node>, ...]               # optional
            },
            "ocr_text": str | null                      # optional
        }
    """

    def parse(self, raw_data: dict) -> ScreenSnapshot:
        """Parse a JSON-dict accessibility tree into a ScreenSnapshot.

        Raises
        ------
        KeyError
            If required keys are missing from *raw_data*.
        TypeError
            If values have unexpected types.
        ValueError
            If width/height are not positive integers.
        """
        if not isinstance(raw_data, dict):
            raise TypeError(f"raw_data must be a dict, got {type(raw_data).__name__}")

        try:
            width = int(raw_data["width"])
            height = int(raw_data["height"])
        except KeyError as exc:
            raise KeyError(f"Missing required key in raw_data: {exc}") from exc

        if width <= 0 or height <= 0:
            raise ValueError(
                f"width and height must be positive; got width={width}, height={height}"
            )

        if "root" not in raw_data:
            raise KeyError("Missing required key in raw_data: 'root'")

        root_node = self._parse_node(raw_data["root"])
        ocr_text = raw_data.get("ocr_text")
        timestamp = float(raw_data.get("timestamp", time.time()))

        return ScreenSnapshot(
            width=width,
            height=height,
            root_node=root_node,
            ocr_text=ocr_text,
            timestamp=timestamp,
        )

    def _parse_node(self, node_dict: dict) -> AccessibilityNode:
        """Recursively parse a node dict into an AccessibilityNode.

        Raises
        ------
        KeyError
            If 'role' or 'name' keys are missing.
        TypeError
            If node_dict is not a dict.
        """
        if not isinstance(node_dict, dict):
            raise TypeError(f"Node must be a dict, got {type(node_dict).__name__}")

        try:
            role = str(node_dict["role"])
            name = str(node_dict["name"])
        except KeyError as exc:
            raise KeyError(f"Node missing required key: {exc}") from exc

        raw_bbox = node_dict.get("bbox")
        bbox: tuple[int, int, int, int] | None = None
        if raw_bbox is not None:
            if len(raw_bbox) != 4:
                raise ValueError(f"bbox must have 4 elements, got {len(raw_bbox)}")
            bbox = (int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3]))

        attributes: dict[str, Any] = dict(node_dict.get("attributes") or {})

        children: list[AccessibilityNode] = [
            self._parse_node(child) for child in (node_dict.get("children") or [])
        ]

        return AccessibilityNode(
            role=role,
            name=name,
            bbox=bbox,
            children=children,
            attributes=attributes,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCREEN_PARSER_REGISTRY: dict[str, type[ScreenParser]] = {
    "json_tree": JSONTreeParser,
}


def register_screen_parser(name: str, cls: type[ScreenParser]) -> None:
    """Register a ScreenParser implementation under *name*.

    Raises
    ------
    TypeError
        If *cls* is not a subclass of ScreenParser.
    """
    if not (isinstance(cls, type) and issubclass(cls, ScreenParser)):
        raise TypeError(f"{cls!r} must be a subclass of ScreenParser")
    SCREEN_PARSER_REGISTRY[name] = cls


def get_screen_parser(name: str) -> type[ScreenParser]:
    """Retrieve a registered ScreenParser class by *name*.

    Raises
    ------
    KeyError
        If *name* is not in the registry.
    """
    if name not in SCREEN_PARSER_REGISTRY:
        raise KeyError(
            f"No ScreenParser registered as {name!r}. Available: {sorted(SCREEN_PARSER_REGISTRY)}"
        )
    return SCREEN_PARSER_REGISTRY[name]
