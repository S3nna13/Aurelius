"""Screen analyzer — pure-Python pixel-grid element detection.

No opencv, pillow, or OS API imports. Operates on 2-D lists of RGB tuples.
"""

from __future__ import annotations

from typing import Any


class ScreenAnalyzer:
    """Analyzes a mock screen buffer (2-D RGB grid) to detect UI elements."""

    def __init__(self) -> None:
        self._threshold = 30.0

    def detect_elements(
        self,
        pixel_data: list[list[tuple[int, int, int]]],
    ) -> list[dict[str, Any]]:
        """Return contiguous regions of similar color as element dicts.

        Each dict contains ``x``, ``y``, ``width``, ``height``,
        ``pixel_count``, and ``dominant_color``.
        """
        if not pixel_data or not pixel_data[0]:
            return []

        height = len(pixel_data)
        width = len(pixel_data[0])
        visited = [[False] * width for _ in range(height)]
        elements: list[dict[str, Any]] = []

        for y in range(height):
            for x in range(width):
                if visited[y][x]:
                    continue
                seed_color = pixel_data[y][x]
                stack = [(x, y)]
                visited[y][x] = True
                pixels: list[tuple[int, int]] = []
                min_x = x
                min_y = y
                max_x = x
                max_y = y

                while stack:
                    cx, cy = stack.pop()
                    pixels.append((cx, cy))
                    if cx < min_x:
                        min_x = cx
                    if cy < min_y:
                        min_y = cy
                    if cx > max_x:
                        max_x = cx
                    if cy > max_y:
                        max_y = cy
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height and not visited[ny][nx]:
                            dist = self._color_distance(pixel_data[ny][nx], seed_color)
                            if dist <= self._threshold:
                                visited[ny][nx] = True
                                stack.append((nx, ny))

                region_pixels = [
                    [pixel_data[py][px] for px in range(min_x, max_x + 1)]
                    for py in range(min_y, max_y + 1)
                ]
                dominant = self.extract_dominant_color(region_pixels)
                elements.append(
                    {
                        "x": min_x,
                        "y": min_y,
                        "width": max_x - min_x + 1,
                        "height": max_y - min_y + 1,
                        "pixel_count": len(pixels),
                        "dominant_color": dominant,
                    }
                )

        return elements

    def find_text_regions(
        self,
        pixel_data: list[list[tuple[int, int, int]]],
    ) -> list[dict[str, Any]]:
        """Heuristic: return regions with high horizontal color uniformity.

        Consecutive rows whose color-change count is low are merged into
        rectangular regions.
        """
        if not pixel_data or not pixel_data[0]:
            return []

        height = len(pixel_data)
        width = len(pixel_data[0])
        regions: list[dict[str, Any]] = []
        in_region = False
        start_y = 0

        for y in range(height):
            changes = self.count_color_changes(pixel_data[y])
            uniform = changes <= max(1, width // 4)
            if uniform and not in_region:
                in_region = True
                start_y = y
            elif not uniform and in_region:
                region_height = y - start_y
                if region_height >= 3:
                    regions.append(
                        {
                            "x": 0,
                            "y": start_y,
                            "width": width,
                            "height": region_height,
                        }
                    )
                in_region = False

        if in_region:
            region_height = height - start_y
            if region_height >= 3:
                regions.append(
                    {
                        "x": 0,
                        "y": start_y,
                        "width": width,
                        "height": region_height,
                    }
                )

        return regions

    def count_color_changes(self, row: list[tuple[int, int, int]]) -> int:
        """Count the number of times the color changes between adjacent pixels."""
        if len(row) <= 1:
            return 0
        count = 0
        prev = row[0]
        for color in row[1:]:
            if color != prev:
                count += 1
                prev = color
        return count

    def extract_dominant_color(
        self,
        region: list[list[tuple[int, int, int]]],
    ) -> tuple[int, int, int]:
        """Return the most frequent RGB value in *region*."""
        frequency: dict[tuple[int, int, int], int] = {}
        for row in region:
            for color in row:
                frequency[color] = frequency.get(color, 0) + 1
        if not frequency:
            return (0, 0, 0)
        return max(frequency, key=lambda k: frequency[k])

    @staticmethod
    def _color_distance(
        a: tuple[int, int, int],
        b: tuple[int, int, int],
    ) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


SCREEN_ANALYZER_REGISTRY: dict[str, type[ScreenAnalyzer]] = {
    "default": ScreenAnalyzer,
}
