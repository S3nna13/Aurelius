"""Screenshot comparison for the Aurelius computer_use surface.

Compares two screenshots and identifies changed regions using pixel-level
diffing.  Pure stdlib with optional Pillow dependency for image I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import IO


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ScreenshotDiffError(Exception):
    """Raised when screenshot diffing fails or receives invalid input."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChangedRegion:
    """A bounding box of changed pixels between two screenshots."""

    x: int
    y: int
    width: int
    height: int
    pixel_count: int
    total_pixels: int
    ratio: float

    @property
    def area(self) -> int:
        """Total pixel area of this bounding box."""
        return self.width * self.height


@dataclass
class DiffResult:
    """Container for the results of a screenshot comparison."""

    changed_regions: list[ChangedRegion] = field(default_factory=list)
    total_changed_pixels: int = 0
    total_pixels: int = 0
    width: int = 0
    height: int = 0
    changed_ratio: float = 0.0


# ---------------------------------------------------------------------------
# Main diffing interface
# ---------------------------------------------------------------------------

class ScreenshotDiff:
    """Compares two screenshots and identifies changed regions.

    Parameters
    ----------
    noise_threshold:
        Minimum number of changed pixels required in a connected component
        for it to be reported.  Components below this size are treated as
        noise and discarded.  Default 50.

    Supports:

    * Per-pixel difference computation on grayscale or packed-RGB grids.
    * Connected-component labelling (4-neighbour) to group changed pixels
      into bounding boxes.
    * Noise filtering via ``noise_threshold``.
    * Optional mask grid to exclude regions from comparison.
    """

    def __init__(self, noise_threshold: int = 50) -> None:
        if noise_threshold < 0:
            raise ScreenshotDiffError(
                f"noise_threshold must be non-negative, got {noise_threshold}"
            )
        self._noise_threshold: int = noise_threshold

    def compare(
        self,
        before: list[list[int]],
        after: list[list[int]],
        mask: list[list[bool]] | None = None,
    ) -> DiffResult:
        """Compare two pixel grids and return changed regions.

        Parameters
        ----------
        before:
            2-D list of integer pixel values (grayscale or packed RGB).
        after:
            2-D list of integer pixel values, same dimensions as *before*.
        mask:
            Optional 2-D boolean grid where ``True`` means *ignore* this
            pixel during comparison.

        Returns
        -------
        DiffResult

        Raises
        ------
        ScreenshotDiffError
            If inputs are empty, dimensions mismatch, or mask has wrong shape.
        """
        if not before or not after:
            raise ScreenshotDiffError("before and after must be non-empty")

        height = len(before)
        width = len(before[0])

        if len(after) != height:
            raise ScreenshotDiffError(
                f"height mismatch: before has {height} rows, after has {len(after)}"
            )
        for y in range(height):
            if len(before[y]) != width:
                raise ScreenshotDiffError(
                    f"before row {y} has inconsistent width"
                )
            if len(after[y]) != width:
                raise ScreenshotDiffError(
                    f"after row {y} has inconsistent width"
                )

        if mask is not None:
            if len(mask) != height:
                raise ScreenshotDiffError("mask height must match image height")
            for y in range(height):
                if len(mask[y]) != width:
                    raise ScreenshotDiffError(
                        f"mask row {y} has inconsistent width"
                    )

        changed_map = self._compute_diff(before, after, width, height, mask)
        total_pixels = width * height

        regions = self._find_regions(changed_map, width, height)

        regions = [r for r in regions if r.pixel_count >= self._noise_threshold]
        total_changed = sum(r.pixel_count for r in regions)

        return DiffResult(
            changed_regions=regions,
            total_changed_pixels=total_changed,
            total_pixels=total_pixels,
            width=width,
            height=height,
            changed_ratio=total_changed / total_pixels if total_pixels > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_diff(
        before: list[list[int]],
        after: list[list[int]],
        width: int,
        height: int,
        mask: list[list[bool]] | None,
    ) -> list[list[bool]]:
        """Return a boolean grid marking pixels that differ between *before* and *after*."""
        changed: list[list[bool]] = [[False] * width for _ in range(height)]

        for y in range(height):
            by = before[y]
            ay = after[y]
            cy = changed[y]
            my = mask[y] if mask else None
            for x in range(width):
                if my is not None and my[x]:
                    continue
                if by[x] != ay[x]:
                    cy[x] = True

        return changed

    def _find_regions(
        self,
        changed: list[list[bool]],
        width: int,
        height: int,
    ) -> list[ChangedRegion]:
        """Group changed pixels into connected-component bounding boxes."""
        visited: list[list[bool]] = [[False] * width for _ in range(height)]
        regions: list[ChangedRegion] = []

        for y in range(height):
            for x in range(width):
                if not changed[y][x] or visited[y][x]:
                    continue
                pixels, min_x, min_y, max_x, max_y = self._flood_fill(
                    changed, visited, x, y, width, height,
                )
                rw = max_x - min_x + 1
                rh = max_y - min_y + 1
                total_region = rw * rh
                regions.append(ChangedRegion(
                    x=min_x,
                    y=min_y,
                    width=rw,
                    height=rh,
                    pixel_count=pixels,
                    total_pixels=total_region,
                    ratio=pixels / total_region if total_region > 0 else 0.0,
                ))

        return regions

    @staticmethod
    def _flood_fill(
        changed: list[list[bool]],
        visited: list[list[bool]],
        start_x: int,
        start_y: int,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int, int]:
        """4-direction flood fill; returns (pixel_count, min_x, min_y, max_x, max_y)."""
        stack = [(start_x, start_y)]
        count = 0
        min_x = max_x = start_x
        min_y = max_y = start_y

        while stack:
            cx, cy = stack.pop()
            if cx < 0 or cx >= width or cy < 0 or cy >= height:
                continue
            if not changed[cy][cx] or visited[cy][cx]:
                continue
            visited[cy][cx] = True
            count += 1

            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy

            stack.append((cx - 1, cy))
            stack.append((cx + 1, cy))
            stack.append((cx, cy - 1))
            stack.append((cx, cy + 1))

        return count, min_x, min_y, max_x, max_y


# ---------------------------------------------------------------------------
# Image I/O helpers (PIL preferred, PPM fallback)
# ---------------------------------------------------------------------------

def load_grayscale_pixels(path: str) -> list[list[int]]:
    """Load an image as a 2-D grayscale pixel array.

    Uses Pillow when available; falls back to a basic PGM/PPM reader.
    """
    try:
        from PIL import Image

        img = Image.open(path).convert("L")
        w, h = img.size
        pixels = list(img.get_flattened_data())
        return [pixels[i * w : (i + 1) * w] for i in range(h)]
    except ImportError:
        return _load_ppm_grayscale(path)


def load_rgb_pixels(path: str) -> list[list[int]]:
    """Load an image as a 2-D array of packed 24-bit RGB integers.

    Requires Pillow.
    """
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        w, h = img.size
        rows: list[list[int]] = []
        for y in range(h):
            row: list[int] = []
            for x in range(w):
                r, g, b = img.getpixel((x, y))
                row.append((r << 16) | (g << 8) | b)
            rows.append(row)
        return rows
    except ImportError:
        raise ScreenshotDiffError(
            "color image loading requires Pillow (PIL)"
        )


def _load_ppm_grayscale(path: str) -> list[list[int]]:
    """Read a PGM or PPM file as grayscale (pure stdlib fallback)."""
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic not in (b"P5", b"P6"):
            raise ScreenshotDiffError(
                f"Unsupported PPM format {magic!r}; "
                f"expected P5 (grayscale) or P6 (colour)"
            )
        # skip comments
        line = f.readline()
        while line.startswith(b"#"):
            line = f.readline()
        dims = line.strip().split()
        w, h = int(dims[0]), int(dims[1])
        max_val = int(f.readline().strip())
        expected = w * h * (3 if magic == b"P6" else 1)
        data = f.read(expected)
        if len(data) < expected:
            raise ScreenshotDiffError(
                "Unexpected end-of-file in PPM image"
            )
        if magic == b"P6":
            pixels = [
                (data[i] + data[i + 1] + data[i + 2]) // 3
                for i in range(0, expected, 3)
            ]
        else:
            pixels = list(data)
        return [pixels[i * w : (i + 1) * w] for i in range(h)]
