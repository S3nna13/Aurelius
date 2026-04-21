"""Programmatic Image Tools for Zero-Vision SFT (Kimi K2.5 §4, arXiv:2602.02276).

These are proxy operations enabling the model to learn visual reasoning via
tool-call patterns that invoke deterministic functions on Tensor image
representations, without requiring real vision+text training data.

Image representation: torch.Tensor[C, H, W], dtype=float32, values in [0, 1].
Box format: (x1, y1, x2, y2) in pixel coords (int), where x=col, y=row.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import Tensor

__all__ = [
    "crop_region",
    "pixel_distance",
    "blob_count",
    "detect_objects",
]


def crop_region(image: Tensor, box: tuple[int, int, int, int]) -> Tensor:
    """Return a cropped sub-image defined by *box*.

    Args:
        image: Float tensor of shape [C, H, W] with values in [0, 1].
        box: (x1, y1, x2, y2) in pixel coords.  x=col, y=row.
             x2 > x1, y2 > y1 (before clamping).

    Returns:
        Cropped tensor of shape [C, (y2-y1), (x2-x1)] after clamping *box*
        to valid image bounds.  A box entirely outside the image returns a
        minimum 1×1 slice so callers never receive an empty tensor.
    """
    _check_image(image)
    _, H, W = image.shape
    x1, y1, x2, y2 = box

    # Clamp to valid range
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))

    return image[:, y1:y2, x1:x2]


def pixel_distance(
    image: Tensor,  # noqa: ARG001 — kept for API symmetry
    pt1: tuple[int, int],
    pt2: tuple[int, int],
) -> float:
    """Return Euclidean distance between two pixel coordinates.

    Args:
        image: Not used in the distance computation but kept in the signature
               for tool-call API consistency.
        pt1: (x, y) of the first point.
        pt2: (x, y) of the second point.

    Returns:
        Euclidean distance as a Python float.
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    return float(math.sqrt(dx * dx + dy * dy))


def blob_count(
    image: Tensor,
    threshold: float = 0.5,
    min_size: int = 1,
) -> int:
    """Count connected blobs of pixels above *threshold*.

    The image is first converted to a single-channel binary mask by averaging
    across the channel dimension.  Connected components are found with BFS
    (4-connectivity).  Blobs with fewer than *min_size* pixels are excluded.

    Args:
        image: Float tensor of shape [C, H, W].
        threshold: Pixel value threshold; a pixel is "bright" when its mean
                   channel value exceeds this.
        min_size: Minimum number of pixels for a region to count as a blob.

    Returns:
        Number of qualifying blobs (int >= 0).
    """
    _check_image(image)
    mask = _binary_mask(image, threshold)
    _, labels = _connected_components(mask)
    return _count_blobs(labels, min_size)


def detect_objects(
    image: Tensor,
    threshold: float = 0.5,
    min_size: int = 4,
) -> list[dict]:
    """Detect blobs as bounding boxes, areas, and centroids.

    Args:
        image: Float tensor of shape [C, H, W].
        threshold: Pixel value threshold (same semantics as *blob_count*).
        min_size: Minimum pixel area for a region to be reported.

    Returns:
        List of dicts, one per qualifying blob::

            {
                "box": (x1, y1, x2, y2),   # tight bounding box (ints)
                "area": int,               # pixel count
                "centroid": (cx, cy),      # float centroid (col, row)
            }

        The list is ordered by discovery (top-left raster scan order).
    """
    _check_image(image)
    mask = _binary_mask(image, threshold)
    num_labels, labels = _connected_components(mask)

    objects: list[dict] = []
    for lbl in range(1, num_labels + 1):
        positions = (labels == lbl).nonzero(as_tuple=False)  # [N, 2] → (row, col)
        area = positions.shape[0]
        if area < min_size:
            continue

        rows = positions[:, 0]
        cols = positions[:, 1]

        y1 = int(rows.min().item())
        y2 = int(rows.max().item()) + 1  # exclusive
        x1 = int(cols.min().item())
        x2 = int(cols.max().item()) + 1  # exclusive

        cx = float(cols.float().mean().item())
        cy = float(rows.float().mean().item())

        objects.append(
            {
                "box": (x1, y1, x2, y2),
                "area": area,
                "centroid": (cx, cy),
            }
        )

    return objects


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_image(image: Tensor) -> None:
    """Raise ValueError for obviously malformed inputs."""
    if not isinstance(image, Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image)}")
    if image.ndim != 3:
        raise ValueError(f"image must have 3 dimensions [C, H, W], got {image.ndim}")


def _binary_mask(image: Tensor, threshold: float) -> Tensor:
    """Return a 2-D boolean mask [H, W] where mean channel value > threshold."""
    # Average across channels → [H, W]
    mean_img = image.float().mean(dim=0)
    return mean_img > threshold


def _connected_components(mask: Tensor) -> tuple[int, Tensor]:
    """Label connected components (4-connectivity) in a 2-D boolean mask.

    Uses iterative BFS implemented in pure Python so there is no scipy
    dependency.

    Returns:
        (num_labels, label_tensor) where label_tensor has the same shape as
        *mask* and contains 0 for background, 1..num_labels for components.
    """
    H, W = mask.shape
    labels = torch.zeros(H, W, dtype=torch.int32)
    current_label = 0

    # Convert mask to a Python bool list-of-lists for O(1) neighbour access
    # without repeated tensor indexing overhead.
    visited = [[False] * W for _ in range(H)]
    bright = mask.tolist()  # list[list[bool]]

    for r0 in range(H):
        for c0 in range(W):
            if bright[r0][c0] and not visited[r0][c0]:
                current_label += 1
                queue: deque[tuple[int, int]] = deque()
                queue.append((r0, c0))
                visited[r0][c0] = True
                labels[r0, c0] = current_label

                while queue:
                    r, c = queue.popleft()
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W:
                            if bright[nr][nc] and not visited[nr][nc]:
                                visited[nr][nc] = True
                                labels[nr, nc] = current_label
                                queue.append((nr, nc))

    return current_label, labels


def _count_blobs(labels: Tensor, min_size: int) -> int:
    """Count labels (1-based) whose pixel count is >= min_size."""
    if labels.max().item() == 0:
        return 0
    count = 0
    max_label = int(labels.max().item())
    for lbl in range(1, max_label + 1):
        if int((labels == lbl).sum().item()) >= min_size:
            count += 1
    return count
