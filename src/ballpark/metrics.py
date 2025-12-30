"""Quality metrics for sphere decomposition evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull, QhullError

# Re-export compute_coverage from _spherize (canonical implementation)
from ._spherize import compute_coverage as compute_coverage

if TYPE_CHECKING:
    from ._spherize import Sphere


def compute_tightness(points: np.ndarray, spheres: list["Sphere"]) -> float:
    """Compute tightness: hull_volume / total_sphere_volume.

    Higher is better (max 1.0 for perfect fit).
    A single bounding sphere typically has tightness ~0.05.

    Args:
        points: (N, 3) array of sampled surface points
        spheres: List of Sphere objects

    Returns:
        Tightness ratio (0.0 to 1.0)
    """
    if len(points) < 4 or len(spheres) == 0:
        return 0.0

    try:
        hull_vol = ConvexHull(points).volume
    except (QhullError, ValueError):
        return 0.0

    total_sphere_vol = sum(
        4 / 3 * np.pi * float(s.radius) ** 3 for s in spheres
    )

    if total_sphere_vol < 1e-10:
        return 0.0

    return float(hull_vol / total_sphere_vol)


def compute_volume_overhead(points: np.ndarray, spheres: list["Sphere"]) -> float:
    """Compute volume overhead: total_sphere_volume / hull_volume.

    Lower is better (1.0 = perfect, >1.0 = over-approximation).

    Args:
        points: (N, 3) array of sampled surface points
        spheres: List of Sphere objects

    Returns:
        Volume overhead ratio (>=1.0 typically)
    """
    if len(points) < 4 or len(spheres) == 0:
        return float("inf")

    try:
        hull_vol = ConvexHull(points).volume
    except (QhullError, ValueError):
        return float("inf")

    if hull_vol < 1e-10:
        return float("inf")

    total_sphere_vol = sum(
        4 / 3 * np.pi * float(s.radius) ** 3 for s in spheres
    )

    return float(total_sphere_vol / hull_vol)


def compute_quality(coverage: float, tightness: float) -> float:
    """Compute combined quality score: coverage * tightness.

    Higher is better. Rewards both high coverage and tight fit.

    Args:
        coverage: Fraction of points inside spheres (0-1)
        tightness: hull_vol / sphere_vol ratio (0-1)

    Returns:
        Quality score
    """
    return coverage * tightness
