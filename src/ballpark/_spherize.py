"""Adaptive tight sphere fitting algorithm."""

from dataclasses import dataclass
from typing import cast, TypedDict

import numpy as np
import trimesh
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, QhullError


@dataclass
class Sphere:
    """A sphere defined by center and radius."""

    center: np.ndarray
    radius: float


class SpherizeConfig(TypedDict):
    """Configuration for spherize function."""

    target_tightness: float
    """max acceptable sphere_vol/hull_vol ratio before splitting"""
    aspect_threshold: float
    """max acceptable aspect ratio before splitting"""
    target_spheres: int
    """target number of spheres to generate (may slightly exceed)"""
    n_samples: int
    """number of surface samples to use"""
    padding: float
    """radius multiplier for safety margin"""
    percentile: float
    """percentile of distances to use for radius (handles outliers)"""
    max_radius_ratio: float
    """cap radius relative to bounding box diagonal"""
    uniform_radius: bool
    """if True, post-process to make radii more uniform (may under-approximate)"""


def spherize(
    mesh: trimesh.Trimesh,
    cfg: SpherizeConfig,
) -> list[Sphere]:
    """
    Adaptive splitting with tight sphere fitting.

    Uses budget-based recursion to target the specified sphere count.
    Splits regions that are too elongated or have poor tightness.
    The actual number of spheres may slightly exceed target_spheres due to
    minimum allocation constraints during recursive splitting.

    Args:
        mesh: The mesh to spherize
        cfg: Configuration parameters

    Returns:
        List of Sphere objects covering the mesh
    """
    # Unpack config
    target_tightness = cfg["target_tightness"]
    aspect_threshold = cfg["aspect_threshold"]
    target_spheres = cfg["target_spheres"]
    n_samples = cfg["n_samples"]
    padding = cfg["padding"]
    percentile = cfg["percentile"]
    max_radius_ratio = cfg["max_radius_ratio"]
    uniform_radius = cfg["uniform_radius"]

    points = cast(np.ndarray, mesh.sample(n_samples))

    # Compute max allowed radius
    bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0)).item()
    max_radius = bbox_diag * max_radius_ratio

    def should_split(pts, sphere, budget):
        if len(pts) < 20 or budget <= 1:
            return False

        aspect = get_aspect_ratio(pts)
        tightness = compute_tightness(pts, sphere)

        # Force split if sphere exceeds max radius
        if sphere.radius > max_radius:
            return True

        # Split if elongated OR loose
        if aspect > aspect_threshold:
            return True
        if tightness > target_tightness:
            return True

        return False

    def split(pts, budget):
        if len(pts) < 15 or budget <= 0:
            if len(pts) > 0 and budget > 0:
                s = fit_sphere_minmax(pts, padding, percentile)
                s.radius = min(s.radius, max_radius)  # cap radius
                return [s]
            return []

        sphere = fit_sphere_minmax(pts, padding, percentile)

        if not should_split(pts, sphere, budget):
            return [sphere]

        # Split along principal axis
        pca = PCA(n_components=1)
        pca.fit(pts)
        proj = pts @ pca.components_[0]
        # Blend median with geometric midpoint for more symmetric splits
        proj_mid = (proj.min() + proj.max()) / 2
        split_point = 0.5 * np.median(proj) + 0.5 * proj_mid

        left = pts[proj <= split_point]
        right = pts[proj > split_point]

        # Allocate budget proportionally to point count
        left_frac = len(left) / len(pts)
        left_budget = max(1, int(round(budget * left_frac)))
        right_budget = max(1, budget - left_budget)

        result = []
        if len(left) >= 10:
            result.extend(split(left, left_budget))
        if len(right) >= 10:
            result.extend(split(right, right_budget))

        return result if result else [sphere]

    spheres = split(points, target_spheres)

    # Post-process: cap radius variance for more uniformity (may cause under-approximation)
    if uniform_radius and len(spheres) > 1:
        radii = np.array([s.radius for s in spheres])
        median_radius = np.median(radii)
        for s in spheres:
            s.radius = np.clip(s.radius, median_radius * 0.4, median_radius * 2.5)

    return spheres


def fit_sphere_minmax(
    points: np.ndarray, padding: float = 1.0, percentile: float = 98.0
) -> Sphere:
    """
    Tighter sphere fitting using iterative refinement.

    Args:
        points: (N, 3) array of points to enclose
        padding: Radius multiplier for safety margin (1.02 = 2% larger)
        percentile: Use this percentile of distances instead of max (handles outliers)

    Returns:
        Sphere that encloses the points
    """
    if len(points) < 4:
        c = points.mean(axis=0) if len(points) > 0 else np.zeros(3)
        if len(points) > 0:
            r = np.linalg.norm(points - c, axis=1).max()
            r = max(r, 1e-4)  # Ensure minimum radius for degenerate cases
        else:
            r = 0.01
        return Sphere(c, r * padding)

    center = points.mean(axis=0)

    # Iterative refinement: move center to reduce max distance
    for _ in range(5):
        dists = np.linalg.norm(points - center, axis=1)
        farthest = points[np.argmax(dists)]
        # Move center slightly toward farthest point
        center = center + 0.1 * (farthest - center)

    dists = np.linalg.norm(points - center, axis=1)
    # Blend percentile with median for more uniform radii (keep mostly percentile to avoid under-approx)
    p_high = np.percentile(dists, percentile)
    p_med = np.median(dists)
    radius = (0.85 * p_high + 0.15 * p_med) * padding

    return Sphere(center, radius)


def get_aspect_ratio(points: np.ndarray) -> float:
    """Compute aspect ratio from PCA eigenvalues."""
    if len(points) < 10:
        return 1.0
    pca = PCA(n_components=min(3, len(points)))
    pca.fit(points)
    var = pca.explained_variance_
    if len(var) < 2 or var[0] < 1e-10 or var[1] < 1e-10:
        return 1.0
    # Cap the ratio to avoid extreme values for degenerate cases
    ratio = np.sqrt(var[0] / var[1])
    return min(ratio, 100.0)


def compute_tightness(points: np.ndarray, sphere: Sphere) -> float:
    """Compute sphere volume / convex hull volume ratio."""
    if len(points) < 4:
        return 1.0
    try:
        hull_vol = ConvexHull(points).volume
        sphere_vol = 4 / 3 * np.pi * sphere.radius**3
        return sphere_vol / (hull_vol + 1e-10)
    except (QhullError, ValueError):
        return 1.0
