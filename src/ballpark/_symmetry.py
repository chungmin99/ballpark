"""Point cloud symmetry detection for sphere decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA


def get_primitive_symmetry_planes(
    points: np.ndarray, geometry_type: str
) -> list[np.ndarray]:
    """
    Get known symmetry planes for geometric primitives.

    Args:
        points: (N, 3) array of points sampled from the primitive
        geometry_type: One of 'box', 'cylinder', 'sphere'

    Returns:
        List of plane normals representing symmetry planes
    """
    if len(points) < 20:
        return []

    if geometry_type == "sphere":
        # Sphere has infinite symmetry, no need for special handling
        return []

    # Center the points to get better PCA alignment
    centroid = points.mean(axis=0)
    centered = points - centroid

    if geometry_type == "box":
        # Boxes have 3 reflection planes along cardinal axes
        # Use PCA to find the oriented axes of the box
        pca = PCA(n_components=3)
        pca.fit(centered)
        # Return all 3 principal axes as symmetry planes
        return [pca.components_[i] / np.linalg.norm(pca.components_[i]) for i in range(3)]

    elif geometry_type == "cylinder":
        # Cylinder has 1 reflection plane perpendicular to its axis
        # and infinite rotational symmetry around the axis
        # Use PCA to find the cylinder axis (largest variance)
        pca = PCA(n_components=3)
        pca.fit(centered)

        # Cylinder axis is the first principal component
        cylinder_axis = pca.components_[0] / np.linalg.norm(pca.components_[0])

        # The perpendicular plane (normal = cylinder axis) is a symmetry plane
        # We also have 2 perpendicular planes through the cylinder axis
        perpendicular_planes = []

        # Get two orthogonal directions perpendicular to cylinder axis
        # Use the second and third PCA components if available
        if pca.n_components_ >= 2:
            perp1 = pca.components_[1] / np.linalg.norm(pca.components_[1])
            perpendicular_planes.append(perp1)
        if pca.n_components_ >= 3:
            perp2 = pca.components_[2] / np.linalg.norm(pca.components_[2])
            perpendicular_planes.append(perp2)

        # Add the perpendicular-to-axis plane (splits cylinder along height)
        perpendicular_planes.append(cylinder_axis)

        return perpendicular_planes

    return []


@dataclass
class SymmetryInfo:
    """Detected symmetry information for a point cloud."""

    reflection_planes: list[np.ndarray]  # List of plane normals
    rotation_axes: list[tuple[np.ndarray, int]]  # (axis, fold) tuples
    symmetry_score: float  # 0-1, how symmetric the shape is

    @property
    def has_symmetry(self) -> bool:
        return len(self.reflection_planes) > 0 or len(self.rotation_axes) > 0

    @property
    def preferred_split_axes(self) -> list[np.ndarray]:
        """Axes that should be preferred for splitting to maintain symmetry."""
        axes = []
        # Reflection planes are natural split boundaries
        for normal in self.reflection_planes:
            axes.append(normal)
        return axes


def detect_symmetry(
    points: np.ndarray,
    n_samples: int = 500,
    tolerance: float = 0.02,
) -> SymmetryInfo:
    """
    Detect reflection and rotational symmetry in a point cloud.

    Uses sampling and distance-based matching to find symmetry planes.

    Args:
        points: (N, 3) array of points
        n_samples: Number of points to use for detection (for efficiency)
        tolerance: Relative tolerance for symmetry matching (scaled by point density)

    Returns:
        SymmetryInfo with detected symmetry elements
    """
    if len(points) < 20:
        return SymmetryInfo([], [], 0.0)

    # Subsample for efficiency
    if len(points) > n_samples:
        indices = np.random.choice(len(points), n_samples, replace=False)
        pts = points[indices]
    else:
        pts = points.copy()

    # Center the points
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    # Get bounding box scale for tolerance
    extents = pts.max(axis=0) - pts.min(axis=0)
    scale = np.max(extents)
    if scale < 1e-10:
        return SymmetryInfo([], [], 0.0)

    # Scale tolerance with point density: sparser samples need larger tolerance
    # Expected spacing for surface samples: sqrt(surface_area / n_samples)
    # Approximate surface area as 6*scale^2 (cube-like)
    expected_spacing = np.sqrt(6 * scale**2 / len(pts))
    # Tolerance should be at least 2x expected spacing to reliably find matches
    density_factor = max(1.0, 2.0 * expected_spacing / (tolerance * scale))
    abs_tol = tolerance * scale * density_factor

    reflection_planes: list[np.ndarray] = []
    rotation_axes: list[tuple[np.ndarray, int]] = []

    # Get candidate axes: PCA + cardinal axes for axis-aligned shapes
    pca_axes = _get_pca_axes(centered)
    candidate_axes = _get_candidate_axes(centered, pca_axes)

    for axis in candidate_axes:
        score = _test_reflection_symmetry(centered, axis, abs_tol)
        if score > 0.9:  # High symmetry score
            # Avoid adding duplicate/similar axes
            is_duplicate = any(abs(np.dot(axis, p)) > 0.95 for p in reflection_planes)
            if not is_duplicate:
                reflection_planes.append(axis)

    # Test for rotational symmetry around candidate axes
    for axis in candidate_axes:
        for fold in [2, 3, 4, 6]:  # Common rotation symmetries
            score = _test_rotational_symmetry(centered, axis, fold, abs_tol)
            if score > 0.85:
                # Avoid duplicate axes
                is_duplicate = any(
                    abs(np.dot(axis, a)) > 0.95 for a, _ in rotation_axes
                )
                if not is_duplicate:
                    rotation_axes.append((axis, fold))
                break  # Only record highest fold

    # Compute overall symmetry score
    if reflection_planes or rotation_axes:
        n_symm = len(reflection_planes) + len(rotation_axes)
        symmetry_score = min(1.0, n_symm / 3.0)
    else:
        symmetry_score = 0.0

    return SymmetryInfo(
        reflection_planes=reflection_planes,
        rotation_axes=rotation_axes,
        symmetry_score=symmetry_score,
    )


def _get_pca_axes(centered_points: np.ndarray) -> list[np.ndarray]:
    """Get normalized PCA axes."""
    if len(centered_points) < 3:
        return [np.array([1.0, 0.0, 0.0])]

    n_components = min(3, len(centered_points))
    pca = PCA(n_components=n_components)
    pca.fit(centered_points)

    axes = []
    for comp in pca.components_:
        norm = np.linalg.norm(comp)
        if norm > 1e-10:
            axes.append(comp / norm)

    return axes if axes else [np.array([1.0, 0.0, 0.0])]


def _get_candidate_axes(
    centered_points: np.ndarray,
    pca_axes: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Get candidate axes for symmetry testing.

    Includes PCA axes plus cardinal axes (X, Y, Z) which are important
    for axis-aligned shapes like cubes where PCA may give rotated axes.
    """
    # Cardinal axes
    cardinal = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    candidates = list(pca_axes)

    # Add cardinal axes if not already similar to PCA axes
    for c_axis in cardinal:
        is_similar = any(abs(np.dot(c_axis, p)) > 0.95 for p in candidates)
        if not is_similar:
            candidates.append(c_axis)

    return candidates


def _test_reflection_symmetry(
    centered: np.ndarray,
    plane_normal: np.ndarray,
    tolerance: float,
) -> float:
    """
    Test how well points match their reflections across a plane.
    Returns score 0-1.
    """
    # Reflect points across plane through origin
    normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-10)
    # Reflection: p' = p - 2*(p.n)*n
    proj = centered @ normal
    reflected = centered - 2 * np.outer(proj, normal)

    # For each reflected point, find nearest original point
    n_test = min(200, len(centered))
    if len(centered) > n_test:
        idx = np.random.choice(len(centered), n_test, replace=False)
        reflected_sub = reflected[idx]
    else:
        reflected_sub = reflected

    dists = cdist(reflected_sub, centered).min(axis=1)

    # Score: fraction of points with a close match
    matched = np.sum(dists < tolerance) / len(dists)
    return float(matched)


def _test_rotational_symmetry(
    centered: np.ndarray,
    axis: np.ndarray,
    fold: int,
    tolerance: float,
) -> float:
    """
    Test n-fold rotational symmetry around an axis.
    Returns score 0-1.
    """
    angle = 2 * np.pi / fold
    axis_normalized = axis / (np.linalg.norm(axis) + 1e-10)
    rot = Rotation.from_rotvec(angle * axis_normalized)
    rotated = rot.apply(centered)

    # Check if rotated points match original
    n_test = min(200, len(centered))
    if len(centered) > n_test:
        idx = np.random.choice(len(centered), n_test, replace=False)
        rotated_sub = rotated[idx]
    else:
        rotated_sub = rotated

    dists = cdist(rotated_sub, centered).min(axis=1)

    matched = np.sum(dists < tolerance) / len(dists)
    return float(matched)


def get_symmetric_split_point(
    points: np.ndarray,
    axis: np.ndarray,
    symmetry_info: SymmetryInfo,
) -> float:
    """
    Get a split point along axis that respects detected symmetry.

    For shapes with reflection symmetry along the split axis,
    returns the centroid projection (symmetric split).
    """
    proj = points @ axis

    # Check if this axis aligns with a reflection plane
    for plane_normal in symmetry_info.reflection_planes:
        alignment = abs(np.dot(axis, plane_normal))
        if alignment > 0.95:
            # Split at centroid for symmetric split
            return float(np.mean(proj))

    # Default: blend of median and geometric center
    return float(0.5 * np.median(proj) + 0.5 * (proj.min() + proj.max()) / 2)
