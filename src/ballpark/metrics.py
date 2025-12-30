"""Quality metrics for sphere decomposition evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull, QhullError
import trimesh

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


def compute_over_extension(
    mesh: trimesh.Trimesh,
    spheres: list["Sphere"],
    resolution: int = 64,
) -> dict[str, float]:
    """Compute over-extension using voxel intersection.

    Measures the volume of regions inside spheres but outside the mesh.
    Uses a 3D voxel grid to approximate the intersection.

    Args:
        mesh: Original trimesh object
        spheres: List of Sphere objects
        resolution: Number of voxels along the longest axis (default 64)

    Returns:
        Dictionary with:
            - over_extension_volume: Volume inside spheres but outside mesh
            - over_extension_ratio: over_extension_volume / mesh_volume
            - mesh_volume: Volume of the mesh (for reference)
    """
    if len(spheres) == 0:
        return {
            "over_extension_volume": 0.0,
            "over_extension_ratio": 0.0,
            "mesh_volume": float(mesh.volume) if mesh.is_volume else 0.0,
        }

    # Get bounding box that covers both mesh and all spheres
    mesh_bounds = mesh.bounds  # (2, 3) array: [min, max]
    sphere_centers = np.array([s.center for s in spheres])
    sphere_radii = np.array([float(s.radius) for s in spheres])

    # Expand bounds to include all spheres
    sphere_mins = sphere_centers - sphere_radii[:, np.newaxis]
    sphere_maxs = sphere_centers + sphere_radii[:, np.newaxis]

    bounds_min = np.minimum(mesh_bounds[0], sphere_mins.min(axis=0))
    bounds_max = np.maximum(mesh_bounds[1], sphere_maxs.max(axis=0))

    # Create voxel grid
    extent = bounds_max - bounds_min
    max_extent = extent.max()
    voxel_size = max_extent / resolution

    # Number of voxels in each dimension
    n_voxels = np.ceil(extent / voxel_size).astype(int)
    n_voxels = np.maximum(n_voxels, 1)  # At least 1 voxel per dimension

    # Generate voxel center coordinates
    x = bounds_min[0] + (np.arange(n_voxels[0]) + 0.5) * voxel_size
    y = bounds_min[1] + (np.arange(n_voxels[1]) + 0.5) * voxel_size
    z = bounds_min[2] + (np.arange(n_voxels[2]) + 0.5) * voxel_size

    # Create meshgrid and flatten to (N, 3) points
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    voxel_centers = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # Check which voxels are inside any sphere
    inside_spheres = np.zeros(len(voxel_centers), dtype=bool)
    for sphere in spheres:
        center = np.array(sphere.center)
        radius = float(sphere.radius)
        distances = np.linalg.norm(voxel_centers - center, axis=1)
        inside_spheres |= distances <= radius

    # Check which voxels are inside the mesh using signed distance
    # trimesh convention: negative = outside, positive = inside
    signed_distances = trimesh.proximity.signed_distance(mesh, voxel_centers)
    inside_mesh = signed_distances > 0

    # Over-extension: inside spheres but outside mesh
    over_extending = inside_spheres & ~inside_mesh

    # Compute volumes
    voxel_volume = voxel_size**3
    over_extension_volume = float(np.sum(over_extending) * voxel_volume)
    mesh_volume = float(mesh.volume) if mesh.is_volume else float(
        np.sum(inside_mesh) * voxel_volume
    )

    # Avoid division by zero
    if mesh_volume < 1e-10:
        over_extension_ratio = float("inf") if over_extension_volume > 0 else 0.0
    else:
        over_extension_ratio = over_extension_volume / mesh_volume

    return {
        "over_extension_volume": over_extension_volume,
        "over_extension_ratio": over_extension_ratio,
        "mesh_volume": mesh_volume,
    }
