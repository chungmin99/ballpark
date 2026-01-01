"""Quality metrics for sphere decomposition evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import cdist
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


def compute_regularity(spheres: list["Sphere"]) -> dict[str, float]:
    """
    Compute regularity/uniformity metrics for sphere placement.

    Regularity measures how organized vs chaotic the sphere placement is.
    Higher values indicate more regular, uniform placement.

    Metrics computed:
    - radius_uniformity: 1 - (std/mean) of radii. Higher = more uniform sizes.
    - spacing_uniformity: 1 - (std/mean) of inter-sphere distances. Higher = more regular grid.
    - overall_regularity: Combined score (average of above).

    Args:
        spheres: List of Sphere objects

    Returns:
        Dictionary with regularity metrics
    """
    if len(spheres) < 2:
        return {
            "radius_uniformity": 1.0,
            "spacing_uniformity": 1.0,
            "overall_regularity": 1.0,
        }

    # Extract radii and centers
    radii = np.array([float(s.radius) for s in spheres])
    centers = np.array([np.array(s.center) for s in spheres])

    # Radius uniformity: 1 - coefficient of variation
    radius_mean = radii.mean()
    radius_std = radii.std()
    if radius_mean > 1e-10:
        radius_cv = radius_std / radius_mean
        radius_uniformity = max(0.0, 1.0 - radius_cv)
    else:
        radius_uniformity = 0.0

    # Spacing uniformity: analyze distances between sphere centers
    if len(spheres) >= 2:
        # Compute pairwise distances
        distances = cdist(centers, centers)
        # Get distances to nearest neighbor (excluding self)
        np.fill_diagonal(distances, np.inf)
        nearest_dists = distances.min(axis=1)

        spacing_mean = nearest_dists.mean()
        spacing_std = nearest_dists.std()
        if spacing_mean > 1e-10:
            spacing_cv = spacing_std / spacing_mean
            spacing_uniformity = max(0.0, 1.0 - spacing_cv)
        else:
            spacing_uniformity = 0.0
    else:
        spacing_uniformity = 1.0

    # Overall regularity: average of uniformity scores
    overall_regularity = (radius_uniformity + spacing_uniformity) / 2

    return {
        "radius_uniformity": float(radius_uniformity),
        "spacing_uniformity": float(spacing_uniformity),
        "overall_regularity": float(overall_regularity),
    }


def compute_symmetry_score(
    spheres: list["Sphere"],
    points: np.ndarray | None = None,
    tolerance: float = 0.1,
) -> dict[str, float]:
    """
    Compute how well the sphere placement respects mesh symmetry.

    Tests reflection symmetry across principal planes (XY, XZ, YZ) and
    measures how well sphere positions mirror across these planes.

    Args:
        spheres: List of Sphere objects
        points: Optional (N, 3) array of mesh surface points for reference
        tolerance: Relative tolerance for symmetry matching

    Returns:
        Dictionary with symmetry scores:
        - xy_symmetry: Symmetry score for reflection across XY plane
        - xz_symmetry: Symmetry score for reflection across XZ plane
        - yz_symmetry: Symmetry score for reflection across YZ plane
        - overall_symmetry: Best of the three plane symmetries
    """
    if len(spheres) < 2:
        return {
            "xy_symmetry": 1.0,
            "xz_symmetry": 1.0,
            "yz_symmetry": 1.0,
            "overall_symmetry": 1.0,
        }

    centers = np.array([np.array(s.center) for s in spheres])
    radii = np.array([float(s.radius) for s in spheres])

    # Compute centroid
    centroid = centers.mean(axis=0)
    centered = centers - centroid

    # Compute bounding box for tolerance scaling
    extent = centered.max(axis=0) - centered.min(axis=0)
    scale = max(extent.max(), 1e-10)
    abs_tolerance = tolerance * scale

    def test_reflection_symmetry(axis_idx: int) -> float:
        """Test symmetry by reflecting points across a plane."""
        # Reflect across plane perpendicular to axis
        reflected = centered.copy()
        reflected[:, axis_idx] *= -1

        # For each reflected point, find best matching original
        total_match = 0.0
        for i in range(len(centered)):
            ref_pt = reflected[i]
            ref_radius = radii[i]

            # Find closest original point
            dists = np.linalg.norm(centered - ref_pt, axis=1)
            closest_idx = np.argmin(dists)
            min_dist = dists[closest_idx]

            # Check if position and radius match
            if min_dist < abs_tolerance:
                radius_error = abs(radii[closest_idx] - ref_radius) / (ref_radius + 1e-10)
                if radius_error < tolerance:
                    total_match += 1.0
                else:
                    total_match += max(0.0, 1.0 - radius_error)

        return total_match / len(spheres)

    # Test symmetry across each plane
    xy_sym = test_reflection_symmetry(2)  # Reflect across XY (flip Z)
    xz_sym = test_reflection_symmetry(1)  # Reflect across XZ (flip Y)
    yz_sym = test_reflection_symmetry(0)  # Reflect across YZ (flip X)

    # Overall: use best symmetry score
    overall_sym = max(xy_sym, xz_sym, yz_sym)

    return {
        "xy_symmetry": float(xy_sym),
        "xz_symmetry": float(xz_sym),
        "yz_symmetry": float(yz_sym),
        "overall_symmetry": float(overall_sym),
    }


def compute_all_metrics(
    mesh: trimesh.Trimesh,
    spheres: list["Sphere"],
    n_samples: int = 5000,
) -> dict[str, float]:
    """
    Compute all quality metrics for a sphere decomposition.

    This is a convenience function that computes all available metrics
    in one call.

    Args:
        mesh: Original mesh
        spheres: List of Sphere objects
        n_samples: Number of surface samples for coverage computation

    Returns:
        Dictionary with all metrics:
        - coverage: Fraction of surface points covered
        - tightness: hull_vol / sphere_vol
        - volume_overhead: sphere_vol / hull_vol
        - quality: coverage * tightness
        - over_extension_ratio: volume outside mesh / mesh volume
        - radius_uniformity: Uniformity of sphere radii
        - spacing_uniformity: Uniformity of sphere spacing
        - overall_regularity: Combined regularity score
        - overall_symmetry: Best plane symmetry score
    """
    # Sample surface points
    points = np.asarray(mesh.sample(n_samples))

    # Core metrics
    coverage = compute_coverage(points, spheres)
    tightness = compute_tightness(points, spheres)
    volume_overhead = compute_volume_overhead(points, spheres)
    quality = compute_quality(coverage, tightness)

    # Over-extension
    over_ext = compute_over_extension(mesh, spheres)

    # Regularity
    regularity = compute_regularity(spheres)

    # Symmetry
    symmetry = compute_symmetry_score(spheres, points)

    return {
        "coverage": coverage,
        "tightness": tightness,
        "volume_overhead": volume_overhead,
        "quality": quality,
        "over_extension_ratio": over_ext["over_extension_ratio"],
        "radius_uniformity": regularity["radius_uniformity"],
        "spacing_uniformity": regularity["spacing_uniformity"],
        "overall_regularity": regularity["overall_regularity"],
        "overall_symmetry": symmetry["overall_symmetry"],
    }
