"""Interior sphere placement using grid-based sampling.

This module provides algorithms for placing spheres inside a mesh volume
rather than just near the surface. This improves coverage for thick/solid
objects by filling the interior with appropriately-sized spheres.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import trimesh

if TYPE_CHECKING:
    from ._spherize import Sphere
    from ._config import SpherizeParams


def sample_interior_grid(
    mesh: trimesh.Trimesh,
    resolution: int = 10,
    padding: float = 0.0,
) -> np.ndarray:
    """
    Sample interior points using a 3D grid.

    Creates a regular grid inside the mesh bounding box and filters to
    points that are inside the mesh volume.

    Args:
        mesh: Watertight mesh to sample from
        resolution: Number of grid points along the longest axis
        padding: Inward padding as fraction of mesh extent (0.05 = 5% inset)

    Returns:
        (N, 3) array of interior points
    """
    if not mesh.is_watertight:
        return np.zeros((0, 3))

    # Get bounding box with optional padding
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]
    max_extent = extent.max()

    # Apply inward padding
    pad_amount = max_extent * padding
    bounds_min = bounds[0] + pad_amount
    bounds_max = bounds[1] - pad_amount

    # Create grid
    n_points = np.ceil(extent / max_extent * resolution).astype(int)
    n_points = np.maximum(n_points, 2)

    x = np.linspace(bounds_min[0], bounds_max[0], n_points[0])
    y = np.linspace(bounds_min[1], bounds_max[1], n_points[1])
    z = np.linspace(bounds_min[2], bounds_max[2], n_points[2])

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    grid_points = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)

    # Filter to points inside mesh
    inside_mask = mesh.contains(grid_points)
    interior_points = grid_points[inside_mask]

    return interior_points


def compute_distance_to_surface(
    points: np.ndarray,
    mesh: trimesh.Trimesh,
) -> np.ndarray:
    """
    Compute distance from each point to nearest mesh surface.

    Args:
        points: (N, 3) array of points
        mesh: Mesh to compute distance to

    Returns:
        (N,) array of distances (positive = inside, negative = outside)
    """
    if len(points) == 0:
        return np.zeros(0)

    # Use signed distance (positive inside, negative outside)
    signed_distances = trimesh.proximity.signed_distance(mesh, points)

    return signed_distances


def greedy_sphere_selection(
    candidate_centers: np.ndarray,
    candidate_radii: np.ndarray,
    target_count: int,
    min_distance_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedily select spheres to maximize coverage with good spatial distribution.

    Algorithm:
    1. Start with sphere with largest radius
    2. For each subsequent sphere, pick the one that:
       - Has a reasonable radius
       - Is far enough from already-selected spheres (spatial diversity)
    3. The score balances radius and distance to nearest selected sphere

    Args:
        candidate_centers: (N, 3) array of candidate sphere centers
        candidate_radii: (N,) array of sphere radii
        target_count: Number of spheres to select
        min_distance_ratio: Minimum distance as ratio of sum of radii (0.5 = 50%)

    Returns:
        Tuple of (selected_centers, selected_radii)
    """
    n_candidates = len(candidate_centers)
    if n_candidates == 0:
        return np.zeros((0, 3)), np.zeros(0)

    if n_candidates <= target_count:
        return candidate_centers, candidate_radii

    # Start with sphere with largest radius
    first_idx = int(np.argmax(candidate_radii))
    selected_idx = [first_idx]
    selected_centers = [candidate_centers[first_idx]]
    selected_radii = [candidate_radii[first_idx]]

    remaining_idx = set(range(n_candidates)) - {first_idx}

    # Normalize radii for scoring
    max_radius = candidate_radii.max()

    while len(selected_idx) < target_count and remaining_idx:
        best_idx = None
        best_score = -float('inf')

        for idx in remaining_idx:
            center = candidate_centers[idx]
            radius = candidate_radii[idx]

            # Compute minimum distance to any selected sphere
            min_dist = float('inf')
            for sel_center, sel_radius in zip(selected_centers, selected_radii):
                dist = np.linalg.norm(center - sel_center)
                # Normalize by sum of radii
                normalized_dist = dist / (radius + sel_radius + 1e-10)
                min_dist = min(min_dist, normalized_dist)

            # Score: balance radius (want large) and distance (want diverse)
            # radius_score: normalized 0-1
            # distance_score: 0 if overlapping too much, increases with distance
            radius_score = radius / (max_radius + 1e-10)

            if min_dist < min_distance_ratio:
                # Too close to existing spheres - heavily penalize
                distance_score = min_dist / min_distance_ratio * 0.5
            else:
                # Good distance - give bonus for being far
                distance_score = 1.0 + (min_dist - min_distance_ratio) * 0.5

            # Combined score emphasizes spatial diversity
            score = radius_score * 0.4 + distance_score * 0.6

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            break

        selected_idx.append(best_idx)
        selected_centers.append(candidate_centers[best_idx])
        selected_radii.append(candidate_radii[best_idx])
        remaining_idx.remove(best_idx)

    return np.array(selected_centers), np.array(selected_radii)


def spherize_interior(
    mesh: trimesh.Trimesh,
    target_spheres: int,
    padding: float = 1.02,
    grid_resolution: int = 15,
    min_radius_ratio: float = 0.05,
) -> list["Sphere"]:
    """
    Spherize a mesh by placing spheres in the interior.

    Uses grid-based interior sampling and greedy sphere selection.

    Args:
        mesh: Watertight mesh to spherize
        target_spheres: Target number of spheres
        padding: Radius padding multiplier
        grid_resolution: Grid resolution for interior sampling
        min_radius_ratio: Minimum sphere radius as fraction of mesh extent

    Returns:
        List of Sphere objects filling the mesh interior
    """
    from ._spherize import Sphere

    if not mesh.is_watertight:
        return []

    # Sample interior points
    interior_points = sample_interior_grid(mesh, resolution=grid_resolution, padding=0.05)

    if len(interior_points) < target_spheres:
        # Not enough interior points, try higher resolution
        interior_points = sample_interior_grid(mesh, resolution=grid_resolution * 2, padding=0.02)

    if len(interior_points) < 5:
        return []  # Mesh too thin for interior spheres

    # Compute distance to surface for each point (this is the max radius)
    distances = compute_distance_to_surface(interior_points, mesh)

    # Filter to points that are actually inside (positive distance)
    inside_mask = distances > 0
    interior_points = interior_points[inside_mask]
    distances = distances[inside_mask]

    if len(interior_points) < 5:
        return []

    # Compute sphere radii (distance to surface with padding)
    # Apply min radius constraint
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    min_radius = bbox_diag * min_radius_ratio

    candidate_radii = np.maximum(distances * padding, min_radius)

    # Greedy selection
    selected_centers, selected_radii = greedy_sphere_selection(
        interior_points,
        candidate_radii,
        target_spheres,
        min_distance_ratio=0.5,
    )

    # Convert to Sphere objects
    spheres = []
    for center, radius in zip(selected_centers, selected_radii):
        spheres.append(Sphere(
            center=jnp.asarray(center),
            radius=jnp.asarray(radius),
        ))

    return spheres


def hybrid_spherize(
    mesh: trimesh.Trimesh,
    target_spheres: int,
    params: "SpherizeParams | None" = None,
) -> list["Sphere"]:
    """
    Hybrid spherization combining interior and surface approaches.

    For thick meshes: primarily uses interior sphere placement
    For thin meshes: primarily uses surface-based adaptive splitting

    The thickness is estimated from the ratio of volume to surface area.

    Args:
        mesh: Mesh to spherize
        target_spheres: Target number of spheres
        params: Algorithm parameters (imported from _config)

    Returns:
        List of Sphere objects
    """
    from ._config import SpherizeParams as SpherizeParamsClass
    from ._spherize import spherize as adaptive_spherize

    p = params or SpherizeParamsClass()

    if not mesh.is_watertight or not mesh.is_volume:
        # Non-watertight mesh: use surface-based approach only
        return adaptive_spherize(mesh, target_spheres, params)

    # Estimate thickness: thick meshes have low surface_area/volume ratio
    volume = mesh.volume
    surface_area = mesh.area
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

    # Normalized thickness measure
    # A sphere has sa/vol = 3/r, so thickness ~ vol^(1/3) / sa^(1/2) * constant
    # We normalize by bbox diagonal
    thickness_ratio = (volume ** (1/3)) / (surface_area ** 0.5 + 1e-10) * (bbox_diag ** 0.5)

    # Thin mesh threshold (empirically tuned)
    thin_threshold = 0.15

    if thickness_ratio < thin_threshold:
        # Thin mesh: use adaptive splitting
        return adaptive_spherize(mesh, target_spheres, params)

    # Thick mesh: try interior spherization
    interior_spheres = spherize_interior(
        mesh,
        target_spheres,
        padding=p.padding,
        grid_resolution=15,
        min_radius_ratio=0.05,
    )

    if len(interior_spheres) >= target_spheres * 0.5:
        # Interior spherization worked well
        return interior_spheres

    # Fallback to adaptive splitting
    return adaptive_spherize(mesh, target_spheres, params)
