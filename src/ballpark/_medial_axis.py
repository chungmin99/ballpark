"""Medial Axis Transform based sphere initialization.

This module provides an alternative sphere initialization strategy using
3D skeletonization. The medial axis (skeleton) of a mesh represents the
set of centers of maximally inscribed spheres, making it ideal for
sphere decomposition.

Algorithm:
1. Voxelize the mesh at specified resolution
2. Compute distance transform (EDT) of interior voxels
3. Extract 3D skeleton using morphological thinning
4. Use EDT values at skeleton points as sphere radii
5. Greedy selection to meet budget while maximizing coverage
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

if TYPE_CHECKING:
    from ._config import SpherizeParams
    from ._spherize import Sphere


def voxelize_mesh(
    mesh: trimesh.Trimesh,
    resolution: int = 48,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Voxelize a mesh into a 3D binary grid.

    Args:
        mesh: Mesh to voxelize (should be watertight for best results)
        resolution: Number of voxels along longest axis

    Returns:
        Tuple of:
            - voxel_grid: (X, Y, Z) boolean array (True = inside mesh)
            - origin: (3,) array, world coordinates of voxel [0,0,0]
            - voxel_size: Size of each voxel in world units
        Returns None if mesh cannot be voxelized.
    """
    if mesh.is_empty or len(mesh.vertices) < 4:
        return None

    # Get mesh bounds
    bounds = mesh.bounds
    extent = bounds[1] - bounds[0]
    max_extent = extent.max()

    if max_extent < 1e-10:
        return None

    # Compute voxel size
    voxel_size = max_extent / resolution

    # Create voxel grid using trimesh
    try:
        voxel_grid = mesh.voxelized(pitch=voxel_size)
        # Get the filled (interior) voxels
        filled = voxel_grid.fill()
        matrix = filled.matrix  # Boolean 3D array
        origin = filled.transform[:3, 3]  # Translation from transform
    except Exception:
        return None

    if matrix.sum() < 10:
        return None

    return matrix, origin, voxel_size


def compute_distance_transform(voxel_grid: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance transform of interior voxels.

    For each interior voxel, computes distance to nearest boundary.
    This distance represents the radius of the largest inscribed
    sphere centered at that point (in voxel units).

    Args:
        voxel_grid: (X, Y, Z) boolean array (True = inside)

    Returns:
        (X, Y, Z) float array of distances (0 for exterior voxels)
    """
    return distance_transform_edt(voxel_grid)


def extract_skeleton_3d(voxel_grid: np.ndarray) -> np.ndarray:
    """
    Extract 3D skeleton (medial axis) using morphological thinning.

    The skeleton is the set of voxels that are equidistant from at least
    two boundary points - the centers of maximally inscribed spheres.

    Args:
        voxel_grid: (X, Y, Z) boolean array (True = inside)

    Returns:
        (X, Y, Z) boolean array (True = on skeleton)
    """
    # skeletonize auto-detects 2D vs 3D from input shape
    return skeletonize(voxel_grid.astype(np.uint8)).astype(bool)


def _ensure_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh | None:
    """
    Attempt to make mesh watertight for MAT computation.

    Args:
        mesh: Input mesh

    Returns:
        Watertight mesh, or None if cannot be made watertight
    """
    if mesh.is_watertight:
        return mesh

    # Try filling holes
    try:
        filled = mesh.copy()
        filled.fill_holes()
        if filled.is_watertight:
            return filled
    except Exception:
        pass

    # Use convex hull as fallback
    try:
        hull = mesh.convex_hull
        if hull.is_watertight:
            return hull
    except Exception:
        pass

    return None


def _is_thin_shell(mesh: trimesh.Trimesh, threshold: float = 0.15) -> bool:
    """
    Check if mesh is too thin for meaningful MAT.

    Args:
        mesh: Mesh to check
        threshold: Thickness ratio below which mesh is considered thin

    Returns:
        True if mesh is a thin shell
    """
    if not mesh.is_watertight:
        return True

    try:
        volume = mesh.volume
        area = mesh.area
        bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])

        # Normalized thickness measure
        # A sphere has sa/vol = 3/r, so thickness ~ vol^(1/3) / sa^(1/2) * constant
        thickness_ratio = (volume ** (1 / 3)) / (area**0.5 + 1e-10) * (bbox_diag**0.5)
        return thickness_ratio < threshold
    except Exception:
        return True


def spherize_medial_axis(
    mesh: trimesh.Trimesh,
    target_spheres: int,
    params: "SpherizeParams | None" = None,
) -> list["Sphere"]:
    """
    Spherize a mesh using Medial Axis Transform.

    This is the main entry point for MAT-based spherization. Places spheres
    along the mesh's skeleton (medial axis), with radii determined by the
    distance transform.

    Algorithm:
    1. Ensure mesh is watertight (or use convex hull fallback)
    2. Voxelize mesh at specified resolution
    3. Compute EDT and 3D skeleton
    4. Place spheres at skeleton points with EDT-derived radii
    5. Greedy selection to meet budget

    Args:
        mesh: Mesh to spherize
        target_spheres: Target number of spheres
        params: Algorithm parameters (uses MAT-specific fields)

    Returns:
        List of Sphere objects along the medial axis.
        Returns empty list if MAT fails (caller should fall back to adaptive).
    """
    from ._config import SpherizeParams
    from ._interior import greedy_sphere_selection
    from ._spherize import Sphere

    p = params or SpherizeParams()

    # Check for thin shells - MAT doesn't work well for these
    if _is_thin_shell(mesh):
        return []

    # Ensure mesh is watertight
    watertight_mesh = _ensure_watertight(mesh)
    if watertight_mesh is None:
        return []

    # Voxelize mesh
    voxel_result = voxelize_mesh(watertight_mesh, resolution=p.mat_voxel_resolution)
    if voxel_result is None:
        return []

    voxel_grid, origin, voxel_size = voxel_result

    # Compute distance transform (gives inscribed radius at each voxel)
    distance_field = compute_distance_transform(voxel_grid)

    # Extract 3D skeleton
    skeleton = extract_skeleton_3d(voxel_grid)

    if skeleton.sum() == 0:
        return []

    # Get skeleton voxel coordinates and their distances
    skeleton_coords = np.argwhere(skeleton)  # (N, 3) voxel indices
    skeleton_distances = distance_field[skeleton]  # (N,) distances in voxels

    # Convert to world coordinates
    world_coords = origin + skeleton_coords * voxel_size
    world_radii = skeleton_distances * voxel_size * p.padding

    # Filter by minimum radius
    bbox_diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    min_radius = bbox_diag * p.mat_min_radius_ratio

    valid_mask = world_radii >= min_radius
    world_coords = world_coords[valid_mask]
    world_radii = world_radii[valid_mask]

    if len(world_coords) == 0:
        return []

    # Optional subsampling for performance
    if p.mat_skeleton_sampling < 1.0 and len(world_coords) > target_spheres * 2:
        n_samples = max(target_spheres * 2, int(len(world_coords) * p.mat_skeleton_sampling))
        indices = np.random.default_rng(42).choice(
            len(world_coords), size=min(n_samples, len(world_coords)), replace=False
        )
        world_coords = world_coords[indices]
        world_radii = world_radii[indices]

    # Greedy selection (reuse from _interior.py)
    selected_centers, selected_radii = greedy_sphere_selection(
        world_coords,
        world_radii,
        target_spheres,
        min_distance_ratio=0.5,
    )

    if len(selected_centers) == 0:
        return []

    # Convert to Sphere objects
    spheres = []
    for center, radius in zip(selected_centers, selected_radii):
        spheres.append(
            Sphere(
                center=jnp.asarray(center),
                radius=jnp.asarray(radius),
            )
        )

    return spheres
