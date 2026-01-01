"""Convex decomposition for complex meshes.

This module provides convex decomposition of concave meshes before
spherization. Each convex part is spherized independently and then merged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import trimesh

if TYPE_CHECKING:
    from ._spherize import Sphere
    from ._config import SpherizeParams


def has_vhacd() -> bool:
    """Check if vhacdx is available for convex decomposition."""
    try:
        import vhacdx  # noqa: F401
        return True
    except ImportError:
        return False


def decompose_convex(
    mesh: trimesh.Trimesh,
    max_convex_hulls: int = 16,
    resolution: int = 100000,
    max_hull_verts: int = 64,
) -> list[trimesh.Trimesh]:
    """
    Decompose a mesh into convex parts using V-HACD.

    If vhacdx is not available, falls back to returning the convex hull
    of the original mesh.

    Args:
        mesh: Mesh to decompose
        max_convex_hulls: Maximum number of convex parts
        resolution: Voxel resolution for V-HACD
        max_hull_verts: Maximum vertices per convex hull

    Returns:
        List of convex mesh parts
    """
    if not has_vhacd():
        # Fallback: return convex hull
        try:
            hull = mesh.convex_hull
            if hull is not None and not hull.is_empty:
                return [hull]
        except Exception:
            pass
        return [mesh]

    try:
        # Use vhacdx via trimesh's decomposition module
        parts = mesh.convex_decomposition(
            maxhulls=max_convex_hulls,
            resolution=resolution,
            maxNumVerticesPerCH=max_hull_verts,
        )
        if parts and len(parts) > 0:
            return list(parts)
    except Exception:
        pass

    # Fallback: return convex hull
    try:
        hull = mesh.convex_hull
        if hull is not None and not hull.is_empty:
            return [hull]
    except Exception:
        pass

    return [mesh]


def spherize_decomposed(
    mesh: trimesh.Trimesh,
    target_spheres: int,
    params: "SpherizeParams | None" = None,
    max_parts: int = 16,
) -> list["Sphere"]:
    """
    Spherize a mesh using convex decomposition.

    Decomposes the mesh into convex parts, spherizes each part independently,
    and allocates the sphere budget proportionally to part volume.

    Args:
        mesh: Mesh to spherize
        target_spheres: Total target number of spheres
        params: Spherization parameters
        max_parts: Maximum number of convex parts

    Returns:
        List of spheres covering all parts
    """
    from ._spherize import spherize as spherize_single

    # Decompose into convex parts
    parts = decompose_convex(mesh, max_convex_hulls=max_parts)

    if len(parts) <= 1:
        # No decomposition needed, just spherize the original
        return spherize_single(mesh, target_spheres, params)

    # Allocate sphere budget proportionally to part volume
    volumes = []
    for part in parts:
        try:
            vol = part.volume if part.is_volume else part.convex_hull.volume
            volumes.append(max(vol, 1e-10))
        except Exception:
            volumes.append(1e-10)

    total_volume = sum(volumes)
    volume_fractions = [v / total_volume for v in volumes]

    # Allocate spheres, ensuring at least 1 per part
    sphere_counts = []
    remaining = target_spheres

    for i, frac in enumerate(volume_fractions):
        if i == len(parts) - 1:
            # Last part gets remaining spheres
            count = max(1, remaining)
        else:
            count = max(1, int(round(frac * target_spheres)))
            remaining -= count

        sphere_counts.append(count)

    # Spherize each part
    all_spheres = []
    for part, count in zip(parts, sphere_counts):
        try:
            part_spheres = spherize_single(part, count, params)
            all_spheres.extend(part_spheres)
        except Exception:
            # If spherization fails for a part, skip it
            pass

    return all_spheres


def is_convex(mesh: trimesh.Trimesh, tolerance: float = 0.01) -> bool:
    """
    Check if a mesh is approximately convex.

    Compares mesh volume to its convex hull volume.

    Args:
        mesh: Mesh to test
        tolerance: Relative tolerance (0.01 = 1% difference allowed)

    Returns:
        True if mesh is approximately convex
    """
    if not mesh.is_volume:
        return False

    try:
        hull = mesh.convex_hull
        if hull is None or hull.is_empty:
            return False

        mesh_vol = mesh.volume
        hull_vol = hull.volume

        if hull_vol < 1e-10:
            return False

        ratio = mesh_vol / hull_vol
        return ratio >= (1 - tolerance)
    except Exception:
        return False


def needs_decomposition(mesh: trimesh.Trimesh, threshold: float = 0.9) -> bool:
    """
    Determine if a mesh would benefit from convex decomposition.

    A mesh benefits from decomposition if:
    1. It's watertight and can be decomposed
    2. Its volume is significantly less than its convex hull (concave)

    Args:
        mesh: Mesh to analyze
        threshold: Volume ratio below which decomposition is recommended

    Returns:
        True if mesh should be decomposed
    """
    if not mesh.is_watertight:
        return False

    try:
        hull = mesh.convex_hull
        if hull is None or hull.is_empty:
            return False

        mesh_vol = mesh.volume
        hull_vol = hull.volume

        if hull_vol < 1e-10:
            return False

        ratio = mesh_vol / hull_vol
        return ratio < threshold
    except Exception:
        return False
