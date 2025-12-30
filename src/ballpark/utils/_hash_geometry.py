"""Geometry hashing utility functions."""

from __future__ import annotations

import numpy as np
import trimesh
from loguru import logger


def get_link_collision_fingerprint(
    urdf, link_name: str, use_extents_for_mesh: bool = True
) -> tuple:
    """Extract a fingerprint for a link's collision geometry.
    Args:
        urdf: yourdfpy URDF object
        link_name: Name of the link to extract fingerprint for
        use_extents_for_mesh: If True, use geometry-based fingerprint for mesh
            geometries (more robust to file differences). If False, use
            filename + scale as fingerprint.

    Returns:
        Tuple fingerprint representing the link's collision geometry. Empty if no geometry.
    """
    assert link_name in urdf.link_map, f"Link '{link_name}' not found in URDF."

    link = urdf.link_map[link_name]
    if not link.collisions:
        return tuple()

    # Generate fingerprint for each collision and combine them
    fingerprints = []
    for collision in link.collisions:
        geom = collision.geometry
        fp = None

        # Check for primitive shapes.
        if geom.box is not None:
            # Box: fingerprint is the sorted extents (rotation-invariant)
            extents = tuple(sorted(geom.box.size))
            fp = ("box", extents)
        elif geom.cylinder is not None:
            fp = ("cylinder", geom.cylinder.radius, geom.cylinder.length)
        elif geom.sphere is not None:
            fp = ("sphere", geom.sphere.radius)

        # Check for mesh geometry.
        elif geom.mesh is not None:
            if use_extents_for_mesh:
                # Use geometry-based fingerprint
                try:
                    mesh_path = geom.mesh.filename
                    # Resolve package:// URLs using URDF's filename handler
                    if hasattr(urdf, "_filename_handler") and urdf._filename_handler is not None:
                        mesh_path = urdf._filename_handler(mesh_path)
                    loaded = trimesh.load(mesh_path, force="mesh", process=False)
                    if isinstance(loaded, trimesh.Scene):
                        mesh = loaded.dump(concatenate=True)
                    else:
                        mesh = loaded

                    if not isinstance(mesh, trimesh.Trimesh):
                        raise ValueError(f"Unexpected mesh type: {type(mesh)}")

                    if geom.mesh.scale is not None:
                        mesh.apply_scale(np.asarray(geom.mesh.scale))

                    if not mesh.is_empty:
                        geom_fp = _get_geometry_fingerprint(mesh)
                        if geom_fp is not None:
                            fp = geom_fp
                except Exception:
                    # Fall back to filename-based fingerprint
                    mesh_file = geom.mesh.filename
                    scale = (
                        tuple(geom.mesh.scale)
                        if geom.mesh.scale is not None
                        else (1.0, 1.0, 1.0)
                    )
                    fp = ("mesh_file", mesh_file, scale)
            else:
                # Mesh from file: use filename + scale as fingerprint
                mesh_file = geom.mesh.filename
                scale = (
                    tuple(geom.mesh.scale)
                    if geom.mesh.scale is not None
                    else (1.0, 1.0, 1.0)
                )
                fp = ("mesh_file", mesh_file, scale)

        # If not known...
        else:
            logger.warning("Unknown geometry type in link '%s'", link_name)
            fp = None

        if fp is not None:
            fingerprints.append(fp)

    return tuple(sorted(fingerprints)) if fingerprints else tuple()


def _get_geometry_fingerprint(mesh, tolerance: float = 0.001) -> tuple | None:
    """Geometry-based fingerprint: sorted extents + volume (mirror-invariant)."""
    if mesh.is_empty:
        return None
    extents = tuple(round(e / tolerance) for e in sorted(mesh.extents))
    try:
        vol = mesh.convex_hull.volume
        # Round volume to 3 significant figures to handle mesh discretization noise
        volume = (
            round(vol, -int(np.floor(np.log10(abs(vol) + 1e-10))) + 2) if vol > 0 else 0
        )
    except Exception:
        volume = 0
    return ("geometry", extents, volume)
