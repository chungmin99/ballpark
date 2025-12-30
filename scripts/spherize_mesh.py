"""Decompose a 3D mesh into spheres for collision approximation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
import tyro
from loguru import logger

from ballpark import spherize


def main(
    mesh_path: Path | None = None,
    primitive: Literal["cube", "sphere"] = "cube",
    target_spheres: int = 32,
    output_path: Path | None = None,
) -> None:
    """Compute sphere decomposition for a 3D mesh.

    Args:
        mesh_path: Path to mesh file (STL, OBJ, PLY, etc.). If not provided, uses primitive.
        primitive: Built-in primitive to use when mesh_path is not provided.
        target_spheres: Target number of spheres to generate.
        output_path: Output JSON file path. Defaults to <mesh_name>_spheres.json.

    Examples:
        python scripts/spherize_mesh.py
        python scripts/spherize_mesh.py --primitive sphere --target-spheres 16
        python scripts/spherize_mesh.py mesh.stl --target-spheres 64
    """
    # Load or create mesh
    if mesh_path is not None:
        logger.info(f"Loading mesh from {mesh_path}...")
        mesh = trimesh.load(mesh_path, force="mesh")

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Failed to load mesh from {mesh_path}")

        mesh_name = mesh_path.stem
    else:
        logger.info(f"Creating {primitive} primitive...")
        if primitive == "cube":
            mesh = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
        else:  # sphere
            mesh = trimesh.creation.icosphere(radius=0.5)
        mesh_name = primitive

    logger.info(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

    # Compute spheres
    logger.info(f"Computing spheres (target={target_spheres})...")
    spheres = spherize(mesh, target_spheres=target_spheres)
    logger.info(f"Generated {len(spheres)} spheres")

    # Print sphere info
    for i, s in enumerate(spheres):
        logger.debug(f"  [{i}] center={s.center}, radius={s.radius:.4f}")

    # Export to JSON
    if output_path is None:
        output_path = Path(f"{mesh_name}_spheres.json")

    import json

    data = {
        "centers": [np.asarray(s.center).tolist() for s in spheres],
        "radii": [float(s.radius) for s in spheres],
    }
    with output_path.open(mode="w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Exported {len(spheres)} spheres to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
