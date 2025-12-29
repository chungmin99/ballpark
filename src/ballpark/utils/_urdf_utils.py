"""URDF utility functions for forward kinematics and joint info."""

from __future__ import annotations

import numpy as np
import trimesh
from loguru import logger
from scipy.spatial.transform import Rotation


def get_joint_limits(urdf) -> tuple[np.ndarray, np.ndarray]:
    """
    Get joint limits from a yourdfpy URDF.

    Args:
        urdf: yourdfpy URDF object

    Returns:
        Tuple of (lower_limits, upper_limits) as numpy arrays
    """
    lower = []
    upper = []
    for jname in urdf.actuated_joint_names:
        joint = urdf.joint_map[jname]
        lower.append(joint.limit.lower if joint.limit else -np.pi)
        upper.append(joint.limit.upper if joint.limit else np.pi)
    return np.array(lower), np.array(upper)


def get_link_transforms(urdf, joint_cfg: np.ndarray | None = None) -> np.ndarray:
    """
    Compute forward kinematics for all links using yourdfpy.

    Args:
        urdf: yourdfpy URDF object
        joint_cfg: Joint configuration array. If None, uses middle of joint limits.

    Returns:
        (num_links, 7) array where each row is [qw, qx, qy, qz, x, y, z]
        (quaternion wxyz format + translation)
    """
    if joint_cfg is None:
        lower, upper = get_joint_limits(urdf)
        joint_cfg = (lower + upper) / 2

    # Update URDF configuration (does FK internally)
    urdf.update_cfg(joint_cfg)

    # Get transforms for all links
    link_names = list(urdf.link_map.keys())
    transforms = np.zeros((len(link_names), 7))

    for i, link_name in enumerate(link_names):
        # Get 4x4 homogeneous transform
        T = urdf.get_transform(link_name)

        # Extract rotation matrix and convert to quaternion (wxyz format)
        rot_matrix = T[:3, :3]
        quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()  # scipy uses xyzw
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        # Extract translation
        translation = T[:3, 3]

        transforms[i, :4] = quat_wxyz
        transforms[i, 4:] = translation

    return transforms


def get_link_names(urdf) -> list[str]:
    """Get ordered list of link names from URDF."""
    return list(urdf.link_map.keys())


def get_num_actuated_joints(urdf) -> int:
    """Get the number of actuated joints in the URDF."""
    return len(urdf.actuated_joint_names)


def link_has_collision(urdf, link_name: str) -> bool:
    """Check if a link has collision geometry."""
    if link_name not in urdf.link_map:
        return False
    return len(urdf.link_map[link_name].collisions) > 0


def get_adjacent_links(urdf) -> set[tuple[str, str]]:
    """Build adjacency set from joint parent-child relationships.

    Returns:
        Set of (link_a, link_b) tuples (sorted alphabetically)
    """
    adjacent = set()
    for joint in urdf.robot.joints:
        pair = tuple(sorted([joint.parent, joint.child]))
        adjacent.add(pair)
    return adjacent


def get_non_contiguous_link_pairs(urdf, link_names: list[str]) -> list[tuple[str, str]]:
    """Get all link pairs that are NOT adjacent (for self-collision checking).

    Args:
        urdf: yourdfpy URDF object
        link_names: List of link names to consider

    Returns:
        List of (link_a, link_b) pairs that are not adjacent in kinematic tree
    """
    adjacent = get_adjacent_links(urdf)
    pairs = []
    for i, link_a in enumerate(link_names):
        for link_b in link_names[i + 1:]:
            if tuple(sorted([link_a, link_b])) not in adjacent:
                pairs.append((link_a, link_b))
    return pairs


def get_collision_mesh_for_link(urdf, link_name: str) -> trimesh.Trimesh:
    """
    Extract collision mesh for a given link from URDF.

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        link_name: Name of the link to extract

    Returns:
        Combined collision mesh for the link (empty Trimesh if no collisions)
    """
    if link_name not in urdf.link_map:
        return trimesh.Trimesh()

    link = urdf.link_map[link_name]
    coll_meshes = []

    for collision in link.collisions:
        geom = collision.geometry
        mesh = None

        if collision.origin is not None:
            transform = collision.origin
        else:
            transform = np.eye(4)

        if geom.box is not None:
            mesh = trimesh.creation.box(extents=geom.box.size)
        elif geom.cylinder is not None:
            mesh = trimesh.creation.cylinder(
                radius=geom.cylinder.radius, height=geom.cylinder.length
            )
        elif geom.sphere is not None:
            mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
        elif geom.mesh is not None:
            mesh_path = geom.mesh.filename
            # Resolve package:// URLs using URDF's filename handler
            if (
                hasattr(urdf, "_filename_handler")
                and urdf._filename_handler is not None
            ):
                mesh_path = urdf._filename_handler(mesh_path)
            try:
                loaded_obj = trimesh.load(
                    mesh_path,
                    force="mesh",
                    process=False,
                )
                if isinstance(loaded_obj, trimesh.Scene):
                    mesh = loaded_obj.dump(concatenate=True)
                else:
                    mesh = loaded_obj

                # Ensure mesh is a Trimesh (not a list)
                if not isinstance(mesh, trimesh.Trimesh):
                    logger.warning(
                        f"Unexpected mesh type from {mesh_path}: {type(mesh)}"
                    )
                    continue

                if geom.mesh.scale is not None:
                    scale = np.asarray(geom.mesh.scale)
                    mesh.apply_scale(scale)
            except Exception as e:
                logger.warning(f"Failed to load mesh {mesh_path}: {e}")
                continue

        if mesh is not None:
            mesh.apply_transform(transform)
            coll_meshes.append(mesh)

    if not coll_meshes:
        return trimesh.Trimesh()

    return trimesh.util.concatenate(coll_meshes)
