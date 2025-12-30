"""Mesh and geometry utilities for collision checking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


@dataclass
class LinkMeshData:
    """Cached mesh data for a single link."""

    sampled_points: np.ndarray  # (n_samples, 3) in link-local coordinates
    bbox_min: np.ndarray  # (3,) axis-aligned bounding box min
    bbox_max: np.ndarray  # (3,) axis-aligned bounding box max
    is_empty: bool


def apply_rotation_vectorized(
    points: np.ndarray,
    wxyz: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:
    """Apply rotation and translation to points using vectorized quaternion ops.

    Args:
        points: (N, 3) array of points
        wxyz: Quaternion in (w, x, y, z) format
        xyz: Translation vector (3,)

    Returns:
        Transformed points (N, 3)
    """
    # scipy uses (x, y, z, w) format, jaxlie uses (w, x, y, z)
    quat_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    rot = R.from_quat(quat_xyzw)
    return rot.apply(points) + xyz


def get_bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Get all 8 corners of an axis-aligned bounding box."""
    return np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],
        [bbox_min[0], bbox_min[1], bbox_max[2]],
        [bbox_min[0], bbox_max[1], bbox_min[2]],
        [bbox_min[0], bbox_max[1], bbox_max[2]],
        [bbox_max[0], bbox_min[1], bbox_min[2]],
        [bbox_max[0], bbox_min[1], bbox_max[2]],
        [bbox_max[0], bbox_max[1], bbox_min[2]],
        [bbox_max[0], bbox_max[1], bbox_max[2]],
    ])


def bbox_distance(
    bbox_min_a: np.ndarray,
    bbox_max_a: np.ndarray,
    bbox_min_b: np.ndarray,
    bbox_max_b: np.ndarray,
) -> float:
    """Compute minimum distance between two axis-aligned bounding boxes."""
    gap = np.maximum(0, np.maximum(bbox_min_a - bbox_max_b, bbox_min_b - bbox_max_a))
    return float(np.linalg.norm(gap))


def precompute_link_mesh_data(
    link_meshes: dict[str, trimesh.Trimesh],
    n_samples: int = 1000,
) -> dict[str, LinkMeshData]:
    """Precompute mesh data for all links (sample once).

    Args:
        link_meshes: Dict mapping link names to their collision meshes.
        n_samples: Number of surface samples per link

    Returns:
        Dict mapping link names to LinkMeshData
    """
    link_data = {}
    for link_name, mesh in link_meshes.items():
        if mesh.is_empty:
            link_data[link_name] = LinkMeshData(
                sampled_points=np.zeros((0, 3)),
                bbox_min=np.zeros(3),
                bbox_max=np.zeros(3),
                is_empty=True,
            )
        else:
            sampled_points = mesh.sample(n_samples)
            bbox_min = sampled_points.min(axis=0)
            bbox_max = sampled_points.max(axis=0)
            link_data[link_name] = LinkMeshData(
                sampled_points=sampled_points,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                is_empty=False,
            )

    return link_data


def transform_link_points_to_world(
    link_data: dict[str, LinkMeshData],
    Ts: np.ndarray,
    link_name_to_idx: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Transform all cached points to world coordinates.

    Args:
        link_data: Dict of precomputed LinkMeshData
        Ts: Link transforms (N, 7) in [wxyz, xyz] format
        link_name_to_idx: Mapping from link names to transform indices

    Returns:
        Tuple of (world_points, world_bbox_min, world_bbox_max) dicts
    """
    world_points = {}
    world_bbox_min = {}
    world_bbox_max = {}

    for link_name, data in link_data.items():
        if data.is_empty:
            world_points[link_name] = np.zeros((0, 3))
            world_bbox_min[link_name] = np.array([np.inf, np.inf, np.inf])
            world_bbox_max[link_name] = np.array([-np.inf, -np.inf, -np.inf])
            continue

        idx = link_name_to_idx[link_name]
        T = Ts[idx]
        wxyz, xyz = T[:4], T[4:]

        pts_world = apply_rotation_vectorized(data.sampled_points, wxyz, xyz)
        world_points[link_name] = pts_world

        corners = get_bbox_corners(data.bbox_min, data.bbox_max)
        corners_world = apply_rotation_vectorized(corners, wxyz, xyz)
        world_bbox_min[link_name] = corners_world.min(axis=0)
        world_bbox_max[link_name] = corners_world.max(axis=0)

    return world_points, world_bbox_min, world_bbox_max


def compute_mesh_distances_batch(
    link_meshes: dict[str, trimesh.Trimesh],
    link_pairs: list[tuple[str, str]],
    all_link_names: list[str],
    joint_limits: tuple[np.ndarray, np.ndarray],
    compute_transforms: Callable[[np.ndarray], np.ndarray],
    n_samples: int = 1000,
    bbox_skip_threshold: float = 0.1,
    joint_cfg: np.ndarray | None = None,
) -> dict[tuple[str, str], float]:
    """Compute mesh distances for multiple link pairs efficiently.

    Uses bounding box checks to skip distant pairs.

    Args:
        link_meshes: Dict mapping link names to their collision meshes.
        link_pairs: List of (link_a, link_b) pairs to check
        all_link_names: Ordered list of all link names (for FK index mapping).
        joint_limits: Tuple of (lower_limits, upper_limits) arrays.
        compute_transforms: Function that takes joint_cfg and returns (N, 7) transforms.
        n_samples: Surface samples per link for distance computation
        bbox_skip_threshold: Skip detailed check if bbox distance exceeds this
        joint_cfg: Joint configuration for FK. If None, uses middle of limits.

    Returns:
        Dict mapping (link_a, link_b) to minimum mesh distance
    """
    if not link_pairs:
        return {}

    unique_links = set()
    for link_a, link_b in link_pairs:
        unique_links.add(link_a)
        unique_links.add(link_b)

    # Filter meshes to only those needed
    relevant_meshes = {k: v for k, v in link_meshes.items() if k in unique_links}
    link_data = precompute_link_mesh_data(relevant_meshes, n_samples)

    link_name_to_idx = {name: idx for idx, name in enumerate(all_link_names)}
    if joint_cfg is None:
        lower, upper = joint_limits
        joint_cfg = (lower + upper) / 2
    Ts = compute_transforms(joint_cfg)

    world_points, world_bbox_min, world_bbox_max = transform_link_points_to_world(
        link_data, Ts, link_name_to_idx
    )

    results = {}

    for link_a, link_b in link_pairs:
        data_a = link_data[link_a]
        data_b = link_data[link_b]

        if data_a.is_empty or data_b.is_empty:
            results[(link_a, link_b)] = float("inf")
            continue

        bbox_dist = bbox_distance(
            world_bbox_min[link_a],
            world_bbox_max[link_a],
            world_bbox_min[link_b],
            world_bbox_max[link_b],
        )

        if bbox_dist > bbox_skip_threshold:
            results[(link_a, link_b)] = bbox_dist
            continue

        points_a_world = world_points[link_a]
        points_b_world = world_points[link_b]

        tree_b = cKDTree(points_b_world)
        distances_a_to_b, _ = tree_b.query(points_a_world)
        min_dist_a_to_b = np.min(distances_a_to_b)

        tree_a = cKDTree(points_a_world)
        distances_b_to_a, _ = tree_a.query(points_b_world)
        min_dist_b_to_a = np.min(distances_b_to_a)

        results[(link_a, link_b)] = float(min(min_dist_a_to_b, min_dist_b_to_a))

    return results
