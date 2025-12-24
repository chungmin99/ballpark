"""Robot-level NLLS refinement with self-collision avoidance."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import jaxlie
import optax
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.optimize import linear_sum_assignment

from ._sphere import Sphere
from ._similarity import SimilarityResult


@dataclass
class _LinkMeshData:
    """Cached mesh data for a single link."""

    sampled_points: np.ndarray  # (n_samples, 3) in link-local coordinates
    bbox_min: np.ndarray  # (3,) axis-aligned bounding box min
    bbox_max: np.ndarray  # (3,) axis-aligned bounding box max
    is_empty: bool


@dataclass
class RobotRefinementResult:
    """Result from robot-level sphere refinement."""

    link_spheres: dict[str, list[Sphere]]
    ignore_pairs: list[tuple[str, str]]  # Adjacent + mesh-proximity skipped pairs


@dataclass
class _FlattenedSphereData:
    """Flattened sphere and point data for JIT-compatible optimization.

    JAX's JIT compiler requires static array shapes at compile time. Per-link
    dictionaries with variable-length lists cannot be traced. We "flatten"
    per-link data into contiguous arrays with index mappings:

    - centers/radii: All spheres concatenated into single arrays
    - sphere_to_link: Maps each sphere index to its link index
    - link_sphere_ranges: Allows slicing back to per-link data

    This enables mask-based filtering (sphere_to_link == link_idx) instead of
    dynamic slicing (centers[start:end]), which is required for JIT.

    Index Relationships
    -------------------
    Multiple parallel structures track how flattened data maps back to links:

        link_names[i]           - The i-th link being processed
        link_sphere_ranges[i]   - Spheres for link i are at indices [start:end]
        link_point_ranges[i]    - Points for link i are at indices [start:end]

    Example for 3 links with [2, 3, 1] spheres:
        link_names = ["link_a", "link_b", "link_c"]
        link_sphere_ranges = [(0, 2), (2, 5), (5, 6)]
        sphere_to_link = [0, 0, 1, 1, 1, 2]  # Maps each sphere to link index
    """

    # Flattened arrays for optimization
    centers: jnp.ndarray  # (N, 3) all sphere centers concatenated
    radii: jnp.ndarray  # (N,) all sphere radii concatenated
    initial_centers: jnp.ndarray  # (N, 3) copy for regularization
    initial_radii: jnp.ndarray  # (N,) copy for regularization
    points_all: jnp.ndarray  # (P, 3) all surface points concatenated

    # Index mappings for per-link operations
    sphere_to_link: jnp.ndarray  # (N,) int32 - which link each sphere belongs to
    point_to_link: jnp.ndarray  # (P,) int32 - which link each point belongs to
    sphere_link_mask: jnp.ndarray  # (N, N) bool - True if spheres on same link
    sphere_transforms: jnp.ndarray  # (N, 7) FK transform (wxyz+xyz) per sphere

    # For unflattening back to per-link dicts
    link_sphere_ranges: list[tuple[int, int]]  # (start, end) for each link
    link_point_ranges: list[tuple[int, int]]  # (start, end) for each link
    link_names: list[str]  # ordered list of link names
    all_centers_list: list  # original centers list for similarity matching

    # Metadata
    n_spheres: int
    n_links: int
    scale: float  # bbox diagonal for normalization


def _build_flattened_sphere_data(
    link_spheres: dict[str, list[Sphere]],
    link_points: dict[str, np.ndarray],
    link_names: list[str],
    Ts: np.ndarray,
    link_name_to_idx: dict[str, int],
) -> _FlattenedSphereData | None:
    """Flatten per-link sphere/point dicts into arrays for JAX optimization.

    This is the first step of the refinement pipeline. It converts the
    user-friendly dict-of-lists format into flat arrays that JAX can trace.

    Args:
        link_spheres: Dict mapping link names to Sphere lists
        link_points: Dict mapping link names to (N, 3) surface point arrays
        link_names: Ordered list of links to include (must have spheres)
        Ts: (num_links, 7) FK transforms from robot.forward_kinematics()
        link_name_to_idx: Maps link name to index in Ts array

    Returns:
        _FlattenedSphereData with all arrays prepared for optimization,
        or None if no spheres to process.
    """
    all_centers = []
    all_radii = []
    all_points = []
    link_sphere_ranges = []
    link_point_ranges = []

    sphere_idx = 0
    point_idx = 0

    for link_name in link_names:
        spheres = link_spheres[link_name]
        points = link_points.get(link_name, np.zeros((0, 3)))

        # Track sphere range for this link
        start_sphere = sphere_idx
        for s in spheres:
            all_centers.append(s.center)
            all_radii.append(s.radius)
            sphere_idx += 1
        link_sphere_ranges.append((start_sphere, sphere_idx))

        # Track point range for this link
        start_point = point_idx
        if len(points) > 0:
            all_points.append(points)
            point_idx += len(points)
        link_point_ranges.append((start_point, point_idx))

    if not all_centers:
        return None

    n_spheres = len(all_centers)
    n_links = len(link_names)

    # Convert to JAX arrays
    centers = jnp.array(all_centers)
    radii = jnp.array(all_radii)
    points_all = jnp.array(np.vstack(all_points) if all_points else np.zeros((1, 3)))

    # Build sphere_to_link mapping
    sphere_to_link_list = []
    for i, link_name in enumerate(link_names):
        n_in_link = len(link_spheres[link_name])
        sphere_to_link_list.extend([i] * n_in_link)
    sphere_to_link = jnp.array(sphere_to_link_list, dtype=jnp.int32)

    # Build point_to_link mapping
    point_to_link_list = []
    for i, link_name in enumerate(link_names):
        points = link_points.get(link_name, np.zeros((0, 3)))
        point_to_link_list.extend([i] * len(points))
    if point_to_link_list:
        point_to_link = jnp.array(point_to_link_list, dtype=jnp.int32)
    else:
        point_to_link = jnp.zeros((1,), dtype=jnp.int32)  # Dummy for JIT

    # Build sphere_link_mask: True if spheres belong to same link
    sphere_link_mask = sphere_to_link[:, None] == sphere_to_link[None, :]

    # Build sphere_transforms: FK transform for each sphere
    sphere_transforms_list = []
    for link_name in link_names:
        transform_idx = link_name_to_idx[link_name]
        T = Ts[transform_idx]
        n_in_link = len(link_spheres[link_name])
        sphere_transforms_list.extend([T] * n_in_link)
    sphere_transforms = jnp.array(sphere_transforms_list)

    # Compute scale for normalization
    if len(all_points) > 0:
        points_stacked = np.vstack(all_points)
        bbox_diag = np.linalg.norm(
            points_stacked.max(axis=0) - points_stacked.min(axis=0)
        )
    else:
        bbox_diag = np.linalg.norm(
            np.array(all_centers).max(axis=0) - np.array(all_centers).min(axis=0)
        )
    scale = float(bbox_diag + 1e-8)

    return _FlattenedSphereData(
        centers=centers,
        radii=radii,
        initial_centers=centers,  # Copy for regularization
        initial_radii=radii,
        points_all=points_all,
        sphere_to_link=sphere_to_link,
        point_to_link=point_to_link,
        sphere_link_mask=sphere_link_mask,
        sphere_transforms=sphere_transforms,
        link_sphere_ranges=link_sphere_ranges,
        link_point_ranges=link_point_ranges,
        link_names=link_names,
        all_centers_list=all_centers,
        n_spheres=n_spheres,
        n_links=n_links,
        scale=scale,
    )


def _build_collision_pair_mask(
    n_spheres: int,
    link_names: list[str],
    link_sphere_ranges: list[tuple[int, int]],
    non_contiguous_pairs: list[tuple[str, str]],
    mesh_distances: dict[tuple[str, str], float],
    mesh_collision_tolerance: float,
) -> tuple[jnp.ndarray, list[tuple[str, str]], list[tuple[str, str, float]]]:
    """Build boolean mask indicating which sphere pairs to check for collision.

    Filters out link pairs where the underlying meshes are already close
    (within mesh_collision_tolerance), as these represent inherent geometry
    that cannot be fixed by adjusting sphere parameters.

    Args:
        n_spheres: Total number of spheres across all links
        link_names: Ordered list of link names
        link_sphere_ranges: (start, end) indices for each link's spheres
        non_contiguous_pairs: Link pairs that are not adjacent
        mesh_distances: Pre-computed mesh distances between link pairs
        mesh_collision_tolerance: Skip pairs with mesh distance below this

    Returns:
        collision_pair_mask: (N, N) JAX array, True if pair should be checked
        valid_pairs: Link pairs included in collision checking
        skipped_pairs: Link pairs excluded, with their mesh distances
    """
    link_name_to_internal_idx = {name: i for i, name in enumerate(link_names)}

    # Start with all False
    collision_pair_mask = np.zeros((n_spheres, n_spheres), dtype=bool)

    # Filter to pairs where both links have spheres
    pairs_with_spheres = []
    for link_a, link_b in non_contiguous_pairs:
        internal_idx_a = link_name_to_internal_idx[link_a]
        internal_idx_b = link_name_to_internal_idx[link_b]
        range_a = link_sphere_ranges[internal_idx_a]
        range_b = link_sphere_ranges[internal_idx_b]
        if range_a[0] < range_a[1] and range_b[0] < range_b[1]:
            pairs_with_spheres.append((link_a, link_b))

    skipped_pairs = []
    valid_pairs = []

    for link_a, link_b in pairs_with_spheres:
        internal_idx_a = link_name_to_internal_idx[link_a]
        internal_idx_b = link_name_to_internal_idx[link_b]
        range_a = link_sphere_ranges[internal_idx_a]
        range_b = link_sphere_ranges[internal_idx_b]

        # Try both key orderings in case cache uses different order
        mesh_dist = mesh_distances.get(
            (link_a, link_b), mesh_distances.get((link_b, link_a), float("inf"))
        )

        if mesh_dist < mesh_collision_tolerance:
            # Skip - meshes are inherently close/overlapping
            skipped_pairs.append((link_a, link_b, mesh_dist))
        else:
            valid_pairs.append((link_a, link_b))
            # Set mask True for all sphere pairs between these links
            for i in range(range_a[0], range_a[1]):
                for j in range(range_b[0], range_b[1]):
                    collision_pair_mask[i, j] = True
                    collision_pair_mask[j, i] = True

    return jnp.array(collision_pair_mask), valid_pairs, skipped_pairs


def _build_similarity_pairs(
    similarity_result: SimilarityResult | None,
    link_names: list[str],
    link_sphere_ranges: list[tuple[int, int]],
    all_centers: list,
    lambda_similarity: float,
) -> tuple[jnp.ndarray, list[tuple[int, int]]]:
    """Build sphere correspondence pairs for similarity regularization.

    Uses the Hungarian algorithm (optimal bipartite matching) to pair spheres
    between similar links based on spatial proximity after alignment.

    Algorithm
    ---------
    For each similarity group (links with identical geometry):
    1. Pick the first link as the "reference"
    2. For each other link in the group:
       a. Transform reference centers using alignment from SimilarityResult
       b. Build cost matrix: squared distance between transformed and actual
       c. Run Hungarian matching for optimal 1:1 pairing
       d. Store matched pairs as (global_sphere_idx_a, global_sphere_idx_b)

    Why World Frame?
    ----------------
    The similarity loss compares matched spheres in world frame because
    spheres on different links exist in different local coordinate frames.
    The alignment transforms account for the geometric relationship between
    similar links' local frames.

    Args:
        similarity_result: Result from detect_similar_links(), or None
        link_names: Ordered list of link names
        link_sphere_ranges: (start, end) indices for each link's spheres
        all_centers: List of sphere centers (for building cost matrix)
        lambda_similarity: Weight for similarity loss (skip if 0)

    Returns:
        similarity_pairs_array: (n_pairs, 2) JAX array of matched indices
        similarity_pairs_list: Python list for later reference
    """
    link_name_to_internal_idx = {name: i for i, name in enumerate(link_names)}
    similarity_pairs = []

    if similarity_result is None or lambda_similarity <= 0:
        return jnp.zeros((0, 2), dtype=jnp.int32), []

    for group in similarity_result.groups:
        # Get links in this group that have spheres
        group_links = [l for l in group if l in link_name_to_internal_idx]
        if len(group_links) < 2:
            continue

        # Use first link as reference for matching
        first_link = group_links[0]
        first_internal_idx = link_name_to_internal_idx[first_link]
        first_range = link_sphere_ranges[first_internal_idx]
        n_first = first_range[1] - first_range[0]

        if n_first == 0:
            continue

        first_centers_np = np.array(all_centers[first_range[0] : first_range[1]])

        for other_link in group_links[1:]:
            other_internal_idx = link_name_to_internal_idx[other_link]
            other_range = link_sphere_ranges[other_internal_idx]
            n_other = other_range[1] - other_range[0]

            if n_other == 0:
                continue

            other_centers_np = np.array(all_centers[other_range[0] : other_range[1]])

            # Get alignment transform from similarity detection
            transform = similarity_result.transforms.get(
                (first_link, other_link), np.eye(4)
            )

            # Transform reference centers to other link's frame
            first_centers_homo = np.hstack([first_centers_np, np.ones((n_first, 1))])
            first_transformed = (transform @ first_centers_homo.T).T[:, :3]

            # Build cost matrix: squared distance between transformed and actual
            cost_matrix = np.zeros((n_first, n_other))
            for i in range(n_first):
                for j in range(n_other):
                    cost_matrix[i, j] = np.sum(
                        (first_transformed[i] - other_centers_np[j]) ** 2
                    )

            # Hungarian matching for optimal 1:1 assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Store matched pairs (using global sphere indices)
            n_match = min(n_first, n_other)
            for i, j in zip(row_ind[:n_match], col_ind[:n_match]):
                global_i = first_range[0] + i
                global_j = other_range[0] + j
                similarity_pairs.append((global_i, global_j))

            # Warn if sphere counts differ significantly
            if abs(n_first - n_other) > max(n_first, n_other) * 0.2:
                print(
                    f"  Warning: Sphere count mismatch between "
                    f"{first_link} ({n_first}) and {other_link} ({n_other})"
                )
                print(
                    f"    Only {len(row_ind)} of {max(n_first, n_other)} "
                    f"spheres will be regularized"
                )

    # Convert to JAX array
    if similarity_pairs:
        similarity_pairs_array = jnp.array(similarity_pairs, dtype=jnp.int32)
    else:
        similarity_pairs_array = jnp.zeros((0, 2), dtype=jnp.int32)

    return similarity_pairs_array, similarity_pairs


def _unflatten_to_link_spheres(
    centers: np.ndarray,
    radii: np.ndarray,
    link_names: list[str],
    link_sphere_ranges: list[tuple[int, int]],
    original_link_spheres: dict[str, list[Sphere]],
) -> dict[str, list[Sphere]]:
    """Convert flattened arrays back to per-link Sphere dictionaries.

    This is the inverse of _build_flattened_sphere_data, used after
    optimization to return results in the user-friendly format.

    Args:
        centers: (N, 3) optimized sphere centers
        radii: (N,) optimized sphere radii
        link_names: Ordered list of link names that were optimized
        link_sphere_ranges: (start, end) indices for each link
        original_link_spheres: Original dict, for links with no spheres

    Returns:
        Dict mapping link names to lists of Sphere objects
    """
    refined_link_spheres = {}

    for i, link_name in enumerate(link_names):
        start, end = link_sphere_ranges[i]
        refined_link_spheres[link_name] = [
            Sphere(center=centers[j], radius=float(radii[j])) for j in range(start, end)
        ]

    # Include links that had no spheres (pass through unchanged)
    for link_name, spheres in original_link_spheres.items():
        if link_name not in refined_link_spheres:
            refined_link_spheres[link_name] = spheres

    return refined_link_spheres


def get_adjacent_links(urdf) -> set[tuple[str, str]]:
    """
    Build adjacency set from joint parent-child relationships.

    Two links are adjacent (contiguous) if they are directly connected by a joint.

    Args:
        urdf: yourdfpy URDF object

    Returns:
        Set of (link_a, link_b) tuples where links are adjacent.
        Tuples are sorted alphabetically for consistent lookup.
    """
    adjacent = set()
    for joint in urdf.robot.joints:
        pair = tuple(sorted([joint.parent, joint.child]))
        adjacent.add(pair)
    return adjacent


def get_non_contiguous_link_pairs(urdf, link_names: list[str]) -> list[tuple[str, str]]:
    """
    Get all link pairs that are NOT adjacent (for self-collision checking).

    Args:
        urdf: yourdfpy URDF object
        link_names: List of link names to consider

    Returns:
        List of (link_a, link_b) tuples for non-contiguous pairs
    """
    adjacent = get_adjacent_links(urdf)
    pairs = []
    for i, link_a in enumerate(link_names):
        for link_b in link_names[i + 1 :]:
            if tuple(sorted([link_a, link_b])) not in adjacent:
                pairs.append((link_a, link_b))
    return pairs


def _apply_rotation_vectorized(
    points: np.ndarray,
    wxyz: np.ndarray,
    xyz: np.ndarray,
) -> np.ndarray:
    """Apply rotation and translation to points using vectorized quaternion ops.

    Uses scipy.spatial.transform.Rotation for efficient batch rotation.

    Args:
        points: (N, 3) points in local coordinates
        wxyz: (4,) quaternion (w, x, y, z) - jaxlie convention
        xyz: (3,) translation

    Returns:
        (N, 3) points in world coordinates
    """
    # scipy uses (x, y, z, w) format, jaxlie uses (w, x, y, z)
    quat_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    rot = R.from_quat(quat_xyzw)

    # Vectorized rotation + translation
    return rot.apply(points) + xyz


def _get_bbox_corners(bbox_min: np.ndarray, bbox_max: np.ndarray) -> np.ndarray:
    """Get all 8 corners of an axis-aligned bounding box."""
    return np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],
            [bbox_min[0], bbox_min[1], bbox_max[2]],
            [bbox_min[0], bbox_max[1], bbox_min[2]],
            [bbox_min[0], bbox_max[1], bbox_max[2]],
            [bbox_max[0], bbox_min[1], bbox_min[2]],
            [bbox_max[0], bbox_min[1], bbox_max[2]],
            [bbox_max[0], bbox_max[1], bbox_min[2]],
            [bbox_max[0], bbox_max[1], bbox_max[2]],
        ]
    )


def _bbox_distance(
    bbox_min_a: np.ndarray,
    bbox_max_a: np.ndarray,
    bbox_min_b: np.ndarray,
    bbox_max_b: np.ndarray,
) -> float:
    """Compute minimum distance between two axis-aligned bounding boxes.

    Returns 0 if boxes overlap, otherwise the minimum distance.
    """
    # Per-axis gap: positive if separated, 0 or negative if overlapping
    gap = np.maximum(0, np.maximum(bbox_min_a - bbox_max_b, bbox_min_b - bbox_max_a))
    return float(np.linalg.norm(gap))


def _precompute_link_mesh_data(
    urdf,
    link_names: list[str],
    n_samples: int = 1000,
) -> dict[str, _LinkMeshData]:
    """Precompute mesh data for all links (load once, sample once).

    Args:
        urdf: yourdfpy URDF object
        link_names: Links to precompute
        n_samples: Points to sample from each mesh

    Returns:
        Dict mapping link name to cached mesh data
    """
    from ._robot import get_collision_mesh_for_link

    link_data = {}
    for link_name in link_names:
        mesh = get_collision_mesh_for_link(urdf, link_name)

        if mesh.is_empty:
            link_data[link_name] = _LinkMeshData(
                sampled_points=np.zeros((0, 3)),
                bbox_min=np.zeros(3),
                bbox_max=np.zeros(3),
                is_empty=True,
            )
        else:
            sampled_points = mesh.sample(n_samples)
            bbox_min = sampled_points.min(axis=0)
            bbox_max = sampled_points.max(axis=0)
            link_data[link_name] = _LinkMeshData(
                sampled_points=sampled_points,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                is_empty=False,
            )

    return link_data


def _transform_link_points_to_world(
    link_data: dict[str, _LinkMeshData],
    Ts_zero: np.ndarray,
    link_name_to_idx: dict[str, int],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Transform all cached points to world coordinates.

    Args:
        link_data: Precomputed mesh data per link
        Ts_zero: (num_links, 7) FK transforms (wxyz + xyz)
        link_name_to_idx: Mapping from link name to FK index

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
        T = Ts_zero[idx]
        wxyz, xyz = T[:4], T[4:]

        # Vectorized transform
        pts_world = _apply_rotation_vectorized(data.sampled_points, wxyz, xyz)
        world_points[link_name] = pts_world

        # Transform bounding box corners to get world-space AABB
        corners = _get_bbox_corners(data.bbox_min, data.bbox_max)
        corners_world = _apply_rotation_vectorized(corners, wxyz, xyz)
        world_bbox_min[link_name] = corners_world.min(axis=0)
        world_bbox_max[link_name] = corners_world.max(axis=0)

    return world_points, world_bbox_min, world_bbox_max


def compute_mesh_distances_batch(
    urdf,
    robot,
    link_pairs: list[tuple[str, str]],
    n_samples: int = 1000,
    bbox_skip_threshold: float = 0.1,
    joint_cfg: np.ndarray | None = None,
) -> dict[tuple[str, str], float]:
    """Compute mesh distances for multiple link pairs efficiently.

    Uses caching, vectorized transforms, and bounding box pre-filtering.

    Args:
        urdf: yourdfpy URDF object
        robot: pyroki Robot object
        link_pairs: List of (link_a, link_b) pairs to compute
        n_samples: Points to sample from each mesh
        bbox_skip_threshold: Skip detailed computation if AABB distance exceeds this
        joint_cfg: Joint configuration to use for FK. If None, uses middle of joint limits.

    Returns:
        Dict mapping (link_a, link_b) to approximate minimum distance
    """
    if not link_pairs:
        return {}

    # Get unique links
    unique_links = set()
    for link_a, link_b in link_pairs:
        unique_links.add(link_a)
        unique_links.add(link_b)

    # Step 1: Precompute mesh data (load + sample once per link)
    link_data = _precompute_link_mesh_data(urdf, list(unique_links), n_samples)

    # Step 2: Compute FK once
    link_name_to_idx = {name: idx for idx, name in enumerate(robot.links.names)}
    if joint_cfg is None:
        # Use middle of joint limits as default neutral pose
        joint_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    Ts = robot.forward_kinematics(joint_cfg)

    # Step 3: Transform all points to world coordinates (vectorized)
    world_points, world_bbox_min, world_bbox_max = _transform_link_points_to_world(
        link_data, Ts, link_name_to_idx
    )

    # Step 4: Compute distances with bounding box pre-filter
    results = {}

    for link_a, link_b in link_pairs:
        data_a = link_data[link_a]
        data_b = link_data[link_b]

        # Handle empty meshes
        if data_a.is_empty or data_b.is_empty:
            results[(link_a, link_b)] = float("inf")
            continue

        # Bounding box pre-filter
        bbox_dist = _bbox_distance(
            world_bbox_min[link_a],
            world_bbox_max[link_a],
            world_bbox_min[link_b],
            world_bbox_max[link_b],
        )

        if bbox_dist > bbox_skip_threshold:
            # AABBs are far apart; use bbox distance as lower bound
            results[(link_a, link_b)] = bbox_dist
            continue

        # Detailed KD-tree distance computation
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


def compute_mesh_distance(
    urdf,
    robot,
    link_a: str,
    link_b: str,
    n_samples: int = 1000,
    joint_cfg: np.ndarray | None = None,
) -> float:
    """
    Compute approximate minimum distance between two link meshes.

    Uses point sampling to approximate the minimum distance between two meshes
    at the given joint configuration.

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        robot: pyroki Robot object for forward kinematics
        link_a: Name of first link
        link_b: Name of second link
        n_samples: Number of points to sample from each mesh
        joint_cfg: Joint configuration to use for FK. If None, uses middle of joint limits.

    Returns:
        Approximate minimum distance between meshes.
        Positive = separation, negative = overlap (approximate).
    """
    # Import here to avoid circular import
    from ._robot import get_collision_mesh_for_link

    # Get meshes
    mesh_a = get_collision_mesh_for_link(urdf, link_a)
    mesh_b = get_collision_mesh_for_link(urdf, link_b)

    if mesh_a.is_empty or mesh_b.is_empty:
        return float("inf")

    # Build link name to index mapping
    link_name_to_idx = {name: idx for idx, name in enumerate(robot.links.names)}

    # Get FK at specified config (or middle of limits)
    if joint_cfg is None:
        joint_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    Ts = robot.forward_kinematics(joint_cfg)

    # Sample points from each mesh
    points_a = mesh_a.sample(n_samples)
    points_b = mesh_b.sample(n_samples)

    # Transform to world coordinates
    idx_a = link_name_to_idx[link_a]
    idx_b = link_name_to_idx[link_b]

    T_a = Ts[idx_a]
    T_b = Ts[idx_b]

    wxyz_a, xyz_a = T_a[:4], T_a[4:]
    wxyz_b, xyz_b = T_b[:4], T_b[4:]

    so3_a = jaxlie.SO3(wxyz=wxyz_a)
    so3_b = jaxlie.SO3(wxyz=wxyz_b)

    # Transform points to world
    points_a_world = np.array([so3_a @ p + xyz_a for p in points_a])
    points_b_world = np.array([so3_b @ p + xyz_b for p in points_b])

    # Use KD-tree for efficient nearest neighbor search
    tree_b = cKDTree(points_b_world)
    distances_a_to_b, _ = tree_b.query(points_a_world)
    min_dist_a_to_b = np.min(distances_a_to_b)

    tree_a = cKDTree(points_a_world)
    distances_b_to_a, _ = tree_a.query(points_b_world)
    min_dist_b_to_a = np.min(distances_b_to_a)

    # Minimum distance (approximate - doesn't account for true surface distance)
    min_dist = min(min_dist_a_to_b, min_dist_b_to_a)

    return float(min_dist)


def compute_min_self_collision_distance(
    urdf,
    robot,
    link_spheres: dict[str, list[Sphere]],
    valid_pairs: list[tuple[str, str]] | None = None,
    joint_cfg: np.ndarray | None = None,
) -> float:
    """
    Compute the minimum signed distance between spheres of non-contiguous links.

    A negative value indicates overlap (collision), zero means touching,
    and positive means separation.

    Args:
        urdf: yourdfpy URDF object
        robot: pyroki Robot object for forward kinematics
        link_spheres: Dict mapping link names to lists of Sphere objects
        valid_pairs: Optional list of (link_a, link_b) pairs to consider.
            If None, all non-contiguous pairs are used.
        joint_cfg: Joint configuration to use for FK. If None, uses middle of joint limits.

    Returns:
        Minimum signed distance between any pair of spheres on non-contiguous links.
        Returns inf if no non-contiguous pairs exist.
    """
    # Get links that have spheres
    link_names = [name for name in robot.links.names if link_spheres.get(name)]
    if not link_names:
        return float("inf")

    # Build link name to index mapping
    link_name_to_idx = {name: idx for idx, name in enumerate(robot.links.names)}

    # Compute FK at specified config (or middle of limits)
    if joint_cfg is None:
        joint_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    Ts = robot.forward_kinematics(joint_cfg)

    # Get non-contiguous link pairs (or use provided valid_pairs)
    if valid_pairs is not None:
        non_contiguous_pairs = valid_pairs
    else:
        non_contiguous_pairs = get_non_contiguous_link_pairs(urdf, link_names)

    min_dist = float("inf")

    for link_a, link_b in non_contiguous_pairs:
        spheres_a = link_spheres.get(link_a, [])
        spheres_b = link_spheres.get(link_b, [])

        if not spheres_a or not spheres_b:
            continue

        # Get transforms for each link
        idx_a = link_name_to_idx[link_a]
        idx_b = link_name_to_idx[link_b]

        T_a = Ts[idx_a]
        T_b = Ts[idx_b]

        wxyz_a, xyz_a = T_a[:4], T_a[4:]
        wxyz_b, xyz_b = T_b[:4], T_b[4:]

        so3_a = jaxlie.SO3(wxyz=wxyz_a)
        so3_b = jaxlie.SO3(wxyz=wxyz_b)

        # Transform sphere centers to world coordinates
        for sphere_i in spheres_a:
            center_i_world = np.array(so3_a @ sphere_i.center) + xyz_a

            for sphere_j in spheres_b:
                center_j_world = np.array(so3_b @ sphere_j.center) + xyz_b

                # Compute signed distance
                dist = np.linalg.norm(center_i_world - center_j_world)
                signed_dist = dist - (sphere_i.radius + sphere_j.radius)

                min_dist = min(min_dist, signed_dist)

    return min_dist


def _compute_robot_loss(
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    initial_centers: jnp.ndarray,
    initial_radii: jnp.ndarray,
    points_all: jnp.ndarray,
    scale: float,
    min_radius: float,
    lambda_under: float,
    lambda_over: float,
    lambda_overlap: float,
    lambda_uniform: float,
    lambda_self_collision: float,
    lambda_center_reg: float,
    lambda_similarity: float,
    n_links: int,  # Number of links (static for JIT)
    # Pre-computed arrays for JIT compatibility
    sphere_to_link: jnp.ndarray,  # (n_spheres,) - which link each sphere belongs to
    point_to_link: jnp.ndarray,  # (n_points,) - which link each point belongs to
    sphere_link_mask: jnp.ndarray,  # (n_spheres, n_spheres) - True if same link (for intra-link overlap)
    collision_pair_mask: jnp.ndarray,  # (n_spheres, n_spheres) - True if spheres are on non-contiguous links
    world_centers: jnp.ndarray,  # (n_spheres, 3) - pre-transformed centers in world frame
    # For world transform computation
    sphere_transforms: jnp.ndarray,  # (n_spheres, 7) - wxyz + xyz transform for each sphere
    # Similarity pairs
    similarity_pairs: jnp.ndarray,  # (n_pairs, 2) - pairs of matched sphere indices
) -> jnp.ndarray:
    """Compute total loss for robot-level sphere refinement (JIT-compatible).

    Args:
        centers: (N, 3) sphere centers in link-local coordinates
        radii: (N,) sphere radii
        initial_centers: (N, 3) starting centers for regularization
        initial_radii: (N,) starting radii for regularization
        points_all: (P, 3) surface points to cover
        scale: Bounding box diagonal for normalization
        min_radius: Clamp radii to this minimum
        lambda_*: Weights for each loss component
        n_links: Number of links (static for JIT)
        sphere_to_link: (N,) maps sphere index to link index
        point_to_link: (P,) maps point index to link index
        sphere_link_mask: (N, N) True if spheres on same link
        collision_pair_mask: (N, N) True if spheres on non-adjacent links
        world_centers: Unused (recomputed internally for gradient flow)
        sphere_transforms: (N, 7) FK transform (wxyz+xyz) per sphere
        similarity_pairs: (M, 2) matched sphere index pairs

    Returns:
        Scalar loss value (sum of weighted components)
    """
    n_spheres = centers.shape[0]
    n_points = points_all.shape[0]
    radii = jnp.maximum(radii, min_radius)

    total_loss = jnp.array(0.0)

    # Transform centers to world coordinates for self-collision
    def transform_single_center(center, transform):
        wxyz = transform[:4]
        xyz = transform[4:]
        so3 = jaxlie.SO3(wxyz=wxyz)
        return so3 @ center + xyz

    world_centers_current = jax.vmap(transform_single_center)(
        centers, sphere_transforms
    )

    # 1. Under-approximation loss (per-link, using masks)
    # For each point, find min signed distance to spheres OF THE SAME LINK
    # points_all: (P, 3), centers: (N, 3)
    if n_points > 0:
        diff_pts = points_all[:, None, :] - centers[None, :, :]  # (P, N, 3)
        dists_to_centers = jnp.sqrt(jnp.sum(diff_pts**2, axis=-1) + 1e-8)  # (P, N)
        signed_dists = dists_to_centers - radii[None, :]  # (P, N)

        # Mask: point p can only be covered by sphere s if they belong to same link
        # point_to_link: (P,), sphere_to_link: (N,)
        same_link_mask = point_to_link[:, None] == sphere_to_link[None, :]  # (P, N)

        # Set signed distance to inf for spheres not on same link (so they don't count)
        signed_dists_masked = jnp.where(same_link_mask, signed_dists, jnp.inf)
        min_signed_dist = jnp.min(signed_dists_masked, axis=1)  # (P,)

        # Points with no valid spheres (min=inf) should be ignored
        valid_points = jnp.isfinite(min_signed_dist)
        under_approx = jnp.sum(
            jnp.where(valid_points, jnp.maximum(0.0, min_signed_dist) ** 2, 0.0)
        )
        under_approx = under_approx / (jnp.sum(valid_points) + 1e-8)
        total_loss = total_loss + lambda_under * under_approx

    # 2. Over-approximation loss (all spheres)
    over_approx = jnp.mean((radii / scale) ** 3)
    total_loss = total_loss + lambda_over * over_approx

    # 3. Intra-link overlap loss (only between spheres of same link)
    if n_spheres > 1:
        center_diff = centers[:, None, :] - centers[None, :, :]  # (N, N, 3)
        center_dists = jnp.sqrt(jnp.sum(center_diff**2, axis=-1) + 1e-8)  # (N, N)
        sum_radii_mat = radii[:, None] + radii[None, :]  # (N, N)
        overlap_depth = jnp.maximum(0.0, sum_radii_mat - center_dists)  # (N, N)

        # Upper triangle mask (avoid double-counting)
        triu_mask = jnp.triu(jnp.ones((n_spheres, n_spheres)), k=1)
        # Combined mask: same link AND upper triangle
        intra_link_mask = sphere_link_mask * triu_mask

        overlap_loss = jnp.sum(intra_link_mask * overlap_depth**2) / (
            jnp.sum(intra_link_mask) + 1e-8
        )
        total_loss = total_loss + lambda_overlap * overlap_loss

    # 4. Uniformity loss (per-link variance, aggregated)
    # Use vectorized segment operations instead of fori_loop
    if n_spheres > 1:
        # Compute per-link statistics using segment_sum
        # Count spheres per link
        ones = jnp.ones(n_spheres)
        link_counts = jax.ops.segment_sum(ones, sphere_to_link, num_segments=n_links)

        # Sum of radii per link
        link_radius_sum = jax.ops.segment_sum(
            radii, sphere_to_link, num_segments=n_links
        )

        # Mean radius per link (broadcast back to spheres)
        link_mean = link_radius_sum / (link_counts + 1e-8)  # (n_links,)
        sphere_mean = link_mean[
            sphere_to_link
        ]  # (n_spheres,) - mean for each sphere's link

        # Variance per link: sum of squared deviations
        squared_dev = (radii - sphere_mean) ** 2
        link_var_sum = jax.ops.segment_sum(
            squared_dev, sphere_to_link, num_segments=n_links
        )
        link_var = link_var_sum / (link_counts + 1e-8)  # (n_links,)

        # Normalized variance (coefficient of variation squared)
        link_uniform = link_var / (link_mean**2 + 1e-8)  # (n_links,)

        # Only count links with > 1 sphere
        valid_links = link_counts > 1
        uniform_loss = jnp.sum(jnp.where(valid_links, link_uniform, 0.0))
        n_valid_links = jnp.sum(valid_links)
        total_loss = total_loss + lambda_uniform * uniform_loss / (n_valid_links + 1e-8)

    # 5. Self-collision loss between non-contiguous links
    # if n_spheres > 1 and lambda_self_collision > 0:
    # Use world-transformed centers
    world_diff = (
        world_centers_current[:, None, :] - world_centers_current[None, :, :]
    )  # (N, N, 3)
    world_dists = jnp.sqrt(jnp.sum(world_diff**2, axis=-1) + 1e-8)  # (N, N)
    sum_radii_mat = radii[:, None] + radii[None, :]  # (N, N)
    signed_dist = world_dists - sum_radii_mat  # (N, N)
    overlap = jnp.maximum(0.0, -signed_dist) ** 2  # (N, N)

    # Upper triangle to avoid double counting
    triu_mask = jnp.triu(jnp.ones((n_spheres, n_spheres)), k=1)
    # Only count pairs on non-contiguous links
    collision_mask = collision_pair_mask * triu_mask

    n_collision_pairs = jnp.sum(collision_mask)
    self_collision_loss = jnp.sum(collision_mask * overlap) / (n_collision_pairs + 1e-8)
    total_loss = total_loss + lambda_self_collision * self_collision_loss

    # 6. Center regularization
    # if lambda_center_reg > 0:
    center_drift = jnp.sum((centers - initial_centers) ** 2, axis=-1)
    center_reg = jnp.mean(center_drift) / (scale**2 + 1e-8)
    radii_drift = (radii - initial_radii) ** 2
    radii_reg = jnp.mean(radii_drift) / (scale**2 + 1e-8)
    total_loss = total_loss + lambda_center_reg * (center_reg + radii_reg)

    # 7. Similarity loss (position correspondence between matched spheres)
    # IMPORTANT: Compare in world frame since spheres are in different local frames
    n_sim_pairs = similarity_pairs.shape[0]
    if n_sim_pairs > 0:
        # Get world-transformed centers for matched pairs
        idx_a = similarity_pairs[:, 0]
        idx_b = similarity_pairs[:, 1]
        world_centers_a = world_centers_current[idx_a]  # (n_pairs, 3)
        world_centers_b = world_centers_current[idx_b]  # (n_pairs, 3)

        # Squared distance between matched centers in world frame (normalized by scale)
        pair_dists_sq = jnp.sum((world_centers_a - world_centers_b) ** 2, axis=-1) / (
            scale**2 + 1e-8
        )
        similarity_loss = jnp.mean(pair_dists_sq)
        total_loss = total_loss + lambda_similarity * similarity_loss

    return total_loss


def _compute_robot_loss_breakdown(
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    initial_centers: jnp.ndarray,
    initial_radii: jnp.ndarray,
    points_all: jnp.ndarray,
    scale: float,
    min_radius: float,
    lambda_under: float,
    lambda_over: float,
    lambda_overlap: float,
    lambda_uniform: float,
    lambda_self_collision: float,
    lambda_center_reg: float,
    lambda_similarity: float,
    n_links: int,
    sphere_to_link: jnp.ndarray,
    point_to_link: jnp.ndarray,
    sphere_link_mask: jnp.ndarray,
    collision_pair_mask: jnp.ndarray,
    sphere_transforms: jnp.ndarray,
    similarity_pairs: jnp.ndarray,
) -> dict[str, float]:
    """Compute individual loss components for display (not JIT-compiled).

    Returns a dict with unweighted loss values and their weighted contributions.
    """
    n_spheres = centers.shape[0]
    n_points = points_all.shape[0]
    radii = jnp.maximum(radii, min_radius)

    # Transform centers to world coordinates for self-collision
    def transform_single_center(center, transform):
        wxyz = transform[:4]
        xyz = transform[4:]
        so3 = jaxlie.SO3(wxyz=wxyz)
        return so3 @ center + xyz

    world_centers_current = jax.vmap(transform_single_center)(
        centers, sphere_transforms
    )

    losses = {}

    # 1. Under-approximation loss
    if n_points > 0:
        diff_pts = points_all[:, None, :] - centers[None, :, :]
        dists_to_centers = jnp.sqrt(jnp.sum(diff_pts**2, axis=-1) + 1e-8)
        signed_dists = dists_to_centers - radii[None, :]
        same_link_mask = point_to_link[:, None] == sphere_to_link[None, :]
        signed_dists_masked = jnp.where(same_link_mask, signed_dists, jnp.inf)
        min_signed_dist = jnp.min(signed_dists_masked, axis=1)
        valid_points = jnp.isfinite(min_signed_dist)
        under_approx = jnp.sum(
            jnp.where(valid_points, jnp.maximum(0.0, min_signed_dist) ** 2, 0.0)
        )
        under_approx = under_approx / (jnp.sum(valid_points) + 1e-8)
    else:
        under_approx = jnp.array(0.0)
    losses["under_approx"] = float(under_approx)
    losses["under_approx_weighted"] = float(lambda_under * under_approx)

    # 2. Over-approximation loss
    over_approx = jnp.mean((radii / scale) ** 3)
    losses["over_approx"] = float(over_approx)
    losses["over_approx_weighted"] = float(lambda_over * over_approx)

    # 3. Intra-link overlap loss
    if n_spheres > 1:
        center_diff = centers[:, None, :] - centers[None, :, :]
        center_dists = jnp.sqrt(jnp.sum(center_diff**2, axis=-1) + 1e-8)
        sum_radii_mat = radii[:, None] + radii[None, :]
        overlap_depth = jnp.maximum(0.0, sum_radii_mat - center_dists)
        triu_mask = jnp.triu(jnp.ones((n_spheres, n_spheres)), k=1)
        intra_link_mask = sphere_link_mask * triu_mask
        overlap_loss = jnp.sum(intra_link_mask * overlap_depth**2) / (
            jnp.sum(intra_link_mask) + 1e-8
        )
    else:
        overlap_loss = jnp.array(0.0)
    losses["intra_overlap"] = float(overlap_loss)
    losses["intra_overlap_weighted"] = float(lambda_overlap * overlap_loss)

    # 4. Uniformity loss
    if n_spheres > 1:
        ones = jnp.ones(n_spheres)
        link_counts = jax.ops.segment_sum(ones, sphere_to_link, num_segments=n_links)
        link_radius_sum = jax.ops.segment_sum(
            radii, sphere_to_link, num_segments=n_links
        )
        link_mean = link_radius_sum / (link_counts + 1e-8)
        sphere_mean = link_mean[sphere_to_link]
        squared_dev = (radii - sphere_mean) ** 2
        link_var_sum = jax.ops.segment_sum(
            squared_dev, sphere_to_link, num_segments=n_links
        )
        link_var = link_var_sum / (link_counts + 1e-8)
        link_uniform = link_var / (link_mean**2 + 1e-8)
        valid_links = link_counts > 1
        uniform_loss = jnp.sum(jnp.where(valid_links, link_uniform, 0.0))
        n_valid_links = jnp.sum(valid_links)
        uniform_loss = uniform_loss / (n_valid_links + 1e-8)
    else:
        uniform_loss = jnp.array(0.0)
    losses["uniformity"] = float(uniform_loss)
    losses["uniformity_weighted"] = float(lambda_uniform * uniform_loss)

    # 5. Self-collision loss
    world_diff = world_centers_current[:, None, :] - world_centers_current[None, :, :]
    world_dists = jnp.sqrt(jnp.sum(world_diff**2, axis=-1) + 1e-8)
    sum_radii_mat = radii[:, None] + radii[None, :]
    signed_dist = world_dists - sum_radii_mat
    overlap = jnp.maximum(0.0, -signed_dist) ** 2
    triu_mask = jnp.triu(jnp.ones((n_spheres, n_spheres)), k=1)
    collision_mask = collision_pair_mask * triu_mask
    n_collision_pairs = jnp.sum(collision_mask)
    self_collision_loss = jnp.sum(collision_mask * overlap) / (n_collision_pairs + 1e-8)
    losses["self_collision"] = float(self_collision_loss)
    losses["self_collision_weighted"] = float(
        lambda_self_collision * self_collision_loss
    )

    # 6. Center regularization
    center_drift = jnp.sum((centers - initial_centers) ** 2, axis=-1)
    center_reg = jnp.mean(center_drift) / (scale**2 + 1e-8)
    radii_drift = (radii - initial_radii) ** 2
    radii_reg = jnp.mean(radii_drift) / (scale**2 + 1e-8)
    reg_loss = center_reg + radii_reg
    losses["center_reg"] = float(reg_loss)
    losses["center_reg_weighted"] = float(lambda_center_reg * reg_loss)

    # 7. Similarity loss (must use world-transformed centers)
    n_sim_pairs = similarity_pairs.shape[0]
    if n_sim_pairs > 0:
        idx_a = similarity_pairs[:, 0]
        idx_b = similarity_pairs[:, 1]
        world_centers_a = world_centers_current[idx_a]
        world_centers_b = world_centers_current[idx_b]
        pair_dists_sq = jnp.sum((world_centers_a - world_centers_b) ** 2, axis=-1) / (
            scale**2 + 1e-8
        )
        similarity_loss = jnp.mean(pair_dists_sq)
    else:
        similarity_loss = jnp.array(0.0)
    losses["similarity"] = float(similarity_loss)
    losses["similarity_weighted"] = float(lambda_similarity * similarity_loss)

    # Total
    losses["total"] = (
        losses["under_approx_weighted"]
        + losses["over_approx_weighted"]
        + losses["intra_overlap_weighted"]
        + losses["uniformity_weighted"]
        + losses["self_collision_weighted"]
        + losses["center_reg_weighted"]
        + losses["similarity_weighted"]
    )

    return losses


@partial(jax.jit, static_argnames=["n_iters", "n_links"])
def _run_robot_optimization(
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    initial_centers: jnp.ndarray,
    initial_radii: jnp.ndarray,
    points_all: jnp.ndarray,
    scale: float,
    min_radius: float,
    lambda_under: float,
    lambda_over: float,
    lambda_overlap: float,
    lambda_uniform: float,
    lambda_self_collision: float,
    lambda_center_reg: float,
    lambda_similarity: float,
    lr: float,
    n_iters: int,
    n_links: int,
    tol: float,
    # Pre-computed masks
    sphere_to_link: jnp.ndarray,
    point_to_link: jnp.ndarray,
    sphere_link_mask: jnp.ndarray,
    collision_pair_mask: jnp.ndarray,
    sphere_transforms: jnp.ndarray,
    similarity_pairs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the full optimization loop (JIT-compiled with early stopping).

    Returns:
        (final_centers, final_radii, final_loss, n_steps)
    """
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init((centers, radii))

    # Dummy world_centers (will be recomputed in loss)
    world_centers = jnp.zeros_like(centers)

    # Initial loss
    init_loss = _compute_robot_loss(
        centers,
        radii,
        initial_centers,
        initial_radii,
        points_all,
        scale,
        min_radius,
        lambda_under,
        lambda_over,
        lambda_overlap,
        lambda_uniform,
        lambda_self_collision,
        lambda_center_reg,
        lambda_similarity,
        n_links,
        sphere_to_link,
        point_to_link,
        sphere_link_mask,
        collision_pair_mask,
        world_centers,
        sphere_transforms,
        similarity_pairs,
    )

    def body_fn(state):
        """Single optimization step."""
        centers, radii, opt_state, prev_loss, curr_loss, i = state

        def loss_fn(params):
            c, r = params
            return _compute_robot_loss(
                c,
                r,
                initial_centers,
                initial_radii,
                points_all,
                scale,
                min_radius,
                lambda_under,
                lambda_over,
                lambda_overlap,
                lambda_uniform,
                lambda_self_collision,
                lambda_center_reg,
                lambda_similarity,
                n_links,
                sphere_to_link,
                point_to_link,
                sphere_link_mask,
                collision_pair_mask,
                world_centers,
                sphere_transforms,
                similarity_pairs,
            )

        params = (centers, radii)
        _, grads = jax.value_and_grad(loss_fn)(params)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)

        new_centers, new_radii = new_params
        new_radii = jnp.maximum(new_radii, min_radius)

        # Compute loss at NEW parameters for convergence check
        new_loss = loss_fn((new_centers, new_radii))

        return (new_centers, new_radii, new_opt_state, curr_loss, new_loss, i + 1)

    def cond_fn(state):
        """Continue while not converged and under max iterations."""
        centers, radii, opt_state, prev_loss, curr_loss, i = state
        # Use relative tolerance to handle varying loss magnitudes
        rel_change = jnp.abs(prev_loss - curr_loss) / (jnp.abs(prev_loss) + 1e-8)
        not_converged = rel_change > tol
        not_max_iters = i < n_iters
        return jnp.logical_and(not_converged, not_max_iters)

    # Run optimization loop with early stopping
    init_state = (centers, radii, opt_state, jnp.inf, init_loss, 0)
    final_centers, final_radii, _, _, final_loss, n_steps = lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return final_centers, final_radii, final_loss, n_steps


def refine_spheres_for_robot(
    urdf,
    robot,
    link_spheres: dict[str, list[Sphere]],
    link_points: dict[str, np.ndarray],
    n_iters: int = 100,
    lr: float = 1e-3,
    lambda_under: float = 1.0,
    lambda_over: float = 0.01,
    lambda_overlap: float = 0.1,
    lambda_uniform: float = 0.0,
    lambda_self_collision: float = 1.0,
    lambda_center_reg: float = 1.0,
    min_radius: float = 1e-4,
    mesh_collision_tolerance: float = 0.01,
    tol: float = 1e-4,
    mesh_distances: dict[tuple[str, str], float] | None = None,
    joint_cfg: np.ndarray | None = None,
    similarity_result: SimilarityResult | None = None,
    lambda_similarity: float = 1.0,
) -> RobotRefinementResult:
    """
    Refine sphere parameters for all robot links jointly with self-collision avoidance.

    Optimizes all sphere centers and radii across all links to minimize:
    1. Per-link under-approximation: points outside all spheres for that link
    2. Per-link over-approximation: sphere volume (radius^3 proxy)
    3. Per-link overlap penalty: excessive sphere intersection within a link
    4. Per-link uniformity: variance of radii within a link
    5. Self-collision: overlap between spheres of non-contiguous links
    6. Center regularization: penalizes centers moving far from initial positions

    Link pairs where the actual meshes are within mesh_collision_tolerance at the
    given joint configuration are skipped, as these represent inherent geometry
    that cannot be fixed by adjusting sphere parameters.

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        robot: pyroki Robot object for forward kinematics
        link_spheres: Dict mapping link names to lists of Sphere objects
        link_points: Dict mapping link names to (N, 3) surface point arrays
        n_iters: Maximum number of optimization iterations
        lr: Learning rate for Adam optimizer
        lambda_under: Weight for under-approximation loss
        lambda_over: Weight for over-approximation loss
        lambda_overlap: Weight for intra-link overlap penalty
        lambda_uniform: Weight for radius uniformity
        lambda_self_collision: Weight for self-collision penalty
        lambda_center_reg: Weight for center regularization (prevents spheres drifting)
        min_radius: Minimum allowed radius
        mesh_collision_tolerance: Skip link pairs where mesh distance < this value (meters)
        tol: Relative convergence tolerance for early stopping (e.g., 1e-4 = 0.01% change)
        mesh_distances: Optional pre-computed mesh distances from compute_mesh_distances_batch().
            If provided, skips recomputation. Keys are (link_a, link_b) tuples.
        joint_cfg: Joint configuration to use for FK. If None, uses middle of joint limits.
        similarity_result: Optional similarity detection result. If provided, adds position
            correspondence loss between similar links to encourage consistent sphere layouts.
        lambda_similarity: Weight for similarity position correspondence loss.

    Returns:
        RobotRefinementResult containing:
        - link_spheres: Dict mapping link names to refined lists of Sphere objects
        - ignore_pairs: List of link pairs to ignore for collision checking
          (adjacent links + pairs with mesh proximity below tolerance)

    Implementation Notes
    --------------------
    This function performs several setup steps before optimization:

    1. FLATTEN: Convert per-link sphere/point dicts to flat arrays
       (see _build_flattened_sphere_data for JAX/JIT rationale)

    2. COLLISION FILTERING: Identify link pairs for self-collision loss
       - Skip adjacent links (connected by joints)
       - Skip pairs where underlying meshes are already close

    3. SIMILARITY SETUP: Match spheres between similar links
       (uses Hungarian algorithm, done once before optimization)

    4. OPTIMIZE: Run JAX/JIT-compiled gradient descent

    5. UNFLATTEN: Convert flat arrays back to per-link dictionaries
    """
    # =========================================================================
    # EARLY RETURN: Handle empty input
    # =========================================================================
    link_names = [name for name in robot.links.names if link_spheres.get(name)]
    if not link_names:
        adjacent_pairs = get_adjacent_links(urdf)
        return RobotRefinementResult(
            link_spheres=link_spheres,
            ignore_pairs=list(adjacent_pairs),
        )

    # Build link name to FK index mapping
    link_name_to_idx = {name: idx for idx, name in enumerate(robot.links.names)}

    # Compute FK at specified config (or middle of limits)
    if joint_cfg is None:
        joint_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
    Ts = robot.forward_kinematics(joint_cfg)  # (num_links, 7) wxyz + xyz

    # =========================================================================
    # STEP 1: FLATTEN - Convert per-link data to flat arrays for JAX
    # =========================================================================
    flat_data = _build_flattened_sphere_data(
        link_spheres, link_points, link_names, Ts, link_name_to_idx
    )

    if flat_data is None:
        adjacent_pairs = get_adjacent_links(urdf)
        return RobotRefinementResult(
            link_spheres=link_spheres,
            ignore_pairs=list(adjacent_pairs),
        )

    # =========================================================================
    # STEP 2: COLLISION FILTERING - Build mask for self-collision checking
    # =========================================================================
    non_contiguous_pairs = get_non_contiguous_link_pairs(urdf, link_names)

    # Compute mesh distances if not provided
    if mesh_distances is None:
        print("Computing mesh distances for link pairs...")
        # First filter to pairs with spheres on both sides
        link_name_to_internal = {name: i for i, name in enumerate(link_names)}
        pairs_with_spheres = []
        for link_a, link_b in non_contiguous_pairs:
            range_a = flat_data.link_sphere_ranges[link_name_to_internal[link_a]]
            range_b = flat_data.link_sphere_ranges[link_name_to_internal[link_b]]
            if range_a[0] < range_a[1] and range_b[0] < range_b[1]:
                pairs_with_spheres.append((link_a, link_b))

        mesh_distances = compute_mesh_distances_batch(
            urdf,
            robot,
            pairs_with_spheres,
            n_samples=1000,
            bbox_skip_threshold=0.1,
        )
    else:
        print("Using cached mesh distances...")

    collision_pair_mask, valid_pairs, skipped_pairs = _build_collision_pair_mask(
        flat_data.n_spheres,
        flat_data.link_names,
        flat_data.link_sphere_ranges,
        non_contiguous_pairs,
        mesh_distances,
        mesh_collision_tolerance,
    )

    # Log skipped pairs
    if skipped_pairs:
        print(f"Skipping {len(skipped_pairs)} link pairs with inherent mesh proximity:")
        for link_a, link_b, dist in skipped_pairs:
            print(f"  {link_a} <-> {link_b}: mesh_dist={dist:.4f}m")
    print(f"Checking self-collision for {len(valid_pairs)} link pairs")

    # =========================================================================
    # STEP 3: SIMILARITY SETUP - Match spheres between similar links
    # =========================================================================
    similarity_pairs_array, similarity_pairs_list = _build_similarity_pairs(
        similarity_result,
        flat_data.link_names,
        flat_data.link_sphere_ranges,
        flat_data.all_centers_list,
        lambda_similarity,
    )

    if similarity_pairs_list:
        print(
            f"Similarity regularization: {len(similarity_pairs_list)} matched sphere pairs"
        )

    # =========================================================================
    # STEP 4: OPTIMIZE - Run JIT-compiled gradient descent
    # =========================================================================
    # Compute and print initial self-collision distance
    initial_min_dist = compute_min_self_collision_distance(
        urdf, robot, link_spheres, joint_cfg=joint_cfg
    )
    initial_min_dist_valid = compute_min_self_collision_distance(
        urdf, robot, link_spheres, valid_pairs=valid_pairs, joint_cfg=joint_cfg
    )
    print(f"Initial min self-collision distance: {initial_min_dist:.6f}")
    if valid_pairs:
        print(f"  (valid pairs only: {initial_min_dist_valid:.6f})")

    # Run optimization
    final_centers, final_radii, final_loss, n_steps = _run_robot_optimization(
        flat_data.centers,
        flat_data.radii,
        flat_data.initial_centers,
        flat_data.initial_radii,
        flat_data.points_all,
        flat_data.scale,
        min_radius,
        lambda_under,
        lambda_over,
        lambda_overlap,
        lambda_uniform,
        lambda_self_collision,
        lambda_center_reg,
        lambda_similarity,
        lr,
        n_iters,
        flat_data.n_links,
        tol,
        flat_data.sphere_to_link,
        flat_data.point_to_link,
        flat_data.sphere_link_mask,
        collision_pair_mask,
        flat_data.sphere_transforms,
        similarity_pairs_array,
    )

    # Compute initial loss for comparison
    world_centers_dummy = jnp.zeros_like(flat_data.centers)
    init_loss = _compute_robot_loss(
        flat_data.centers,
        flat_data.radii,
        flat_data.initial_centers,
        flat_data.initial_radii,
        flat_data.points_all,
        flat_data.scale,
        min_radius,
        lambda_under,
        lambda_over,
        lambda_overlap,
        lambda_uniform,
        lambda_self_collision,
        lambda_center_reg,
        lambda_similarity,
        flat_data.n_links,
        flat_data.sphere_to_link,
        flat_data.point_to_link,
        flat_data.sphere_link_mask,
        collision_pair_mask,
        world_centers_dummy,
        flat_data.sphere_transforms,
        similarity_pairs_array,
    )
    print(
        f"  Optimization converged in {int(n_steps)} iterations "
        f"(init loss: {float(init_loss):.6f}, final loss: {float(final_loss):.6f})"
    )

    # Compute and print loss breakdown
    loss_breakdown = _compute_robot_loss_breakdown(
        final_centers,
        final_radii,
        flat_data.initial_centers,
        flat_data.initial_radii,
        flat_data.points_all,
        flat_data.scale,
        min_radius,
        lambda_under,
        lambda_over,
        lambda_overlap,
        lambda_uniform,
        lambda_self_collision,
        lambda_center_reg,
        lambda_similarity,
        flat_data.n_links,
        flat_data.sphere_to_link,
        flat_data.point_to_link,
        flat_data.sphere_link_mask,
        collision_pair_mask,
        flat_data.sphere_transforms,
        similarity_pairs_array,
    )
    print("  Loss breakdown (weighted):")
    print(
        f"    Under-approximation: {loss_breakdown['under_approx_weighted']:.6f} (points outside spheres)"
    )
    print(
        f"    Over-approximation:  {loss_breakdown['over_approx_weighted']:.6f} (sphere volume)"
    )
    print(
        f"    Intra-link overlap:  {loss_breakdown['intra_overlap_weighted']:.6f} (overlap within links)"
    )
    print(
        f"    Self-collision:      {loss_breakdown['self_collision_weighted']:.6f} (overlap between links)"
    )
    print(
        f"    Center regularization: {loss_breakdown['center_reg_weighted']:.6f} (drift from initial)"
    )
    if similarity_pairs_list:
        print(
            f"    Similarity:          {loss_breakdown['similarity_weighted']:.6f} (position correspondence)"
        )

    # =========================================================================
    # STEP 5: UNFLATTEN - Convert flat arrays back to per-link dicts
    # =========================================================================
    centers_np = np.array(final_centers)
    radii_np = np.array(final_radii)

    refined_link_spheres = _unflatten_to_link_spheres(
        centers_np,
        radii_np,
        flat_data.link_names,
        flat_data.link_sphere_ranges,
        link_spheres,
    )

    # Compute and print final self-collision distance
    final_min_dist = compute_min_self_collision_distance(
        urdf, robot, refined_link_spheres, joint_cfg=joint_cfg
    )
    final_min_dist_valid = compute_min_self_collision_distance(
        urdf, robot, refined_link_spheres, valid_pairs=valid_pairs, joint_cfg=joint_cfg
    )
    print(f"Final min self-collision distance: {final_min_dist:.6f}")
    if valid_pairs:
        print(f"  (valid pairs only: {final_min_dist_valid:.6f})")

    # Build ignore_pairs: adjacent links + skipped pairs (mesh proximity)
    adjacent_pairs = get_adjacent_links(urdf)
    ignore_pairs: list[tuple[str, str]] = list(adjacent_pairs) + [
        (link_a, link_b) for link_a, link_b, _ in skipped_pairs
    ]

    return RobotRefinementResult(
        link_spheres=refined_link_spheres,
        ignore_pairs=ignore_pairs,
    )
