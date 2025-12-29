"""Robot-level sphere refinement with JAX/optax optimization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import jaxlie
import optax
from loguru import logger
from scipy.optimize import linear_sum_assignment

from ._spherize import Sphere
from ._similarity import SimilarityResult
from ._config import RefineParams
from .utils._urdf_utils import (
    get_collision_mesh_for_link,
    get_joint_limits,
    get_link_transforms,
    get_link_names,
    get_non_contiguous_link_pairs,
)
from .utils._mesh_utils import compute_mesh_distances_batch


# =============================================================================
# Internal dataclasses
# =============================================================================


@dataclass
class _FlattenedSphereData:
    """Flattened sphere and point data for JIT-compatible optimization.

    JAX's JIT compiler requires static array shapes at compile time. Per-link
    dictionaries with variable-length lists cannot be traced. We "flatten"
    per-link data into contiguous arrays with index mappings.
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


# =============================================================================
# Self-collision utilities
# =============================================================================


def _compute_min_self_collision_distance(
    urdf,
    link_spheres: dict[str, list[Sphere]],
    valid_pairs: list[tuple[str, str]] | None = None,
    joint_cfg: np.ndarray | None = None,
) -> float:
    """Compute the minimum signed distance between spheres of non-contiguous links."""
    all_link_names = get_link_names(urdf)
    links_with_spheres = [name for name in all_link_names if link_spheres.get(name)]
    if not links_with_spheres:
        return float("inf")

    link_name_to_idx = {name: idx for idx, name in enumerate(all_link_names)}

    if joint_cfg is None:
        lower, upper = get_joint_limits(urdf)
        joint_cfg = (lower + upper) / 2
    Ts = get_link_transforms(urdf, joint_cfg)

    if valid_pairs is not None:
        non_contiguous_pairs = valid_pairs
    else:
        non_contiguous_pairs = get_non_contiguous_link_pairs(urdf, links_with_spheres)

    min_dist = float("inf")

    for link_a, link_b in non_contiguous_pairs:
        spheres_a = link_spheres.get(link_a, [])
        spheres_b = link_spheres.get(link_b, [])

        if not spheres_a or not spheres_b:
            continue

        idx_a = link_name_to_idx[link_a]
        idx_b = link_name_to_idx[link_b]

        T_a = Ts[idx_a]
        T_b = Ts[idx_b]

        wxyz_a, xyz_a = T_a[:4], T_a[4:]
        wxyz_b, xyz_b = T_b[:4], T_b[4:]

        so3_a = jaxlie.SO3(wxyz=wxyz_a)
        so3_b = jaxlie.SO3(wxyz=wxyz_b)

        for sphere_i in spheres_a:
            center_i_world = np.array(so3_a @ sphere_i.center) + xyz_a

            for sphere_j in spheres_b:
                center_j_world = np.array(so3_b @ sphere_j.center) + xyz_b

                dist = np.linalg.norm(center_i_world - center_j_world)
                signed_dist = dist - (sphere_i.radius + sphere_j.radius)

                min_dist = min(min_dist, signed_dist)

    return min_dist


# =============================================================================
# Flattening utilities (internal)
# =============================================================================


def _build_flattened_sphere_data(
    link_spheres: dict[str, list[Sphere]],
    link_points: dict[str, np.ndarray],
    link_names: list[str],
    Ts: np.ndarray,
    link_name_to_idx: dict[str, int],
) -> _FlattenedSphereData | None:
    """Flatten per-link sphere/point dicts into arrays for JAX optimization."""
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

        start_sphere = sphere_idx
        for s in spheres:
            all_centers.append(s.center)
            all_radii.append(s.radius)
            sphere_idx += 1
        link_sphere_ranges.append((start_sphere, sphere_idx))

        start_point = point_idx
        if len(points) > 0:
            all_points.append(points)
            point_idx += len(points)
        link_point_ranges.append((start_point, point_idx))

    if not all_centers:
        return None

    n_spheres = len(all_centers)
    n_links = len(link_names)

    centers = jnp.array(all_centers)
    radii = jnp.array(all_radii)
    points_all = jnp.array(np.vstack(all_points) if all_points else np.zeros((1, 3)))

    sphere_to_link_list = []
    for i, link_name in enumerate(link_names):
        n_in_link = len(link_spheres[link_name])
        sphere_to_link_list.extend([i] * n_in_link)
    sphere_to_link = jnp.array(sphere_to_link_list, dtype=jnp.int32)

    point_to_link_list = []
    for i, link_name in enumerate(link_names):
        points = link_points.get(link_name, np.zeros((0, 3)))
        point_to_link_list.extend([i] * len(points))
    if point_to_link_list:
        point_to_link = jnp.array(point_to_link_list, dtype=jnp.int32)
    else:
        point_to_link = jnp.zeros((1,), dtype=jnp.int32)

    sphere_link_mask = sphere_to_link[:, None] == sphere_to_link[None, :]

    sphere_transforms_list = []
    for link_name in link_names:
        transform_idx = link_name_to_idx[link_name]
        T = Ts[transform_idx]
        n_in_link = len(link_spheres[link_name])
        sphere_transforms_list.extend([T] * n_in_link)
    sphere_transforms = jnp.array(sphere_transforms_list)

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
        initial_centers=centers,
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
    """Build boolean mask indicating which sphere pairs to check for collision."""
    link_name_to_internal_idx = {name: i for i, name in enumerate(link_names)}

    collision_pair_mask = np.zeros((n_spheres, n_spheres), dtype=bool)

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

        mesh_dist = mesh_distances.get(
            (link_a, link_b), mesh_distances.get((link_b, link_a), float("inf"))
        )

        if mesh_dist < mesh_collision_tolerance:
            skipped_pairs.append((link_a, link_b, mesh_dist))
        else:
            valid_pairs.append((link_a, link_b))
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
    """Build sphere correspondence pairs for similarity regularization."""
    link_name_to_internal_idx = {name: i for i, name in enumerate(link_names)}
    similarity_pairs = []

    if similarity_result is None or lambda_similarity <= 0:
        return jnp.zeros((0, 2), dtype=jnp.int32), []

    for group in similarity_result.groups:
        group_links = [l for l in group if l in link_name_to_internal_idx]
        if len(group_links) < 2:
            continue

        first_link = group_links[0]
        first_internal_idx = link_name_to_internal_idx[first_link]
        first_range = link_sphere_ranges[first_internal_idx]
        n_first = first_range[1] - first_range[0]

        if n_first == 0:
            continue

        first_centers_np = np.array(all_centers[first_range[0]:first_range[1]])

        for other_link in group_links[1:]:
            other_internal_idx = link_name_to_internal_idx[other_link]
            other_range = link_sphere_ranges[other_internal_idx]
            n_other = other_range[1] - other_range[0]

            if n_other == 0:
                continue

            other_centers_np = np.array(all_centers[other_range[0]:other_range[1]])

            transform = similarity_result.transforms.get(
                (first_link, other_link), np.eye(4)
            )

            first_centers_homo = np.hstack([first_centers_np, np.ones((n_first, 1))])
            first_transformed = (transform @ first_centers_homo.T).T[:, :3]

            cost_matrix = np.zeros((n_first, n_other))
            for i in range(n_first):
                for j in range(n_other):
                    cost_matrix[i, j] = np.sum(
                        (first_transformed[i] - other_centers_np[j]) ** 2
                    )

            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            n_match = min(n_first, n_other)
            for i, j in zip(row_ind[:n_match], col_ind[:n_match]):
                global_i = first_range[0] + i
                global_j = other_range[0] + j
                similarity_pairs.append((global_i, global_j))

            if abs(n_first - n_other) > max(n_first, n_other) * 0.2:
                logger.warning(
                    f"Sphere count mismatch between "
                    f"{first_link} ({n_first}) and {other_link} ({n_other}). "
                    f"Only {len(row_ind)} of {max(n_first, n_other)} "
                    f"spheres will be regularized"
                )

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
    """Convert flattened arrays back to per-link Sphere dictionaries."""
    refined_link_spheres = {}

    for i, link_name in enumerate(link_names):
        start, end = link_sphere_ranges[i]
        refined_link_spheres[link_name] = [
            Sphere(center=centers[j], radius=float(radii[j])) for j in range(start, end)
        ]

    for link_name, spheres in original_link_spheres.items():
        if link_name not in refined_link_spheres:
            refined_link_spheres[link_name] = spheres

    return refined_link_spheres


# =============================================================================
# Loss functions (internal)
# =============================================================================


def _compute_robot_loss(
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    initial_centers: jnp.ndarray,
    initial_radii: jnp.ndarray,
    points_all: jnp.ndarray,
    scale: float,
    n_links: int,
    sphere_to_link: jnp.ndarray,
    point_to_link: jnp.ndarray,
    sphere_link_mask: jnp.ndarray,
    collision_pair_mask: jnp.ndarray,
    sphere_transforms: jnp.ndarray,
    similarity_pairs: jnp.ndarray,
    params: RefineParams,
) -> jnp.ndarray:
    """Compute total loss for robot-level sphere refinement (JIT-compatible)."""
    n_spheres = centers.shape[0]
    n_points = points_all.shape[0]
    radii = jnp.maximum(radii, params.min_radius)

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
        total_loss = total_loss + params.lambda_under * under_approx

    # 2. Over-approximation loss (all spheres)
    over_approx = jnp.mean((radii / scale) ** 3)
    total_loss = total_loss + params.lambda_over * over_approx

    # 3. Intra-link overlap loss (only between spheres of same link)
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
        total_loss = total_loss + params.lambda_overlap * overlap_loss

    # 4. Uniformity loss (per-link variance, aggregated)
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
        total_loss = total_loss + params.lambda_uniform * uniform_loss / (n_valid_links + 1e-8)

    # 5. Self-collision loss between non-contiguous links
    world_diff = (
        world_centers_current[:, None, :] - world_centers_current[None, :, :]
    )
    world_dists = jnp.sqrt(jnp.sum(world_diff**2, axis=-1) + 1e-8)
    sum_radii_mat = radii[:, None] + radii[None, :]
    signed_dist = world_dists - sum_radii_mat
    overlap = jnp.maximum(0.0, -signed_dist) ** 2

    triu_mask = jnp.triu(jnp.ones((n_spheres, n_spheres)), k=1)
    collision_mask = collision_pair_mask * triu_mask

    n_collision_pairs = jnp.sum(collision_mask)
    self_collision_loss = jnp.sum(collision_mask * overlap) / (n_collision_pairs + 1e-8)
    total_loss = total_loss + params.lambda_self_collision * self_collision_loss

    # 6. Center regularization
    center_drift = jnp.sum((centers - initial_centers) ** 2, axis=-1)
    center_reg = jnp.mean(center_drift) / (scale**2 + 1e-8)
    radii_drift = (radii - initial_radii) ** 2
    radii_reg = jnp.mean(radii_drift) / (scale**2 + 1e-8)
    total_loss = total_loss + params.lambda_center_reg * (center_reg + radii_reg)

    # 7. Similarity loss (position correspondence between matched spheres)
    n_sim_pairs = similarity_pairs.shape[0]
    if n_sim_pairs > 0:
        idx_a = similarity_pairs[:, 0]
        idx_b = similarity_pairs[:, 1]
        local_centers_a = centers[idx_a]
        local_centers_b = centers[idx_b]

        pair_dists_sq = jnp.sum((local_centers_a - local_centers_b) ** 2, axis=-1) / (
            scale**2 + 1e-8
        )
        similarity_loss = jnp.mean(pair_dists_sq)
        total_loss = total_loss + params.lambda_similarity * similarity_loss

    return total_loss


# =============================================================================
# Optimization loop (internal, JIT-compiled)
# =============================================================================


@partial(jax.jit, static_argnames=["n_iters", "n_links"])
def _run_robot_optimization(
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    initial_centers: jnp.ndarray,
    initial_radii: jnp.ndarray,
    points_all: jnp.ndarray,
    scale: float,
    n_links: int,
    refine_params: RefineParams,
    lr: float,
    n_iters: int,
    tol: float,
    sphere_to_link: jnp.ndarray,
    point_to_link: jnp.ndarray,
    sphere_link_mask: jnp.ndarray,
    collision_pair_mask: jnp.ndarray,
    sphere_transforms: jnp.ndarray,
    similarity_pairs: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the full optimization loop (JIT-compiled with early stopping).

    Returns:
        Tuple of (final_centers, final_radii, init_loss, final_loss, n_steps)
    """
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))
    opt_state = optimizer.init((centers, radii))

    init_loss = _compute_robot_loss(
        centers,
        radii,
        initial_centers,
        initial_radii,
        points_all,
        scale,
        n_links,
        sphere_to_link,
        point_to_link,
        sphere_link_mask,
        collision_pair_mask,
        sphere_transforms,
        similarity_pairs,
        refine_params,
    )

    def body_fn(state):
        """Single optimization step."""
        centers, radii, opt_state, prev_loss, curr_loss, i = state

        def loss_fn(opt_params):
            c, r = opt_params
            return _compute_robot_loss(
                c,
                r,
                initial_centers,
                initial_radii,
                points_all,
                scale,
                n_links,
                sphere_to_link,
                point_to_link,
                sphere_link_mask,
                collision_pair_mask,
                sphere_transforms,
                similarity_pairs,
                refine_params,
            )

        opt_params = (centers, radii)
        _, grads = jax.value_and_grad(loss_fn)(opt_params)
        updates, new_opt_state = optimizer.update(grads, opt_state, opt_params)
        new_opt_params = optax.apply_updates(opt_params, updates)

        new_centers, new_radii = new_opt_params
        new_radii = jnp.maximum(new_radii, refine_params.min_radius)

        new_loss = loss_fn((new_centers, new_radii))

        return (new_centers, new_radii, new_opt_state, curr_loss, new_loss, i + 1)

    def cond_fn(state):
        """Continue while not converged and under max iterations."""
        centers, radii, opt_state, prev_loss, curr_loss, i = state
        rel_change = jnp.abs(prev_loss - curr_loss) / (jnp.abs(prev_loss) + 1e-8)
        not_converged = jnp.logical_or(jnp.isinf(prev_loss), rel_change > tol)
        not_max_iters = i < n_iters
        return jnp.logical_and(not_converged, not_max_iters)

    init_state = (centers, radii, opt_state, jnp.inf, init_loss, 0)
    final_centers, final_radii, _, _, final_loss, n_steps = lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return final_centers, final_radii, init_loss, final_loss, n_steps


# =============================================================================
# Main entry point (called by Robot.refine)
# =============================================================================


def _refine_robot_spheres(
    urdf,
    link_spheres: dict[str, list[Sphere]],
    similarity_result: SimilarityResult | None,
    refine_params: RefineParams | None = None,
) -> dict[str, list[Sphere]]:
    """
    Refine sphere parameters for all robot links jointly.

    This is the internal entry point called by Robot.refine().

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        link_spheres: Dict mapping link names to lists of Sphere objects
        similarity_result: Result from detect_similar_links() for similarity regularization
        refine_params: Refinement parameters. If None, uses defaults.

    Returns:
        Dict mapping link names to refined lists of Sphere objects
    """
    # Apply defaults
    p = refine_params or RefineParams()
    joint_cfg = None  # Use middle of limits by default

    # Get link names
    all_link_names = get_link_names(urdf)
    link_names = [name for name in all_link_names if link_spheres.get(name)]
    if not link_names:
        return link_spheres

    link_name_to_idx = {name: idx for idx, name in enumerate(all_link_names)}

    # Compute FK
    if joint_cfg is None:
        lower, upper = get_joint_limits(urdf)
        joint_cfg = (lower + upper) / 2
    Ts = get_link_transforms(urdf, joint_cfg)

    # Sample points for each link
    link_points: dict[str, np.ndarray] = {}
    for link_name in link_names:
        mesh = get_collision_mesh_for_link(urdf, link_name)
        if not mesh.is_empty:
            link_points[link_name] = mesh.sample(p.n_samples)
        else:
            link_points[link_name] = np.zeros((0, 3))

    # Build flattened data
    flat_data = _build_flattened_sphere_data(
        link_spheres, link_points, link_names, Ts, link_name_to_idx
    )

    if flat_data is None:
        return link_spheres

    # Build collision pair mask
    non_contiguous_pairs = get_non_contiguous_link_pairs(urdf, link_names)

    logger.info("Computing mesh distances for link pairs...")
    link_name_to_internal = {name: i for i, name in enumerate(link_names)}
    pairs_with_spheres = []
    for link_a, link_b in non_contiguous_pairs:
        range_a = flat_data.link_sphere_ranges[link_name_to_internal[link_a]]
        range_b = flat_data.link_sphere_ranges[link_name_to_internal[link_b]]
        if range_a[0] < range_a[1] and range_b[0] < range_b[1]:
            pairs_with_spheres.append((link_a, link_b))

    mesh_distances = compute_mesh_distances_batch(
        urdf,
        pairs_with_spheres,
        n_samples=1000,
        bbox_skip_threshold=0.1,
        joint_cfg=joint_cfg,
    )

    collision_pair_mask, valid_pairs, skipped_pairs = _build_collision_pair_mask(
        flat_data.n_spheres,
        flat_data.link_names,
        flat_data.link_sphere_ranges,
        non_contiguous_pairs,
        mesh_distances,
        p.mesh_collision_tolerance,
    )

    if skipped_pairs:
        logger.debug(f"Skipping {len(skipped_pairs)} link pairs with inherent mesh proximity")
    logger.info(f"Checking self-collision for {len(valid_pairs)} link pairs")

    # Build similarity pairs
    similarity_pairs_array, similarity_pairs_list = _build_similarity_pairs(
        similarity_result,
        flat_data.link_names,
        flat_data.link_sphere_ranges,
        flat_data.all_centers_list,
        p.lambda_similarity,
    )

    if similarity_pairs_list:
        logger.info(
            f"Similarity regularization: {len(similarity_pairs_list)} matched sphere pairs"
        )

    # Log initial self-collision distance (excludes adjacent links and inherently close meshes)
    initial_min_dist = _compute_min_self_collision_distance(
        urdf, link_spheres, valid_pairs=valid_pairs, joint_cfg=joint_cfg
    )
    if initial_min_dist < 0:
        logger.info(f"Self-collision: {-initial_min_dist:.4f} penetration (initial)")
    else:
        logger.info(f"Self-collision: {initial_min_dist:.4f} clearance (initial)")

    # Run optimization
    final_centers, final_radii, init_loss, final_loss, n_steps = _run_robot_optimization(
        flat_data.centers,
        flat_data.radii,
        flat_data.initial_centers,
        flat_data.initial_radii,
        flat_data.points_all,
        flat_data.scale,
        flat_data.n_links,
        p,
        p.lr,
        p.n_iters,
        p.tol,
        flat_data.sphere_to_link,
        flat_data.point_to_link,
        flat_data.sphere_link_mask,
        collision_pair_mask,
        flat_data.sphere_transforms,
        similarity_pairs_array,
    )

    logger.info(
        f"Optimization: {int(n_steps)} iterations, "
        f"loss {float(init_loss):.4f} -> {float(final_loss):.4f}"
    )

    # Unflatten results
    centers_np = np.array(final_centers)
    radii_np = np.array(final_radii)

    refined_link_spheres = _unflatten_to_link_spheres(
        centers_np,
        radii_np,
        flat_data.link_names,
        flat_data.link_sphere_ranges,
        link_spheres,
    )

    # Log final self-collision distance
    final_min_dist = _compute_min_self_collision_distance(
        urdf, refined_link_spheres, valid_pairs=valid_pairs, joint_cfg=joint_cfg
    )
    if final_min_dist < 0:
        logger.info(f"Self-collision: {-final_min_dist:.4f} penetration (final)")
    else:
        logger.info(f"Self-collision: {final_min_dist:.4f} clearance (final)")

    return refined_link_spheres
