"""Robot-level sphere refinement with jaxls nonlinear least squares optimization."""

from __future__ import annotations

from typing import Callable

import numpy as np
import trimesh
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
from loguru import logger

from ._spherize import Sphere
from ._config import RefineParams
from ._primitives import (
    Ellipsoid,
    ellipsoid_effective_radius,
    sphere_to_ellipsoid,
)


# =============================================================================
# Helper functions
# =============================================================================


def _transform_point(center: jax.Array, transform: jax.Array) -> jax.Array:
    """Transform a point from link frame to world frame.

    Args:
        center: (3,) point in link frame
        transform: (7,) array [qw, qx, qy, qz, x, y, z]

    Returns:
        (3,) point in world frame
    """
    wxyz = transform[:4]
    xyz = transform[4:]
    so3 = jaxlie.SO3(wxyz=wxyz)
    return so3 @ center + xyz


# =============================================================================
# jaxls Variable Type
# =============================================================================


class SphereVar(
    jaxls.Var[Sphere],
    default_factory=lambda: Sphere(center=jnp.zeros(3), radius=jnp.array(0.01)),
):
    """Variable type for sphere parameters (center + radius)."""

    pass


class EllipsoidVar(
    jaxls.Var[Ellipsoid],
    default_factory=lambda: Ellipsoid(
        center=jnp.zeros(3), semi_axes=jnp.array([0.01, 0.01, 0.01])
    ),
):
    """Variable type for ellipsoid parameters (center + semi-axes)."""

    pass


# =============================================================================
# Internal dataclasses
# =============================================================================


@jdc.pytree_dataclass
class _FlattenedSphereData:
    """Flattened sphere and point data for JIT-compatible optimization.

    JAX's JIT compiler requires static array shapes at compile time. Per-link
    dictionaries with variable-length lists cannot be traced. We "flatten"
    per-link data into contiguous arrays with index mappings.
    """

    # Batched spheres
    spheres: Sphere  # center: (N, 3), radius: (N,)

    # Point data
    points_all: jnp.ndarray  # (P, 3) all surface points concatenated

    # Index mappings for per-link operations
    sphere_to_link: jnp.ndarray  # (N,) int32 - which link each sphere belongs to
    point_to_link: jnp.ndarray  # (P,) int32 - which link each point belongs to

    # FK transforms for self-collision
    sphere_transforms: jnp.ndarray  # (N, 7) FK transform (wxyz+xyz) per sphere

    # Metadata (static - not traced by JAX)
    link_sphere_ranges: jdc.Static[tuple[tuple[int, int], ...]]
    link_point_ranges: jdc.Static[tuple[tuple[int, int], ...]]
    link_names: jdc.Static[tuple[str, ...]]
    n_spheres: jdc.Static[int]
    n_links: jdc.Static[int]
    scale: float


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
    """Flatten per-link sphere/point dicts into arrays for JAX optimization.

    Args:
        link_spheres: Dict mapping link names to lists of Sphere objects.
        link_points: Dict mapping link names to surface point arrays.
        link_names: Ordered list of link names with spheres.
        Ts: FK transforms for all links, shape (num_all_links, 7).
        link_name_to_idx: Maps link name to index in Ts.

    Returns:
        Flattened data or None if no spheres.
    """
    all_centers: list[np.ndarray] = []
    all_radii: list[np.ndarray] = []
    all_points: list[np.ndarray] = []
    link_sphere_ranges: list[tuple[int, int]] = []
    link_point_ranges: list[tuple[int, int]] = []

    sphere_idx = 0
    point_idx = 0

    for link_name in link_names:
        spheres = link_spheres[link_name]
        points = link_points.get(link_name, np.zeros((0, 3)))

        start_sphere = sphere_idx
        for s in spheres:
            all_centers.append(np.asarray(s.center))
            all_radii.append(np.asarray(s.radius))
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

    # Create batched Sphere objects
    spheres = Sphere(center=centers, radius=radii)

    sphere_to_link_list: list[int] = []
    for i, link_name in enumerate(link_names):
        n_in_link = len(link_spheres[link_name])
        sphere_to_link_list.extend([i] * n_in_link)
    sphere_to_link = jnp.array(sphere_to_link_list, dtype=jnp.int32)

    point_to_link_list: list[int] = []
    for i, link_name in enumerate(link_names):
        points = link_points.get(link_name, np.zeros((0, 3)))
        point_to_link_list.extend([i] * len(points))
    if point_to_link_list:
        point_to_link = jnp.array(point_to_link_list, dtype=jnp.int32)
    else:
        point_to_link = jnp.zeros((1,), dtype=jnp.int32)

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

    # Build sphere transforms (assign each sphere its link's FK transform)
    sphere_transforms_list: list[np.ndarray] = []
    for link_name in link_names:
        transform_idx = link_name_to_idx[link_name]
        T = Ts[transform_idx]
        n_in_link = len(link_spheres[link_name])
        sphere_transforms_list.extend([T] * n_in_link)
    sphere_transforms = jnp.array(sphere_transforms_list)

    return _FlattenedSphereData(
        spheres=spheres,
        points_all=points_all,
        sphere_to_link=sphere_to_link,
        point_to_link=point_to_link,
        sphere_transforms=sphere_transforms,
        link_sphere_ranges=tuple(link_sphere_ranges),
        link_point_ranges=tuple(link_point_ranges),
        link_names=tuple(link_names),
        n_spheres=n_spheres,
        n_links=n_links,
        scale=scale,
    )


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
            Sphere(center=centers[j], radius=radii[j]) for j in range(start, end)
        ]

    for link_name, spheres in original_link_spheres.items():
        if link_name not in refined_link_spheres:
            refined_link_spheres[link_name] = spheres

    return refined_link_spheres


# =============================================================================
# jaxls Cost Functions
# =============================================================================


@jaxls.Cost.factory
def _under_approx_cost(
    vals: jaxls.VarValues,
    sphere_vars: tuple[SphereVar, ...],
    points: jax.Array,
    sphere_mask: jax.Array,
    point_mask: jax.Array,
    lambda_under: float,
) -> jax.Array:
    """Vectorized under-approximation: penalize points outside ALL spheres.

    Uses fixed-size inputs with masking to enable single JIT compilation
    regardless of actual sphere/point counts per link.

    Args:
        vals: Variable values from optimizer
        sphere_vars: Tuple of sphere variables (fixed length, padded)
        points: Surface points (fixed shape, padded), shape (max_P, 3)
        sphere_mask: Boolean mask for valid spheres, shape (max_S,)
        point_mask: Boolean mask for valid points, shape (max_P,)
        lambda_under: Weight for under-approximation penalty

    Returns:
        Residual vector of shape (max_P,), masked for invalid points
    """
    # Stack sphere centers and radii (tuple length is static, loop unrolls)
    centers = jnp.stack([vals[sv].center for sv in sphere_vars])  # (S, 3)
    radii = jnp.stack([vals[sv].radius for sv in sphere_vars])  # (S,)

    # Broadcast: points (P, 1, 3) - centers (1, S, 3) -> diff (P, S, 3)
    diff = points[:, None, :] - centers[None, :, :]
    dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)  # (P, S)
    signed_dists = dists - radii[None, :]  # (P, S)

    # Mask invalid spheres (set to +inf so they don't affect min)
    signed_dists = jnp.where(sphere_mask[None, :], signed_dists, jnp.inf)

    # Min over spheres: point is covered if inside ANY sphere
    min_signed_dist = jnp.min(signed_dists, axis=1)  # (P,)

    # Compute residuals, zeroing out invalid points
    residuals = jnp.sqrt(lambda_under) * jnp.maximum(0.0, min_signed_dist)
    return jnp.where(point_mask, residuals, 0.0)


@jaxls.Cost.factory
def _over_approx_cost(
    vals: jaxls.VarValues,
    sphere_var: SphereVar,
    scale: float,
    lambda_over: float,
) -> jax.Array:
    """Over-approximation cost: penalize large sphere volumes."""
    sphere = vals[sphere_var]
    radius = jnp.maximum(sphere.radius, 1e-4)
    # Residual proportional to radius^1.5 (volume^0.5)
    return jnp.sqrt(lambda_over) * (radius / scale) ** 1.5


@jaxls.Cost.factory
def _center_reg_cost(
    vals: jaxls.VarValues,
    sphere_var: SphereVar,
    init_center: jax.Array,
    scale: float,
    lambda_center_reg: float,
) -> jax.Array:
    """Center regularization: penalize deviation from initial center position.

    Args:
        vals: Variable values from optimizer
        sphere_var: Sphere variable
        init_center: Initial center position, shape (3,)
        scale: Scale factor for normalization
        lambda_center_reg: Weight for regularization

    Returns:
        Residual vector of shape (3,), one per coordinate
    """
    sphere = vals[sphere_var]
    return jnp.sqrt(lambda_center_reg) * (sphere.center - init_center) / scale


@jaxls.Cost.factory
def _radius_reg_cost(
    vals: jaxls.VarValues,
    sphere_var: SphereVar,
    init_radius: jax.Array,
    scale: float,
    lambda_radius_reg: float,
) -> jax.Array:
    """Radius regularization: penalize deviation from initial radius.

    Args:
        vals: Variable values from optimizer
        sphere_var: Sphere variable
        init_radius: Initial radius value
        scale: Scale factor for normalization
        lambda_radius_reg: Weight for regularization

    Returns:
        Scalar residual
    """
    sphere = vals[sphere_var]
    return jnp.sqrt(lambda_radius_reg) * (sphere.radius - init_radius) / scale


@jaxls.Cost.factory
def _self_collision_cost(
    vals: jaxls.VarValues,
    sphere_var_i: SphereVar,
    sphere_var_j: SphereVar,
    transform_i: jax.Array,
    transform_j: jax.Array,
    lambda_self_collision: float,
) -> jax.Array:
    """Self-collision cost for a pair of spheres from non-contiguous links.

    Penalizes penetration (negative signed distance) between spheres.

    Args:
        vals: Variable values from optimizer
        sphere_var_i: First sphere variable
        sphere_var_j: Second sphere variable
        transform_i: FK transform for first sphere's link, (7,) wxyz+xyz
        transform_j: FK transform for second sphere's link, (7,) wxyz+xyz
        lambda_self_collision: Weight for self-collision penalty

    Returns:
        Scalar residual (sqrt(lambda) * penetration_depth)
    """
    sphere_i = vals[sphere_var_i]
    sphere_j = vals[sphere_var_j]

    # Transform centers to world frame
    center_i_world = _transform_point(sphere_i.center, transform_i)
    center_j_world = _transform_point(sphere_j.center, transform_j)

    # Compute signed distance (negative = penetration)
    dist = jnp.sqrt(jnp.sum((center_i_world - center_j_world) ** 2) + 1e-8)
    sum_radii = sphere_i.radius + sphere_j.radius
    signed_dist = dist - sum_radii

    # Penalize penetration only: max(0, -signed_dist)
    penetration = jnp.maximum(0.0, -signed_dist)
    return jnp.sqrt(lambda_self_collision) * penetration


def _build_collision_pairs(
    link_names: list[str],
    link_sphere_ranges: list[tuple[int, int]],
    non_contiguous_pairs: list[tuple[str, str]],
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[str, str]]]:
    """Build list of sphere index pairs to check for self-collision.

    Args:
        link_names: Ordered list of link names with spheres.
        link_sphere_ranges: (start, end) sphere indices for each link.
        non_contiguous_pairs: Link pairs that are not adjacent in kinematic chain.
        excluded_pairs: Link pairs to skip (user-disabled).

    Returns:
        Tuple of (sphere_pairs, valid_link_pairs) where:
        - sphere_pairs: List of (sphere_idx_i, sphere_idx_j) to check
        - valid_link_pairs: Link pairs that passed filtering (for logging)
    """
    link_name_to_internal_idx = {name: i for i, name in enumerate(link_names)}

    sphere_pairs: list[tuple[int, int]] = []
    valid_link_pairs: list[tuple[str, str]] = []

    for link_a, link_b in non_contiguous_pairs:
        # Skip user-excluded pairs
        if excluded_pairs:
            if (link_a, link_b) in excluded_pairs or (link_b, link_a) in excluded_pairs:
                continue
        # Skip if either link not in our optimization set
        if link_a not in link_name_to_internal_idx:
            continue
        if link_b not in link_name_to_internal_idx:
            continue

        internal_idx_a = link_name_to_internal_idx[link_a]
        internal_idx_b = link_name_to_internal_idx[link_b]
        range_a = link_sphere_ranges[internal_idx_a]
        range_b = link_sphere_ranges[internal_idx_b]

        # Skip if either link has no spheres
        if range_a[0] >= range_a[1] or range_b[0] >= range_b[1]:
            continue

        valid_link_pairs.append((link_a, link_b))

        # Add all sphere pairs between these links
        for i in range(range_a[0], range_a[1]):
            for j in range(range_b[0], range_b[1]):
                sphere_pairs.append((i, j))

    return sphere_pairs, valid_link_pairs


def _build_jaxls_costs(
    data: _FlattenedSphereData,
    params: RefineParams,
    collision_pairs: list[tuple[int, int]],
) -> list[jaxls.Cost]:
    """Build optimization costs for sphere refinement.

    Includes:
    - Under-approximation: penalize points outside spheres
    - Over-approximation: penalize large sphere volumes
    - Center regularization: penalize deviation from initial positions
    - Radius regularization: penalize deviation from initial radii
    - Self-collision: penalize overlap between non-adjacent link spheres

    Args:
        data: Flattened sphere/point data
        params: Refinement parameters
        collision_pairs: List of (sphere_idx_i, sphere_idx_j) pairs to check

    Returns:
        List of jaxls.Cost objects
    """
    costs: list[jaxls.Cost] = []
    n_spheres = data.n_spheres
    sphere_vars = SphereVar(jnp.arange(n_spheres))

    # Find max spheres and points per link for padding
    max_spheres_per_link = max(end - start for start, end in data.link_sphere_ranges)
    max_points_per_link = max(end - start for start, end in data.link_point_ranges)

    # Under-approximation costs: one per link with padded fixed dimensions
    for link_idx in range(data.n_links):
        point_start, point_end = data.link_point_ranges[link_idx]
        sphere_start, sphere_end = data.link_sphere_ranges[link_idx]

        n_link_spheres = sphere_end - sphere_start
        n_link_points = point_end - point_start

        if n_link_points == 0 or n_link_spheres == 0:
            continue

        # Get link data
        link_points = data.points_all[point_start:point_end]

        # Pad points to max size
        padded_points = jnp.pad(
            link_points,
            ((0, max_points_per_link - n_link_points), (0, 0)),
        )

        # Create padded sphere vars tuple (use SphereVar(0) as padding placeholder)
        link_sphere_vars = tuple(
            SphereVar(sphere_start + i) if i < n_link_spheres else SphereVar(0)
            for i in range(max_spheres_per_link)
        )

        # Create masks
        sphere_mask = jnp.arange(max_spheres_per_link) < n_link_spheres
        point_mask = jnp.arange(max_points_per_link) < n_link_points

        costs.append(
            _under_approx_cost(
                link_sphere_vars,
                padded_points,
                sphere_mask,
                point_mask,
                params.lambda_under,
            )
        )

    # Over-approximation costs: penalize large sphere volumes
    costs.append(
        _over_approx_cost(
            sphere_vars,
            data.scale,
            params.lambda_over,
        )
    )

    # Center regularization: penalize deviation from initial positions
    costs.append(
        _center_reg_cost(
            sphere_vars,
            data.spheres.center,
            data.scale,
            params.lambda_center_reg,
        )
    )

    # Radius regularization: penalize deviation from initial radii
    costs.append(
        _radius_reg_cost(
            sphere_vars,
            data.spheres.radius,
            data.scale,
            params.lambda_radius_reg,
        )
    )

    # Self-collision costs: one per collision pair
    # Note: The lambda_self_collision > 0 check is done before calling this function
    # (in refine_robot_spheres) to avoid JIT tracing issues with the conditional
    for i, j in collision_pairs:
        costs.append(
            _self_collision_cost(
                SphereVar(i),
                SphereVar(j),
                data.sphere_transforms[i],
                data.sphere_transforms[j],
                params.lambda_self_collision,
            )
        )

    return costs


# =============================================================================
# Optimization loop
# =============================================================================


@jdc.jit
def _run_robot_optimization(
    data: _FlattenedSphereData,
    params: RefineParams,
    n_iters: jdc.Static[int],
    tol: float,
    collision_pairs: jdc.Static[tuple[tuple[int, int], ...]],
) -> tuple[Sphere, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the full optimization loop using jaxls nonlinear least squares.

    Args:
        data: Flattened sphere/point data with initial spheres
        params: Refinement parameters
        n_iters: Maximum number of optimization iterations
        tol: Relative convergence tolerance for early stopping
        collision_pairs: Static tuple of (sphere_idx_i, sphere_idx_j) pairs

    Returns:
        Tuple of (final_spheres, init_loss, final_loss, n_steps)
    """
    n_spheres = data.n_spheres

    # Create sphere variables
    sphere_vars = SphereVar(jnp.arange(n_spheres))

    # Build costs
    costs = _build_jaxls_costs(data, params, list(collision_pairs))

    if not costs:
        # No costs to optimize, return initial spheres
        return data.spheres, jnp.array(0.0), jnp.array(0.0), jnp.array(0)

    # Create initial values
    initial_vals = jaxls.VarValues.make(
        [
            sphere_vars[i].with_value(
                Sphere(
                    center=data.spheres.center[i],
                    radius=data.spheres.radius[i],
                )
            )
            for i in range(n_spheres)
        ]
    )

    # Build and solve the problem
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[sphere_vars])
    analyzed = problem.analyze()

    # Compute initial cost
    init_residual = analyzed.compute_residual_vector(initial_vals)
    init_loss = jnp.sum(init_residual**2)

    # Solve
    solution, summary = analyzed.solve(
        initial_vals=initial_vals,
        termination=jaxls.TerminationConfig(
            max_iterations=n_iters,
            cost_tolerance=tol,
        ),
        trust_region=jaxls.TrustRegionConfig(),
        verbose=False,
        return_summary=True,
    )

    # Extract optimized spheres
    centers_list = []
    radii_list = []
    for i in range(n_spheres):
        sphere = solution[SphereVar(i)]
        centers_list.append(sphere.center)
        radii_list.append(sphere.radius)

    final_centers = jnp.stack(centers_list)
    final_radii = jnp.stack(radii_list)

    # Clamp radii to minimum
    final_radii = jnp.maximum(final_radii, params.min_radius)

    final_spheres = Sphere(center=final_centers, radius=final_radii)

    # Compute final cost
    final_vals = jaxls.VarValues.make(
        [
            sphere_vars[i].with_value(
                Sphere(center=final_centers[i], radius=final_radii[i])
            )
            for i in range(n_spheres)
        ]
    )
    final_residual = analyzed.compute_residual_vector(final_vals)
    final_loss = jnp.sum(final_residual**2)

    return final_spheres, init_loss, final_loss, summary.iterations


# =============================================================================
# Main entry point (called by Robot.refine)
# =============================================================================


def refine_robot_spheres(
    link_spheres: dict[str, list[Sphere]],
    link_meshes: dict[str, trimesh.Trimesh],
    all_link_names: list[str],
    joint_limits: tuple[np.ndarray, np.ndarray],
    compute_transforms: Callable[[np.ndarray], np.ndarray],
    non_contiguous_pairs: list[tuple[str, str]],
    refine_params: RefineParams | None = None,
    joint_cfg: np.ndarray | None = None,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, list[Sphere]]:
    """
    Refine sphere parameters for all robot links jointly.

    Uses under-approximation (spheres must cover mesh surface),
    over-approximation (minimize sphere volumes), and self-collision
    (penalize overlap between non-adjacent links) costs.

    This is the internal entry point called by Robot.refine().

    Args:
        link_spheres: Dict mapping link names to lists of Sphere objects.
        link_meshes: Dict mapping link names to their collision meshes.
        all_link_names: Ordered list of all link names.
        joint_limits: Tuple of (lower_limits, upper_limits) arrays.
        compute_transforms: Function that takes joint_cfg and returns (N, 7) transforms.
        non_contiguous_pairs: List of (link_a, link_b) pairs that are not adjacent.
        refine_params: Refinement parameters. If None, uses defaults.
        joint_cfg: Joint configuration for FK computation. If None, uses middle of
            joint limits.
        excluded_pairs: Link pairs to skip for collision checking (user-disabled).

    Returns:
        Dict mapping link names to refined lists of Sphere objects
    """
    p = refine_params or RefineParams()

    # Get link names with spheres
    link_names = [name for name in all_link_names if link_spheres.get(name)]
    if not link_names:
        return link_spheres

    link_name_to_idx = {name: idx for idx, name in enumerate(all_link_names)}

    # Compute FK at specified config or middle of joint limits
    if joint_cfg is None:
        lower, upper = joint_limits
        joint_cfg = (lower + upper) / 2
    Ts = compute_transforms(joint_cfg)

    # Sample points for each link
    link_points: dict[str, np.ndarray] = {}
    for link_name in link_names:
        mesh = link_meshes.get(link_name)
        if mesh is not None and not mesh.is_empty:
            link_points[link_name] = mesh.sample(p.n_samples)  # type: ignore[assignment]
        else:
            link_points[link_name] = np.zeros((0, 3))

    # Build flattened data (with transforms)
    flat_data = _build_flattened_sphere_data(
        link_spheres, link_points, link_names, Ts, link_name_to_idx
    )

    if flat_data is None:
        return link_spheres

    # Build collision pairs only if self-collision is enabled
    # This check must be done here (not in JIT) to avoid tracing issues
    collision_pairs: list[tuple[int, int]] = []
    if p.lambda_self_collision > 0:
        # Build collision pairs
        collision_pairs, valid_link_pairs = _build_collision_pairs(
            list(flat_data.link_names),
            list(flat_data.link_sphere_ranges),
            non_contiguous_pairs,
            excluded_pairs,
        )

        if collision_pairs:
            logger.info(
                f"Self-collision: checking {len(collision_pairs)} sphere pairs "
                f"across {len(valid_link_pairs)} link pairs"
            )

    # Run optimization
    final_spheres, init_loss, final_loss, n_steps = _run_robot_optimization(
        flat_data,
        p,
        p.n_iters,
        p.tol,
        tuple(collision_pairs),
    )

    logger.info(
        f"Optimization: {int(n_steps)} iterations, "
        f"loss {float(init_loss):.4f} -> {float(final_loss):.4f}"
    )

    # Unflatten results
    centers_np = np.array(final_spheres.center)
    radii_np = np.array(final_spheres.radius)

    refined_link_spheres = _unflatten_to_link_spheres(
        centers_np,
        radii_np,
        list(flat_data.link_names),
        list(flat_data.link_sphere_ranges),
        link_spheres,
    )

    return refined_link_spheres


# =============================================================================
# Ellipsoid Refinement
# =============================================================================


@jdc.pytree_dataclass
class _FlattenedEllipsoidData:
    """Flattened ellipsoid and point data for JIT-compatible optimization."""

    # Batched ellipsoids
    ellipsoids: Ellipsoid  # center: (N, 3), semi_axes: (N, 3)

    # Point data
    points_all: jnp.ndarray  # (P, 3) all surface points concatenated

    # Index mappings for per-link operations
    ellipsoid_to_link: jnp.ndarray  # (N,) int32 - which link each ellipsoid belongs to
    point_to_link: jnp.ndarray  # (P,) int32 - which link each point belongs to

    # FK transforms for self-collision
    ellipsoid_transforms: jnp.ndarray  # (N, 7) FK transform (wxyz+xyz) per ellipsoid

    # Metadata (static - not traced by JAX)
    link_ellipsoid_ranges: jdc.Static[tuple[tuple[int, int], ...]]
    link_point_ranges: jdc.Static[tuple[tuple[int, int], ...]]
    link_names: jdc.Static[tuple[str, ...]]
    n_ellipsoids: jdc.Static[int]
    n_links: jdc.Static[int]
    scale: float


def _build_flattened_ellipsoid_data(
    link_ellipsoids: dict[str, list[Ellipsoid]],
    link_points: dict[str, np.ndarray],
    link_names: list[str],
    Ts: np.ndarray,
    link_name_to_idx: dict[str, int],
) -> _FlattenedEllipsoidData | None:
    """Flatten per-link ellipsoid data into contiguous arrays for optimization."""
    centers_list = []
    semi_axes_list = []
    ellipsoid_to_link_list = []
    ellipsoid_transforms_list = []
    link_ellipsoid_ranges = []

    points_list = []
    point_to_link_list = []
    link_point_ranges = []

    ellipsoid_offset = 0
    point_offset = 0

    for link_idx, link_name in enumerate(link_names):
        ellips = link_ellipsoids.get(link_name, [])
        pts = link_points.get(link_name, np.zeros((0, 3)))

        # Ellipsoids for this link
        n_ellips = len(ellips)
        link_ellipsoid_ranges.append((ellipsoid_offset, ellipsoid_offset + n_ellips))
        for ell in ellips:
            centers_list.append(np.array(ell.center))
            semi_axes_list.append(np.array(ell.semi_axes))
            ellipsoid_to_link_list.append(link_idx)
            # Get FK transform for this link
            global_link_idx = link_name_to_idx[link_name]
            ellipsoid_transforms_list.append(Ts[global_link_idx])
        ellipsoid_offset += n_ellips

        # Points for this link
        n_pts = len(pts)
        link_point_ranges.append((point_offset, point_offset + n_pts))
        if n_pts > 0:
            points_list.append(pts)
            point_to_link_list.extend([link_idx] * n_pts)
        point_offset += n_pts

    if ellipsoid_offset == 0:
        return None

    # Stack arrays
    centers = np.stack(centers_list)  # (N, 3)
    semi_axes = np.stack(semi_axes_list)  # (N, 3)
    ellipsoid_to_link = np.array(ellipsoid_to_link_list, dtype=np.int32)
    ellipsoid_transforms = np.stack(ellipsoid_transforms_list)  # (N, 7)

    if points_list:
        points_all = np.concatenate(points_list)  # (P, 3)
    else:
        points_all = np.zeros((0, 3))
    point_to_link = (
        np.array(point_to_link_list, dtype=np.int32)
        if point_to_link_list
        else np.zeros(0, dtype=np.int32)
    )

    # Compute scale from bounding box
    all_positions = (
        np.concatenate([centers, points_all]) if len(points_all) > 0 else centers
    )
    bbox_min = all_positions.min(axis=0)
    bbox_max = all_positions.max(axis=0)
    scale = float(np.linalg.norm(bbox_max - bbox_min) + 1e-6)

    return _FlattenedEllipsoidData(
        ellipsoids=Ellipsoid(
            center=jnp.array(centers),
            semi_axes=jnp.array(semi_axes),
        ),
        points_all=jnp.array(points_all),
        ellipsoid_to_link=jnp.array(ellipsoid_to_link),
        point_to_link=jnp.array(point_to_link),
        ellipsoid_transforms=jnp.array(ellipsoid_transforms),
        link_ellipsoid_ranges=tuple(link_ellipsoid_ranges),
        link_point_ranges=tuple(link_point_ranges),
        link_names=tuple(link_names),
        n_ellipsoids=ellipsoid_offset,
        n_links=len(link_names),
        scale=scale,
    )


def _unflatten_to_link_ellipsoids(
    centers: np.ndarray,
    semi_axes: np.ndarray,
    link_names: list[str],
    link_ellipsoid_ranges: list[tuple[int, int]],
    original_link_ellipsoids: dict[str, list[Ellipsoid]],
) -> dict[str, list[Ellipsoid]]:
    """Convert flattened arrays back to per-link Ellipsoid dictionaries."""
    refined_link_ellipsoids = {}

    for i, link_name in enumerate(link_names):
        start, end = link_ellipsoid_ranges[i]
        refined_link_ellipsoids[link_name] = [
            Ellipsoid(center=centers[j], semi_axes=semi_axes[j])
            for j in range(start, end)
        ]

    # Keep links not in optimization
    for link_name, ellipsoids in original_link_ellipsoids.items():
        if link_name not in refined_link_ellipsoids:
            refined_link_ellipsoids[link_name] = ellipsoids

    return refined_link_ellipsoids


# =============================================================================
# Ellipsoid jaxls Cost Functions
# =============================================================================


def safe_sqrt(x, epsilon=1e-12):
    """
    Returns sqrt(x) with a gradient that is well-defined (0.0) at x=0.
    """
    safe_x = jnp.where(x > epsilon, x, epsilon)
    return jnp.where(x > epsilon, jnp.sqrt(safe_x), 0.0)


@jaxls.Cost.factory
def _ellipsoid_under_approx_cost(
    vals: jaxls.VarValues,
    ellipsoid_vars: tuple[EllipsoidVar, ...],
    points: jax.Array,
    ellipsoid_mask: jax.Array,
    point_mask: jax.Array,
    lambda_under: float,
) -> jax.Array:
    """Under-approximation for ellipsoids: penalize points outside ALL ellipsoids.

    Uses effective radius approximation for smooth, differentiable distance.
    """
    # 1. Collect and clamp geometry
    centers = jnp.stack([vals[ev].center for ev in ellipsoid_vars])  # (E, 3)
    semi_axes_raw = jnp.stack([vals[ev].semi_axes for ev in ellipsoid_vars])  # (E, 3)

    # Clamp semi_axes to a reasonable minimum to avoid division by zero
    semi_axes = jnp.maximum(semi_axes_raw, 1e-4)

    # 2. Compute distance to center
    # diff: (P, E, 3)
    diff = points[:, None, :] - centers[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    dist_to_center = safe_sqrt(dist_sq)

    # 3. Compute effective radius (r_eff)
    # Avoid division by zero in direction if point is exactly at center
    direction = diff / (dist_to_center[:, :, None] + 1e-9)

    # Vectorized r_eff: 1 / sqrt(sum((dir/axis)^2))
    scaled_dir_sq = (direction / semi_axes[None, :, :]) ** 2
    inv_r_sq = jnp.sum(scaled_dir_sq, axis=-1)
    r_eff = 1.0 / safe_sqrt(inv_r_sq)

    # 4. Signed distance
    signed_dists = dist_to_center - r_eff

    # 5. Stable Masking
    # Using jnp.inf can cause NaNs in gradients if all entries are masked.
    # We use a large finite value instead.
    LARGE_VAL = 1e8
    masked_dists = jnp.where(ellipsoid_mask[None, :], signed_dists, LARGE_VAL)

    # 6. Final Residual Calculation
    min_signed_dist = jnp.min(masked_dists, axis=1)

    # residuals: lambda * max(0, dist)
    # We use point_mask at the very end to zero out invalid points
    raw_residuals = jnp.sqrt(lambda_under) * jnp.maximum(0.0, min_signed_dist)

    return jnp.where(point_mask, raw_residuals, 0.0)


@jaxls.Cost.factory
def _ellipsoid_over_approx_cost(
    vals: jaxls.VarValues,
    ellipsoid_var: EllipsoidVar,
    scale: float,
    lambda_over: float,
) -> jax.Array:
    """Over-approximation cost: penalize large ellipsoid semi-axes.

    Uses per-axis penalties so each axis has independent gradient.
    Unlike volume-based penalty where ∂cost/∂a depends on (b, c),
    this gives consistent shrinkage pressure on each axis.
    """
    ell = vals[ellipsoid_var]
    semi_axes = jnp.maximum(ell.semi_axes, 1e-4)
    # Per-axis penalty: each axis contributes independently to the loss
    # The /3 normalizes so that for a=b=c=r, total loss matches sphere
    per_axis = (semi_axes / scale) ** 1.5
    return jnp.sqrt(lambda_over / 3) * per_axis


@jaxls.Cost.factory
def _ellipsoid_center_reg_cost(
    vals: jaxls.VarValues,
    ellipsoid_var: EllipsoidVar,
    init_center: jax.Array,
    scale: float,
    lambda_center_reg: float,
) -> jax.Array:
    """Center regularization: penalize deviation from initial center position."""
    ell = vals[ellipsoid_var]
    return jnp.sqrt(lambda_center_reg) * (ell.center - init_center) / scale


@jaxls.Cost.factory
def _ellipsoid_semi_axes_reg_cost(
    vals: jaxls.VarValues,
    ellipsoid_var: EllipsoidVar,
    init_semi_axes: jax.Array,
    scale: float,
    lambda_radius_reg: float,
) -> jax.Array:
    """Semi-axes regularization: penalize change in mean radius only.

    Unlike per-axis regularization, this allows shape changes (squishing)
    while still penalizing overall size drift.
    """
    ell = vals[ellipsoid_var]
    mean_init = jnp.mean(init_semi_axes)
    mean_curr = jnp.mean(ell.semi_axes)
    return jnp.sqrt(lambda_radius_reg) * (mean_curr - mean_init) / scale


@jaxls.Cost.factory
def _ellipsoid_axis_ratio_cost(
    vals: jaxls.VarValues,
    ellipsoid_var: EllipsoidVar,
    max_ratio: float,
    lambda_axis_ratio: float,
) -> jax.Array:
    """Axis ratio regularization: penalize extreme axis ratios.

    Prevents degenerate ellipsoids where one axis is much larger/smaller than others.
    """
    ell = vals[ellipsoid_var]
    semi_axes = jnp.maximum(ell.semi_axes, 1e-6)
    max_axis = jnp.max(semi_axes)
    min_axis = jnp.min(semi_axes)
    ratio = max_axis / min_axis

    # Penalize ratio exceeding max_ratio
    excess = jnp.maximum(0.0, ratio - max_ratio)
    return jnp.sqrt(lambda_axis_ratio) * excess


@jaxls.Cost.factory
def _ellipsoid_self_collision_cost(
    vals: jaxls.VarValues,
    ellipsoid_var_i: EllipsoidVar,
    ellipsoid_var_j: EllipsoidVar,
    transform_i: jax.Array,
    transform_j: jax.Array,
    lambda_self_collision: float,
) -> jax.Array:
    """Self-collision cost for a pair of ellipsoids from non-contiguous links.

    Uses mutual scaled-sphere approximation for smooth collision distance.
    """
    ell_i = vals[ellipsoid_var_i]
    ell_j = vals[ellipsoid_var_j]

    # Clamp semi_axes to prevent numerical issues with negative values
    semi_axes_i = jnp.maximum(ell_i.semi_axes, 1e-4)
    semi_axes_j = jnp.maximum(ell_j.semi_axes, 1e-4)

    # Transform centers to world frame
    center_i_world = _transform_point(ell_i.center, transform_i)
    center_j_world = _transform_point(ell_j.center, transform_j)

    # Direction from i to j
    center_to_center = center_j_world - center_i_world
    dist_centers = jnp.sqrt(jnp.sum(center_to_center**2) + 1e-8)
    direction = center_to_center / (dist_centers + 1e-8)

    # For axis-aligned ellipsoids, use direction directly
    # (In Phase 2 with rotated ellipsoids, we'd transform to local frame)
    r_i_eff = ellipsoid_effective_radius(semi_axes_i, direction)
    r_j_eff = ellipsoid_effective_radius(semi_axes_j, -direction)

    # Signed distance (negative = penetration)
    signed_dist = dist_centers - r_i_eff - r_j_eff

    # Penalize penetration only
    penetration = jnp.maximum(0.0, -signed_dist)
    return jnp.sqrt(lambda_self_collision) * penetration


def _build_ellipsoid_jaxls_costs(
    data: _FlattenedEllipsoidData,
    params: RefineParams,
    collision_pairs: list[tuple[int, int]],
) -> list:
    """Build jaxls cost functions for ellipsoid optimization."""
    costs = []
    n_ellipsoids = data.n_ellipsoids

    # Create ellipsoid variable indices
    ellipsoid_vars = EllipsoidVar(jnp.arange(n_ellipsoids))

    # Per-link costs
    for link_idx in range(data.n_links):
        ell_start, ell_end = data.link_ellipsoid_ranges[link_idx]
        pt_start, pt_end = data.link_point_ranges[link_idx]

        n_ell = ell_end - ell_start
        n_pts = pt_end - pt_start

        if n_ell == 0:
            continue

        link_ellipsoid_vars = tuple(
            ellipsoid_vars[i] for i in range(ell_start, ell_end)
        )

        # Under-approximation cost
        # Note: Don't check params.lambda_under > 0 here - it's a traced value in JIT
        # The lambda weight handles "disabled" costs by returning 0 contribution
        if n_pts > 0:
            link_points = data.points_all[pt_start:pt_end]

            # Pad to fixed size for JIT
            max_pts = max(n_pts, 1)
            max_ell = max(n_ell, 1)

            padded_points = jnp.zeros((max_pts, 3))
            padded_points = padded_points.at[:n_pts].set(link_points)

            ellipsoid_mask = jnp.arange(max_ell) < n_ell
            point_mask = jnp.arange(max_pts) < n_pts

            costs.append(
                _ellipsoid_under_approx_cost(
                    link_ellipsoid_vars,
                    padded_points,
                    ellipsoid_mask,
                    point_mask,
                    params.lambda_under,
                )
            )

        # Per-ellipsoid costs
        for i, ell_idx in enumerate(range(ell_start, ell_end)):
            ell_var = ellipsoid_vars[ell_idx]

            # Over-approximation
            costs.append(
                _ellipsoid_over_approx_cost(
                    ell_var,
                    data.scale,
                    params.lambda_over,
                )
            )

            # Center regularization
            init_center = data.ellipsoids.center[ell_idx]
            costs.append(
                _ellipsoid_center_reg_cost(
                    ell_var,
                    init_center,
                    data.scale,
                    params.lambda_center_reg,
                )
            )

            # Semi-axes regularization
            init_semi_axes = data.ellipsoids.semi_axes[ell_idx]
            costs.append(
                _ellipsoid_semi_axes_reg_cost(
                    ell_var,
                    init_semi_axes,
                    data.scale,
                    params.lambda_radius_reg,
                )
            )

            # Axis ratio regularization
            costs.append(
                _ellipsoid_axis_ratio_cost(
                    ell_var,
                    params.max_axis_ratio,
                    params.lambda_axis_ratio,
                )
            )
    #
    # # Self-collision costs
    # # Note: The lambda_self_collision > 0 check is done before calling this function
    # # (in refine_robot_ellipsoids) to avoid JIT tracing issues with the conditional
    # for i, j in collision_pairs:
    #     costs.append(
    #         _ellipsoid_self_collision_cost(
    #             ellipsoid_vars[i],
    #             ellipsoid_vars[j],
    #             data.ellipsoid_transforms[i],
    #             data.ellipsoid_transforms[j],
    #             params.lambda_self_collision,
    #         )
    #     )

    return costs


@jdc.jit
def _run_robot_ellipsoid_optimization(
    data: _FlattenedEllipsoidData,
    params: RefineParams,
    n_iters: jdc.Static[int],
    tol: float,
    collision_pairs: jdc.Static[tuple[tuple[int, int], ...]],
) -> tuple[Ellipsoid, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run ellipsoid optimization loop using jaxls nonlinear least squares."""
    n_ellipsoids = data.n_ellipsoids

    # Create ellipsoid variables
    ellipsoid_vars = EllipsoidVar(jnp.arange(n_ellipsoids))

    # Build costs
    costs = _build_ellipsoid_jaxls_costs(data, params, list(collision_pairs))

    if not costs:
        return data.ellipsoids, jnp.array(0.0), jnp.array(0.0), jnp.array(0)

    # Create initial values
    initial_vals = jaxls.VarValues.make(
        [
            ellipsoid_vars[i].with_value(
                Ellipsoid(
                    center=data.ellipsoids.center[i],
                    semi_axes=data.ellipsoids.semi_axes[i],
                )
            )
            for i in range(n_ellipsoids)
        ]
    )

    # Build and solve
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[ellipsoid_vars])
    analyzed = problem.analyze()

    # Compute initial cost
    init_residual = analyzed.compute_residual_vector(initial_vals)
    init_loss = jnp.sum(init_residual**2)

    # Solve
    solution, summary = analyzed.solve(
        initial_vals=initial_vals,
        termination=jaxls.TerminationConfig(
            max_iterations=n_iters,
            cost_tolerance=tol,
        ),
        trust_region=jaxls.TrustRegionConfig(),
        verbose=False,
        return_summary=True,
    )

    # Extract optimized ellipsoids
    centers_list = []
    semi_axes_list = []
    for i in range(n_ellipsoids):
        ell = solution[EllipsoidVar(i)]
        centers_list.append(ell.center)
        semi_axes_list.append(ell.semi_axes)

    final_centers = jnp.stack(centers_list)
    final_semi_axes = jnp.stack(semi_axes_list)

    # Clamp semi-axes to minimum
    final_semi_axes = jnp.maximum(final_semi_axes, params.min_semi_axis)

    final_ellipsoids = Ellipsoid(center=final_centers, semi_axes=final_semi_axes)

    # Compute final cost
    final_vals = jaxls.VarValues.make(
        [
            ellipsoid_vars[i].with_value(
                Ellipsoid(center=final_centers[i], semi_axes=final_semi_axes[i])
            )
            for i in range(n_ellipsoids)
        ]
    )
    final_residual = analyzed.compute_residual_vector(final_vals)
    final_loss = jnp.sum(final_residual**2)

    return final_ellipsoids, init_loss, final_loss, summary.iterations


# =============================================================================
# Ellipsoid Main Entry Point
# =============================================================================


def refine_robot_ellipsoids(
    link_spheres: dict[str, list[Sphere]],
    link_meshes: dict[str, trimesh.Trimesh],
    all_link_names: list[str],
    joint_limits: tuple[np.ndarray, np.ndarray],
    compute_transforms: Callable[[np.ndarray], np.ndarray],
    non_contiguous_pairs: list[tuple[str, str]],
    refine_params: RefineParams | None = None,
    joint_cfg: np.ndarray | None = None,
    excluded_pairs: set[tuple[str, str]] | None = None,
) -> dict[str, list[Ellipsoid]]:
    """
    Convert spheres to ellipsoids and refine jointly using jaxls optimization.

    Takes initial sphere placement and refines to axis-aligned ellipsoids,
    allowing each primitive to stretch along axes to better fit the geometry.

    Args:
        link_spheres: Dict mapping link names to lists of Sphere objects.
        link_meshes: Dict mapping link names to their collision meshes.
        all_link_names: Ordered list of all link names.
        joint_limits: Tuple of (lower_limits, upper_limits) arrays.
        compute_transforms: Function that takes joint_cfg and returns (N, 7) transforms.
        non_contiguous_pairs: List of (link_a, link_b) pairs that are not adjacent.
        refine_params: Refinement parameters. If None, uses defaults.
        joint_cfg: Joint configuration for FK computation. If None, uses middle.
        excluded_pairs: Link pairs to skip for collision checking.

    Returns:
        Dict mapping link names to refined lists of Ellipsoid objects.
    """
    p = refine_params or RefineParams()

    # Get link names with spheres
    link_names = [name for name in all_link_names if link_spheres.get(name)]
    if not link_names:
        return {}

    link_name_to_idx = {name: idx for idx, name in enumerate(all_link_names)}

    # Convert spheres to ellipsoids
    link_ellipsoids: dict[str, list[Ellipsoid]] = {}
    for link_name, spheres in link_spheres.items():
        link_ellipsoids[link_name] = [sphere_to_ellipsoid(s) for s in spheres]

    # Compute FK
    if joint_cfg is None:
        lower, upper = joint_limits
        joint_cfg = (lower + upper) / 2
    Ts = compute_transforms(joint_cfg)

    # Sample points
    link_points: dict[str, np.ndarray] = {}
    for link_name in link_names:
        mesh = link_meshes.get(link_name)
        if mesh is not None and not mesh.is_empty:
            link_points[link_name] = mesh.sample(p.n_samples)
        else:
            link_points[link_name] = np.zeros((0, 3))

    # Build flattened data
    flat_data = _build_flattened_ellipsoid_data(
        link_ellipsoids, link_points, link_names, Ts, link_name_to_idx
    )

    if flat_data is None:
        return link_ellipsoids

    # Build collision pairs
    collision_pairs: list[tuple[int, int]] = []
    if p.lambda_self_collision > 0:
        collision_pairs, valid_link_pairs = _build_collision_pairs(
            list(flat_data.link_names),
            list(flat_data.link_ellipsoid_ranges),
            non_contiguous_pairs,
            excluded_pairs,
        )

        if collision_pairs:
            logger.info(
                f"Self-collision: checking {len(collision_pairs)} ellipsoid pairs "
                f"across {len(valid_link_pairs)} link pairs"
            )

    # Run optimization
    final_ellipsoids, init_loss, final_loss, n_steps = (
        _run_robot_ellipsoid_optimization(
            flat_data,
            p,
            p.n_iters,
            p.tol,
            tuple(collision_pairs),
        )
    )

    logger.info(
        f"Ellipsoid optimization: {int(n_steps)} iterations, "
        f"loss {float(init_loss):.4f} -> {float(final_loss):.4f}"
    )

    # Unflatten results
    centers_np = np.array(final_ellipsoids.center)
    semi_axes_np = np.array(final_ellipsoids.semi_axes)

    refined_link_ellipsoids = _unflatten_to_link_ellipsoids(
        centers_np,
        semi_axes_np,
        list(flat_data.link_names),
        list(flat_data.link_ellipsoid_ranges),
        link_ellipsoids,
    )

    return refined_link_ellipsoids
