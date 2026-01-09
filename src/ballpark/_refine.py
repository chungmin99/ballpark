"""Robot-level sphere refinement with jaxls nonlinear least squares optimization."""

from __future__ import annotations

import numpy as np
import trimesh
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxls
from loguru import logger

from ._spherize import Sphere
from ._config import RefineParams


# =============================================================================
# jaxls Variable Type
# =============================================================================


class SphereVar(
    jaxls.Var[Sphere],
    default_factory=lambda: Sphere(center=jnp.zeros(3), radius=jnp.array(0.01)),
):
    """Variable type for sphere parameters (center + radius)."""

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
) -> _FlattenedSphereData | None:
    """Flatten per-link sphere/point dicts into arrays for JAX optimization.

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

    return _FlattenedSphereData(
        spheres=spheres,
        points_all=points_all,
        sphere_to_link=sphere_to_link,
        point_to_link=point_to_link,
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


def _build_jaxls_costs(
    data: _FlattenedSphereData,
    params: RefineParams,
) -> list[jaxls.Cost]:
    """Build optimization costs for sphere refinement.

    Includes:
    - Under-approximation: penalize points outside spheres
    - Over-approximation: penalize large sphere volumes
    - Center regularization: penalize deviation from initial positions
    - Radius regularization: penalize deviation from initial radii

    Args:
        data: Flattened sphere/point data
        params: Refinement parameters

    Returns:
        List of jaxls.Cost objects
    """
    costs: list[jaxls.Cost] = []
    n_spheres = data.n_spheres
    sphere_vars = SphereVar(jnp.arange(n_spheres))

    # Find max spheres and points per link for padding
    max_spheres_per_link = max(
        end - start for start, end in data.link_sphere_ranges
    )
    max_points_per_link = max(
        end - start for start, end in data.link_point_ranges
    )

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
) -> tuple[Sphere, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the full optimization loop using jaxls nonlinear least squares.

    Args:
        data: Flattened sphere/point data with initial spheres
        params: Refinement parameters
        n_iters: Maximum number of optimization iterations
        tol: Relative convergence tolerance for early stopping

    Returns:
        Tuple of (final_spheres, init_loss, final_loss, n_steps)
    """
    n_spheres = data.n_spheres

    # Create sphere variables
    sphere_vars = SphereVar(jnp.arange(n_spheres))

    # Build costs
    costs = _build_jaxls_costs(data, params)

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
    refine_params: RefineParams | None = None,
) -> dict[str, list[Sphere]]:
    """
    Refine sphere parameters for all robot links jointly.

    Uses under-approximation (spheres must cover mesh surface) and
    over-approximation (minimize sphere volumes) costs.

    This is the internal entry point called by Robot.refine().

    Args:
        link_spheres: Dict mapping link names to lists of Sphere objects.
        link_meshes: Dict mapping link names to their collision meshes.
        all_link_names: Ordered list of all link names.
        refine_params: Refinement parameters. If None, uses defaults.

    Returns:
        Dict mapping link names to refined lists of Sphere objects
    """
    p = refine_params or RefineParams()

    # Get link names with spheres
    link_names = [name for name in all_link_names if link_spheres.get(name)]
    if not link_names:
        return link_spheres

    # Sample points for each link
    link_points: dict[str, np.ndarray] = {}
    for link_name in link_names:
        mesh = link_meshes.get(link_name)
        if mesh is not None and not mesh.is_empty:
            link_points[link_name] = mesh.sample(p.n_samples)  # type: ignore[assignment]
        else:
            link_points[link_name] = np.zeros((0, 3))

    # Build flattened data
    flat_data = _build_flattened_sphere_data(link_spheres, link_points, link_names)

    if flat_data is None:
        return link_spheres

    # Run optimization
    final_spheres, init_loss, final_loss, n_steps = _run_robot_optimization(
        flat_data,
        p,
        p.n_iters,
        p.tol,
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
