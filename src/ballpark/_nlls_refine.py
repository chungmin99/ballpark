"""NLLS sphere refinement with JAX/optax."""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax

from ._sphere import Sphere


def _compute_surface_loss(
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    surface_points: jnp.ndarray,
) -> jnp.ndarray:
    """
    Surface matching loss: minimize absolute distance from surface samples to sphere surfaces.

    For each surface point, find the closest sphere and measure how far the point is from
    that sphere's surface. Positive = outside, negative = inside.

    Args:
        centers: (N, 3) sphere centers
        radii: (N,) sphere radii
        surface_points: (S, 3) points sampled on mesh surface

    Returns:
        Scalar loss value (mean of absolute signed distances)
    """
    # Distance from each surface point to all sphere centers
    diff = surface_points[:, None, :] - centers[None, :, :]  # (S, N, 3)
    dists_to_centers = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)  # (S, N)

    # Signed distance: positive = outside sphere, negative = inside
    signed_dists = dists_to_centers - radii[None, :]  # (S, N)

    # For each point, find minimum signed distance (closest sphere)
    min_signed_dist = jnp.min(signed_dists, axis=1)  # (S,)

    # Loss: mean of absolute distances
    return jnp.mean(jnp.abs(min_signed_dist))


def _compute_sqem_loss(
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    surface_points: jnp.ndarray,
    surface_normals: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute SQEM (Squared Error with Normal Projection) loss.

    Measures geometric fit quality using surface normals. For each surface point,
    computes the signed perpendicular distance to the closest sphere along the
    normal direction.

    Args:
        centers: (N, 3) sphere centers
        radii: (N,) sphere radii
        surface_points: (P, 3) points on mesh surface
        surface_normals: (P, 3) unit normals at those points

    Returns:
        Scalar loss: mean(signed_distance^2) to closest sphere
    """
    # Direction vectors from surface samples to sphere centers
    diff_vec = surface_points[:, None, :] - centers[None, :, :]  # (P, N, 3)

    # Unsigned Euclidean distances (for finding closest sphere)
    dists_to_centers = jnp.sqrt(jnp.sum(diff_vec**2, axis=-1) + 1e-8)  # (P, N)

    # Signed distances using normal projection
    # dot(diff_vec, normal) gives distance along normal direction
    signed_dist = jnp.sum(
        diff_vec * surface_normals[:, None, :], axis=-1
    ) - radii[None, :]  # (P, N)

    # Find closest sphere for each surface point (using unsigned distance)
    closest_sphere_idx = jnp.argmin(dists_to_centers, axis=1)  # (P,)

    # Extract signed distance to closest sphere
    closest_signed_dist = jnp.take_along_axis(
        signed_dist, closest_sphere_idx[:, None], axis=1
    ).squeeze(1)  # (P,)

    # SQEM loss: mean squared signed distance
    return jnp.mean(closest_signed_dist**2)


def _compute_loss(
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    points: jnp.ndarray,
    scale: float,
    min_radius: float,
    lambda_under: float,
    lambda_over: float,
    lambda_overlap: float,
    lambda_uniform: float,
    lambda_surface: float = 0.0,
    lambda_sqem: float = 0.0,
    surface_points: jnp.ndarray | None = None,
    surface_normals: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Compute total loss for sphere refinement.

    Args:
        centers: (N, 3) sphere centers
        radii: (N,) sphere radii
        points: (P, 3) points to cover
        scale: Normalization scale factor
        min_radius: Minimum allowed radius
        lambda_under: Weight for under-approximation loss
        lambda_over: Weight for over-approximation loss
        lambda_overlap: Weight for overlap penalty
        lambda_uniform: Weight for radius uniformity
        lambda_surface: Weight for surface matching loss
        lambda_sqem: Weight for SQEM loss (surface signed error with normal projection)
        surface_points: (S, 3) points sampled on mesh surface, optional
        surface_normals: (S, 3) unit normals at surface points, optional

    Returns:
        Scalar loss value
    """
    # Ensure positive radii
    radii = jnp.maximum(radii, min_radius)
    n_spheres = radii.shape[0]

    total_loss = jnp.array(0.0)

    # 1. Under-approximation: points outside all spheres
    diff = points[:, None, :] - centers[None, :, :]  # (P, N, 3)
    dists_to_centers = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)  # (P, N)
    signed_dists = dists_to_centers - radii[None, :]  # (P, N)
    min_signed_dist = jnp.min(signed_dists, axis=1)  # (P,)
    under_approx = jnp.mean(jnp.maximum(0.0, min_signed_dist) ** 2)
    total_loss = total_loss + lambda_under * under_approx

    # 2. Over-approximation: minimize total sphere volume
    over_approx = jnp.mean((radii / scale) ** 3)
    total_loss = total_loss + lambda_over * over_approx

    # 3. Overlap penalty
    center_diff = centers[:, None, :] - centers[None, :, :]  # (N, N, 3)
    center_dists = jnp.sqrt(jnp.sum(center_diff**2, axis=-1) + 1e-8)  # (N, N)
    sum_radii = radii[:, None] + radii[None, :]  # (N, N)
    overlap_depth = jnp.maximum(0.0, sum_radii - center_dists)  # (N, N)
    mask = jnp.triu(jnp.ones((n_spheres, n_spheres)), k=1)
    overlap_loss = jnp.sum(mask * overlap_depth**2) / (n_spheres + 1e-8)
    total_loss = total_loss + lambda_overlap * overlap_loss

    # 4. Uniformity
    mean_radius = jnp.mean(radii)
    uniform_loss = jnp.var(radii) / (mean_radius**2 + 1e-8)
    total_loss = total_loss + lambda_uniform * uniform_loss

    # 5. Surface loss (boundary matching)
    # Note: We always compute and add these losses, but with lambda=0 they contribute nothing.
    # This avoids JAX tracing issues with conditional branches.
    if surface_points is not None:
        surface_loss = _compute_surface_loss(centers, radii, surface_points)
        total_loss = total_loss + lambda_surface * surface_loss

    # 6. SQEM loss (surface signed error with normal projection)
    if surface_points is not None and surface_normals is not None:
        sqem_loss = _compute_sqem_loss(centers, radii, surface_points, surface_normals)
        total_loss = total_loss + lambda_sqem * sqem_loss

    return total_loss


@partial(jax.jit, static_argnames=["n_iters"])
def _run_optimization(
    centers: jnp.ndarray,
    radii: jnp.ndarray,
    points: jnp.ndarray,
    scale: float,
    min_radius: float,
    lambda_under: float,
    lambda_over: float,
    lambda_overlap: float,
    lambda_uniform: float,
    lambda_surface: float,
    lambda_sqem: float,
    surface_points: jnp.ndarray | None,
    surface_normals: jnp.ndarray | None,
    lr_center: float,
    lr_radius: float,
    n_iters: int,
    tol: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run the full optimization loop (JIT-compiled).

    Args:
        centers: Initial sphere centers (N, 3)
        radii: Initial sphere radii (N,)
        points: Point cloud to cover (P, 3)
        scale: Normalization scale (bbox diagonal)
        min_radius: Minimum allowed radius
        lambda_under: Weight for under-approximation loss
        lambda_over: Weight for over-approximation loss
        lambda_overlap: Weight for overlap penalty
        lambda_uniform: Weight for uniformity penalty
        lambda_surface: Weight for surface matching loss
        lambda_sqem: Weight for SQEM loss
        surface_points: Surface points for surface/SQEM loss (S, 3), optional
        surface_normals: Surface normals for SQEM loss (S, 3), optional
        lr_center: Learning rate for sphere centers
        lr_radius: Learning rate for sphere radii
        n_iters: Maximum number of iterations
        tol: Relative convergence tolerance for early stopping (e.g., 1e-4 = 0.01% change)

    Returns:
        (final_centers, final_radii, final_loss, n_steps)
    """
    # Create separate optimizers for centers and radii
    optimizer_center = optax.adam(lr_center)
    optimizer_radius = optax.adam(lr_radius)
    opt_state_center = optimizer_center.init(centers)
    opt_state_radius = optimizer_radius.init(radii)

    # Initial loss
    init_loss = _compute_loss(
        centers, radii, points, scale, min_radius,
        lambda_under, lambda_over, lambda_overlap, lambda_uniform,
        lambda_surface, lambda_sqem, surface_points, surface_normals
    )

    def body_fn(state):
        """Single optimization step."""
        centers, radii, opt_state_c, opt_state_r, prev_loss, curr_loss, i = state

        def loss_fn(params):
            c, r = params
            return _compute_loss(
                c, r, points, scale, min_radius,
                lambda_under, lambda_over, lambda_overlap, lambda_uniform,
                lambda_surface, lambda_sqem, surface_points, surface_normals
            )

        # Compute gradients for the tuple
        loss, grads = jax.value_and_grad(loss_fn)((centers, radii))
        grad_c, grad_r = grads

        # Update centers with center optimizer
        updates_c, new_opt_state_c = optimizer_center.update(grad_c, opt_state_c, centers)
        new_centers = optax.apply_updates(centers, updates_c)

        # Update radii with radius optimizer
        updates_r, new_opt_state_r = optimizer_radius.update(grad_r, opt_state_r, radii)
        new_radii = optax.apply_updates(radii, updates_r)
        new_radii = jnp.maximum(new_radii, min_radius)

        return (new_centers, new_radii, new_opt_state_c, new_opt_state_r, curr_loss, loss, i + 1)

    def cond_fn(state):
        """Continue while not converged and under max iterations."""
        centers, radii, opt_state_c, opt_state_r, prev_loss, curr_loss, i = state
        # Use relative tolerance to handle varying loss magnitudes
        rel_change = jnp.abs(prev_loss - curr_loss) / (jnp.abs(prev_loss) + 1e-8)
        not_converged = rel_change > tol
        not_max_iters = i < n_iters
        return jnp.logical_and(not_converged, not_max_iters)

    # Run optimization loop with early stopping
    init_state = (centers, radii, opt_state_center, opt_state_radius, jnp.inf, init_loss, 0)
    final_centers, final_radii, _, _, _, final_loss, n_steps = lax.while_loop(
        cond_fn, body_fn, init_state
    )

    return final_centers, final_radii, final_loss, n_steps


def refine_spheres_nlls(
    spheres: list[Sphere],
    points: np.ndarray,
    n_iters: int = 100,
    lr: float = 1e-3,
    lr_center: float | None = None,
    lr_radius: float | None = None,
    lambda_under: float = 1.0,
    lambda_over: float = 0.01,
    lambda_overlap: float = 0.1,
    lambda_uniform: float = 0.0,
    lambda_surface: float = 0.0,
    lambda_sqem: float = 0.0,
    surface_points: np.ndarray | None = None,
    surface_normals: np.ndarray | None = None,
    min_radius: float = 1e-4,
    tol: float = 1e-4,
) -> list[Sphere]:
    """
    Refine sphere parameters using gradient descent with JAX/optax.

    Jointly optimizes all sphere centers and radii to minimize:
    1. Under-approximation: points outside all spheres
    2. Over-approximation: sphere volume (radius^3 proxy)
    3. Overlap penalty: excessive sphere intersection
    4. Uniformity: variance of radii (encourages similar-sized spheres)
    5. Surface matching: absolute distance from surface samples to sphere surfaces
    6. SQEM: surface signed error with normal projection (geometric accuracy)

    Args:
        spheres: Initial sphere configuration from adaptive_tight
        points: (N, 3) surface point cloud to cover
        n_iters: Maximum number of optimization iterations
        lr: Learning rate for Adam optimizer (used as default for both center and radius)
        lr_center: Learning rate for sphere centers. If None, uses lr.
        lr_radius: Learning rate for sphere radii. If None, uses lr * 0.1.
        lambda_under: Weight for under-approximation loss (points outside spheres)
        lambda_over: Weight for over-approximation loss (sphere volume)
        lambda_overlap: Weight for overlap penalty
        lambda_uniform: Weight for radius uniformity (variance penalty)
        lambda_surface: Weight for surface matching loss (boundary approximation)
        lambda_sqem: Weight for SQEM loss (surface signed error with normal projection)
        surface_points: (S, 3) points sampled on mesh surface for surface/SQEM loss
        surface_normals: (S, 3) unit normals at surface points for SQEM loss
        min_radius: Minimum allowed radius (for numerical stability)
        tol: Relative convergence tolerance for early stopping (e.g., 1e-4 = 0.01% change)

    Returns:
        Refined list of Sphere objects
    """
    if len(spheres) == 0 or len(points) == 0:
        return spheres

    # Set default learning rates (MorphIt-inspired 10:1 ratio)
    if lr_center is None:
        lr_center = lr
    if lr_radius is None:
        lr_radius = lr * 0.1

    # Convert to JAX arrays
    centers = jnp.array([s.center for s in spheres])  # (N, 3)
    radii = jnp.array([s.radius for s in spheres])  # (N,)
    points_jax = jnp.array(points)  # (P, 3)

    # Convert surface data if provided
    surface_points_jax = None
    surface_normals_jax = None
    if surface_points is not None:
        surface_points_jax = jnp.array(surface_points)
    if surface_normals is not None:
        surface_normals_jax = jnp.array(surface_normals)

    # Compute scale for normalization
    bbox_diag = jnp.linalg.norm(points_jax.max(axis=0) - points_jax.min(axis=0))
    scale = float(bbox_diag + 1e-8)

    # Run JIT-compiled optimization
    final_centers, final_radii, final_loss, n_steps = _run_optimization(
        centers, radii, points_jax, scale, min_radius,
        lambda_under, lambda_over, lambda_overlap, lambda_uniform,
        lambda_surface, lambda_sqem, surface_points_jax, surface_normals_jax,
        lr_center, lr_radius, n_iters, tol
    )

    # Convert back to Sphere objects
    centers_np = np.array(final_centers)
    radii_np = np.array(final_radii)

    refined_spheres = [
        Sphere(center=centers_np[i], radius=float(radii_np[i]))
        for i in range(len(spheres))
    ]

    return refined_spheres
