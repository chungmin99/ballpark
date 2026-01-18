"""Ellipsoid primitives for collision approximation.

Adopts the effective radius approach from pyroki for smooth, differentiable
collision detection suitable for optimization-based refinement.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie

from ._spherize import Sphere

_SAFE_EPS = 1e-6


# =============================================================================
# Ellipsoid Dataclasses
# =============================================================================


@jdc.pytree_dataclass
class Ellipsoid:
    """Axis-aligned ellipsoid defined by center and semi-axes.

    Can represent a single ellipsoid (center: (3,), semi_axes: (3,)) or
    a batch of ellipsoids (center: (N, 3), semi_axes: (N, 3)).

    The semi-axes (a, b, c) represent half-lengths along the x, y, z axes.
    """

    center: jnp.ndarray
    semi_axes: jnp.ndarray


# =============================================================================
# Utility Functions
# =============================================================================


def _normalize_with_norm(
    x: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Normalize a vector and return (normalized_vector, norm).

    Handles zero vectors gracefully by returning zero vector with zero norm.
    """
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    safe_norm = jnp.where(norm < _SAFE_EPS, 1.0, norm)
    normalized = x / safe_norm
    result_vec = jnp.where(norm < _SAFE_EPS, jnp.zeros_like(x), normalized)
    result_norm = norm[..., 0] if norm.ndim > 0 else norm
    return result_vec, jnp.squeeze(result_norm)


# =============================================================================
# Effective Radius
# =============================================================================


def ellipsoid_effective_radius(
    semi_axes: jax.Array,
    direction: jax.Array,
) -> jax.Array:
    """Compute effective radius of ellipsoid along a direction.

    For an ellipsoid with semi-axes (a, b, c), the effective radius
    along a unit direction d = (dx, dy, dz) is:
        r_eff = 1 / sqrt((dx/a)^2 + (dy/b)^2 + (dz/c)^2)

    This is the distance from center to surface along that direction.

    Args:
        semi_axes: Ellipsoid semi-axes (a, b, c), shape (..., 3).
        direction: Direction vector, shape (..., 3). Will be normalized.

    Returns:
        Effective radius along the direction, shape (...).
    """
    d_normalized, norm = _normalize_with_norm(direction)

    # Handle zero direction case - return geometric mean
    is_zero = norm < _SAFE_EPS
    d_safe = jnp.where(is_zero[..., None], jnp.ones_like(d_normalized), d_normalized)

    # Scale direction by inverse semi-axes
    scaled_dir = d_safe / (semi_axes + _SAFE_EPS)
    inv_r_sq = jnp.sum(scaled_dir**2, axis=-1)
    r_eff = 1.0 / jnp.sqrt(inv_r_sq + _SAFE_EPS)

    # For zero direction, return geometric mean of semi-axes
    mean_radius = jnp.cbrt(semi_axes[..., 0] * semi_axes[..., 1] * semi_axes[..., 2])
    return jnp.where(is_zero, mean_radius, r_eff)


# =============================================================================
# Distance Functions
# =============================================================================


def ellipsoid_point_distance(
    center: jax.Array,
    semi_axes: jax.Array,
    point: jax.Array,
    rotation: jaxlie.SO3 | None = None,
) -> jax.Array:
    """Compute signed distance from point to ellipsoid surface.

    Uses effective radius approximation: distance from point to center
    minus the effective radius in that direction.

    Args:
        center: Ellipsoid center, shape (..., 3).
        semi_axes: Ellipsoid semi-axes (a, b, c), shape (..., 3).
        point: Query point, shape (..., 3).
        rotation: Optional SO(3) rotation for rotated ellipsoid.

    Returns:
        Signed distance (negative = inside), shape (...).
    """
    diff = point - center
    _, dist = _normalize_with_norm(diff)

    # Transform direction to local frame if rotated
    if rotation is not None:
        direction_local = rotation.inverse().apply(diff)
    else:
        direction_local = diff

    r_eff = ellipsoid_effective_radius(semi_axes, direction_local)
    return dist - r_eff


def ellipsoid_ellipsoid_distance(
    center1: jax.Array,
    semi_axes1: jax.Array,
    center2: jax.Array,
    semi_axes2: jax.Array,
    rotation1: jaxlie.SO3 | None = None,
    rotation2: jaxlie.SO3 | None = None,
) -> jax.Array:
    """Compute signed distance between two ellipsoids.

    Uses mutual scaled-sphere approximation:
    1. Compute center-to-center direction
    2. Get effective radius of each ellipsoid along this direction
    3. Distance = center_dist - r1_eff - r2_eff

    This is an approximation but provides smooth gradients suitable
    for optimization-based refinement.

    Args:
        center1: First ellipsoid center, shape (..., 3).
        semi_axes1: First ellipsoid semi-axes, shape (..., 3).
        center2: Second ellipsoid center, shape (..., 3).
        semi_axes2: Second ellipsoid semi-axes, shape (..., 3).
        rotation1: Optional SO(3) rotation for first ellipsoid.
        rotation2: Optional SO(3) rotation for second ellipsoid.

    Returns:
        Signed distance (negative = penetrating), shape (...).
    """
    # Direction from ellipsoid 1 to ellipsoid 2
    center_to_center = center2 - center1
    _, dist_centers = _normalize_with_norm(center_to_center)

    # Transform directions to local frames
    if rotation1 is not None:
        dir_local1 = rotation1.inverse().apply(center_to_center)
    else:
        dir_local1 = center_to_center

    if rotation2 is not None:
        dir_local2 = rotation2.inverse().apply(-center_to_center)
    else:
        dir_local2 = -center_to_center

    # Compute effective radii
    r1_eff = ellipsoid_effective_radius(semi_axes1, dir_local1)
    r2_eff = ellipsoid_effective_radius(semi_axes2, dir_local2)

    return dist_centers - r1_eff - r2_eff


# =============================================================================
# Conversion Functions
# =============================================================================


def sphere_to_ellipsoid(sphere: Sphere) -> Ellipsoid:
    """Convert a sphere to an axis-aligned ellipsoid with equal semi-axes."""
    # Handle both single and batched spheres
    if sphere.radius.ndim == 0:
        semi_axes = jnp.broadcast_to(sphere.radius, (3,))
    else:
        semi_axes = jnp.broadcast_to(sphere.radius[..., None], sphere.center.shape)
    return Ellipsoid(center=sphere.center, semi_axes=semi_axes)


def ellipsoid_to_sphere(ellipsoid: Ellipsoid, method: str = "volume") -> Sphere:
    """Convert an ellipsoid to a sphere.

    Args:
        ellipsoid: The ellipsoid to convert.
        method: Conversion method:
            - "volume": Same volume (geometric mean of semi-axes).
            - "bounding": Circumscribed sphere (max semi-axis).
            - "inscribed": Inscribed sphere (min semi-axis).

    Returns:
        Equivalent sphere.
    """
    if method == "volume":
        # Geometric mean preserves volume
        radius = jnp.cbrt(
            ellipsoid.semi_axes[..., 0]
            * ellipsoid.semi_axes[..., 1]
            * ellipsoid.semi_axes[..., 2]
        )
    elif method == "bounding":
        radius = jnp.max(ellipsoid.semi_axes, axis=-1)
    elif method == "inscribed":
        radius = jnp.min(ellipsoid.semi_axes, axis=-1)
    else:
        raise ValueError(f"Unknown method: {method}")

    return Sphere(center=ellipsoid.center, radius=radius)


def ellipsoid_volume(semi_axes: jax.Array) -> jax.Array:
    """Compute ellipsoid volume: (4/3) * pi * a * b * c."""
    return (4.0 / 3.0) * jnp.pi * jnp.prod(semi_axes, axis=-1)


def sphere_volume(radius: jax.Array) -> jax.Array:
    """Compute sphere volume: (4/3) * pi * r^3."""
    return (4.0 / 3.0) * jnp.pi * radius**3
