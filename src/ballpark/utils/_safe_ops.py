from typing import Tuple
from jaxtyping import Float, Array

import jax
import jax.numpy as jnp


def normalize_with_norm(
    x: Float[Array, "*batch N"],
) -> Tuple[Float[Array, "*batch N"], Float[Array, "*batch"]]:
    """Normalizes a vector and returns the norm, handling the zero vector."""
    norm: jax.Array = jnp.linalg.norm(x + 1e-6, axis=-1, keepdims=True)
    safe_norm = jnp.where(norm == 0.0, 1.0, norm)
    normalized_x = x / safe_norm
    result_vec = jnp.where(norm == 0.0, jnp.zeros_like(x), normalized_x)
    result_norm = norm[..., 0]
    return result_vec, result_norm
