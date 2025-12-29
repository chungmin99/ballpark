"""Configuration for sphere decomposition."""

from __future__ import annotations

from enum import Enum

import jax_dataclasses as jdc


class SpherePreset(Enum):
    """Preset configurations for sphere decomposition.

    SURFACE: Tight fit to mesh surface. Best for visualization and
             precise collision bounds. May under-approximate concavities.

    BALANCED: Default. Balanced between coverage and efficiency.
              Good general-purpose setting for most robots.

    CONSERVATIVE: Over-approximates to ensure full coverage.
                  Larger spheres, safer for collision checking but
                  less precise. Good when false negatives are costly.
    """

    SURFACE = "surface"
    BALANCED = "balanced"
    CONSERVATIVE = "conservative"


@jdc.pytree_dataclass
class SpherizeParams:
    """Parameters for the adaptive splitting algorithm."""

    target_tightness: float = 1.2
    """Max acceptable sphere_vol/hull_vol ratio before splitting."""

    aspect_threshold: float = 1.3
    """Max acceptable aspect ratio before splitting."""

    n_samples: int = 5000
    """Number of surface samples to use."""

    padding: float = 1.02
    """Radius multiplier for safety margin (1.02 = 2% larger)."""

    percentile: float = 98.0
    """Percentile of distances to use for radius (handles outliers)."""

    max_radius_ratio: float = 0.5
    """Cap radius relative to bounding box diagonal."""

    uniform_radius: bool = False
    """If True, post-process to make radii more uniform."""


@jdc.pytree_dataclass
class RefineParams:
    """Parameters for gradient-based refinement."""

    # Optimization params
    n_iters: int = 100
    """Maximum number of optimization iterations."""

    lr: float = 1e-3
    """Learning rate for Adam optimizer."""

    tol: float = 1e-4
    """Relative convergence tolerance for early stopping."""

    min_radius: float = 1e-4
    """Minimum allowed sphere radius."""

    n_samples: int = 5000
    """Points to sample per link for loss computation."""

    # Per-link loss weights
    lambda_under: float = 1.0
    """Weight for under-approximation loss (points outside spheres)."""

    lambda_over: float = 0.01
    """Weight for over-approximation loss (sphere volume)."""

    lambda_overlap: float = 0.1
    """Weight for intra-link sphere overlap penalty."""

    lambda_uniform: float = 0.0
    """Weight for radius uniformity within links."""

    lambda_surface: float = 0.0
    """Weight for surface matching loss."""

    lambda_sqem: float = 0.0
    """Weight for SQEM loss (signed error with normals)."""

    # Robot-level loss weights
    lambda_self_collision: float = 1.0
    """Weight for inter-link self-collision penalty."""

    lambda_center_reg: float = 1.0
    """Weight for center drift regularization."""

    lambda_similarity: float = 1.0
    """Weight for similar link correspondence."""

    mesh_collision_tolerance: float = 0.01
    """Skip link pairs with mesh distance below this."""


@jdc.pytree_dataclass
class BallparkConfig:
    """Unified configuration for sphere decomposition.

    Combines spherize and refine parameters with optional preset support.

    Usage:
        # Use a preset
        config = BallparkConfig.from_preset(SpherePreset.CONSERVATIVE)

        # Customize from preset
        config = BallparkConfig.from_preset(SpherePreset.BALANCED)
        config = jdc.replace(config, spherize=jdc.replace(config.spherize, padding=1.03))

        # Fully custom
        config = BallparkConfig(
            spherize=SpherizeParams(target_tightness=1.1),
            refine=RefineParams(n_iters=200),
        )
    """

    spherize: SpherizeParams = jdc.field(default_factory=SpherizeParams)
    """Parameters for adaptive splitting algorithm."""

    refine: RefineParams = jdc.field(default_factory=RefineParams)
    """Parameters for gradient-based refinement."""

    @classmethod
    def from_preset(cls, preset: SpherePreset) -> "BallparkConfig":
        """Create config from a preset.

        Args:
            preset: Base preset to use

        Returns:
            BallparkConfig with preset values
        """
        return jdc.replace(_PRESET_CONFIGS[preset])


# Preset definitions
_PRESET_CONFIGS: dict[SpherePreset, BallparkConfig] = {
    SpherePreset.SURFACE: BallparkConfig(
        spherize=SpherizeParams(
            target_tightness=1.1,  # Tighter splits
            aspect_threshold=1.2,  # Split elongated shapes more aggressively
            padding=1.01,  # Minimal padding
            percentile=99.0,  # Use more of the points
            max_radius_ratio=0.4,  # Smaller max spheres
            uniform_radius=False,
        ),
        refine=RefineParams(
            lambda_under=2.0,  # Prioritize coverage
            lambda_over=0.02,  # Less volume penalty
            lambda_overlap=0.05,  # Allow more overlap for tightness
            lambda_uniform=0.0,
            n_iters=150,  # More iterations for precision
        ),
    ),
    SpherePreset.BALANCED: BallparkConfig(
        spherize=SpherizeParams(
            target_tightness=1.2,
            aspect_threshold=1.3,
            padding=1.02,
            percentile=98.0,
            max_radius_ratio=0.5,
            uniform_radius=False,
        ),
        refine=RefineParams(
            lambda_under=1.0,
            lambda_over=0.01,
            lambda_overlap=0.1,
            lambda_uniform=0.0,
            n_iters=100,
        ),
    ),
    SpherePreset.CONSERVATIVE: BallparkConfig(
        spherize=SpherizeParams(
            target_tightness=1.4,  # Looser splits - fewer, larger spheres
            aspect_threshold=1.5,  # More tolerant of elongation
            padding=1.05,  # More padding for safety
            percentile=95.0,  # Ignore more outliers
            max_radius_ratio=0.6,  # Allow larger spheres
            uniform_radius=True,  # More uniform sizes
        ),
        refine=RefineParams(
            lambda_under=0.5,  # Less strict on coverage
            lambda_over=0.005,  # Allow larger volumes
            lambda_overlap=0.2,  # Discourage overlap more
            lambda_uniform=0.1,  # Encourage uniform radii
            n_iters=80,  # Fewer iterations needed
        ),
    ),
}
