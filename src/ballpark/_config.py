"""Configuration presets for Ballpark sphere decomposition.

This module provides dataclass-based configuration for the three stages of
sphere decomposition:
1. Adaptive tight fitting (initial sphere generation)
2. NLLS refinement (single-link optimization)
3. Robot-level refinement (self-collision aware optimization)

Three presets are available:
- conservative: Safety-first collision avoidance
- balanced: General purpose, current defaults
- surface: Tight surface matching for contact-rich tasks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any


class _UnsetType:
    """Sentinel type for unset parameters."""

    _instance: "_UnsetType | None" = None

    def __new__(cls) -> "_UnsetType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET: _UnsetType = _UnsetType()
"""Sentinel value indicating a parameter was not explicitly provided."""


@dataclass
class AdaptiveTightConfig:
    """Configuration for adaptive tight sphere fitting (stage 1).

    Controls the recursive splitting algorithm that generates initial spheres.

    Attributes:
        target_tightness: Max sphere_vol/hull_vol ratio before splitting.
            Lower values produce more splits and tighter fits.
        aspect_threshold: Max aspect ratio before splitting.
            Lower values split elongated regions more aggressively.
        padding: Radius safety margin multiplier (e.g., 1.02 = 2% larger).
        percentile: Use percentile distances for outlier robustness.
            100.0 uses max distance (most conservative).
        max_radius_ratio: Cap radius at this fraction of bbox diagonal.
        uniform_radius: Post-process to equalize radii (may under-approximate).
    """

    target_tightness: float = 1.2
    aspect_threshold: float = 1.3
    padding: float = 1.02
    percentile: float = 98.0
    max_radius_ratio: float = 0.15
    uniform_radius: bool = False


@dataclass
class RefinementConfig:
    """Configuration for NLLS refinement (stage 2).

    Controls the gradient-based optimization of sphere parameters.

    Attributes:
        n_iters: Maximum number of optimization iterations.
        lr: Base learning rate for Adam optimizer.
        lr_center: Learning rate for sphere centers. If None, uses lr.
        lr_radius: Learning rate for sphere radii. If None, uses lr * 0.1.
        lambda_under: Weight for under-approximation loss (points outside spheres).
        lambda_over: Weight for over-approximation loss (sphere volume).
        lambda_overlap: Weight for overlap penalty.
        lambda_uniform: Weight for radius uniformity (variance penalty).
        lambda_surface: Weight for surface matching loss.
        lambda_sqem: Weight for SQEM loss (signed distance along surface normals).
        min_radius: Minimum allowed radius (for numerical stability).
        tol: Relative convergence tolerance for early stopping (e.g., 1e-4 = 0.01% change).
    """

    n_iters: int = 500
    lr: float = 1e-3
    lr_center: float | None = None
    lr_radius: float | None = None
    lambda_under: float = 1.0
    lambda_over: float = 0.01
    lambda_overlap: float = 0.1
    lambda_uniform: float = 0.0
    lambda_surface: float = 0.0
    lambda_sqem: float = 0.0
    min_radius: float = 1e-4
    tol: float = 1e-4


@dataclass
class RobotRefinementConfig:
    """Configuration for robot-level self-collision refinement (stage 3).

    Controls the joint optimization across all robot links.

    Attributes:
        enabled: Whether to perform robot-level refinement.
        n_iters: Maximum number of optimization iterations.
        lr: Learning rate for Adam optimizer.
        lambda_self_collision: Weight for self-collision avoidance penalty.
        lambda_center_reg: Regularization to prevent center drift.
        lambda_overlap: Weight for overlap penalty within each link.
        lambda_uniform: Weight for radius uniformity.
        mesh_collision_tolerance: Skip pairs with inherent mesh proximity.
        lambda_similarity: Weight for similarity position correspondence loss.
            Encourages similar links (same mesh) to have consistent sphere layouts.
    """

    enabled: bool = True
    n_iters: int = 500
    lr: float = 1e-3
    lambda_self_collision: float = 10.0
    lambda_center_reg: float = 1.0
    lambda_overlap: float = 0.1
    lambda_uniform: float = 0.0
    mesh_collision_tolerance: float = 0.005
    lambda_similarity: float = 1.0


@dataclass
class BallparkConfig:
    """Top-level configuration combining all stages.

    Attributes:
        adaptive_tight: Configuration for stage 1 (initial sphere generation).
        refinement: Configuration for stage 2 (single-link NLLS optimization).
        robot_refinement: Configuration for stage 3 (robot-level refinement).
    """

    adaptive_tight: AdaptiveTightConfig = field(default_factory=AdaptiveTightConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    robot_refinement: RobotRefinementConfig = field(default_factory=RobotRefinementConfig)


# Preset configurations
PRESET_CONFIGS: dict[str, BallparkConfig] = {
    "conservative": BallparkConfig(
        # Conservative: Safety-first collision avoidance
        # Best for: Industrial bin picking, high-speed manipulation, safety-critical applications
        adaptive_tight=AdaptiveTightConfig(
            target_tightness=0.8,  # More aggressive splitting
            aspect_threshold=1.1,  # Split elongated regions more
            padding=1.05,  # 5% safety margin
            percentile=100.0,  # Use max distance (no outlier exclusion)
            max_radius_ratio=0.20,  # Allow larger individual spheres
            uniform_radius=False,
        ),
        refinement=RefinementConfig(
            n_iters=200,
            lr=1e-3,
            lambda_under=5.0,  # Strong coverage penalty
            lambda_over=0.005,  # Weak volume penalty
            lambda_overlap=0.01,  # Allow more overlap (larger radii)
            lambda_uniform=0.0,
        ),
        robot_refinement=RobotRefinementConfig(
            enabled=True,
            n_iters=500,
            lambda_self_collision=50.0,  # Very strong self-collision avoidance
            lambda_center_reg=5.0,  # Keep spheres from drifting
        ),
    ),
    "balanced": BallparkConfig(
        # Balanced: General purpose, current defaults
        # Best for: Most applications, balanced tradeoffs
        adaptive_tight=AdaptiveTightConfig(
            target_tightness=1.2,
            aspect_threshold=1.3,
            padding=1.02,
            percentile=98.0,
            max_radius_ratio=0.15,
            uniform_radius=False,
        ),
        refinement=RefinementConfig(
            n_iters=500,
            lr=1e-3,
            lambda_under=1.0,
            lambda_over=0.01,
            lambda_overlap=0.1,
            lambda_uniform=0.0,
        ),
        robot_refinement=RobotRefinementConfig(
            enabled=True,
            n_iters=500,
            lambda_self_collision=10.0,
            lambda_center_reg=1.0,
        ),
    ),
    "surface": BallparkConfig(
        # Surface: Tight surface matching for contact-rich tasks
        # Best for: Grasp planning, shape modeling, precision manipulation
        adaptive_tight=AdaptiveTightConfig(
            target_tightness=1.5,  # Less splitting, more refined later
            aspect_threshold=1.5,  # Allow more elongation
            padding=1.01,  # Minimal 1% margin
            percentile=95.0,  # Exclude outliers for shape accuracy
            max_radius_ratio=0.10,  # Smaller max radius for detail
            uniform_radius=False,
        ),
        refinement=RefinementConfig(
            n_iters=500,
            lr=1e-3,
            lambda_under=0.5,  # Looser coverage constraint
            lambda_over=0.1,  # Stronger volume penalty (tighter fit)
            lambda_overlap=0.05,  # Allow more overlap for tight packing
            lambda_uniform=0.5,  # Encourage similar-sized spheres
            lambda_sqem=0.1,  # Surface normal awareness for contact quality
        ),
        robot_refinement=RobotRefinementConfig(
            enabled=False,  # Skip robot-level refinement
        ),
    ),
}


def get_config(preset: str = "balanced") -> BallparkConfig:
    """Get configuration with specified preset.

    Args:
        preset: One of:
            - "conservative": Safety-first collision avoidance.
                Tighter fits, larger safety margins, strong self-collision avoidance.
                Best for: Industrial bin picking, high-speed manipulation.
            - "balanced": General-purpose default.
                Balanced tradeoffs across all objectives.
                Best for: Most applications.
            - "surface": Tight surface matching.
                Minimal padding, uniform spheres, optimized for contact.
                Best for: Grasp planning, shape modeling, precision manipulation.

    Returns:
        BallparkConfig instance with preset parameters (deep copy).

    Raises:
        ValueError: If preset is not recognized.

    Example:
        >>> config = get_config("conservative")
        >>> config.adaptive_tight.padding
        1.05
    """
    if preset not in PRESET_CONFIGS:
        available = ", ".join(f'"{k}"' for k in PRESET_CONFIGS.keys())
        raise ValueError(f'Unknown preset: "{preset}". Available: {available}')

    return deepcopy(PRESET_CONFIGS[preset])


def update_config_from_dict(
    config: BallparkConfig, updates: dict[str, Any]
) -> BallparkConfig:
    """Update configuration from a dictionary with nested keys.

    Supports dot notation for nested updates (e.g., "adaptive_tight.padding").

    Args:
        config: Base configuration to update.
        updates: Dictionary of updates. Keys use dot notation for nested fields.

    Returns:
        New BallparkConfig with updates applied (original unchanged).

    Raises:
        ValueError: If a key path is invalid or field doesn't exist.

    Example:
        >>> config = get_config("ballpark-b")
        >>> config = update_config_from_dict(config, {
        ...     "adaptive_tight.padding": 1.05,
        ...     "refinement.lambda_under": 2.0,
        ... })
    """
    config = deepcopy(config)

    for key, value in updates.items():
        if "." not in key:
            raise ValueError(
                f'Invalid key format: "{key}". Use "section.param" format '
                f'(e.g., "adaptive_tight.padding").'
            )

        section, param = key.split(".", 1)

        if not hasattr(config, section):
            raise ValueError(
                f'Unknown section: "{section}". '
                f"Available: adaptive_tight, refinement, robot_refinement"
            )

        section_config = getattr(config, section)

        # Handle nested params (e.g., "robot_refinement.enabled")
        if "." in param:
            raise ValueError(
                f'Nested params beyond two levels not supported: "{key}"'
            )

        if not hasattr(section_config, param):
            available = [f for f in dir(section_config) if not f.startswith("_")]
            raise ValueError(
                f'Unknown parameter: "{param}" in section "{section}". '
                f"Available: {', '.join(available)}"
            )

        setattr(section_config, param, value)

    return config


# Mapping from function parameter names to config paths
_PARAM_MAP: dict[str, tuple[str, str]] = {
    # AdaptiveTightConfig
    "target_tightness": ("adaptive_tight", "target_tightness"),
    "aspect_threshold": ("adaptive_tight", "aspect_threshold"),
    "padding": ("adaptive_tight", "padding"),
    "percentile": ("adaptive_tight", "percentile"),
    "max_radius_ratio": ("adaptive_tight", "max_radius_ratio"),
    "uniform_radius": ("adaptive_tight", "uniform_radius"),
    # RefinementConfig
    "refine_iters": ("refinement", "n_iters"),
    "refine_lr": ("refinement", "lr"),
    "refine_lr_center": ("refinement", "lr_center"),
    "refine_lr_radius": ("refinement", "lr_radius"),
    "lambda_under": ("refinement", "lambda_under"),
    "lambda_over": ("refinement", "lambda_over"),
    "lambda_overlap": ("refinement", "lambda_overlap"),
    "lambda_uniform": ("refinement", "lambda_uniform"),
    "lambda_surface": ("refinement", "lambda_surface"),
    "lambda_sqem": ("refinement", "lambda_sqem"),
    # RobotRefinementConfig
    "refine_self_collision": ("robot_refinement", "enabled"),
    "lambda_self_collision": ("robot_refinement", "lambda_self_collision"),
    "lambda_center_reg": ("robot_refinement", "lambda_center_reg"),
    "mesh_collision_tolerance": ("robot_refinement", "mesh_collision_tolerance"),
    "lambda_similarity": ("robot_refinement", "lambda_similarity"),
}


def resolve_params(
    config: BallparkConfig | None = None,
    preset: str | None = None,
    **explicit_params: Any,
) -> dict[str, Any]:
    """Resolve parameters from config/preset with explicit overrides.

    Resolution order (highest to lowest priority):
    1. Explicit params (if not UNSET)
    2. Provided config object
    3. Preset config (loaded from preset string)
    4. Default BallparkConfig values

    Args:
        config: Optional BallparkConfig object.
        preset: Optional preset name ("conservative", "balanced", "surface").
        **explicit_params: Individual parameters that override config values.
            Values equal to UNSET are treated as "not provided".

    Returns:
        Dict of resolved parameter values.

    Raises:
        ValueError: If both config and preset are provided.

    Example:
        >>> params = resolve_params(
        ...     preset="conservative",
        ...     padding=UNSET,  # Will use preset value (1.05)
        ...     lambda_under=5.0,  # Explicit override
        ... )
        >>> params["padding"]
        1.05
        >>> params["lambda_under"]
        5.0
    """
    if config is not None and preset is not None:
        raise ValueError("Cannot specify both 'config' and 'preset'. Choose one.")

    # Build base config
    if config is not None:
        base_config = config
    elif preset is not None:
        base_config = get_config(preset)
    else:
        base_config = BallparkConfig()

    # Resolve each parameter
    result: dict[str, Any] = {}
    for param_name, param_value in explicit_params.items():
        if param_value is not UNSET:
            # Explicit value provided - use it
            result[param_name] = param_value
        elif param_name in _PARAM_MAP:
            # Get from config
            section, attr = _PARAM_MAP[param_name]
            section_config = getattr(base_config, section)
            result[param_name] = getattr(section_config, attr)
        else:
            # Unknown param - this shouldn't happen, but pass through None
            result[param_name] = None

    return result
