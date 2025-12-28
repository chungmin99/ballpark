"""Robot collision mesh extraction and sphere visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from loguru import logger

from ._sphere import Sphere
from ._adaptive_tight import spherize_adaptive_tight
from ._robot_refine import (
    refine_spheres_for_robot,
    get_adjacent_links,
)
from ._similarity import SimilarityResult, detect_similar_links
from ._config import BallparkConfig, UNSET, resolve_params, _UnsetType
from ._urdf_utils import (
    get_joint_limits,
    get_link_transforms,
    get_link_names,
    get_num_actuated_joints,
)


@dataclass
class RobotSpheresResult:
    """Result from robot sphere decomposition."""

    link_spheres: dict[str, list[Sphere]]
    ignore_pairs: list[tuple[str, str]] | None  # None if refine_self_collision=False


def get_collision_mesh_for_link(urdf, link_name: str) -> trimesh.Trimesh:
    """
    Extract collision mesh for a given link from URDF.

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        link_name: Name of the link to extract

    Returns:
        Combined collision mesh for the link (empty Trimesh if no collisions)
    """
    if link_name not in urdf.link_map:
        return trimesh.Trimesh()

    link = urdf.link_map[link_name]
    filename_handler = urdf._filename_handler
    coll_meshes = []

    for collision in link.collisions:
        geom = collision.geometry
        mesh = None

        if collision.origin is not None:
            transform = collision.origin
        else:
            transform = np.eye(4)

        if geom.box is not None:
            mesh = trimesh.creation.box(extents=geom.box.size)
        elif geom.cylinder is not None:
            mesh = trimesh.creation.cylinder(
                radius=geom.cylinder.radius, height=geom.cylinder.length
            )
        elif geom.sphere is not None:
            mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
        elif geom.mesh is not None:
            try:
                mesh_path = geom.mesh.filename
                loaded_obj = trimesh.load(
                    file_obj=filename_handler(mesh_path), force="mesh"
                )
                scale = (
                    geom.mesh.scale if geom.mesh.scale is not None else [1.0, 1.0, 1.0]
                )

                if isinstance(loaded_obj, trimesh.Trimesh):
                    mesh = loaded_obj.copy()
                    mesh.apply_scale(scale)
                elif isinstance(loaded_obj, trimesh.Scene):
                    if len(loaded_obj.geometry) > 0:
                        geom_candidate = list(loaded_obj.geometry.values())[0]
                        if isinstance(geom_candidate, trimesh.Trimesh):
                            mesh = geom_candidate.copy()
                            mesh.apply_scale(scale)
            except Exception as e:
                logger.warning(f"Failed loading mesh for {link_name}: {e}")
                continue

        if mesh is not None:
            mesh.apply_transform(transform)
            coll_meshes.append(mesh)

    if not coll_meshes:
        return trimesh.Trimesh()
    return sum(coll_meshes, trimesh.Trimesh())


def allocate_spheres_for_robot(
    urdf,
    target_spheres: int = 100,
    min_spheres_per_link: int = 1,
) -> dict[str, int]:
    """
    Allocate sphere budget across robot links based on geometry complexity.

    Distributes a sphere budget across links proportionally based on
    how poorly each link is approximated by a single bounding sphere.
    The actual total may slightly exceed target_spheres when there are
    more links than the target allows with min_spheres_per_link constraint.

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        target_spheres: Target sphere budget across the entire robot (may slightly exceed)
        min_spheres_per_link: Minimum spheres for any non-empty link

    Returns:
        Dict mapping link names to allocated sphere counts (only non-empty links)
    """
    # Extract meshes and compute allocation weights
    # Weight based on "sphere inefficiency": how much a single bounding sphere wastes
    link_weights = {}
    for link_name in urdf.link_map.keys():
        mesh = get_collision_mesh_for_link(urdf, link_name)
        if not mesh.is_empty:
            # Bounding sphere radius (half of bbox diagonal)
            bbox_diag = np.linalg.norm(mesh.extents)
            bounding_sphere_radius = bbox_diag / 2
            bounding_sphere_vol = 4 / 3 * np.pi * bounding_sphere_radius**3

            # Mesh volume (use convex hull for robustness)
            try:
                mesh_vol = mesh.convex_hull.volume
            except Exception:
                # Convex hull computation can fail for degenerate meshes
                mesh_vol = mesh.volume if mesh.volume > 0 else bounding_sphere_vol

            # Inefficiency = how much the bounding sphere over-approximates
            # Higher inefficiency = needs more spheres to get tight fit
            inefficiency = bounding_sphere_vol / (mesh_vol + 1e-10)

            # Weight = inefficiency (capped to avoid extreme values)
            # Meshes that are poorly approximated by one sphere get more budget
            link_weights[link_name] = min(inefficiency, 20.0)

    # Handle empty robot
    if not link_weights:
        return {}

    # Allocate spheres proportionally to inefficiency
    total_weight = sum(link_weights.values())
    link_budgets = {}
    for link_name, weight in link_weights.items():
        budget = max(min_spheres_per_link, round(target_spheres * weight / total_weight))
        link_budgets[link_name] = budget

    # Adjust if over budget (subtract from largest allocations)
    # Note: if there are more links than target_spheres allows with min_spheres_per_link,
    # the total will exceed target_spheres (this is intentional - we document it as a target)
    while sum(link_budgets.values()) > target_spheres:
        max_link = max(link_budgets, key=link_budgets.get)
        if link_budgets[max_link] > min_spheres_per_link:
            link_budgets[max_link] -= 1
        else:
            break  # Can't reduce further without going below minimum

    return link_budgets


def compute_spheres_for_link(
    urdf,
    link_name: str,
    num_spheres: int,
    padding: float = 1.02,
    target_tightness: float = 1.2,
    aspect_threshold: float = 1.3,
    percentile: float = 98.0,
    max_radius_ratio: float = 0.15,
    uniform_radius: bool = False,
    refine: bool = False,
    refine_iters: int = 500,
    refine_lr: float = 1e-3,
    refine_lr_center: float | None = None,
    refine_lr_radius: float | None = None,
    lambda_under: float = 1.0,
    lambda_over: float = 0.01,
    lambda_overlap: float = 0.1,
    lambda_uniform: float = 0.0,
    lambda_surface: float = 0.0,
    lambda_sqem: float = 0.0,
    mesh: trimesh.Trimesh | None = None,
) -> list[Sphere]:
    """
    Compute sphere decomposition for a single robot link.

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        link_name: Name of the link to process
        num_spheres: Maximum number of spheres to generate for this link
        padding: Radius multiplier for safety margin
        target_tightness: Max acceptable sphere_vol/hull_vol ratio
        aspect_threshold: Max acceptable aspect ratio before splitting
        percentile: Percentile of distances for radius computation
        max_radius_ratio: Cap radius relative to link bbox diagonal
        uniform_radius: If True, cap radii for more uniformity
        refine: If True, refine spheres with NLLS optimization
        refine_iters: Number of optimization iterations for refinement
        refine_lr: Learning rate for refinement optimization
        refine_lr_center: Learning rate for sphere centers. If None, uses refine_lr.
        refine_lr_radius: Learning rate for sphere radii. If None, uses refine_lr * 0.1.
        lambda_under: Weight for under-approximation loss in refinement
        lambda_over: Weight for over-approximation loss in refinement
        lambda_overlap: Weight for overlap penalty in refinement
        lambda_uniform: Weight for radius uniformity in refinement
        lambda_surface: Weight for surface matching loss in refinement
        lambda_sqem: Weight for SQEM loss (surface signed error with normal projection)
        mesh: Optional pre-loaded mesh. If None, loads mesh from URDF.

    Returns:
        List of spheres for the link (in link-local coordinates)
    """
    if num_spheres <= 0:
        return []

    if mesh is None:
        mesh = get_collision_mesh_for_link(urdf, link_name)
    if mesh.is_empty:
        return []

    try:
        spheres = spherize_adaptive_tight(
            mesh,
            target_tightness=target_tightness,
            aspect_threshold=aspect_threshold,
            target_spheres=num_spheres,
            padding=padding,
            percentile=percentile,
            max_radius_ratio=max_radius_ratio,
            uniform_radius=uniform_radius,
            refine=refine,
            refine_iters=refine_iters,
            refine_lr=refine_lr,
            refine_lr_center=refine_lr_center,
            refine_lr_radius=refine_lr_radius,
            lambda_under=lambda_under,
            lambda_over=lambda_over,
            lambda_overlap=lambda_overlap,
            lambda_uniform=lambda_uniform,
            lambda_surface=lambda_surface,
            lambda_sqem=lambda_sqem,
        )
        return spheres
    except Exception as e:
        logger.warning(f"Spherization failed for {link_name}: {e}")
        return []


def compute_spheres_for_robot(
    urdf,
    target_spheres: int = 100,
    min_spheres_per_link: int = 1,
    link_budgets: dict[str, int] | None = None,
    *,
    config: BallparkConfig | None = None,
    preset: str | None = None,
    padding: float | _UnsetType = UNSET,
    target_tightness: float | _UnsetType = UNSET,
    aspect_threshold: float | _UnsetType = UNSET,
    percentile: float | _UnsetType = UNSET,
    max_radius_ratio: float | _UnsetType = UNSET,
    uniform_radius: bool | _UnsetType = UNSET,
    refine: bool = False,
    refine_iters: int | _UnsetType = UNSET,
    refine_lr: float | _UnsetType = UNSET,
    refine_lr_center: float | None | _UnsetType = UNSET,
    refine_lr_radius: float | None | _UnsetType = UNSET,
    lambda_under: float | _UnsetType = UNSET,
    lambda_over: float | _UnsetType = UNSET,
    lambda_overlap: float | _UnsetType = UNSET,
    lambda_uniform: float | _UnsetType = UNSET,
    lambda_surface: float | _UnsetType = UNSET,
    lambda_sqem: float | _UnsetType = UNSET,
    refine_self_collision: bool | _UnsetType = UNSET,
    lambda_self_collision: float | _UnsetType = UNSET,
    lambda_center_reg: float | _UnsetType = UNSET,
    mesh_collision_tolerance: float | _UnsetType = UNSET,
    n_samples: int = 8000,
    mesh_distances: dict[tuple[str, str], float] | None = None,
    joint_cfg=None,
    similarity_result: SimilarityResult | None = None,
    similarity_groups: list[list[str]] | None = None,
    lambda_similarity: float | _UnsetType = UNSET,
) -> RobotSpheresResult:
    """
    Compute sphere decomposition for all robot links.

    Parameters can be specified in three ways (in order of precedence):
    1. Explicit keyword arguments (highest priority)
    2. A BallparkConfig object via `config=`
    3. A preset name via `preset=` ("conservative", "balanced", "surface")

    If no config or preset is specified, uses "balanced" preset defaults.

    Example usage:
        # Simple preset-based API
        result = compute_spheres_for_robot(urdf, preset="conservative")

        # Config object with overrides
        cfg = get_config("balanced")
        result = compute_spheres_for_robot(urdf, config=cfg, padding=1.05)

        # Traditional explicit params (backward compatible)
        result = compute_spheres_for_robot(urdf, padding=1.02, lambda_under=2.0)

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        target_spheres: Target sphere count across the robot (may slightly exceed)
        min_spheres_per_link: Minimum spheres for any non-empty link
        link_budgets: Optional manual override for per-link sphere counts.
            If provided, uses these counts instead of auto-allocation.
        config: Optional BallparkConfig object with all parameters.
        preset: Optional preset name ("conservative", "balanced", "surface").
        padding: Radius multiplier for safety margin
        target_tightness: Max acceptable sphere_vol/hull_vol ratio
        aspect_threshold: Max acceptable aspect ratio before splitting
        percentile: Percentile of distances for radius computation
        max_radius_ratio: Cap radius relative to link bbox diagonal
        uniform_radius: If True, cap radii for more uniformity (may under-approximate)
        refine: If True, refine spheres with per-link NLLS optimization
        refine_iters: Number of optimization iterations for refinement
        refine_lr: Learning rate for refinement optimization
        refine_lr_center: Learning rate for sphere centers. If None, uses refine_lr.
        refine_lr_radius: Learning rate for sphere radii. If None, uses refine_lr * 0.1.
        lambda_under: Weight for under-approximation loss in refinement
        lambda_over: Weight for over-approximation loss in refinement
        lambda_overlap: Weight for intra-link overlap penalty in refinement
        lambda_uniform: Weight for radius uniformity in refinement
        lambda_surface: Weight for surface matching loss in refinement
        lambda_sqem: Weight for SQEM loss (surface signed error with normal projection)
        refine_self_collision: If True, apply robot-level refinement with
            self-collision avoidance between non-contiguous links at zero config
        lambda_self_collision: Weight for self-collision penalty in refinement
        lambda_center_reg: Weight for center/radius regularization (prevents drifting)
        mesh_collision_tolerance: Skip link pairs where mesh distance < this value (meters).
            Link pairs with inherent mesh proximity are skipped as unfixable.
        n_samples: Number of surface samples per link for refinement
        mesh_distances: Optional pre-computed mesh distances cache from
            compute_mesh_distances_batch(). If provided, skips recomputation.
        joint_cfg: Joint configuration to use for self-collision checking.
            If None, uses middle of joint limits.
        similarity_result: Optional pre-computed similarity result from
            detect_similar_links(). If provided, equalizes sphere counts
            within similarity groups and adds similarity loss during refinement.
        similarity_groups: Optional manual override for similarity groups.
            Each group is a list of link names that should have consistent
            sphere decompositions. Takes precedence over similarity_result.
        lambda_similarity: Weight for similarity position correspondence loss
            in robot-level refinement.

    Returns:
        RobotSpheresResult containing:
        - link_spheres: Dict mapping link names to lists of spheres (in link-local coordinates)
        - ignore_pairs: List of link pairs to ignore for collision checking.
          If refine_self_collision=True, includes adjacent links + mesh-proximity pairs.
          If refine_self_collision=False, includes only adjacent links.

    Raises:
        ValueError: If both config and preset are provided.
    """
    # Resolve parameters from config/preset with explicit overrides
    params = resolve_params(
        config=config,
        preset=preset,
        padding=padding,
        target_tightness=target_tightness,
        aspect_threshold=aspect_threshold,
        percentile=percentile,
        max_radius_ratio=max_radius_ratio,
        uniform_radius=uniform_radius,
        refine_iters=refine_iters,
        refine_lr=refine_lr,
        refine_lr_center=refine_lr_center,
        refine_lr_radius=refine_lr_radius,
        lambda_under=lambda_under,
        lambda_over=lambda_over,
        lambda_overlap=lambda_overlap,
        lambda_uniform=lambda_uniform,
        lambda_surface=lambda_surface,
        lambda_sqem=lambda_sqem,
        refine_self_collision=refine_self_collision,
        lambda_self_collision=lambda_self_collision,
        lambda_center_reg=lambda_center_reg,
        mesh_collision_tolerance=mesh_collision_tolerance,
        lambda_similarity=lambda_similarity,
    )

    # Extract resolved values
    _padding = params["padding"]
    _target_tightness = params["target_tightness"]
    _aspect_threshold = params["aspect_threshold"]
    _percentile = params["percentile"]
    _max_radius_ratio = params["max_radius_ratio"]
    _uniform_radius = params["uniform_radius"]
    _refine_iters = params["refine_iters"]
    _refine_lr = params["refine_lr"]
    _refine_lr_center = params["refine_lr_center"]
    _refine_lr_radius = params["refine_lr_radius"]
    _lambda_under = params["lambda_under"]
    _lambda_over = params["lambda_over"]
    _lambda_overlap = params["lambda_overlap"]
    _lambda_uniform = params["lambda_uniform"]
    _lambda_surface = params["lambda_surface"]
    _lambda_sqem = params["lambda_sqem"]
    _refine_self_collision = params["refine_self_collision"]
    _lambda_self_collision = params["lambda_self_collision"]
    _lambda_center_reg = params["lambda_center_reg"]
    _mesh_collision_tolerance = params["mesh_collision_tolerance"]
    _lambda_similarity = params["lambda_similarity"]

    # Resolve similarity groups (manual override takes precedence)
    effective_similarity: SimilarityResult | None = None
    if similarity_groups is not None:
        # Manual override - create a SimilarityResult from groups
        effective_similarity = SimilarityResult(groups=similarity_groups, transforms={})
    elif similarity_result is not None:
        effective_similarity = similarity_result

    # Get sphere budget per link (auto-allocate or use provided)
    if link_budgets is None:
        budgets = allocate_spheres_for_robot(urdf, target_spheres, min_spheres_per_link)
    else:
        budgets = dict(link_budgets)  # Copy to avoid modifying input

    # Equalize sphere counts within similarity groups
    if effective_similarity is not None and effective_similarity.groups:
        for group in effective_similarity.groups:
            # Get budgets for links in this group that have budgets
            group_budgets = [budgets[link] for link in group if link in budgets]
            if group_budgets:
                # Use the maximum budget in the group (ensure all can be covered)
                max_budget = max(group_budgets)
                for link in group:
                    if link in budgets:
                        budgets[link] = max_budget

    # Generate spheres for each link
    link_spheres = {}
    link_points = {}  # Store sampled points for robot-level refinement

    for link_name in urdf.link_map.keys():
        budget = budgets.get(link_name, 0)

        # Load mesh once and reuse for both sphere computation and point sampling
        mesh = get_collision_mesh_for_link(urdf, link_name)

        link_spheres[link_name] = compute_spheres_for_link(
            urdf,
            link_name,
            budget,
            padding=_padding,
            target_tightness=_target_tightness,
            aspect_threshold=_aspect_threshold,
            percentile=_percentile,
            max_radius_ratio=_max_radius_ratio,
            uniform_radius=_uniform_radius,
            refine=refine,
            refine_iters=_refine_iters,
            refine_lr=_refine_lr,
            refine_lr_center=_refine_lr_center,
            refine_lr_radius=_refine_lr_radius,
            lambda_under=_lambda_under,
            lambda_over=_lambda_over,
            lambda_overlap=_lambda_overlap,
            lambda_uniform=_lambda_uniform,
            lambda_surface=_lambda_surface,
            lambda_sqem=_lambda_sqem,
            mesh=mesh,
        )

        # Sample points for robot-level refinement if needed
        if _refine_self_collision and link_spheres[link_name]:
            if not mesh.is_empty:
                link_points[link_name] = mesh.sample(n_samples)

    # Apply robot-level refinement with self-collision avoidance
    if _refine_self_collision:
        refinement_result = refine_spheres_for_robot(
            urdf=urdf,
            link_spheres=link_spheres,
            link_points=link_points,
            n_iters=_refine_iters,
            lr=_refine_lr,
            lambda_under=_lambda_under,
            lambda_over=_lambda_over,
            lambda_overlap=_lambda_overlap,
            lambda_uniform=_lambda_uniform,
            lambda_self_collision=_lambda_self_collision,
            lambda_center_reg=_lambda_center_reg,
            mesh_collision_tolerance=_mesh_collision_tolerance,
            mesh_distances=mesh_distances,
            joint_cfg=joint_cfg,
            similarity_result=effective_similarity,
            lambda_similarity=_lambda_similarity,
        )
        return RobotSpheresResult(
            link_spheres=refinement_result.link_spheres,
            ignore_pairs=refinement_result.ignore_pairs,
        )

    # No self-collision refinement - just return adjacent pairs as ignore_pairs
    adjacent_pairs = get_adjacent_links(urdf)
    return RobotSpheresResult(
        link_spheres=link_spheres,
        ignore_pairs=list(adjacent_pairs),
    )


def visualize_robot_spheres_viser(
    urdf,
    link_spheres: dict[str, list[Sphere]],
    server=None,
):
    """
    Visualize spheres on robot using viser.

    Args:
        urdf: yourdfpy URDF object
        link_spheres: Dict mapping link names to lists of spheres
        server: Optional existing viser server (creates new one if None)

    Returns:
        viser.ViserServer instance

    Raises:
        ImportError: If viser is not installed
    """
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError:
        raise ImportError(
            "viser is required for visualization. "
            "Install with: pip install ballpark[robot]"
        )

    import time

    if server is None:
        server = viser.ViserServer()

    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Get joint limits and link names
    lower_limits, upper_limits = get_joint_limits(urdf)
    all_link_names = get_link_names(urdf)

    # Joint sliders
    joint_sliders = []
    with server.gui.add_folder("Joints"):
        for i in range(len(lower_limits)):
            lower = float(lower_limits[i])
            upper = float(upper_limits[i])
            initial = (lower + upper) / 2
            slider = server.gui.add_slider(
                f"Joint {i}", min=lower, max=upper, step=0.01, initial_value=initial
            )
            joint_sliders.append(slider)

    # Sphere visualization controls
    show_spheres = server.gui.add_checkbox("Show Spheres", initial_value=True)
    sphere_opacity = server.gui.add_slider(
        "Sphere Opacity", min=0.1, max=1.0, step=0.1, initial_value=0.4
    )

    # Colors for spheres (per link)
    sphere_colors = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 255, 100),
        (255, 100, 255),
        (100, 255, 255),
        (255, 180, 100),
        (180, 100, 255),
        (100, 180, 100),
        (180, 180, 180),
    ]

    # Create sphere frames and handles
    sphere_frames: dict[str, viser.FrameHandle] = {}
    sphere_handles: dict[str, viser.IcosphereHandle] = {}

    def create_sphere_visuals():
        nonlocal sphere_frames, sphere_handles

        for handle in sphere_handles.values():
            handle.remove()
        for handle in sphere_frames.values():
            handle.remove()
        sphere_handles.clear()
        sphere_frames.clear()

        if not show_spheres.value:
            return

        for link_idx, link_name in enumerate(all_link_names):
            if link_name not in link_spheres:
                continue
            spheres = link_spheres[link_name]
            if not spheres:
                continue

            color = sphere_colors[link_idx % len(sphere_colors)]
            rgba = (
                color[0] / 255.0,
                color[1] / 255.0,
                color[2] / 255.0,
                sphere_opacity.value,
            )

            for sphere_idx, sphere in enumerate(spheres):
                key = f"{link_name}_{sphere_idx}"

                frame = server.scene.add_frame(
                    f"/sphere_frames/{key}",
                    wxyz=(1, 0, 0, 0),
                    position=(0, 0, 0),
                    show_axes=False,
                )
                sphere_frames[key] = frame

                sphere_handle = server.scene.add_icosphere(
                    f"/sphere_frames/{key}/sphere",
                    radius=sphere.radius,
                    position=tuple(sphere.center),
                    color=rgba[:3],
                    opacity=rgba[3],
                )
                sphere_handles[key] = sphere_handle

    def update_sphere_transforms(Ts_link_world):
        for link_idx, link_name in enumerate(all_link_names):
            if link_name not in link_spheres:
                continue
            spheres = link_spheres[link_name]
            if not spheres:
                continue

            T_wxyz_xyz = Ts_link_world[link_idx]
            wxyz = T_wxyz_xyz[:4]
            pos = T_wxyz_xyz[4:]

            for sphere_idx, _ in enumerate(spheres):
                key = f"{link_name}_{sphere_idx}"
                if key in sphere_frames:
                    sphere_frames[key].wxyz = wxyz
                    sphere_frames[key].position = pos

    create_sphere_visuals()
    last_show_spheres = show_spheres.value
    last_opacity = sphere_opacity.value

    while True:
        if (
            show_spheres.value != last_show_spheres
            or sphere_opacity.value != last_opacity
        ):
            last_show_spheres = show_spheres.value
            last_opacity = sphere_opacity.value
            create_sphere_visuals()

        cfg = np.array([s.value for s in joint_sliders])
        urdf_vis.update_cfg(cfg)
        Ts_link_world = get_link_transforms(urdf, cfg)

        if show_spheres.value:
            update_sphere_transforms(Ts_link_world)

        time.sleep(0.05)
