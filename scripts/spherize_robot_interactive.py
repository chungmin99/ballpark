#!/usr/bin/env python3
"""Visualize sphere decomposition on a robot with interactive controls."""

from __future__ import annotations

import time
from typing import Literal

import jax
import numpy as np
import tyro
import viser
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

import ballpark


def main(
    robot_name: Literal[
        "ur5",
        "panda",
        "yumi",
        "g1",
        "iiwa14",
        "gen2",  # kinova
    ] = "panda",
) -> None:
    """Visualize sphere decomposition on a robot with interactive controls.

    Args:
        robot_name: Name of the robot to load from robot_descriptions.
    """
    print(f"Loading robot: {robot_name}...")

    # Load URDF for visualization
    urdf = load_robot_description(f"{robot_name}_description")

    # Reload URDF with collision meshes
    urdf_coll = yourdfpy.URDF(
        robot=urdf.robot,
        filename_handler=urdf._filename_handler,
        load_collision_meshes=True,
    )

    # Get joint limits and link names (used throughout)
    lower_limits, upper_limits = ballpark.get_joint_limits(urdf)
    link_names = ballpark.get_link_names(urdf)

    # Identify links with collision geometry
    links_with_collision = []
    for link_name in urdf_coll.link_map.keys():
        mesh = ballpark.get_collision_mesh_for_link(urdf_coll, link_name)
        if not mesh.is_empty:
            links_with_collision.append(link_name)

    # Get initial joint config (middle of limits)
    initial_joint_cfg = (lower_limits + upper_limits) / 2

    # Pre-compute mesh distances for self-collision checking (cached once at startup)
    print("Pre-computing mesh distances for self-collision checking...")
    t0 = time.perf_counter()
    non_contiguous_pairs = ballpark.get_non_contiguous_link_pairs(
        urdf_coll, links_with_collision
    )
    mesh_distances_cache = ballpark.compute_mesh_distances_batch(
        urdf_coll,
        non_contiguous_pairs,
        n_samples=1000,
        bbox_skip_threshold=0.1,
        joint_cfg=initial_joint_cfg,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Cached {len(mesh_distances_cache)} mesh distances in {elapsed_ms:.1f}ms")

    # Detect similar links for consistency regularization
    print("Detecting similar links...")
    similarity_result = ballpark.detect_similar_links(urdf_coll)

    # Set up viser server
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # --- GUI Controls (organized into tabs) ---
    tab_group = server.gui.add_tab_group()

    # Spheres tab - all sphere-related controls
    with tab_group.add_tab("Spheres", icon="circles"):
        # Visualization folder
        with server.gui.add_folder("Visualization"):
            show_spheres = server.gui.add_checkbox("Show Spheres", initial_value=True)
            sphere_opacity = server.gui.add_slider(
                "Opacity", min=0.1, max=1.0, step=0.1, initial_value=0.9
            )

        # Config folder - preset, self-collision, padding
        config_folder = server.gui.add_folder("Config")
        with config_folder:
            preset_dropdown = server.gui.add_dropdown(
                "Preset",
                options=["Balanced", "Conservative", "Surface", "Custom"],
                initial_value="Balanced",
            )
            refine_self_collision_checkbox = server.gui.add_checkbox(
                "Self-Collision Refinement", initial_value=False
            )
            padding_slider = server.gui.add_slider(
                "Padding", min=1.0, max=1.2, step=0.01, initial_value=1.05
            )

        # Allocation folder
        with server.gui.add_folder("Allocation"):
            mode_dropdown = server.gui.add_dropdown(
                "Mode", options=["Auto", "Manual"], initial_value="Auto"
            )
            total_spheres_slider = server.gui.add_slider(
                "Total Spheres", min=0, max=100, step=1, initial_value=40
            )
            link_sphere_sliders: dict[str, viser.GuiInputHandle] = {}
            per_link_folder = server.gui.add_folder("Per-Link", expand_by_default=False)
            with per_link_folder:
                for link_name in links_with_collision:
                    display_name = (
                        link_name[:20] + "..." if len(link_name) > 20 else link_name
                    )
                    slider = server.gui.add_slider(
                        display_name,
                        min=0,
                        max=20,
                        step=1,
                        initial_value=1,
                        disabled=True,  # Start disabled (auto mode)
                    )
                    link_sphere_sliders[link_name] = slider

        # Export folder
        with server.gui.add_folder("Export"):
            export_filename = server.gui.add_text(
                "Filename", initial_value="spheres.json"
            )
            export_button = server.gui.add_button("Export to JSON")

    # Joints tab - robot pose configuration (last for less clutter)
    joint_sliders = []
    with tab_group.add_tab("Joints", icon="adjustments"):
        for i in range(len(lower_limits)):
            lower = float(lower_limits[i])
            upper = float(upper_limits[i])
            initial = (lower + upper) / 2
            slider = server.gui.add_slider(
                f"Joint {i}", min=lower, max=upper, step=0.01, initial_value=initial
            )
            joint_sliders.append(slider)

    # Colors for spheres (per link) - avoid grey as it blends with robot meshes
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
        (255, 200, 150),  # Peach/coral instead of grey
    ]

    # Track state
    sphere_frames: dict[str, viser.FrameHandle] = {}
    sphere_handles: dict[str, viser.IcosphereHandle] = {}
    link_spheres: dict[str, list[ballpark.Sphere]] = {}
    current_link_budgets: dict[str, int] = {}
    current_ignore_pairs: list[tuple[str, str]] = []

    # Map UI preset names to internal config names
    preset_name_map = {
        "Balanced": "balanced",
        "Conservative": "conservative",
        "Surface": "surface",
        "Custom": None,
    }

    # Current config values (updated by presets or custom sliders)
    current_config = ballpark.get_config("balanced")  # Start with Balanced

    # Track Params folder and its sliders
    params_folder_handle = None
    params_sliders: dict[str, viser.GuiInputHandle] = {}

    def create_params_folder() -> None:
        """Create the Params folder with all sliders for Custom mode."""
        nonlocal params_folder_handle, params_sliders

        # Create folder inside Config folder
        with config_folder:
            params_folder_handle = server.gui.add_folder("Params")

        with params_folder_handle:
            with server.gui.add_folder("Learning Rates"):
                params_sliders["lr_center"] = server.gui.add_slider(
                    "lr_center",
                    min=0.0001,
                    max=0.01,
                    step=0.0001,
                    initial_value=current_config.refinement.lr_center or 0.001,
                )
                params_sliders["lr_radius"] = server.gui.add_slider(
                    "lr_radius",
                    min=0.00001,
                    max=0.001,
                    step=0.00001,
                    initial_value=current_config.refinement.lr_radius or 0.0001,
                )

            with server.gui.add_folder("Loss Weights"):
                params_sliders["lambda_under"] = server.gui.add_slider(
                    "lambda_under",
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    initial_value=current_config.refinement.lambda_under,
                )
                params_sliders["lambda_over"] = server.gui.add_slider(
                    "lambda_over",
                    min=0.0,
                    max=0.1,
                    step=0.0001,
                    initial_value=current_config.refinement.lambda_over,
                )
                params_sliders["lambda_overlap"] = server.gui.add_slider(
                    "lambda_overlap",
                    min=0.0,
                    max=0.5,
                    step=0.001,
                    initial_value=current_config.refinement.lambda_overlap,
                )
                params_sliders["lambda_uniform"] = server.gui.add_slider(
                    "lambda_uniform",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=current_config.refinement.lambda_uniform,
                )
                params_sliders["lambda_surface"] = server.gui.add_slider(
                    "lambda_surface",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=current_config.refinement.lambda_surface,
                )
                params_sliders["lambda_sqem"] = server.gui.add_slider(
                    "lambda_sqem",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    initial_value=current_config.refinement.lambda_sqem,
                )

            with server.gui.add_folder("Self-Collision"):
                params_sliders["lambda_self_collision"] = server.gui.add_slider(
                    "lambda_self_collision",
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    initial_value=current_config.robot_refinement.lambda_self_collision,
                )
                params_sliders["lambda_center_reg"] = server.gui.add_slider(
                    "lambda_center_reg",
                    min=0.0,
                    max=10.0,
                    step=0.1,
                    initial_value=current_config.robot_refinement.lambda_center_reg,
                )
                params_sliders["mesh_collision_tol"] = server.gui.add_slider(
                    "mesh_collision_tol",
                    min=0.0,
                    max=0.05,
                    step=0.001,
                    initial_value=current_config.robot_refinement.mesh_collision_tolerance,
                )

    def remove_params_folder() -> None:
        """Remove the Params folder."""
        nonlocal params_folder_handle, params_sliders
        if params_folder_handle is not None:
            params_folder_handle.remove()
            params_folder_handle = None
            params_sliders.clear()

    def apply_preset(preset_name: str) -> None:
        """Apply a configuration preset."""
        nonlocal current_config
        internal_name = preset_name_map.get(preset_name)

        if internal_name is None:
            # Custom mode: show Params folder
            if params_folder_handle is None:
                create_params_folder()
            return

        # Preset mode: hide Params folder and load config
        remove_params_folder()
        current_config = ballpark.get_config(internal_name)
        # Update padding and self-collision checkbox from preset
        padding_slider.value = current_config.adaptive_tight.padding
        refine_self_collision_checkbox.value = current_config.robot_refinement.enabled

    def get_current_params() -> dict:
        """Get current parameter values from config or sliders."""
        if params_folder_handle is not None:
            # Custom mode: read from sliders
            return {
                "lr_center": params_sliders["lr_center"].value,
                "lr_radius": params_sliders["lr_radius"].value,
                "lambda_under": params_sliders["lambda_under"].value,
                "lambda_over": params_sliders["lambda_over"].value,
                "lambda_overlap": params_sliders["lambda_overlap"].value,
                "lambda_uniform": params_sliders["lambda_uniform"].value,
                "lambda_surface": params_sliders["lambda_surface"].value,
                "lambda_sqem": params_sliders["lambda_sqem"].value,
                "lambda_self_collision": params_sliders["lambda_self_collision"].value,
                "lambda_center_reg": params_sliders["lambda_center_reg"].value,
                "mesh_collision_tol": params_sliders["mesh_collision_tol"].value,
            }
        else:
            # Preset mode: read from config
            return {
                "lr_center": current_config.refinement.lr_center or 0.001,
                "lr_radius": current_config.refinement.lr_radius or 0.0001,
                "lambda_under": current_config.refinement.lambda_under,
                "lambda_over": current_config.refinement.lambda_over,
                "lambda_overlap": current_config.refinement.lambda_overlap,
                "lambda_uniform": current_config.refinement.lambda_uniform,
                "lambda_surface": current_config.refinement.lambda_surface,
                "lambda_sqem": current_config.refinement.lambda_sqem,
                "lambda_self_collision": current_config.robot_refinement.lambda_self_collision,
                "lambda_center_reg": current_config.robot_refinement.lambda_center_reg,
                "mesh_collision_tol": current_config.robot_refinement.mesh_collision_tolerance,
            }

    # Track last values for change detection
    last_mode = mode_dropdown.value
    last_total_spheres = total_spheres_slider.value
    last_padding = padding_slider.value
    last_preset = preset_dropdown.value
    last_refine_self_collision = refine_self_collision_checkbox.value
    last_show_spheres = show_spheres.value
    last_opacity = sphere_opacity.value
    last_link_budgets: dict[str, int] = {}
    last_params = get_current_params()
    needs_sphere_rebuild = True

    def get_link_budgets_from_sliders() -> dict[str, int]:
        """Read current per-link sphere counts from sliders."""
        return {
            link_name: int(slider.value)
            for link_name, slider in link_sphere_sliders.items()
        }

    def update_link_sliders_from_budgets(budgets: dict[str, int]) -> None:
        """Update per-link sliders to reflect given budgets."""
        for link_name, slider in link_sphere_sliders.items():
            slider.value = budgets.get(link_name, 0)

    def set_link_sliders_enabled(enabled: bool) -> None:
        """Enable or disable all per-link sphere sliders."""
        for slider in link_sphere_sliders.values():
            slider.disabled = not enabled

    def update_total_from_link_sliders() -> None:
        """Update total spheres slider to reflect sum of per-link allocations."""
        total = sum(int(s.value) for s in link_sphere_sliders.values())
        total_spheres_slider.value = total

    def sync_similar_link_sliders(
        old_budgets: dict[str, int], new_budgets: dict[str, int]
    ) -> None:
        """Synchronize sphere counts for similar/duplicate links.

        When a slider changes for a link that's part of a similarity group,
        update all other links in that group to the same value.
        """
        for link_name, new_val in new_budgets.items():
            old_val = old_budgets.get(link_name, 0)
            if new_val != old_val:
                # This slider changed - check if it's in a similarity group
                group = ballpark.get_group_for_link(similarity_result, link_name)
                if group is not None and len(group) > 1:
                    # Update all other links in the group to the same value
                    for other_link in group:
                        if other_link != link_name and other_link in link_sphere_sliders:
                            link_sphere_sliders[other_link].value = new_val

    def compute_spheres():
        nonlocal link_spheres, current_link_budgets
        is_auto = mode_dropdown.value == "Auto"

        if is_auto:
            total = int(total_spheres_slider.value)
            if total == 0:
                link_spheres = {
                    link_name: [] for link_name in urdf_coll.link_map.keys()
                }
                current_link_budgets = {
                    link_name: 0 for link_name in links_with_collision
                }
                update_link_sliders_from_budgets(current_link_budgets)
                return

            # Auto-allocate spheres
            current_link_budgets = ballpark.allocate_spheres_for_robot(
                urdf_coll,
                target_spheres=total,
                min_spheres_per_link=1,
            )
            update_link_sliders_from_budgets(current_link_budgets)

            print(
                f"Computing spheres (auto, total={total}, padding={padding_slider.value:.2f}, preset={preset_dropdown.value})..."
            )
        else:
            # Manual mode: read budgets from sliders
            current_link_budgets = get_link_budgets_from_sliders()
            total = sum(current_link_budgets.values())
            print(
                f"Computing spheres (manual, total={total}, padding={padding_slider.value:.2f}, preset={preset_dropdown.value})..."
            )

        # Get current joint configuration from sliders
        current_joint_cfg = np.array([s.value for s in joint_sliders])

        # Build effective config: either from preset or from custom sliders
        if params_folder_handle is not None:
            # Custom mode: update current_config from slider values
            params = get_current_params()
            effective_config = ballpark.update_config_from_dict(
                current_config,
                {
                    "refinement.lr_center": params["lr_center"],
                    "refinement.lr_radius": params["lr_radius"],
                    "refinement.lambda_under": params["lambda_under"],
                    "refinement.lambda_over": params["lambda_over"],
                    "refinement.lambda_overlap": params["lambda_overlap"],
                    "refinement.lambda_uniform": params["lambda_uniform"],
                    "refinement.lambda_surface": params["lambda_surface"],
                    "refinement.lambda_sqem": params["lambda_sqem"],
                    "robot_refinement.lambda_self_collision": params[
                        "lambda_self_collision"
                    ],
                    "robot_refinement.lambda_center_reg": params["lambda_center_reg"],
                    "robot_refinement.mesh_collision_tolerance": params[
                        "mesh_collision_tol"
                    ],
                },
            )
        else:
            # Preset mode: use current_config directly
            effective_config = current_config

        t0 = time.perf_counter()
        result = ballpark.compute_spheres_for_robot(
            urdf_coll,
            link_budgets=current_link_budgets,
            config=effective_config,
            # Override from GUI controls
            padding=padding_slider.value,
            refine=True,
            refine_self_collision=refine_self_collision_checkbox.value,
            mesh_distances=mesh_distances_cache,
            joint_cfg=current_joint_cfg,
            similarity_result=similarity_result,
        )
        link_spheres = result.link_spheres
        current_ignore_pairs[:] = result.ignore_pairs
        jax.block_until_ready(None)  # Ensure JAX computations complete
        elapsed_ms = (time.perf_counter() - t0) * 1000
        total_generated = sum(len(s) for s in link_spheres.values())
        print(f"Generated {total_generated} spheres in {elapsed_ms:.1f}ms")

    def create_sphere_visuals():
        nonlocal sphere_frames, sphere_handles

        # Clear existing
        for handle in sphere_handles.values():
            handle.remove()
        for handle in sphere_frames.values():
            handle.remove()
        sphere_handles.clear()
        sphere_frames.clear()

        if not show_spheres.value:
            return

        for link_idx, link_name in enumerate(link_names):
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
        for link_idx, link_name in enumerate(link_names):
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

    # Export button callback
    @export_button.on_click
    def _(_) -> None:
        filename = export_filename.value
        if not filename:
            print("Error: No filename specified")
            return
        ballpark.export_spheres_to_json(
            link_spheres=link_spheres,
            output_path=filename,
            ignore_pairs=current_ignore_pairs,
        )
        total_spheres = sum(len(s) for s in link_spheres.values())
        print(f"Exported {total_spheres} spheres to {filename}")

    # Apply initial preset and compute
    apply_preset("Balanced")
    compute_spheres()
    create_sphere_visuals()

    print("Starting visualization (open browser to view)...")

    while True:
        # Check if mode changed
        if mode_dropdown.value != last_mode:
            last_mode = mode_dropdown.value
            is_manual = last_mode == "Manual"
            set_link_sliders_enabled(is_manual)
            # Toggle total spheres slider: enabled in auto, disabled in manual
            total_spheres_slider.disabled = is_manual
            # Expand per-link folder in manual mode, collapse in auto mode
            per_link_folder.expand = is_manual
            # Recompute spheres on mode change
            compute_spheres()
            last_link_budgets = get_link_budgets_from_sliders()
            needs_sphere_rebuild = True

        # Check for preset changes
        if preset_dropdown.value != last_preset:
            last_preset = preset_dropdown.value
            apply_preset(preset_dropdown.value)
            last_params = get_current_params()
            compute_spheres()
            needs_sphere_rebuild = True

        # Check for hyperparameter changes (applies to both modes)
        current_params = get_current_params()
        hyperparams_changed = (
            padding_slider.value != last_padding
            or refine_self_collision_checkbox.value != last_refine_self_collision
            or current_params != last_params
        )

        # Check for total spheres change (only relevant in auto mode)
        total_changed = (
            mode_dropdown.value == "Auto"
            and total_spheres_slider.value != last_total_spheres
        )

        # Check for per-link slider changes (only relevant in manual mode)
        current_budgets = get_link_budgets_from_sliders()
        link_budgets_changed = (
            mode_dropdown.value == "Manual" and current_budgets != last_link_budgets
        )

        # Synchronize similar links and update total when per-link sliders change
        if link_budgets_changed:
            # Sync similar links first (uses last_link_budgets to detect which changed)
            sync_similar_link_sliders(last_link_budgets, current_budgets)
            # Re-read budgets after sync
            current_budgets = get_link_budgets_from_sliders()
            update_total_from_link_sliders()

        if hyperparams_changed or total_changed or link_budgets_changed:
            last_total_spheres = total_spheres_slider.value
            last_padding = padding_slider.value
            last_refine_self_collision = refine_self_collision_checkbox.value
            last_params = current_params
            last_link_budgets = current_budgets
            compute_spheres()
            needs_sphere_rebuild = True

        # Check if visibility or opacity changed
        if (
            show_spheres.value != last_show_spheres
            or sphere_opacity.value != last_opacity
        ):
            last_show_spheres = show_spheres.value
            last_opacity = sphere_opacity.value
            needs_sphere_rebuild = True

        # Rebuild sphere visuals if needed
        if needs_sphere_rebuild:
            create_sphere_visuals()
            needs_sphere_rebuild = False

        # Get current joint configuration
        cfg = np.array([s.value for s in joint_sliders])

        # Update robot visualization
        urdf_vis.update_cfg(cfg)

        # Get link transforms
        Ts_link_world = ballpark.get_link_transforms(urdf, cfg)

        # Update sphere positions
        if show_spheres.value:
            update_sphere_transforms(Ts_link_world)

        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)
