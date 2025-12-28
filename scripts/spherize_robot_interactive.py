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
# Import visualization utilities from internal modules
from ballpark._urdf_utils import get_joint_limits, get_link_names, get_link_transforms


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

    # Get joint limits and link names (used for visualization)
    lower_limits, upper_limits = get_joint_limits(urdf)
    link_names = get_link_names(urdf)

    # Create Robot instance (pre-computes similarity, mesh distances, etc.)
    print("Analyzing robot structure...")
    t0 = time.perf_counter()
    robot = ballpark.Robot(urdf_coll)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"Found {len(robot.links)} links with collision geometry in {elapsed_ms:.1f}ms")

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

        # Config folder - preset, refinement, padding
        with server.gui.add_folder("Config"):
            preset_dropdown = server.gui.add_dropdown(
                "Preset",
                options=["Balanced", "Conservative", "Surface"],
                initial_value="Balanced",
            )
            refine_checkbox = server.gui.add_checkbox(
                "Refinement", initial_value=False
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
                for link_name in robot.links:
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

    # Joints tab - robot pose configuration
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
        (255, 200, 150),
    ]

    # Track state
    sphere_frames: dict[str, viser.FrameHandle] = {}
    sphere_handles: dict[str, viser.IcosphereHandle] = {}
    link_spheres: dict[str, list[ballpark.Sphere]] = {}
    current_link_budgets: dict[str, int] = {}

    # Map UI preset names to internal config names
    preset_name_map = {
        "Balanced": "balanced",
        "Conservative": "conservative",
        "Surface": "surface",
    }

    # Track last values for change detection
    last_mode = mode_dropdown.value
    last_total_spheres = total_spheres_slider.value
    last_padding = padding_slider.value
    last_preset = preset_dropdown.value
    last_refine = refine_checkbox.value
    last_show_spheres = show_spheres.value
    last_opacity = sphere_opacity.value
    last_link_budgets: dict[str, int] = {}
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

    def get_group_for_link(link_name: str) -> list[str] | None:
        """Find the similarity group containing a link."""
        for group in robot.similarity.groups:
            if link_name in group:
                return group
        return None

    def sync_similar_link_sliders(
        old_budgets: dict[str, int], new_budgets: dict[str, int]
    ) -> None:
        """Synchronize sphere counts for similar/duplicate links."""
        for link_name, new_val in new_budgets.items():
            old_val = old_budgets.get(link_name, 0)
            if new_val != old_val:
                group = get_group_for_link(link_name)
                if group is not None and len(group) > 1:
                    for other_link in group:
                        if other_link != link_name and other_link in link_sphere_sliders:
                            link_sphere_sliders[other_link].value = new_val

    def compute_spheres():
        nonlocal link_spheres, current_link_budgets, current_result
        is_auto = mode_dropdown.value == "Auto"

        if is_auto:
            total = int(total_spheres_slider.value)
            if total == 0:
                link_spheres = {link_name: [] for link_name in urdf_coll.link_map.keys()}
                current_link_budgets = {link_name: 0 for link_name in robot.links}
                update_link_sliders_from_budgets(current_link_budgets)
                return

            # Auto-allocate spheres
            current_link_budgets = robot.allocate(total)
            update_link_sliders_from_budgets(current_link_budgets)

            print(
                f"Computing spheres (auto, total={total}, padding={padding_slider.value:.2f}, "
                f"preset={preset_dropdown.value})..."
            )
        else:
            # Manual mode: read budgets from sliders
            current_link_budgets = get_link_budgets_from_sliders()
            total = sum(current_link_budgets.values())
            print(
                f"Computing spheres (manual, total={total}, padding={padding_slider.value:.2f}, "
                f"preset={preset_dropdown.value})..."
            )

        t0 = time.perf_counter()
        current_result = robot.spherize(
            budgets=current_link_budgets,
            refine=refine_checkbox.value,
            preset=preset_name_map[preset_dropdown.value],
            padding=padding_slider.value,
        )
        link_spheres = current_result.link_spheres
        jax.block_until_ready(None)
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

    # Store the latest result for export
    current_result: ballpark.RobotSpheresResult | None = None

    # Export button callback
    @export_button.on_click
    def _(_) -> None:
        filename = export_filename.value
        if not filename:
            print("Error: No filename specified")
            return
        if current_result is None:
            print("Error: No spheres computed yet")
            return
        current_result.export(filename)
        total_spheres = sum(len(s) for s in current_result.link_spheres.values())
        print(f"Exported {total_spheres} spheres to {filename}")

    # Initial computation
    compute_spheres()
    create_sphere_visuals()

    print("Starting visualization (open browser to view)...")

    while True:
        # Check if mode changed
        if mode_dropdown.value != last_mode:
            last_mode = mode_dropdown.value
            is_manual = last_mode == "Manual"
            set_link_sliders_enabled(is_manual)
            total_spheres_slider.disabled = is_manual
            per_link_folder.expand = is_manual
            compute_spheres()
            last_link_budgets = get_link_budgets_from_sliders()
            needs_sphere_rebuild = True

        # Check for config changes
        config_changed = (
            preset_dropdown.value != last_preset
            or padding_slider.value != last_padding
            or refine_checkbox.value != last_refine
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

        # Synchronize similar links when per-link sliders change
        if link_budgets_changed:
            sync_similar_link_sliders(last_link_budgets, current_budgets)
            current_budgets = get_link_budgets_from_sliders()
            update_total_from_link_sliders()

        if config_changed or total_changed or link_budgets_changed:
            last_total_spheres = total_spheres_slider.value
            last_padding = padding_slider.value
            last_preset = preset_dropdown.value
            last_refine = refine_checkbox.value
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

        # Get link transforms and update sphere positions
        Ts_link_world = get_link_transforms(urdf, cfg)
        if show_spheres.value:
            update_sphere_transforms(Ts_link_world)

        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)
