#!/usr/bin/env python3
"""Visualize sphere decomposition on a robot with interactive controls.

This script demonstrates the Robot class API for sphere decomposition:
- Robot(urdf) - wraps a URDF with collision meshes
- robot.auto_allocate(total) - distributes sphere budget across links by complexity
- robot.spherize(allocation=...) - generates spheres for each link
- robot.refine(result) - optimizes sphere positions and radii
- robot.compute_transforms(cfg) - forward kinematics for all links
- result.save_json(path) - exports spheres to JSON
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import tyro
import viser
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from ballpark import Robot, RobotSpheresResult, Sphere


def main(
    robot_name: Literal["ur5", "panda", "yumi", "g1", "iiwa14", "gen2"] = "panda",
) -> None:
    """Visualize sphere decomposition on a robot with interactive controls."""
    print(f"Loading robot: {robot_name}...")

    # Load URDF with collision meshes for sphere computation
    urdf = load_robot_description(f"{robot_name}_description")
    urdf_coll = yourdfpy.URDF(
        robot=urdf.robot,
        filename_handler=urdf._filename_handler,
        load_collision_meshes=True,
    )

    # Create Robot instance - analyzes collision geometry and detects similar links
    robot = Robot(urdf_coll)
    print(f"Found {len(robot.collision_links)} links with collision geometry")

    # Set up viser visualization
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    gui = _SpheresGui(server, robot)
    sphere_visuals = _SphereVisuals(server, robot.links)

    # Current sphere result (updated when settings change)
    result: RobotSpheresResult | None = None

    def on_export() -> None:
        if result:
            path = Path(gui.export_filename)
            result.save_json(path)
            print(f"Exported {result.num_spheres} spheres to {path}")

    gui.on_export(on_export)

    print("Starting visualization (open browser to view)...")
    while True:
        gui.poll()

        # Recompute spheres when allocation settings change
        if gui.needs_recompute:
            # Two allocation modes:
            # - Auto: robot.auto_allocate() distributes spheres by link complexity
            # - Manual: user specifies per-link counts (similar links stay synced)
            if gui.is_auto_mode:
                if gui.total_spheres == 0:
                    allocation = {name: 0 for name in robot.collision_links}
                else:
                    allocation = robot.auto_allocate(gui.total_spheres)
                gui.update_sliders_from_allocation(allocation)
            else:
                allocation = gui.manual_allocation

            # Generate spheres for each link
            total = sum(allocation.values())
            if total > 0:
                t0 = time.perf_counter()
                result = robot.spherize(allocation=allocation)
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"Generated {result.num_spheres} spheres in {elapsed:.1f}ms")

                # Optionally refine spheres
                if gui.refine_enabled:
                    t0 = time.perf_counter()
                    result = robot.refine(result)
                    elapsed = (time.perf_counter() - t0) * 1000
                    print(f"Refined spheres in {elapsed:.1f}ms")
            else:
                result = RobotSpheresResult(link_spheres={})

            sphere_visuals.update(result, gui.opacity, gui.show_spheres)
            gui.mark_computed()
            gui.mark_visuals_updated()

        # Update sphere visuals if only appearance changed (opacity, visibility)
        if gui.needs_visual_update and result:
            sphere_visuals.update(result, gui.opacity, gui.show_spheres)
            gui.mark_visuals_updated()

        # Update robot pose from joint sliders
        cfg = gui.joint_config
        urdf_vis.update_cfg(cfg)

        # Transform spheres to match current link poses
        if gui.show_spheres:
            Ts = robot.compute_transforms(cfg)
            sphere_visuals.update_transforms(Ts)

        time.sleep(0.05)


# -----------------------------------------------------------------------------
# GUI helpers (implementation details below)
# -----------------------------------------------------------------------------


class _SpheresGui:
    """GUI controls for sphere visualization."""

    def __init__(self, server: viser.ViserServer, robot: Robot):
        self._robot = robot
        self._export_callback: Callable[[], None] | None = None

        # Track state for change detection
        self._last_mode: str = "Auto"
        self._last_total: int = 40
        self._last_link_budgets: dict[str, int] = {}
        self._last_show: bool = True
        self._last_opacity: float = 0.9
        self._last_refine: bool = False
        self._needs_recompute = True
        self._needs_visual_update = True

        # Build GUI
        tab_group = server.gui.add_tab_group()

        # Spheres tab
        with tab_group.add_tab("Spheres"):
            with server.gui.add_folder("Visualization"):
                self._show_spheres = server.gui.add_checkbox(
                    "Show Spheres", initial_value=True
                )
                self._opacity = server.gui.add_slider(
                    "Opacity", min=0.1, max=1.0, step=0.1, initial_value=0.9
                )
                self._refine = server.gui.add_checkbox(
                    "Refine (optimize)", initial_value=False
                )

            with server.gui.add_folder("Allocation"):
                self._mode = server.gui.add_dropdown(
                    "Mode", options=["Auto", "Manual"], initial_value="Auto"
                )
                self._total_spheres = server.gui.add_slider(
                    "Total Spheres", min=0, max=100, step=1, initial_value=40
                )
                self._link_sliders: dict[str, viser.GuiInputHandle] = {}
                with server.gui.add_folder("Per-Link", expand_by_default=False):
                    for link_name in robot.collision_links:
                        display = (
                            link_name[:20] + "..."
                            if len(link_name) > 20
                            else link_name
                        )
                        self._link_sliders[link_name] = server.gui.add_slider(
                            display,
                            min=0,
                            max=20,
                            step=1,
                            initial_value=1,
                            disabled=True,
                        )

            with server.gui.add_folder("Export"):
                self._export_filename = server.gui.add_text(
                    "Filename", initial_value="spheres.json"
                )
                export_button = server.gui.add_button("Export to JSON")

                @export_button.on_click
                def _(_) -> None:
                    if self._export_callback:
                        self._export_callback()

        # Joints tab
        lower, upper = robot.joint_limits
        self._joint_sliders = []
        with tab_group.add_tab("Joints"):
            for i in range(len(lower)):
                slider = server.gui.add_slider(
                    f"Joint {i}",
                    min=float(lower[i]),
                    max=float(upper[i]),
                    step=0.01,
                    initial_value=(float(lower[i]) + float(upper[i])) / 2,
                )
                self._joint_sliders.append(slider)

    def poll(self) -> None:
        """Check for GUI changes and update internal state."""
        # Mode change
        if self._mode.value != self._last_mode:
            self._last_mode = self._mode.value
            is_manual = self._last_mode == "Manual"
            for slider in self._link_sliders.values():
                slider.disabled = not is_manual
            self._total_spheres.disabled = is_manual
            self._needs_recompute = True

        # Total spheres change (auto mode)
        if self._mode.value == "Auto" and self._total_spheres.value != self._last_total:
            self._last_total = int(self._total_spheres.value)
            self._needs_recompute = True

        # Per-link slider change (manual mode)
        if self._mode.value == "Manual":
            current = {name: int(s.value) for name, s in self._link_sliders.items()}
            if current != self._last_link_budgets:
                # Sync similar links: when one link in a similarity group changes,
                # update all others in the group to match
                for name, new_val in current.items():
                    if new_val != self._last_link_budgets.get(name, 0):
                        group = self._get_group_for_link(name)
                        if group and len(group) > 1:
                            for other in group:
                                if other != name and other in self._link_sliders:
                                    self._link_sliders[other].value = new_val
                current = {name: int(s.value) for name, s in self._link_sliders.items()}
                self._total_spheres.value = sum(current.values())
                self._last_link_budgets = current
                self._needs_recompute = True

        # Refine checkbox change
        if self._refine.value != self._last_refine:
            self._last_refine = self._refine.value
            self._needs_recompute = True

        # Visibility/opacity change
        if (
            self._show_spheres.value != self._last_show
            or self._opacity.value != self._last_opacity
        ):
            self._last_show = self._show_spheres.value
            self._last_opacity = self._opacity.value
            self._needs_visual_update = True

    def _get_group_for_link(self, link_name: str) -> list[str] | None:
        for group in self._robot._similarity.groups:
            if link_name in group:
                return group
        return None

    @property
    def is_auto_mode(self) -> bool:
        return self._mode.value == "Auto"

    @property
    def total_spheres(self) -> int:
        return int(self._total_spheres.value)

    @property
    def manual_allocation(self) -> dict[str, int]:
        """Per-link allocation from manual sliders."""
        return {name: int(s.value) for name, s in self._link_sliders.items()}

    def update_sliders_from_allocation(self, alloc: dict[str, int]) -> None:
        """Update per-link sliders to reflect an allocation."""
        for name, slider in self._link_sliders.items():
            slider.value = alloc.get(name, 0)
        self._last_link_budgets = alloc

    @property
    def needs_recompute(self) -> bool:
        return self._needs_recompute

    def mark_computed(self) -> None:
        self._needs_recompute = False

    @property
    def needs_visual_update(self) -> bool:
        return self._needs_visual_update

    def mark_visuals_updated(self) -> None:
        self._needs_visual_update = False

    @property
    def show_spheres(self) -> bool:
        return self._show_spheres.value

    @property
    def opacity(self) -> float:
        return self._opacity.value

    @property
    def refine_enabled(self) -> bool:
        return self._refine.value

    @property
    def joint_config(self) -> np.ndarray:
        return np.array([s.value for s in self._joint_sliders])

    @property
    def export_filename(self) -> str:
        return self._export_filename.value

    def on_export(self, callback: Callable[[], None]) -> None:
        self._export_callback = callback


class _SphereVisuals:
    """Manages sphere visualization in viser."""

    COLORS = [
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

    def __init__(self, server: viser.ViserServer, link_names: list[str]):
        self._server = server
        self._link_names = link_names
        self._frames: dict[str, viser.FrameHandle] = {}
        self._handles: dict[str, viser.IcosphereHandle] = {}
        self._link_spheres: dict[str, list[Sphere]] = {}

    def update(
        self,
        result: RobotSpheresResult,
        opacity: float,
        visible: bool,
    ) -> None:
        """Rebuild sphere visuals from result."""
        # Clear existing
        for h in self._handles.values():
            h.remove()
        for f in self._frames.values():
            f.remove()
        self._handles.clear()
        self._frames.clear()
        self._link_spheres = result.link_spheres

        if not visible:
            return

        for link_idx, link_name in enumerate(self._link_names):
            spheres = self._link_spheres.get(link_name, [])
            if not spheres:
                continue

            color = self.COLORS[link_idx % len(self.COLORS)]
            rgb = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

            for sphere_idx, sphere in enumerate(spheres):
                key = f"{link_name}_{sphere_idx}"
                frame = self._server.scene.add_frame(
                    f"/sphere_frames/{key}",
                    wxyz=(1, 0, 0, 0),
                    position=(0, 0, 0),
                    show_axes=False,
                )
                self._frames[key] = frame
                self._handles[key] = self._server.scene.add_icosphere(
                    f"/sphere_frames/{key}/sphere",
                    radius=sphere.radius,
                    position=tuple(sphere.center),
                    color=rgb,
                    opacity=opacity,
                )

    def update_transforms(self, Ts_link_world: np.ndarray) -> None:
        """Update sphere positions from link transforms."""
        for link_idx, link_name in enumerate(self._link_names):
            spheres = self._link_spheres.get(link_name, [])
            if not spheres:
                continue

            T = Ts_link_world[link_idx]
            wxyz, pos = T[:4], T[4:]

            for sphere_idx in range(len(spheres)):
                key = f"{link_name}_{sphere_idx}"
                if key in self._frames:
                    self._frames[key].wxyz = wxyz
                    self._frames[key].position = pos


if __name__ == "__main__":
    tyro.cli(main)
