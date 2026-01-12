#!/usr/bin/env python3
"""Interactive editor for robot sphere JSON files.

This script allows manual refinement of sphere positions and radii
from a robot spheres JSON file. Click to select spheres, drag to
reposition, adjust radius with a slider, and save changes.

Usage:
    python scripts/edit_robot_spheres.py --robot_name panda --json_path spheres.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import tyro
import viser
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

from ballpark import Robot, SPHERE_COLORS


# -----------------------------------------------------------------------------
# Data Store
# -----------------------------------------------------------------------------


class SphereDataStore:
    """Manages mutable sphere data with save functionality."""

    def __init__(self, json_path: Path):
        self.path = json_path
        with open(json_path) as f:
            raw = json.load(f)

        # Validate format
        if "centers" in raw and "radii" in raw:
            raise ValueError(
                "This appears to be a mesh sphere export, not a robot export. "
                "Use view_robot_spheres.py for robot sphere files."
            )

        # Store as mutable lists
        self.data: dict[str, dict[str, list]] = {
            link: {
                "centers": [list(c) for c in v["centers"]],
                "radii": list(v["radii"]),
            }
            for link, v in raw.items()
        }
        self.dirty = False

    def update_sphere_position(self, link: str, idx: int, new_pos: list[float]) -> None:
        """Update a sphere's center position."""
        self.data[link]["centers"][idx] = new_pos
        self.dirty = True

    def update_sphere_radius(self, link: str, idx: int, new_radius: float) -> None:
        """Update a sphere's radius."""
        self.data[link]["radii"][idx] = new_radius
        self.dirty = True

    def delete_sphere(self, link: str, idx: int) -> None:
        """Delete a sphere from the data."""
        del self.data[link]["centers"][idx]
        del self.data[link]["radii"][idx]
        # Remove link entry if no spheres remain
        if len(self.data[link]["radii"]) == 0:
            del self.data[link]
        self.dirty = True

    def save(self) -> None:
        """Write current data back to JSON file."""
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        self.dirty = False

    @property
    def num_spheres(self) -> int:
        """Total number of spheres across all links."""
        return sum(len(v["radii"]) for v in self.data.values())


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------


class _EditorGui:
    """GUI controls for sphere editor."""

    def __init__(
        self,
        server: viser.ViserServer,
        robot: Robot,
        data_store: SphereDataStore,
    ):
        self._server = server
        self._robot = robot
        self._data_store = data_store

        # Callbacks
        self._on_save: Callable[[], None] | None = None
        self._on_deselect: Callable[[], None] | None = None
        self._on_delete: Callable[[], None] | None = None
        self._on_radius_change: Callable[[float], None] | None = None

        # Selection info folder (created dynamically)
        self._selection_folder: viser.GuiFolderHandle | None = None
        self._selection_link_text: viser.GuiInputHandle | None = None
        self._selection_index_text: viser.GuiInputHandle | None = None
        self._radius_slider: viser.GuiInputHandle | None = None

        # Build GUI
        with server.gui.add_folder("Editor"):
            server.gui.add_text(
                "File", initial_value=str(data_store.path), disabled=True
            )
            self._status_text = server.gui.add_text(
                "Status", initial_value="Saved", disabled=True
            )

            save_button = server.gui.add_button("Save")

            @save_button.on_click
            def _(_) -> None:
                if self._on_save:
                    self._on_save()

            self._deselect_button = server.gui.add_button("Deselect", disabled=True)

            @self._deselect_button.on_click
            def _(_) -> None:
                if self._on_deselect:
                    self._on_deselect()

        with server.gui.add_folder("Visualization"):
            self._show_spheres = server.gui.add_checkbox(
                "Show Spheres", initial_value=True
            )
            self._opacity = server.gui.add_slider(
                "Opacity", min=0.1, max=1.0, step=0.1, initial_value=0.7
            )

        # Joint sliders
        lower, upper = robot.joint_limits
        self._joint_sliders = []
        with server.gui.add_folder("Joints"):
            for i in range(len(lower)):
                slider = server.gui.add_slider(
                    f"Joint {i}",
                    min=float(lower[i]),
                    max=float(upper[i]),
                    step=0.01,
                    initial_value=(float(lower[i]) + float(upper[i])) / 2,
                )
                self._joint_sliders.append(slider)

    def set_on_save(self, callback: Callable[[], None]) -> None:
        self._on_save = callback

    def set_on_deselect(self, callback: Callable[[], None]) -> None:
        self._on_deselect = callback

    def set_on_delete(self, callback: Callable[[], None]) -> None:
        self._on_delete = callback

    def set_on_radius_change(self, callback: Callable[[float], None]) -> None:
        self._on_radius_change = callback

    def update_status(self) -> None:
        """Update status text based on dirty state."""
        self._status_text.value = "Unsaved*" if self._data_store.dirty else "Saved"

    def show_selection_info(self, link_name: str, idx: int, radius: float) -> None:
        """Show the selected sphere info folder."""
        # Remove existing folder if present
        self.hide_selection_info()

        # Enable deselect button
        self._deselect_button.disabled = False

        self._selection_folder = self._server.gui.add_folder("Selected Sphere")
        with self._selection_folder:
            self._selection_link_text = self._server.gui.add_text(
                "Link", initial_value=link_name, disabled=True
            )
            self._selection_index_text = self._server.gui.add_text(
                "Index", initial_value=str(idx), disabled=True
            )
            self._radius_slider = self._server.gui.add_slider(
                "Radius", min=0.005, max=0.3, step=0.005, initial_value=radius
            )

            @self._radius_slider.on_update
            def _(_) -> None:
                if self._on_radius_change and self._radius_slider is not None:
                    self._on_radius_change(float(self._radius_slider.value))

            delete_button = self._server.gui.add_button("Delete", color="red")

            @delete_button.on_click
            def _(_) -> None:
                if self._on_delete:
                    self._on_delete()

    def hide_selection_info(self) -> None:
        """Hide the selected sphere info folder."""
        # Disable deselect button
        self._deselect_button.disabled = True

        if self._selection_folder is not None:
            self._selection_folder.remove()
            self._selection_folder = None
            self._selection_link_text = None
            self._selection_index_text = None
            self._radius_slider = None

    def update_radius_slider(self, radius: float) -> None:
        """Update the radius slider value without triggering callback."""
        if self._radius_slider is not None:
            self._radius_slider.value = radius

    @property
    def show_spheres(self) -> bool:
        return self._show_spheres.value

    @property
    def opacity(self) -> float:
        return self._opacity.value

    @property
    def joint_config(self) -> np.ndarray:
        return np.array([s.value for s in self._joint_sliders])


# -----------------------------------------------------------------------------
# Sphere Visuals with Selection
# -----------------------------------------------------------------------------


class _EditableSphereVisuals:
    """Manages sphere visualization with selection and editing."""

    def __init__(
        self,
        server: viser.ViserServer,
        link_names: list[str],
        data_store: SphereDataStore,
    ):
        self._server = server
        self._link_names = link_names
        self._data_store = data_store

        # Visual handles
        self._frames: dict[str, viser.FrameHandle] = {}
        self._handles: dict[str, viser.IcosphereHandle] = {}
        self._colors: dict[str, tuple[float, float, float]] = {}

        # Selection state
        self._selected_key: str | None = None
        self._transform_control: viser.TransformControlsHandle | None = None

        # Current link transforms (updated each frame)
        self._link_transforms: np.ndarray | None = None

        # Callbacks
        self._on_select: Callable[[str, int, float], None] | None = None
        self._on_position_change: Callable[[str, int, list[float]], None] | None = None
        self._on_deselect: Callable[[], None] | None = None

        # Visual settings
        self._normal_opacity = 0.7
        self._selected_opacity = 0.9
        self._highlight_color = (0.2, 1.0, 0.8)  # Bright cyan/teal highlight

    def set_on_select(self, callback: Callable[[str, int, float], None]) -> None:
        """Set callback for sphere selection. Args: link_name, idx, radius."""
        self._on_select = callback

    def set_on_position_change(
        self, callback: Callable[[str, int, list[float]], None]
    ) -> None:
        """Set callback for position change. Args: link_name, idx, new_local_pos."""
        self._on_position_change = callback

    def set_on_deselect(self, callback: Callable[[], None]) -> None:
        """Set callback for deselection."""
        self._on_deselect = callback

    def rebuild(self, opacity: float, visible: bool) -> None:
        """Rebuild all sphere visuals from data store."""
        self._normal_opacity = opacity

        # Clear existing
        for h in self._handles.values():
            h.remove()
        for f in self._frames.values():
            f.remove()
        if self._transform_control is not None:
            self._transform_control.remove()
            self._transform_control = None
        self._handles.clear()
        self._frames.clear()
        self._colors.clear()
        self._selected_key = None

        if not visible:
            return

        for link_idx, link_name in enumerate(self._link_names):
            if link_name not in self._data_store.data:
                continue

            centers = self._data_store.data[link_name]["centers"]
            radii = self._data_store.data[link_name]["radii"]
            color = SPHERE_COLORS[link_idx % len(SPHERE_COLORS)]
            rgb = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

            for i, (center, radius) in enumerate(zip(centers, radii)):
                key = f"{link_name}_{i}"
                self._colors[key] = rgb

                frame = self._server.scene.add_frame(
                    f"/sphere_frames/{key}",
                    wxyz=(1, 0, 0, 0),
                    position=(0, 0, 0),
                    show_axes=False,
                )
                self._frames[key] = frame

                handle = self._server.scene.add_icosphere(
                    f"/sphere_frames/{key}/sphere",
                    radius=float(radius),
                    position=(float(center[0]), float(center[1]), float(center[2])),
                    color=rgb,
                    opacity=opacity,
                )
                self._handles[key] = handle

                # Register click handler
                self._register_click_handler(key, handle)

    def _register_click_handler(self, key: str, handle: viser.IcosphereHandle) -> None:
        """Register click handler for a sphere."""

        @handle.on_click
        def _(_: viser.SceneNodePointerEvent) -> None:
            self._select_sphere(key)

    def _select_sphere(self, key: str) -> None:
        """Select a sphere and show transform controls."""
        # Ignore clicks while another sphere is selected
        # User must deselect first to prevent interference with transform controls
        if self._selected_key is not None:
            return

        self._selected_key = key

        # Parse link name and index
        parts = key.rsplit("_", 1)
        link_name, idx = parts[0], int(parts[1])

        # Disable clicks on all other spheres to prevent interference with transform control
        for other_key, handle in self._handles.items():
            if other_key != key:
                handle.remove_click_callback("all")

        # Highlight selected sphere by recreating it with highlight color
        # (color is read-only after creation)
        if key in self._handles:
            center = self._data_store.data[link_name]["centers"][idx]
            radius = self._data_store.data[link_name]["radii"][idx]
            self._handles[key].remove()
            self._handles[key] = self._server.scene.add_icosphere(
                f"/sphere_frames/{key}/sphere",
                radius=float(radius),
                position=(float(center[0]), float(center[1]), float(center[2])),
                color=self._highlight_color,
                opacity=self._selected_opacity,
            )

        # Get current local position and world position
        local_center = self._data_store.data[link_name]["centers"][idx]
        radius = self._data_store.data[link_name]["radii"][idx]

        if self._link_transforms is not None:
            world_center = self._local_to_world(link_name, np.array(local_center))

            # Create transform control at world position
            self._transform_control = self._server.scene.add_transform_controls(
                "/transform_control",
                position=tuple(world_center),
                wxyz=(1, 0, 0, 0),
                scale=0.2,
                depth_test=False,
            )

            # Register update callback
            @self._transform_control.on_update
            def _(_) -> None:
                self._on_transform_update()

        # Notify callback
        if self._on_select:
            self._on_select(link_name, idx, radius)

    def deselect(self) -> None:
        """Deselect the current sphere."""
        if self._selected_key is None:
            return

        # Restore original color by recreating the sphere (color is read-only)
        if self._selected_key in self._handles:
            parts = self._selected_key.rsplit("_", 1)
            link_name, idx = parts[0], int(parts[1])
            center = self._data_store.data[link_name]["centers"][idx]
            radius = self._data_store.data[link_name]["radii"][idx]
            original_color = self._colors.get(self._selected_key, (0.5, 0.5, 0.5))

            self._handles[self._selected_key].remove()
            handle = self._server.scene.add_icosphere(
                f"/sphere_frames/{self._selected_key}/sphere",
                radius=float(radius),
                position=(float(center[0]), float(center[1]), float(center[2])),
                color=original_color,
                opacity=self._normal_opacity,
            )
            self._handles[self._selected_key] = handle
            self._register_click_handler(self._selected_key, handle)

        # Remove transform control
        if self._transform_control is not None:
            self._transform_control.remove()
            self._transform_control = None

        # Re-enable clicks on all spheres
        for key, handle in self._handles.items():
            self._register_click_handler(key, handle)

        self._selected_key = None

        # Notify callback
        if self._on_deselect:
            self._on_deselect()

    def _on_transform_update(self) -> None:
        """Handle transform control position change."""
        if self._transform_control is None or self._selected_key is None:
            return
        if self._link_transforms is None:
            return

        # Get new world position from control
        new_world_pos = np.array(self._transform_control.position)

        # Parse link and index
        parts = self._selected_key.rsplit("_", 1)
        link_name, idx = parts[0], int(parts[1])

        # Convert to local coordinates
        new_local_pos = self._world_to_local(link_name, new_world_pos)

        # Update visual (local position relative to frame)
        if self._selected_key in self._handles:
            self._handles[self._selected_key].position = tuple(new_local_pos)

        # Notify callback
        if self._on_position_change:
            self._on_position_change(link_name, idx, new_local_pos.tolist())

    def update_selected_radius(self, new_radius: float) -> None:
        """Update the radius of the selected sphere."""
        if self._selected_key is None:
            return

        # Update visual
        if self._selected_key in self._handles:
            # Need to recreate the sphere since viser doesn't support radius update
            parts = self._selected_key.rsplit("_", 1)
            link_name, idx = parts[0], int(parts[1])
            center = self._data_store.data[link_name]["centers"][idx]

            # Remove old handle
            old_handle = self._handles[self._selected_key]
            old_handle.remove()

            # Create new handle with updated radius
            # Note: no click handler needed - clicks are disabled during selection
            handle = self._server.scene.add_icosphere(
                f"/sphere_frames/{self._selected_key}/sphere",
                radius=float(new_radius),
                position=(float(center[0]), float(center[1]), float(center[2])),
                color=self._highlight_color,
                opacity=self._selected_opacity,
            )
            self._handles[self._selected_key] = handle

    def update_transforms(self, Ts_link_world: np.ndarray) -> None:
        """Update sphere positions from link transforms."""
        self._link_transforms = Ts_link_world

        for link_idx, link_name in enumerate(self._link_names):
            if link_name not in self._data_store.data:
                continue

            T = Ts_link_world[link_idx]
            wxyz, pos = T[:4], T[4:]
            num_spheres = len(self._data_store.data[link_name]["radii"])

            for sphere_idx in range(num_spheres):
                key = f"{link_name}_{sphere_idx}"
                if key in self._frames:
                    self._frames[key].wxyz = wxyz
                    self._frames[key].position = pos

        # Update transform control position if selection exists
        if self._selected_key is not None and self._transform_control is not None:
            parts = self._selected_key.rsplit("_", 1)
            link_name, idx = parts[0], int(parts[1])
            local_center = self._data_store.data[link_name]["centers"][idx]
            world_center = self._local_to_world(link_name, np.array(local_center))
            self._transform_control.position = tuple(world_center)

    def _world_to_local(self, link_name: str, world_pos: np.ndarray) -> np.ndarray:
        """Convert world position to link-local position."""
        if self._link_transforms is None:
            return world_pos

        link_idx = self._link_names.index(link_name)
        T = self._link_transforms[link_idx]
        wxyz, pos = T[:4], T[4:]

        # Quaternion to rotation matrix (wxyz -> xyzw for scipy)
        quat_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        R = Rotation.from_quat(quat_xyzw).as_matrix()

        # local = R^T @ (world - pos)
        local = R.T @ (world_pos - pos)
        return local

    def _local_to_world(self, link_name: str, local_pos: np.ndarray) -> np.ndarray:
        """Convert link-local position to world position."""
        if self._link_transforms is None:
            return local_pos

        link_idx = self._link_names.index(link_name)
        T = self._link_transforms[link_idx]
        wxyz, pos = T[:4], T[4:]

        # Quaternion to rotation matrix (wxyz -> xyzw for scipy)
        quat_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        R = Rotation.from_quat(quat_xyzw).as_matrix()

        # world = R @ local + pos
        world = R @ local_pos + pos
        return world

    @property
    def has_selection(self) -> bool:
        return self._selected_key is not None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main(
    robot_name: Literal["ur5", "panda", "yumi", "g1", "iiwa14", "gen2"],
    json_path: Path,
    port: int = 8080,
) -> None:
    """Edit robot spheres interactively.

    Args:
        robot_name: Name of the robot (must match the robot used to generate spheres).
        json_path: Path to the JSON file containing sphere data.
        port: Port for the viser web server.
    """
    # Load sphere data
    print(f"Loading spheres from {json_path}...")
    data_store = SphereDataStore(json_path)
    print(
        f"Loaded {data_store.num_spheres} spheres across {len(data_store.data)} links"
    )

    # Load robot
    print(f"Loading robot: {robot_name}...")
    urdf = load_robot_description(f"{robot_name}_description")
    urdf_coll = yourdfpy.URDF(
        robot=urdf.robot,
        filename_handler=urdf._filename_handler,
        load_collision_meshes=True,
    )
    robot = Robot(urdf_coll)

    # Check for missing links
    missing = set(data_store.data.keys()) - set(robot.links)
    if missing:
        print(f"Warning: JSON contains links not in robot: {missing}")

    # Set up viser
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Create GUI and visuals
    gui = _EditorGui(server, robot, data_store)
    sphere_visuals = _EditableSphereVisuals(server, robot.links, data_store)

    # Track state for change detection
    last_show = gui.show_spheres
    last_opacity = gui.opacity

    # Wire up callbacks
    def on_save() -> None:
        data_store.save()
        gui.update_status()
        print(f"Saved {data_store.num_spheres} spheres to {json_path}")

    def on_deselect() -> None:
        sphere_visuals.deselect()
        gui.hide_selection_info()

    def on_select(link_name: str, idx: int, radius: float) -> None:
        gui.show_selection_info(link_name, idx, radius)

    def on_position_change(link_name: str, idx: int, new_pos: list[float]) -> None:
        data_store.update_sphere_position(link_name, idx, new_pos)
        gui.update_status()

    def on_radius_change(new_radius: float) -> None:
        if sphere_visuals._selected_key is not None:
            parts = sphere_visuals._selected_key.rsplit("_", 1)
            link_name, idx = parts[0], int(parts[1])
            data_store.update_sphere_radius(link_name, idx, new_radius)
            sphere_visuals.update_selected_radius(new_radius)
            gui.update_status()

    def on_delete() -> None:
        if sphere_visuals._selected_key is not None:
            parts = sphere_visuals._selected_key.rsplit("_", 1)
            link_name, idx = parts[0], int(parts[1])
            # Deselect first, then delete
            sphere_visuals.deselect()
            gui.hide_selection_info()
            data_store.delete_sphere(link_name, idx)
            # Rebuild visuals to reflect deletion
            sphere_visuals.rebuild(gui.opacity, gui.show_spheres)
            gui.update_status()

    gui.set_on_save(on_save)
    gui.set_on_deselect(on_deselect)
    gui.set_on_delete(on_delete)
    gui.set_on_radius_change(on_radius_change)
    sphere_visuals.set_on_select(on_select)
    sphere_visuals.set_on_position_change(on_position_change)
    sphere_visuals.set_on_deselect(gui.hide_selection_info)

    # Initial render
    sphere_visuals.rebuild(gui.opacity, gui.show_spheres)

    print(f"\nEditor ready at http://localhost:{port}")
    print("Click on a sphere to select it, then drag to move or adjust radius.")
    print("Press 'Save' to write changes back to JSON.\n")

    while True:
        # Check for visual changes
        if gui.show_spheres != last_show or gui.opacity != last_opacity:
            last_show = gui.show_spheres
            last_opacity = gui.opacity
            sphere_visuals.rebuild(last_opacity, last_show)

        # Update robot pose
        cfg = gui.joint_config
        urdf_vis.update_cfg(cfg)

        # Transform spheres
        if gui.show_spheres:
            Ts = robot.compute_transforms(cfg)
            sphere_visuals.update_transforms(Ts)

        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)
