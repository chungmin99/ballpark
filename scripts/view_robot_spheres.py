#!/usr/bin/env python3
"""View pre-exported robot spheres/ellipsoids with proper forward kinematics.

This script loads sphere or ellipsoid data from a JSON file and visualizes them
on a robot with correct link transforms. Use this to view primitives
exported from spherize_robot_interactive.py.

Usage:
    python scripts/view_robot_spheres.py --robot_name panda --json_path spheres.json
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import trimesh
import tyro
import viser
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf

from ballpark import Robot, SPHERE_COLORS


def main(
    robot_name: Literal["ur5", "panda", "yumi", "g1", "iiwa14", "gen2"],
    json_path: Path,
    port: int = 8080,
) -> None:
    """View robot spheres from a JSON file with forward kinematics.

    Args:
        robot_name: Name of the robot (must match the robot used to generate spheres).
        json_path: Path to the JSON file containing sphere data.
        port: Port for the viser web server.
    """
    # Load sphere data
    with open(json_path) as f:
        data = json.load(f)

    # Validate format
    if "centers" in data and ("radii" in data or "semi_axes" in data):
        print("Error: This appears to be a mesh sphere export, not a robot export.")
        print("Use: python scripts/visualize_spheres.py", json_path)
        return

    # Detect format: sphere (radii) vs ellipsoid (semi_axes)
    first_link_data = next(iter(data.values()), {})
    is_ellipsoid_format = "semi_axes" in first_link_data
    primitive_key = "semi_axes" if is_ellipsoid_format else "radii"
    primitive_name = "ellipsoids" if is_ellipsoid_format else "spheres"

    total = sum(len(v[primitive_key]) for v in data.values())
    print(f"Loaded {total} {primitive_name} across {len(data)} links from {json_path}")

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
    missing = set(data.keys()) - set(robot.links)
    if missing:
        print(f"Warning: JSON contains links not in robot: {missing}")

    # Set up viser
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Build GUI (viewer-only: no allocation/generation controls)
    with server.gui.add_folder("Visualization"):
        show_spheres = server.gui.add_checkbox("Show Spheres", initial_value=True)
        opacity_slider = server.gui.add_slider(
            "Opacity", min=0.1, max=1.0, step=0.1, initial_value=0.9
        )

    # Joint sliders
    lower, upper = robot.joint_limits
    joint_sliders = []
    with server.gui.add_folder("Joints"):
        for i in range(len(lower)):
            slider = server.gui.add_slider(
                f"Joint {i}",
                min=float(lower[i]),
                max=float(upper[i]),
                step=0.01,
                initial_value=(float(lower[i]) + float(upper[i])) / 2,
            )
            joint_sliders.append(slider)

    # Create primitive visuals (spheres or ellipsoids)
    link_names = robot.links
    frames: dict[str, viser.FrameHandle] = {}
    handles: dict[str, viser.IcosphereHandle | viser.MeshHandle] = {}

    def rebuild_primitives(opacity: float, visible: bool) -> None:
        # Clear existing
        for h in handles.values():
            h.remove()
        for f in frames.values():
            f.remove()
        handles.clear()
        frames.clear()

        if not visible:
            return

        for link_idx, link_name in enumerate(link_names):
            if link_name not in data:
                continue

            centers = data[link_name]["centers"]
            primitives = data[link_name][primitive_key]
            color = SPHERE_COLORS[link_idx % len(SPHERE_COLORS)]
            rgb = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

            for i, (center, prim) in enumerate(zip(centers, primitives)):
                key = f"{link_name}_{i}"
                frame = server.scene.add_frame(
                    f"/primitive_frames/{key}",
                    wxyz=(1, 0, 0, 0),
                    position=(0, 0, 0),
                    show_axes=False,
                )
                frames[key] = frame

                if is_ellipsoid_format:
                    # Ellipsoid: create scaled icosphere mesh
                    semi_axes = prim
                    sphere_mesh = trimesh.creation.icosphere(radius=1.0, subdivisions=2)
                    sphere_mesh.vertices = sphere_mesh.vertices * np.array([
                        float(semi_axes[0]),
                        float(semi_axes[1]),
                        float(semi_axes[2]),
                    ])
                    handles[key] = server.scene.add_mesh_simple(
                        f"/primitive_frames/{key}/ellipsoid",
                        vertices=sphere_mesh.vertices.astype(np.float32),
                        faces=sphere_mesh.faces.astype(np.uint32),
                        position=(float(center[0]), float(center[1]), float(center[2])),
                        color=rgb,
                        opacity=opacity,
                    )
                else:
                    # Sphere: use icosphere with single radius
                    radius = prim
                    handles[key] = server.scene.add_icosphere(
                        f"/primitive_frames/{key}/sphere",
                        radius=float(radius),
                        position=(float(center[0]), float(center[1]), float(center[2])),
                        color=rgb,
                        opacity=opacity,
                    )

    def update_transforms(cfg: np.ndarray) -> None:
        Ts = robot.compute_transforms(cfg)
        for link_idx, link_name in enumerate(link_names):
            if link_name not in data:
                continue
            T = Ts[link_idx]
            wxyz, pos = T[:4], T[4:]
            for i in range(len(data[link_name][primitive_key])):
                key = f"{link_name}_{i}"
                if key in frames:
                    frames[key].wxyz = wxyz
                    frames[key].position = pos

    # Initial render
    rebuild_primitives(opacity_slider.value, show_spheres.value)

    # Track state for change detection
    last_show = show_spheres.value
    last_opacity = opacity_slider.value

    print(f"Visualization ready at http://localhost:{port}")
    while True:
        # Check for visual changes
        if show_spheres.value != last_show or opacity_slider.value != last_opacity:
            last_show = show_spheres.value
            last_opacity = opacity_slider.value
            rebuild_primitives(last_opacity, last_show)

        # Update robot pose
        cfg = np.array([s.value for s in joint_sliders])
        urdf_vis.update_cfg(cfg)

        # Transform spheres
        if show_spheres.value:
            update_transforms(cfg)

        time.sleep(0.05)


if __name__ == "__main__":
    tyro.cli(main)
