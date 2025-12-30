#!/usr/bin/env python3
"""Visualize spheres from a JSON file exported by ballpark.

Usage:
    python scripts/visualize_spheres.py spheres.json
    python scripts/visualize_spheres.py spheres.json --opacity 0.5
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import tyro
import viser

from ballpark import SPHERE_COLORS


def main(
    json_path: Path,
    opacity: float = 0.9,
    port: int = 8080,
) -> None:
    """Visualize spheres from a JSON file.

    Args:
        json_path: Path to the JSON file containing sphere data.
        opacity: Sphere opacity (0.1 to 1.0).
        port: Port for the viser web server.
    """
    # Load sphere data
    with open(json_path) as f:
        data = json.load(f)

    # Detect format: flat {centers, radii} vs grouped {link: {centers, radii}, ...}
    is_robot_format = "centers" not in data or "radii" not in data
    if not is_robot_format:
        # Flat format (single mesh) - wrap in a group
        data = {"mesh": data}

    # Count spheres
    total = sum(len(v["radii"]) for v in data.values())
    print(f"Loaded {total} spheres across {len(data)} groups from {json_path}")

    if is_robot_format:
        print(
            "\n⚠️  WARNING: This appears to be a robot sphere export.\n"
            "   Spheres are in link-local frames and will not appear in correct poses.\n"
            "   For proper visualization, use: python scripts/view_robot_spheres.py\n"
        )

    # Start viser server
    server = viser.ViserServer(port=port)
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    if is_robot_format:
        server.gui.add_markdown(
            "⚠️ **Robot format detected**\n\n"
            "Spheres are in link-local frames and may not appear in correct poses. "
            "Use `view_robot_spheres.py` for proper visualization."
        )

    # Add spheres
    for group_idx, (group_name, group_data) in enumerate(data.items()):
        centers = group_data["centers"]
        radii = group_data["radii"]
        color = SPHERE_COLORS[group_idx % len(SPHERE_COLORS)]
        rgb = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

        for i, (center, radius) in enumerate(zip(centers, radii)):
            server.scene.add_icosphere(
                f"/spheres/{group_name}/{i}",
                radius=float(radius),
                position=(float(center[0]), float(center[1]), float(center[2])),
                color=rgb,
                opacity=opacity,
            )

    print(f"Visualization ready at http://localhost:{port}")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    tyro.cli(main)
