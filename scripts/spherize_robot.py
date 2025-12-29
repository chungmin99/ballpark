#!/usr/bin/env python3
"""Headless sphere decomposition for robot URDFs.

Computes sphere decomposition for a robot URDF and exports to JSON.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass

import yourdfpy
import tyro

import ballpark


@dataclass
class Args:
    """Compute sphere decomposition for a robot URDF.

    Examples:
        python scripts/spherize_robot.py --robot-name panda
        python scripts/spherize_robot.py --urdf my_robot.urdf --target-spheres 150
    """

    # Input (one required)
    urdf: str | None = None
    """Path to URDF file."""

    robot_name: str | None = None
    """Robot name from robot_descriptions (e.g., panda, ur5, iiwa14)."""

    # Output
    output: str = "spheres.json"
    """Output JSON file path."""

    # Config
    target_spheres: int = 100
    """Target sphere count across robot (may slightly exceed)."""

    # Output control
    quiet: bool = False
    """Suppress progress output."""


def load_urdf(urdf_path: str | None, robot_name: str | None) -> yourdfpy.URDF:
    """Load URDF from path or robot_descriptions."""
    if robot_name:
        try:
            from robot_descriptions.loaders.yourdfpy import load_robot_description

            urdf = load_robot_description(f"{robot_name}_description")
            # Reload with collision meshes
            return yourdfpy.URDF(
                robot=urdf.robot,
                filename_handler=urdf._filename_handler,
                load_collision_meshes=True,
            )
        except ImportError:
            print(
                "Error: robot_descriptions is required for --robot-name. "
                "Install with: pip install robot_descriptions",
                file=sys.stderr,
            )
            sys.exit(1)
        except Exception as e:
            print(f"Error loading robot '{robot_name}': {e}", file=sys.stderr)
            sys.exit(1)
    elif urdf_path:
        try:
            return yourdfpy.URDF.load(urdf_path, load_collision_meshes=True)
        except Exception as e:
            print(f"Error loading URDF '{urdf_path}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print("Error: Must specify either --urdf or --robot-name", file=sys.stderr)
        sys.exit(1)


def export_to_json(result: ballpark._robot.RobotSpheresResult, output_path: str) -> None:
    """Export sphere decomposition to JSON file."""
    data = {
        "spheres": {
            link_name: {
                "centers": [sphere.center.tolist() for sphere in spheres],
                "radii": [sphere.radius for sphere in spheres],
            }
            for link_name, spheres in result.link_spheres.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def main(args: Args) -> None:
    """Run sphere decomposition."""

    def log(msg: str) -> None:
        if not args.quiet:
            print(msg)

    # Validate input
    if args.urdf is None and args.robot_name is None:
        print("Error: Must specify either --urdf or --robot-name", file=sys.stderr)
        sys.exit(1)
    if args.urdf is not None and args.robot_name is not None:
        print("Error: Cannot specify both --urdf and --robot-name", file=sys.stderr)
        sys.exit(1)

    # Load URDF
    log("Loading URDF...")
    urdf = load_urdf(args.urdf, args.robot_name)

    # Create Robot instance
    log("Analyzing robot structure...")
    t0 = time.perf_counter()
    robot = ballpark.Robot(urdf)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    log(f"Found {len(robot.links)} links with collision geometry in {elapsed_ms:.1f}ms")

    # Compute spheres
    log(f"Computing spheres (target={args.target_spheres})...")
    t0 = time.perf_counter()
    result = robot.spherize(target_spheres=args.target_spheres)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    total_spheres_count = sum(len(s) for s in result.link_spheres.values())
    log(f"Generated {total_spheres_count} spheres in {elapsed_ms:.1f}ms")

    # Export to JSON
    export_to_json(result, args.output)
    log(f"Exported to {args.output}")

    if not args.quiet:
        # Print summary
        print("\nPer-link sphere counts:")
        for link_name, spheres in sorted(result.link_spheres.items()):
            if spheres:
                print(f"  {link_name}: {len(spheres)}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
