#!/usr/bin/env python3
"""Headless sphere decomposition for robot URDFs.

Computes sphere decomposition for a robot URDF and exports to JSON.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yourdfpy
import tyro

import ballpark


@dataclass
class Args:
    """Compute sphere decomposition for a robot URDF.

    Examples:
        python scripts/spherize_robot.py --robot-name panda --preset conservative
        python scripts/spherize_robot.py --urdf my_robot.urdf --total-spheres 150
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
    preset: Literal["conservative", "balanced", "surface"] = "balanced"
    """Configuration preset."""

    total_spheres: int = 100
    """Total sphere budget across robot."""

    padding: float | None = None
    """Override padding value (default: from preset)."""

    # Refinement
    refine: bool = True
    """Enable per-link NLLS refinement."""

    self_collision: bool = False
    """Enable robot-level self-collision refinement (requires pyroki)."""

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

    # Load robot object if self-collision refinement is enabled
    robot = None
    if args.self_collision:
        try:
            import pyroki as pk

            # Need to load visual URDF for pyroki
            if args.robot_name:
                from robot_descriptions.loaders.yourdfpy import load_robot_description

                urdf_vis = load_robot_description(f"{args.robot_name}_description")
                robot = pk.Robot.from_urdf(urdf_vis)
            else:
                urdf_vis = yourdfpy.URDF.load(args.urdf)
                robot = pk.Robot.from_urdf(urdf_vis)
            log("Loaded robot kinematics for self-collision refinement")
        except ImportError:
            print(
                "Warning: pyroki is required for --self-collision. "
                "Proceeding without self-collision refinement.",
                file=sys.stderr,
            )
            args.self_collision = False

    # Build kwargs for compute_spheres_for_robot
    kwargs: dict = {
        "preset": args.preset,
        "refine": args.refine,
    }

    # Add padding override if specified
    if args.padding is not None:
        kwargs["padding"] = args.padding

    # Add self-collision options if enabled
    if args.self_collision and robot is not None:
        kwargs["robot"] = robot
        kwargs["refine_self_collision"] = True
        # Get initial joint config
        initial_cfg = (robot.joints.lower_limits + robot.joints.upper_limits) / 2
        kwargs["joint_cfg"] = initial_cfg

        # Pre-compute mesh distances
        log("Pre-computing mesh distances...")
        t0 = time.perf_counter()
        links_with_collision = [
            link_name
            for link_name in urdf.link_map.keys()
            if not ballpark.get_collision_mesh_for_link(urdf, link_name).is_empty
        ]
        non_contiguous_pairs = ballpark.get_non_contiguous_link_pairs(
            urdf, links_with_collision
        )
        mesh_distances = ballpark.compute_mesh_distances_batch(
            urdf,
            robot,
            non_contiguous_pairs,
            n_samples=1000,
            joint_cfg=initial_cfg,
        )
        kwargs["mesh_distances"] = mesh_distances
        elapsed_ms = (time.perf_counter() - t0) * 1000
        log(f"Cached {len(mesh_distances)} mesh distances in {elapsed_ms:.1f}ms")

    # Compute spheres
    log(
        f"Computing spheres (preset={args.preset}, total={args.total_spheres}, "
        f"refine={args.refine}, self_collision={args.self_collision})..."
    )
    t0 = time.perf_counter()
    result = ballpark.compute_spheres_for_robot(
        urdf,
        total_spheres=args.total_spheres,
        **kwargs,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    total_spheres_count = sum(len(s) for s in result.link_spheres.values())
    log(f"Generated {total_spheres_count} spheres in {elapsed_ms:.1f}ms")

    # Export to JSON
    ballpark.export_spheres_to_json(
        link_spheres=result.link_spheres,
        output_path=args.output,
        ignore_pairs=result.ignore_pairs,
    )
    log(f"Exported to {args.output}")

    if not args.quiet:
        # Print summary
        print("\nPer-link sphere counts:")
        for link_name, spheres in sorted(result.link_spheres.items()):
            if spheres:
                print(f"  {link_name}: {len(spheres)}")


if __name__ == "__main__":
    tyro.cli(main)
