"""Data classes for various results in Ballpark."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json


@dataclass
class Sphere:
    """A sphere defined by center and radius."""

    center: np.ndarray
    radius: float


@dataclass
class SpherizedRobot:
    """A robot described in terms of spheres."""

    link_spheres: dict[str, list[Sphere]]
    ignore_pairs: list[tuple[str, str]] | None  # None if refine_self_collision=False

    def export_spheres_to_json(
        self,
        link_spheres: dict[str, list[Sphere]],
        output_path: Path,
        ignore_pairs: list[tuple[str, str]] | None = None,
    ) -> None:
        """
        Export sphere decomposition to JSON file.

        Args:
            link_spheres: Dict mapping link names to lists of Sphere objects
            output_path: Path to write JSON file (must have .json extension)
            ignore_pairs: Optional list of link pairs to ignore for collision checking.
                Each pair is a tuple of (link_a, link_b).

        Raises:
            ValueError: If output_path doesn't have .json extension

        Output format:
            {
                "spheres": {
                    "<link_name>": {
                        "centers": [[x, y, z], ...],
                        "radii": [r1, r2, ...]
                    },
                    ...
                },
                "ignore_pairs": [["link_a", "link_b"], ...]  # if provided
            }
        """
        # Validate output path
        resolved_path = Path(output_path).resolve()
        if resolved_path.suffix.lower() != ".json":
            raise ValueError(
                f"Output file must have .json extension, got: '{resolved_path.suffix}'"
            )

        data: dict = {
            "spheres": {
                link_name: {
                    "centers": [sphere.center.tolist() for sphere in spheres],
                    "radii": [sphere.radius for sphere in spheres],
                }
                for link_name, spheres in link_spheres.items()
            },
        }

        if ignore_pairs is not None:
            # Sort each pair alphabetically and sort the list for consistency
            data["ignore_pairs"] = sorted([sorted(pair) for pair in ignore_pairs])

        with open(resolved_path, "w") as f:
            json.dump(data, f, indent=2)
