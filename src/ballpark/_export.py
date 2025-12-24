"""Export sphere decomposition to JSON."""

from __future__ import annotations

import json
from pathlib import Path

from ._sphere import Sphere


def export_spheres_to_json(
    link_spheres: dict[str, list[Sphere]],
    output_path: str | Path,
    ignore_pairs: list[tuple[str, str]] | None = None,
) -> None:
    """
    Export sphere decomposition to JSON file.

    Args:
        link_spheres: Dict mapping link names to lists of Sphere objects
        output_path: Path to write JSON file
        ignore_pairs: Optional list of link pairs to ignore for collision checking.
            Each pair is a tuple of (link_a, link_b).

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

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
