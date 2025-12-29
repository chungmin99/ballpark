"""Robot collision mesh extraction and sphere generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._spherize import Sphere, spherize, SpherizeConfig
from ._similarity import SimilarityResult, detect_similar_links
from .utils._urdf_utils import (
    get_collision_mesh_for_link,
    get_joint_limits,
    get_link_names,
    get_link_transforms,
    link_has_collision,
)


@dataclass
class RobotSpheresResult:
    """Result from robot spherization."""

    link_spheres: dict[str, list[Sphere]]

    @property
    def num_spheres(self) -> int:
        """Total number of spheres across all links."""
        return sum(len(spheres) for spheres in self.link_spheres.values())

    def save_json(self, path: Path) -> None:
        """Save spheres to JSON file."""
        import json

        data = {
            link_name: {
                "centers": [s.center.tolist() for s in spheres],
                "radii": [s.radius for s in spheres],
            }
            for link_name, spheres in self.link_spheres.items()
        }
        with path.open(mode="w") as f:
            json.dump(data, f, indent=2)


class Robot:
    """Robot collision geometry analysis and sphere generation."""

    def __init__(self, urdf):
        """
        Initialize robot from URDF.

        Args:
            urdf: yourdfpy URDF object with collision meshes loaded
        """
        self.urdf = urdf

        # Compute links with collision geometry
        self._links = [
            link_name
            for link_name in urdf.link_map.keys()
            if link_has_collision(urdf, link_name)
        ]

        # Compute similarity
        self._similarity = detect_similar_links(urdf, self._links, verbose=False)

    @property
    def collision_links(self) -> list[str]:
        """Links with collision geometry."""
        return self._links

    @property
    def links(self) -> list[str]:
        """All link names in the URDF."""
        return get_link_names(self.urdf)

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint limits as (lower, upper) arrays."""
        return get_joint_limits(self.urdf)

    def compute_transforms(self, cfg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for all links.

        Args:
            cfg: Joint configuration array

        Returns:
            (num_links, 7) array where each row is [qw, qx, qy, qz, x, y, z]
        """
        return get_link_transforms(self.urdf, cfg)

    def auto_allocate(
        self,
        target_spheres: int,
        min_per_link: int = 1,
    ) -> dict[str, int]:
        """
        Automatically allocate sphere budget across links.

        Allocates proportionally based on geometry complexity. Similar
        links share allocations - only the primary link in each
        similarity group gets spheres allocated, and secondary links
        will reuse them.

        Args:
            target_spheres: Total number of spheres
            min_per_link: Minimum spheres per link

        Returns:
            Dict mapping link names to sphere counts
        """
        # Build set of secondary links (those that will reuse from primary)
        secondary_links = set()
        for group in self._similarity.groups:
            for link in group[1:]:  # All but first (primary)
                secondary_links.add(link)

        # Only allocate to primary links
        primary_links = [l for l in self._links if l not in secondary_links]

        allocation = _allocate_spheres_for_robot(
            self.urdf,
            primary_links,
            target_spheres=target_spheres,
            min_per_link=min_per_link,
        )

        # Secondary links get same allocation as their primary
        for group in self._similarity.groups:
            primary = group[0]
            for secondary in group[1:]:
                allocation[secondary] = allocation.get(primary, min_per_link)

        return allocation

    def spherize(
        self,
        target_spheres: int | None = None,
        allocation: dict[str, int] | None = None,
    ) -> RobotSpheresResult:
        """
        Generate spheres for the robot.

        Args:
            target_spheres: Total spheres (auto-allocates). Mutually exclusive with allocation.
            allocation: Explicit per-link allocation. Mutually exclusive with target_spheres.

        Returns:
            RobotSpheresResult with link_spheres

        Raises:
            ValueError: If neither or both of target_spheres and allocation are provided.
        """
        if (target_spheres is None) == (allocation is None):
            raise ValueError("Provide exactly one of target_spheres or allocation")

        if allocation is None:
            assert target_spheres is not None
            allocation = self.auto_allocate(target_spheres)

        return _compute_spheres_for_robot(
            self.urdf,
            self._links,
            link_budgets=allocation,
            similarity_result=self._similarity,
        )


def _allocate_spheres_for_robot(
    urdf,
    links: list[str],
    target_spheres: int,
    min_per_link: int = 1,
) -> dict[str, int]:
    """
    Allocate sphere budget across links based on geometry complexity.

    Uses "sphere inefficiency" as weight - how poorly each link is
    approximated by a single bounding sphere. Complex/elongated shapes
    get more spheres.

    Args:
        urdf: yourdfpy URDF object
        links: List of link names to allocate for
        target_spheres: Total number of spheres to allocate
        min_per_link: Minimum spheres per link

    Returns:
        Dict mapping link names to sphere counts
    """
    # Compute weights based on sphere inefficiency
    weights = {}
    for link_name in links:
        mesh = get_collision_mesh_for_link(urdf, link_name)
        if not mesh.is_empty:
            # Bounding sphere radius (half of bbox diagonal)
            bbox_diag = np.linalg.norm(mesh.extents)
            bounding_sphere_radius = bbox_diag / 2
            bounding_sphere_vol = 4 / 3 * np.pi * bounding_sphere_radius**3

            # Mesh volume (use convex hull for robustness)
            try:
                mesh_vol = mesh.convex_hull.volume
            except Exception:
                mesh_vol = mesh.bounding_box.volume

            # Inefficiency = how much the bounding sphere over-approximates
            # Higher inefficiency = needs more spheres to get tight fit
            inefficiency = bounding_sphere_vol / (mesh_vol + 1e-10)
            weights[link_name] = min(inefficiency, 20.0)  # Cap extreme values
        else:
            weights[link_name] = 1.0

    total_weight = sum(weights.values())
    if total_weight <= 0:
        per_link = max(1, target_spheres // len(links))
        return {link: per_link for link in links}

    # Allocate proportionally to inefficiency
    allocation = {}
    for link_name in links:
        frac = weights[link_name] / total_weight
        allocation[link_name] = max(min_per_link, round(target_spheres * frac))

    # Adjust if over budget (subtract from largest allocations)
    while sum(allocation.values()) > target_spheres:
        max_link = max(allocation, key=lambda k: allocation[k])
        if allocation[max_link] > min_per_link:
            allocation[max_link] -= 1
        else:
            break

    return allocation


def _compute_spheres_for_robot(
    urdf,
    links: list[str],
    link_budgets: dict[str, int],
    similarity_result: SimilarityResult | None = None,
) -> RobotSpheresResult:
    """
    Compute spheres for all links.

    Args:
        urdf: yourdfpy URDF object
        links: List of link names
        link_budgets: Dict mapping link names to sphere counts
        similarity_result: Optional similarity info for reusing spheres

    Returns:
        RobotSpheresResult with link_spheres
    """
    link_spheres: dict[str, list[Sphere]] = {}

    # Build map of which links can reuse spheres from others
    reuse_from: dict[str, str] = {}
    if similarity_result is not None:
        for group in similarity_result.groups:
            primary = group[0]
            for other in group[1:]:
                reuse_from[other] = primary

    # Spherize each link
    for link_name in links:
        budget = link_budgets.get(link_name, 1)

        # Check if we can reuse from a similar link
        if link_name in reuse_from and similarity_result is not None:
            primary = reuse_from[link_name]
            if primary in link_spheres:
                # Transform spheres from primary to this link
                transform = similarity_result.transforms.get((primary, link_name))
                if transform is not None:
                    link_spheres[link_name] = _transform_spheres(
                        link_spheres[primary], transform
                    )
                    continue

        # Spherize this link
        mesh = get_collision_mesh_for_link(urdf, link_name)
        if mesh.is_empty:
            link_spheres[link_name] = []
            continue

        cfg = SpherizeConfig(
            target_tightness=1.2,
            aspect_threshold=1.3,
            target_spheres=budget,
            n_samples=5000,
            padding=1.02,
            percentile=98.0,
            max_radius_ratio=0.5,
            uniform_radius=False,
        )
        link_spheres[link_name] = spherize(mesh, cfg)

    return RobotSpheresResult(link_spheres=link_spheres)


def _transform_spheres(spheres: list[Sphere], transform: np.ndarray) -> list[Sphere]:
    """Apply 4x4 transform to sphere centers."""
    result = []
    for s in spheres:
        center_h = np.append(s.center, 1.0)
        new_center = (transform @ center_h)[:3]
        result.append(Sphere(center=new_center, radius=s.radius))
    return result
