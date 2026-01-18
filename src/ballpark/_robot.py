"""Robot collision mesh extraction and sphere generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jaxlie
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from loguru import logger

from ._config import BallparkConfig, SpherePreset, SpherizeParams
from ._spherize import Sphere, spherize
from ._similarity import SimilarityResult, detect_similar_links
from ._refine import refine_robot_spheres, refine_robot_ellipsoids
from ._primitives import Ellipsoid
from .utils._hash_geometry import get_link_collision_fingerprint


@dataclass
class RobotSpheresResult:
    """Result from robot spherization."""

    link_spheres: dict[str, list[Sphere]]

    @property
    def num_spheres(self) -> int:
        """Total number of spheres across all links."""
        return sum(len(spheres) for spheres in self.link_spheres.values())

    @property
    def n_primitives(self) -> int:
        """Total number of primitives (spheres)."""
        return self.num_spheres

    def save_json(self, path: Path) -> None:
        """Save spheres to JSON file."""
        import json

        data = {
            link_name: {
                "centers": [s.center.tolist() for s in spheres],
                "radii": [float(s.radius) for s in spheres],
            }
            for link_name, spheres in self.link_spheres.items()
        }
        with path.open(mode="w") as f:
            json.dump(data, f, indent=2)


@dataclass
class RobotEllipsoidsResult:
    """Result from robot ellipsoid refinement."""

    link_ellipsoids: dict[str, list[Ellipsoid]]

    @property
    def num_ellipsoids(self) -> int:
        """Total number of ellipsoids across all links."""
        return sum(len(ellipsoids) for ellipsoids in self.link_ellipsoids.values())

    @property
    def n_primitives(self) -> int:
        """Total number of primitives (ellipsoids)."""
        return self.num_ellipsoids

    def save_json(self, path: Path) -> None:
        """Save ellipsoids to JSON file."""
        import json

        data = {
            link_name: {
                "centers": [e.center.tolist() for e in ellipsoids],
                "semi_axes": [e.semi_axes.tolist() for e in ellipsoids],
            }
            for link_name, ellipsoids in self.link_ellipsoids.items()
        }
        with path.open(mode="w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_json(path: Path) -> "RobotEllipsoidsResult":
        """Load ellipsoids from JSON file."""
        import json
        import jax.numpy as jnp

        with path.open(mode="r") as f:
            data = json.load(f)

        link_ellipsoids = {}
        for link_name, ellipsoid_data in data.items():
            centers = ellipsoid_data["centers"]
            semi_axes = ellipsoid_data["semi_axes"]
            link_ellipsoids[link_name] = [
                Ellipsoid(center=jnp.array(c), semi_axes=jnp.array(s))
                for c, s in zip(centers, semi_axes)
            ]
        return RobotEllipsoidsResult(link_ellipsoids=link_ellipsoids)


class Robot:
    """Robot collision geometry analysis and sphere generation."""

    def __init__(self, urdf):
        """
        Initialize robot from URDF.

        Args:
            urdf: yourdfpy URDF object with collision meshes loaded
        """
        self._urdf = urdf

        # Cache all link names (ordered)
        self._all_link_names: list[str] = list(urdf.link_map.keys())

        # Compute links with collision geometry
        self._links = [
            link_name
            for link_name in self._all_link_names
            if self._link_has_collision(link_name)
        ]

        # Cache collision meshes for all collision links
        self._link_meshes: dict[str, trimesh.Trimesh] = {
            link_name: self._get_collision_mesh_for_link(link_name)
            for link_name in self._links
        }

        # Cache collision fingerprints for similarity detection
        self._link_fingerprints: dict[str, tuple] = {
            link_name: get_link_collision_fingerprint(urdf, link_name)
            for link_name in self._links
        }

        # Compute non-contiguous pairs for self-collision checking
        self._non_contiguous_pairs = self._get_non_contiguous_link_pairs(self._links)

        # Cached mesh distances for self-collision filtering (computed lazily)
        self._cached_mesh_distances: dict[tuple[str, str], float] | None = None

        # Compute similarity using cached data
        self._similarity = detect_similar_links(
            self._link_meshes, self._link_fingerprints
        )

        # Log robot summary
        logger.info(
            f"Robot: {len(self._links)} collision links, "
            f"{len(self._similarity.groups)} similarity group(s)"
        )

    # =========================================================================
    # Public properties
    # =========================================================================

    @property
    def collision_links(self) -> list[str]:
        """Links with collision geometry."""
        return self._links

    @property
    def links(self) -> list[str]:
        """All link names in the URDF."""
        return self._all_link_names

    @property
    def joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Joint limits as (lower, upper) arrays."""
        return self._get_joint_limits()

    @property
    def non_contiguous_pairs(self) -> list[tuple[str, str]]:
        """Link pairs that are not adjacent in the kinematic chain."""
        return self._non_contiguous_pairs

    def get_mesh_distances(
        self, joint_cfg: np.ndarray | None = None
    ) -> dict[tuple[str, str], float]:
        """
        Get mesh distances between non-contiguous link pairs.

        Args:
            joint_cfg: Joint configuration for FK. If None, uses middle of limits
                and returns cached values.

        Returns:
            Dict mapping (link_a, link_b) to minimum mesh distance.
        """
        if joint_cfg is None:
            return self._get_mesh_distances()
        # Compute for specific config (not cached)
        from ballpark.utils._mesh_utils import compute_mesh_distances_batch

        return compute_mesh_distances_batch(
            self._link_meshes,
            self._non_contiguous_pairs,
            self._all_link_names,
            self.joint_limits,
            self.compute_transforms,
            n_samples=1000,
            bbox_skip_threshold=0.1,
            joint_cfg=joint_cfg,
        )

    def compute_transforms(self, cfg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for all links.

        Args:
            cfg: Joint configuration array

        Returns:
            (num_links, 7) array where each row is [qw, qx, qy, qz, x, y, z]
        """
        return self._get_link_transforms(cfg)

    # =========================================================================
    # Private URDF utility methods (inlined from _urdf_utils)
    # =========================================================================

    def _link_has_collision(self, link_name: str) -> bool:
        """Check if a link has collision geometry."""
        if link_name not in self._urdf.link_map:
            return False
        return len(self._urdf.link_map[link_name].collisions) > 0

    def _get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Get joint limits from URDF."""
        lower = []
        upper = []
        for jname in self._urdf.actuated_joint_names:
            joint = self._urdf.joint_map[jname]
            lower.append(joint.limit.lower if joint.limit else -np.pi)
            upper.append(joint.limit.upper if joint.limit else np.pi)
        return np.array(lower), np.array(upper)

    def _get_link_transforms(self, joint_cfg: np.ndarray) -> np.ndarray:
        """Compute forward kinematics for all links."""
        # Update URDF configuration (does FK internally)
        self._urdf.update_cfg(joint_cfg)

        # Get transforms for all links
        transforms = np.zeros((len(self._all_link_names), 7))

        for i, link_name in enumerate(self._all_link_names):
            # Get 4x4 homogeneous transform
            T = self._urdf.get_transform(link_name)

            # Extract rotation matrix and convert to quaternion (wxyz format)
            rot_matrix = T[:3, :3]
            quat_xyzw = Rotation.from_matrix(rot_matrix).as_quat()  # scipy uses xyzw
            quat_wxyz = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )

            # Extract translation
            translation = T[:3, 3]

            transforms[i, :4] = quat_wxyz
            transforms[i, 4:] = translation

        return transforms

    def _get_adjacent_links(self) -> set[tuple[str, str]]:
        """Build adjacency set from joint parent-child relationships."""
        adjacent = set()
        for joint in self._urdf.robot.joints:
            pair = tuple(sorted([joint.parent, joint.child]))
            adjacent.add(pair)
        return adjacent

    def _get_non_contiguous_link_pairs(
        self, link_names: list[str]
    ) -> list[tuple[str, str]]:
        """Get all link pairs that are NOT adjacent (for self-collision checking)."""
        adjacent = self._get_adjacent_links()
        pairs = []
        for i, link_a in enumerate(link_names):
            for link_b in link_names[i + 1 :]:
                if tuple(sorted([link_a, link_b])) not in adjacent:
                    pairs.append((link_a, link_b))
        return pairs

    def _get_mesh_distances(self) -> dict[tuple[str, str], float]:
        """Get cached mesh distances for non-contiguous link pairs.

        Computed lazily on first call using FK at middle of joint limits.
        """
        if self._cached_mesh_distances is None:
            from .utils._mesh_utils import compute_mesh_distances_batch

            lower, upper = self.joint_limits
            q_mid = (lower + upper) / 2

            self._cached_mesh_distances = compute_mesh_distances_batch(
                self._link_meshes,
                self._non_contiguous_pairs,
                self._all_link_names,
                self.joint_limits,
                self.compute_transforms,
                n_samples=1000,
                bbox_skip_threshold=0.1,
                joint_cfg=q_mid,
            )
        return self._cached_mesh_distances

    def _get_collision_mesh_for_link(self, link_name: str) -> trimesh.Trimesh:
        """Extract collision mesh for a given link from URDF."""
        if link_name not in self._urdf.link_map:
            return trimesh.Trimesh()

        link = self._urdf.link_map[link_name]
        coll_meshes = []

        for collision in link.collisions:
            geom = collision.geometry
            mesh = None

            if collision.origin is not None:
                transform = collision.origin
            else:
                transform = np.eye(4)

            if geom.box is not None:
                mesh = trimesh.creation.box(extents=geom.box.size)
            elif geom.cylinder is not None:
                mesh = trimesh.creation.cylinder(
                    radius=geom.cylinder.radius, height=geom.cylinder.length
                )
            elif geom.sphere is not None:
                mesh = trimesh.creation.icosphere(radius=geom.sphere.radius)
            elif geom.mesh is not None:
                mesh_path = geom.mesh.filename
                # Resolve package:// URLs using URDF's filename handler
                if (
                    hasattr(self._urdf, "_filename_handler")
                    and self._urdf._filename_handler is not None
                ):
                    mesh_path = self._urdf._filename_handler(mesh_path)
                try:
                    loaded_obj = trimesh.load(
                        mesh_path,
                        force="mesh",
                        process=False,
                    )
                    if isinstance(loaded_obj, trimesh.Scene):
                        mesh = loaded_obj.dump(concatenate=True)
                    else:
                        mesh = loaded_obj

                    # Ensure mesh is a Trimesh (not a list)
                    if not isinstance(mesh, trimesh.Trimesh):
                        logger.warning(
                            f"Unexpected mesh type from {mesh_path}: {type(mesh)}"
                        )
                        continue

                    if geom.mesh.scale is not None:
                        scale = np.asarray(geom.mesh.scale)
                        mesh.apply_scale(scale)
                except Exception as e:
                    logger.warning(f"Failed to load mesh {mesh_path}: {e}")
                    continue

            if mesh is not None:
                mesh.apply_transform(transform)
                coll_meshes.append(mesh)

        if not coll_meshes:
            return trimesh.Trimesh()

        return trimesh.util.concatenate(coll_meshes)

    # =========================================================================
    # Public methods
    # =========================================================================

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

        The allocation accounts for similarity multipliers: if a primary
        link has N secondary copies, each sphere allocated to it counts
        (1 + N) times in the final output.

        Args:
            target_spheres: Total number of spheres in final output
            min_per_link: Minimum spheres per link

        Returns:
            Dict mapping link names to sphere counts
        """
        # Build set of secondary links and compute multipliers for primaries
        secondary_links = set()
        primary_multiplier: dict[
            str, int
        ] = {}  # primary -> count of copies (including itself)

        for group in self._similarity.groups:
            primary = group[0]
            primary_multiplier[primary] = len(group)  # primary + all secondaries
            for link in group[1:]:
                secondary_links.add(link)

        # Only allocate to primary links
        primary_links = [l for l in self._links if l not in secondary_links]

        # For links not in any similarity group, multiplier is 1
        for link in primary_links:
            if link not in primary_multiplier:
                primary_multiplier[link] = 1

        allocation = _allocate_spheres_for_robot(
            self._link_meshes,
            primary_links,
            target_spheres=target_spheres,
            min_per_link=min_per_link,
            multipliers=primary_multiplier,
        )

        # Secondary links get same allocation as their primary
        for group in self._similarity.groups:
            primary = group[0]
            for secondary in group[1:]:
                allocation[secondary] = allocation.get(primary, min_per_link)

        return allocation

    def _sync_similar_allocations(self, allocation: dict[str, int]) -> dict[str, int]:
        """
        Sync allocations for similar links to the maximum count in each group.

        Args:
            allocation: Per-link sphere allocation

        Returns:
            New allocation dict with similar links synced to max count
        """
        synced = dict(allocation)

        for group in self._similarity.groups:
            # Get counts for links in this group that have allocations
            group_counts = {
                link: synced.get(link, 0) for link in group if link in synced
            }
            if len(group_counts) < 2:
                continue

            max_count = max(group_counts.values())
            links_to_sync = [
                link for link, count in group_counts.items() if count < max_count
            ]

            if links_to_sync:
                logger.warning(
                    f"Syncing similar links to max count {max_count}: "
                    f"{links_to_sync} (was {[group_counts[l] for l in links_to_sync]})"
                )
                for link in links_to_sync:
                    synced[link] = max_count

        return synced

    def spherize(
        self,
        target_spheres: int | None = None,
        allocation: dict[str, int] | None = None,
        config: BallparkConfig | None = None,
        sync_similar: bool = True,
    ) -> RobotSpheresResult:
        """
        Generate spheres for the robot.

        Args:
            target_spheres: Total spheres (auto-allocates). Mutually exclusive with allocation.
            allocation: Explicit per-link allocation. Mutually exclusive with target_spheres.
            config: Configuration for spherization. If None, uses BALANCED preset.
            sync_similar: If True and allocation is provided, sync similar links to the
                highest count in each similarity group (with warning). Default True.

        Returns:
            RobotSpheresResult with link_spheres

        Raises:
            ValueError: If neither or both of target_spheres and allocation are provided.
        """
        if (target_spheres is None) == (allocation is None):
            raise ValueError("Provide exactly one of target_spheres or allocation")

        cfg = config or BallparkConfig.from_preset(SpherePreset.BALANCED)

        if allocation is None:
            assert target_spheres is not None
            allocation = self.auto_allocate(target_spheres)
        elif sync_similar:
            allocation = self._sync_similar_allocations(allocation)

        return _compute_spheres_for_robot(
            self._link_meshes,
            self._links,
            link_budgets=allocation,
            similarity_result=self._similarity,
            spherize_params=cfg.spherize,
        )

    def refine(
        self,
        spheres_result: RobotSpheresResult,
        config: BallparkConfig | None = None,
        joint_cfg: np.ndarray | None = None,
        excluded_pairs: set[tuple[str, str]] | None = None,
    ) -> RobotSpheresResult:
        """
        Refine sphere parameters using gradient-based optimization.

        Jointly optimizes all spheres with:
        - Per-link losses: coverage, volume, overlap, uniformity
        - Robot-level losses: self-collision avoidance, similarity matching

        Args:
            spheres_result: Result from robot.spherize()
            config: Configuration for refinement. If None, uses BALANCED preset.
            joint_cfg: Joint configuration for FK and mesh distances. If None,
                uses middle of joint limits with cached mesh distances.
            excluded_pairs: Link pairs to skip for collision checking.

        Returns:
            RobotSpheresResult with refined spheres
        """
        cfg = config or BallparkConfig.from_preset(SpherePreset.BALANCED)

        refined_link_spheres = refine_robot_spheres(
            spheres_result.link_spheres,
            self._link_meshes,
            self._all_link_names,
            self.joint_limits,
            self.compute_transforms,
            self._non_contiguous_pairs,
            refine_params=cfg.refine,
            joint_cfg=joint_cfg,
            excluded_pairs=excluded_pairs,
        )
        return RobotSpheresResult(link_spheres=refined_link_spheres)

    def refine_ellipsoids(
        self,
        spheres_result: RobotSpheresResult,
        config: BallparkConfig | None = None,
        joint_cfg: np.ndarray | None = None,
        excluded_pairs: set[tuple[str, str]] | None = None,
    ) -> RobotEllipsoidsResult:
        """
        Refine spheres to ellipsoids using gradient-based optimization.

        Converts initial spheres to axis-aligned ellipsoids and optimizes
        allowing each primitive to stretch along axes for better geometry fit.

        Uses the effective radius approach for smooth, differentiable collision
        detection suitable for optimization.

        Args:
            spheres_result: Result from robot.spherize()
            config: Configuration for refinement. If None, uses BALANCED preset.
            joint_cfg: Joint configuration for FK and mesh distances. If None,
                uses middle of joint limits.
            excluded_pairs: Link pairs to skip for collision checking.

        Returns:
            RobotEllipsoidsResult with refined ellipsoids
        """
        cfg = config or BallparkConfig.from_preset(SpherePreset.BALANCED)

        refined_link_ellipsoids = refine_robot_ellipsoids(
            spheres_result.link_spheres,
            self._link_meshes,
            self._all_link_names,
            self.joint_limits,
            self.compute_transforms,
            self._non_contiguous_pairs,
            refine_params=cfg.refine,
            joint_cfg=joint_cfg,
            excluded_pairs=excluded_pairs,
        )
        return RobotEllipsoidsResult(link_ellipsoids=refined_link_ellipsoids)

    def check_self_collision(
        self,
        spheres_result: RobotSpheresResult,
        joint_cfg: np.ndarray | None = None,
    ) -> float:
        """
        Check self-collision between robot spheres.

        Computes the minimum signed distance between spheres of non-adjacent
        links. Negative values indicate penetration (collision).

        Args:
            spheres_result: Result from robot.spherize() or robot.refine()
            joint_cfg: Joint configuration for FK. If None, uses middle of limits.

        Returns:
            Minimum signed distance. Negative = collision, positive = clearance.
        """
        link_spheres = spheres_result.link_spheres
        links_with_spheres = [
            name for name in self._all_link_names if link_spheres.get(name)
        ]
        if not links_with_spheres:
            return float("inf")

        link_name_to_idx = {
            name: idx for idx, name in enumerate(self._all_link_names)
        }

        if joint_cfg is None:
            lower, upper = self.joint_limits
            joint_cfg = (lower + upper) / 2
        Ts = self.compute_transforms(joint_cfg)

        min_dist = float("inf")

        for link_a, link_b in self._non_contiguous_pairs:
            spheres_a = link_spheres.get(link_a, [])
            spheres_b = link_spheres.get(link_b, [])

            if not spheres_a or not spheres_b:
                continue

            idx_a = link_name_to_idx[link_a]
            idx_b = link_name_to_idx[link_b]

            T_a = Ts[idx_a]
            T_b = Ts[idx_b]

            wxyz_a, xyz_a = T_a[:4], T_a[4:]
            wxyz_b, xyz_b = T_b[:4], T_b[4:]

            so3_a = jaxlie.SO3(wxyz=wxyz_a)
            so3_b = jaxlie.SO3(wxyz=wxyz_b)

            for sphere_i in spheres_a:
                center_i_world = np.array(so3_a @ sphere_i.center) + xyz_a

                for sphere_j in spheres_b:
                    center_j_world = np.array(so3_b @ sphere_j.center) + xyz_b

                    dist = np.linalg.norm(center_i_world - center_j_world)
                    signed_dist = float(dist - (sphere_i.radius + sphere_j.radius))

                    min_dist = min(min_dist, signed_dist)

        return min_dist


def _allocate_spheres_for_robot(
    link_meshes: dict[str, trimesh.Trimesh],
    links: list[str],
    target_spheres: int,
    min_per_link: int = 1,
    multipliers: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Allocate sphere budget across links based on geometry complexity.

    Uses "sphere inefficiency" as weight - how poorly each link is
    approximated by a single bounding sphere. Complex/elongated shapes
    get more spheres.

    Args:
        link_meshes: Dict mapping link names to their collision meshes.
        links: List of link names to allocate for
        target_spheres: Total number of spheres in final output
        min_per_link: Minimum spheres per link
        multipliers: Dict mapping link names to their replication count
            (1 = no copies, 2 = one secondary copy, etc.)

    Returns:
        Dict mapping link names to sphere counts
    """
    if multipliers is None:
        multipliers = {link: 1 for link in links}

    # Compute weights based on sphere inefficiency
    weights = {}
    for link_name in links:
        mesh = link_meshes.get(link_name)
        if mesh is not None and not mesh.is_empty:
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

    # Compute effective budget: each sphere on a link counts multiplier times
    # We want: sum(allocation[link] * multiplier[link]) ≈ target_spheres
    # So we allocate based on weight / multiplier to balance it out
    total_weighted = sum(weights[link] / multipliers.get(link, 1) for link in links)

    allocation = {}
    for link_name in links:
        mult = multipliers.get(link_name, 1)
        # Weight adjusted by multiplier: high-multiplier links get fewer spheres
        adjusted_weight = weights[link_name] / mult
        frac = adjusted_weight / total_weighted
        allocation[link_name] = max(min_per_link, round(target_spheres * frac / mult))

    # Compute effective total (accounting for multipliers)
    def effective_total():
        return sum(allocation[link] * multipliers.get(link, 1) for link in links)

    # Adjust if over budget (subtract from largest allocations, preferring high-multiplier links)
    while effective_total() > target_spheres:
        # Prefer reducing links with high multipliers (more impact per reduction)
        candidates = [k for k in allocation if allocation[k] > min_per_link]
        if not candidates:
            break
        # Sort by multiplier descending, then by allocation descending
        candidates.sort(key=lambda k: (-multipliers.get(k, 1), -allocation[k]))
        allocation[candidates[0]] -= 1

    return allocation


def _compute_spheres_for_robot(
    link_meshes: dict[str, trimesh.Trimesh],
    links: list[str],
    link_budgets: dict[str, int],
    similarity_result: SimilarityResult | None = None,
    spherize_params: SpherizeParams | None = None,
) -> RobotSpheresResult:
    """
    Compute spheres for all links.

    Args:
        link_meshes: Dict mapping link names to their collision meshes.
        links: List of link names
        link_budgets: Dict mapping link names to sphere counts
        similarity_result: Optional similarity info for reusing spheres
        spherize_params: Parameters for the spherization algorithm

    Returns:
        RobotSpheresResult with link_spheres
    """
    params = spherize_params or SpherizeParams()
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
        mesh = link_meshes.get(link_name)
        if mesh is None or mesh.is_empty:
            link_spheres[link_name] = []
            continue

        link_spheres[link_name] = spherize(mesh, budget, params)

    return RobotSpheresResult(link_spheres=link_spheres)


def _transform_spheres(spheres: list[Sphere], transform: np.ndarray) -> list[Sphere]:
    """Apply 4x4 transform to sphere centers."""
    import jax.numpy as jnp

    result = []
    for s in spheres:
        center_h = np.append(s.center, 1.0)
        new_center = (transform @ center_h)[:3]
        result.append(Sphere(center=jnp.asarray(new_center), radius=s.radius))
    return result
