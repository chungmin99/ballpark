"""Robot collision mesh extraction and sphere generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from loguru import logger

from ._spheres import Sphere
from ._spherize import spherize, SpherizeConfig
from ._similarity import SimilarityResult, detect_similar_links
from .utils._urdf_utils import (
    get_joint_limits,
    get_link_transforms,
    get_link_names,
)


@dataclass
class RobotSpheresResult:
    """Result from robot spherization."""

    link_spheres: dict[str, list[Sphere]]

    def total_spheres(self) -> int:
        """Total number of spheres across all links."""
        return sum(len(spheres) for spheres in self.link_spheres.values())


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
            if _link_has_collision(urdf, link_name)
        ]

        # Compute similarity
        self._similarity = detect_similar_links(urdf, self._links, verbose=False)

    @property
    def links(self) -> list[str]:
        """Links with collision geometry."""
        return self._links

    @property
    def similarity(self) -> SimilarityResult:
        """Similarity detection result."""
        return self._similarity

    def allocate(
        self,
        target_spheres: int,
        min_per_link: int = 1,
    ) -> dict[str, int]:
        """
        Allocate sphere budget across links proportionally.

        Similar links share allocations - only the primary link in each
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

        allocation = allocate_spheres_for_robot(
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
        budgets: dict[str, int] | None = None,
    ) -> RobotSpheresResult:
        """
        Generate spheres for the robot.

        Args:
            target_spheres: Total spheres (auto-allocates). Mutually exclusive with budgets.
            budgets: Explicit per-link allocation. Mutually exclusive with target_spheres.

        Returns:
            RobotSpheresResult with link_spheres

        Raises:
            ValueError: If neither or both of target_spheres and budgets are provided.
        """
        if (target_spheres is None) == (budgets is None):
            raise ValueError("Provide exactly one of target_spheres or budgets")

        if budgets is None:
            assert target_spheres is not None
            budgets = self.allocate(target_spheres)

        return compute_spheres_for_robot(
            self.urdf,
            self._links,
            link_budgets=budgets,
            similarity_result=self._similarity,
        )


def get_collision_mesh_for_link(urdf, link_name: str) -> trimesh.Trimesh:
    """
    Extract collision mesh for a given link from URDF.

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        link_name: Name of the link to extract

    Returns:
        Combined collision mesh for the link (empty Trimesh if no collisions)
    """
    if link_name not in urdf.link_map:
        return trimesh.Trimesh()

    link = urdf.link_map[link_name]
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
            if hasattr(urdf, "_filename_handler") and urdf._filename_handler is not None:
                mesh_path = urdf._filename_handler(mesh_path)
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
                    logger.warning(f"Unexpected mesh type from {mesh_path}: {type(mesh)}")
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


def _link_has_collision(urdf, link_name: str) -> bool:
    """Check if a link has collision geometry."""
    if link_name not in urdf.link_map:
        return False
    return len(urdf.link_map[link_name].collisions) > 0


def allocate_spheres_for_robot(
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
        max_link = max(allocation, key=allocation.get)
        if allocation[max_link] > min_per_link:
            allocation[max_link] -= 1
        else:
            break

    return allocation


def compute_spheres_for_robot(
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


def visualize_robot_spheres_viser(
    urdf,
    link_spheres: dict[str, list[Sphere]],
    server=None,
):
    """
    Visualize spheres on robot using viser.

    Args:
        urdf: yourdfpy URDF object
        link_spheres: Dict mapping link names to lists of spheres
        server: Optional existing viser server (creates new one if None)

    Returns:
        viser.ViserServer instance

    Raises:
        ImportError: If viser is not installed
    """
    try:
        import viser
        from viser.extras import ViserUrdf
    except ImportError:
        raise ImportError(
            "viser is required for visualization. "
            "Install with: pip install ballpark[robot]"
        )

    import time

    if server is None:
        server = viser.ViserServer()

    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/robot")

    # Get joint limits and link names
    lower_limits, upper_limits = get_joint_limits(urdf)
    all_link_names = get_link_names(urdf)

    # Joint sliders
    joint_sliders = []
    with server.gui.add_folder("Joints"):
        for i in range(len(lower_limits)):
            lower = float(lower_limits[i])
            upper = float(upper_limits[i])
            initial = (lower + upper) / 2
            slider = server.gui.add_slider(
                f"Joint {i}", min=lower, max=upper, step=0.01, initial_value=initial
            )
            joint_sliders.append(slider)

    # Sphere visualization controls
    show_spheres = server.gui.add_checkbox("Show Spheres", initial_value=True)
    sphere_opacity = server.gui.add_slider(
        "Sphere Opacity", min=0.1, max=1.0, step=0.1, initial_value=0.4
    )

    # Colors for spheres (per link)
    sphere_colors = [
        (255, 100, 100),
        (100, 255, 100),
        (100, 100, 255),
        (255, 255, 100),
        (255, 100, 255),
        (100, 255, 255),
        (255, 180, 100),
        (180, 100, 255),
        (100, 180, 100),
        (180, 180, 180),
    ]

    # Create sphere frames and handles
    sphere_frames: dict[str, viser.FrameHandle] = {}
    sphere_handles: dict[str, viser.IcosphereHandle] = {}

    def create_sphere_visuals():
        nonlocal sphere_frames, sphere_handles

        for handle in sphere_handles.values():
            handle.remove()
        for handle in sphere_frames.values():
            handle.remove()
        sphere_handles.clear()
        sphere_frames.clear()

        if not show_spheres.value:
            return

        for link_idx, link_name in enumerate(all_link_names):
            if link_name not in link_spheres:
                continue
            spheres = link_spheres[link_name]
            if not spheres:
                continue

            color = sphere_colors[link_idx % len(sphere_colors)]
            rgba = (
                color[0] / 255.0,
                color[1] / 255.0,
                color[2] / 255.0,
                sphere_opacity.value,
            )

            for sphere_idx, sphere in enumerate(spheres):
                key = f"{link_name}_{sphere_idx}"

                frame = server.scene.add_frame(
                    f"/sphere_frames/{key}",
                    wxyz=(1, 0, 0, 0),
                    position=(0, 0, 0),
                    show_axes=False,
                )
                sphere_frames[key] = frame

                sphere_handle = server.scene.add_icosphere(
                    f"/sphere_frames/{key}/sphere",
                    radius=sphere.radius,
                    position=tuple(sphere.center),
                    color=rgba[:3],
                    opacity=rgba[3],
                )
                sphere_handles[key] = sphere_handle

    def update_sphere_transforms(Ts_link_world):
        for link_idx, link_name in enumerate(all_link_names):
            if link_name not in link_spheres:
                continue
            spheres = link_spheres[link_name]
            if not spheres:
                continue

            T_wxyz_xyz = Ts_link_world[link_idx]
            wxyz = T_wxyz_xyz[:4]
            pos = T_wxyz_xyz[4:]

            for sphere_idx, _ in enumerate(spheres):
                key = f"{link_name}_{sphere_idx}"
                if key in sphere_frames:
                    sphere_frames[key].wxyz = wxyz
                    sphere_frames[key].position = pos

    create_sphere_visuals()
    last_show_spheres = show_spheres.value
    last_opacity = sphere_opacity.value

    while True:
        if (
            show_spheres.value != last_show_spheres
            or sphere_opacity.value != last_opacity
        ):
            last_show_spheres = show_spheres.value
            last_opacity = sphere_opacity.value
            create_sphere_visuals()

        cfg = np.array([s.value for s in joint_sliders])
        urdf_vis.update_cfg(cfg)
        Ts_link_world = get_link_transforms(urdf, cfg)

        if show_spheres.value:
            update_sphere_transforms(Ts_link_world)

        time.sleep(0.05)
