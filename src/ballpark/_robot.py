"""Robot collision mesh extraction and sphere visualization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from loguru import logger

from ._sphere import Sphere
from ._spherize import spherize_adaptive_tight
from ._robot_refine import (
    refine_spheres_for_robot,
    get_adjacent_links,
    get_non_contiguous_link_pairs,
    compute_mesh_distances_batch,
)
from ._similarity import SimilarityResult, detect_similar_links
from ._config import BallparkConfig, UNSET, resolve_params, _UnsetType
from ._urdf_utils import (
    get_joint_limits,
    get_link_transforms,
    get_link_names,
    get_num_actuated_joints,
)


class Robot:
    """ """

    def __init__(
        self,
        urdf,
        joint_cfg: np.ndarray | None = None,
    ):
        """
        Analyze robot structure. Computes similarity, mesh distances, etc.

        Args:
            urdf: yourdfpy URDF object with collision meshes loaded
            joint_cfg: Joint configuration for FK-based analysis (default: zeros)
        """
        self.urdf = urdf

        # Set joint configuration
        num_joints = get_num_actuated_joints(urdf)
        self._joint_cfg = joint_cfg if joint_cfg is not None else np.zeros(num_joints)

        # Compute links with collision geometry
        self._links = [
            link_name
            for link_name in urdf.link_map.keys()
            if _link_has_collision(urdf, link_name)
        ]

        # Compute similarity
        self._similarity = detect_similar_links(urdf, self._links, verbose=False)

        # Compute non-contiguous pairs
        self._non_contiguous_pairs = get_non_contiguous_link_pairs(urdf, self._links)

        # Compute mesh distances for collision filtering
        self._mesh_distances = compute_mesh_distances_batch(
            urdf,
            self._non_contiguous_pairs,
            joint_cfg=self._joint_cfg,
        )

    @property
    def links(self) -> list[str]:
        """Links with collision geometry."""
        return self._links

    @property
    def similarity(self) -> SimilarityResult:
        """Similarity detection result."""
        return self._similarity

    @property
    def mesh_distances(self) -> dict[tuple[str, str], float]:
        """Pre-computed mesh distances between non-contiguous link pairs."""
        return self._mesh_distances

    @property
    def non_contiguous_pairs(self) -> list[tuple[str, str]]:
        """Link pairs that are not adjacent."""
        return self._non_contiguous_pairs

    @property
    def joint_cfg(self) -> np.ndarray:
        """Joint configuration used for analysis."""
        return self._joint_cfg

    def allocate(
        self,
        target_spheres: int,
        min_per_link: int = 1,
    ) -> dict[str, int]:
        """
        Allocate sphere budget across links.

        Args:
            target_spheres: Total number of spheres
            min_per_link: Minimum spheres per link

        Returns:
            Dict mapping link names to sphere counts
        """
        return allocate_spheres_for_robot(
            self.urdf,
            target_spheres=target_spheres,
            min_spheres_per_link=min_per_link,
        )

    def spherize(
        self,
        target_spheres: int | None = None,
        budgets: dict[str, int] | None = None,
        refine: bool = False,
        preset: str = "balanced",
        **kwargs,
    ) -> RobotSpheresResult:
        """
        Generate spheres for the robot.

        Args:
            target_spheres: Total spheres (auto-allocates). Mutually exclusive with budgets.
            budgets: Explicit per-link allocation. Mutually exclusive with target_spheres.
            refine: Run robot-level NLLS refinement with self-collision avoidance.
            preset: Config preset ("conservative", "balanced", "surface")
            **kwargs: Override specific hyperparameters (padding, target_tightness, etc.)

        Returns:
            RobotSpheresResult with link_spheres and ignore_pairs

        Raises:
            ValueError: If neither or both of target_spheres and budgets are provided.
        """
        if (target_spheres is None) == (budgets is None):
            raise ValueError("Provide exactly one of target_spheres or budgets")

        if budgets is None:
            assert target_spheres is not None  # Guaranteed by check above
            budgets = self.allocate(target_spheres)

        return compute_spheres_for_robot(
            self.urdf,
            link_budgets=budgets,
            similarity_result=self._similarity,
            mesh_distances=self._mesh_distances,
            joint_cfg=self._joint_cfg,
            refine_self_collision=refine,
            preset=preset,
            **kwargs,
        )


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
