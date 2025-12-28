"""Sphere decomposition for collision approximation."""

from ._sphere import Sphere as Sphere
from ._adaptive_tight import spherize_adaptive_tight as spherize_adaptive_tight
from ._nlls_refine import refine_spheres_nlls as refine_spheres_nlls
from ._robot import get_collision_mesh_for_link as get_collision_mesh_for_link
from ._robot import get_joint_limits as get_joint_limits
from ._robot import get_link_transforms as get_link_transforms
from ._robot import get_link_names as get_link_names
from ._robot import allocate_spheres_for_robot as allocate_spheres_for_robot
from ._robot import compute_spheres_for_link as compute_spheres_for_link
from ._robot import compute_spheres_for_robot as compute_spheres_for_robot
from ._robot import visualize_robot_spheres_viser as visualize_robot_spheres_viser
from ._robot import RobotSpheresResult as RobotSpheresResult
from ._robot_refine import refine_spheres_for_robot as refine_spheres_for_robot
from ._robot_refine import get_adjacent_links as get_adjacent_links
from ._robot_refine import get_non_contiguous_link_pairs as get_non_contiguous_link_pairs
from ._robot_refine import compute_min_self_collision_distance as compute_min_self_collision_distance
from ._robot_refine import compute_mesh_distances_batch as compute_mesh_distances_batch
from ._robot_refine import RobotRefinementResult as RobotRefinementResult
from ._similarity import detect_similar_links as detect_similar_links
from ._similarity import SimilarityResult as SimilarityResult
from ._similarity import get_group_for_link as get_group_for_link
from ._export import export_spheres_to_json as export_spheres_to_json
from ._config import BallparkConfig as BallparkConfig
from ._config import AdaptiveTightConfig as AdaptiveTightConfig
from ._config import RefinementConfig as RefinementConfig
from ._config import RobotRefinementConfig as RobotRefinementConfig
from ._config import get_config as get_config
from ._config import update_config_from_dict as update_config_from_dict
from ._config import PRESET_CONFIGS as PRESET_CONFIGS
from ._config import UNSET as UNSET
from ._config import resolve_params as resolve_params

__version__ = "0.0.0"
