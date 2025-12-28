"""Sphere decomposition for collision approximation."""

# Core types
from ._sphere import Sphere as Sphere
from ._robot import Robot as Robot
from ._robot import RobotSpheresResult as RobotSpheresResult
from ._similarity import SimilarityResult as SimilarityResult

# Mesh-only API
from ._adaptive_tight import spherize_adaptive_tight as spherize

# Utilities
from ._export import export_spheres_to_json as export_spheres_to_json
from ._robot import visualize_robot_spheres_viser as visualize_robot_spheres_viser

# Configuration
from ._config import BallparkConfig as BallparkConfig
from ._config import get_config as get_config
from ._config import PRESET_CONFIGS as PRESET_CONFIGS

__version__ = "0.0.0"
