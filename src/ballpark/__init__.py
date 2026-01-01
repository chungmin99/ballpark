"""Sphere decomposition for collision approximation."""

from ._config import BallparkConfig as BallparkConfig
from ._config import RefineParams as RefineParams
from ._config import SpherePreset as SpherePreset
from ._config import SpherizeParams as SpherizeParams
from ._decompose import decompose_convex as decompose_convex
from ._decompose import has_vhacd as has_vhacd
from ._decompose import spherize_decomposed as spherize_decomposed
from ._interior import spherize_interior as spherize_interior
from ._medial_axis import spherize_medial_axis as spherize_medial_axis
from ._primitives import detect_primitive as detect_primitive
from ._primitives import PrimitiveInfo as PrimitiveInfo
from ._primitives import PrimitiveType as PrimitiveType
from ._robot import Robot as Robot
from ._robot import RobotSpheresResult as RobotSpheresResult
from ._spherize import Sphere as Sphere
from ._spherize import spherize as spherize
from . import metrics as metrics

__version__ = "0.0.0"

# Colors for sphere visualization (RGB tuples, 0-255)
SPHERE_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 100, 100),
    (100, 255, 100),
    (100, 100, 255),
    (255, 255, 100),
    (255, 100, 255),
    (100, 255, 255),
    (255, 180, 100),
    (180, 100, 255),
    (100, 180, 100),
    (255, 200, 150),
)
