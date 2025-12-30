"""Sphere decomposition for collision approximation."""

from ._config import BallparkConfig as BallparkConfig
from ._config import RefineParams as RefineParams
from ._config import SpherePreset as SpherePreset
from ._config import SpherizeParams as SpherizeParams
from ._robot import Robot as Robot
from ._robot import RobotSpheresResult as RobotSpheresResult
from ._spherize import Sphere as Sphere
from ._spherize import spherize as spherize

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
