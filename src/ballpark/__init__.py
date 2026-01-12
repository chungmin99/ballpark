"""Sphere and ellipsoid decomposition for collision approximation."""

from ._config import BallparkConfig as BallparkConfig
from ._config import RefineParams as RefineParams
from ._config import SpherePreset as SpherePreset
from ._config import SpherizeParams as SpherizeParams
from ._robot import Robot as Robot
from ._robot import RobotSpheresResult as RobotSpheresResult
from ._robot import RobotEllipsoidsResult as RobotEllipsoidsResult
from ._spherize import Sphere as Sphere
from ._spherize import spherize as spherize
from ._primitives import Ellipsoid as Ellipsoid
from ._primitives import RotatedEllipsoid as RotatedEllipsoid
from ._primitives import ellipsoid_effective_radius as ellipsoid_effective_radius
from ._primitives import ellipsoid_point_distance as ellipsoid_point_distance
from ._primitives import ellipsoid_ellipsoid_distance as ellipsoid_ellipsoid_distance
from ._primitives import sphere_to_ellipsoid as sphere_to_ellipsoid
from ._primitives import ellipsoid_to_sphere as ellipsoid_to_sphere
from ._primitives import ellipsoid_volume as ellipsoid_volume

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
