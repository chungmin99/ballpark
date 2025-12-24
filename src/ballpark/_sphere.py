"""Sphere dataclass for collision approximation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Sphere:
    """A sphere defined by center and radius."""

    center: np.ndarray
    radius: float
