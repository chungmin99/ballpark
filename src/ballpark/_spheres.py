"""Core data types for sphere decomposition."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sphere:
    """A sphere defined by center and radius."""

    center: np.ndarray
    radius: float
