"""Pytest configuration and fixtures for ballpark tests."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from .shapes import get_all_shapes, get_shape_by_name, ShapeSpec


# =============================================================================
# PATHS
# =============================================================================

TESTS_DIR = Path(__file__).parent
SNAPSHOTS_DIR = TESTS_DIR / "snapshots"
SHAPE_SNAPSHOTS_FILE = SNAPSHOTS_DIR / "shape_snapshots.json"


# =============================================================================
# TOLERANCE SETTINGS
# =============================================================================

# Coverage tolerance: allow 6% deviation from snapshot
# (due to sampling randomness in the algorithm)
COVERAGE_TOLERANCE = 0.06

# Sphere count tolerance: deprecated in favor of relative tolerance
# (kept for backward compatibility, not recommended for use)
SPHERE_COUNT_TOLERANCE = 5  # deprecated: use relative tolerance instead


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SpherizeResult:
    """Result of a spherization test run."""

    shape_name: str
    budget: int
    sphere_count: int
    coverage: float
    quality: float
    elapsed_ms: float
    valid: bool  # All spheres have finite centers and positive radii


@dataclass
class SnapshotData:
    """Stored snapshot data for regression testing."""

    sphere_count: int
    coverage: float
    quality: float


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def all_shapes() -> list[ShapeSpec]:
    """Get all available test shapes."""
    return get_all_shapes(include_csg=True)


@pytest.fixture(scope="session")
def shape_names() -> list[str]:
    """Get all shape names for parametrization."""
    return [s.name for s in get_all_shapes(include_csg=True)]


@pytest.fixture
def shape_spec(request) -> ShapeSpec:
    """Get a specific shape spec by name (used with indirect parametrization)."""
    shape_name = request.param
    spec = get_shape_by_name(shape_name, include_csg=True)
    if spec is None:
        pytest.fail(f"Unknown shape: {shape_name}")
    return spec


# =============================================================================
# SNAPSHOT MANAGEMENT
# =============================================================================


def load_snapshots() -> dict[str, SnapshotData]:
    """Load snapshot data from file."""
    if not SHAPE_SNAPSHOTS_FILE.exists():
        return {}

    with open(SHAPE_SNAPSHOTS_FILE) as f:
        raw = json.load(f)

    return {
        key: SnapshotData(
            sphere_count=val["sphere_count"],
            coverage=val["coverage"],
            quality=val["quality"],
        )
        for key, val in raw.items()
    }


def save_snapshots(snapshots: dict[str, SnapshotData]) -> None:
    """Save snapshot data to file."""
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = {
        key: {
            "sphere_count": val.sphere_count,
            "coverage": val.coverage,
            "quality": val.quality,
        }
        for key, val in snapshots.items()
    }

    with open(SHAPE_SNAPSHOTS_FILE, "w") as f:
        json.dump(raw, f, indent=2, sort_keys=True)


def get_snapshot_key(shape_name: str, budget: int) -> str:
    """Generate a snapshot key for a shape/budget combination."""
    return f"{shape_name}_{budget}"


@pytest.fixture(scope="session")
def snapshots() -> dict[str, SnapshotData]:
    """Load snapshots for the test session."""
    return load_snapshots()


# =============================================================================
# TIMING UTILITIES
# =============================================================================


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self):
        self.start_time: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000


@pytest.fixture
def timer() -> Timer:
    """Get a Timer instance for measuring execution time."""
    return Timer()


# =============================================================================
# TEST PARAMETERS
# =============================================================================

# Sphere budgets to test
BUDGETS = [4, 16, 64]


# Note: pytest_generate_tests removed - parametrization is handled
# directly in test files using @pytest.mark.parametrize


# =============================================================================
# PYTEST HOOKS
# =============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "robot: marks tests as robot-specific (requires robot_descriptions)"
    )


# =============================================================================
# QUALITY METRIC THRESHOLDS
# =============================================================================

# Minimum tightness (hull_vol / sphere_vol) - penalizes over-approximation
# A single bounding sphere has tightness ~0.05, good decomposition > 0.2
MIN_TIGHTNESS = 0.15

# Maximum volume overhead (sphere_vol / hull_vol)
# Should not exceed 10x the hull volume
MAX_VOLUME_OVERHEAD = 10.0

# Minimum quality score (coverage * tightness)
MIN_QUALITY = 0.10


# =============================================================================
# QUALITY METRIC FUNCTIONS (imported from ballpark.metrics)
# =============================================================================

from ballpark.metrics import (
    compute_coverage,
    compute_quality,
    compute_tightness,
    compute_volume_overhead,
)


# =============================================================================
# ASSERTION HELPERS
# =============================================================================


def assert_coverage_within_tolerance(
    actual: float,
    expected: float,
    tolerance: float = COVERAGE_TOLERANCE,
    msg: str = "",
) -> None:
    """Assert that actual coverage is within tolerance of expected."""
    diff = abs(actual - expected)
    if diff > tolerance:
        raise AssertionError(
            f"Coverage {actual:.4f} differs from expected {expected:.4f} "
            f"by {diff:.4f} (tolerance: {tolerance}). {msg}"
        )


def assert_sphere_count_within_tolerance(
    actual: int,
    expected: int,
    tolerance_percent: float = 0.10,
    msg: str = "",
) -> None:
    """Assert that actual sphere count is within relative tolerance of expected.

    Args:
        actual: Actual sphere count
        expected: Expected sphere count
        tolerance_percent: Relative tolerance as a fraction (default 0.10 = 10%)
        msg: Optional message to append to assertion error

    The tolerance is calculated as: abs(actual - expected) <= expected * tolerance_percent
    For small expected counts (< 10), a minimum absolute tolerance of 1 is used.
    """
    # Calculate relative tolerance, with minimum of 1 for small counts
    relative_tolerance = expected * tolerance_percent
    effective_tolerance = max(relative_tolerance, 1.0) if expected < 10 else relative_tolerance

    diff = abs(actual - expected)
    if diff > effective_tolerance:
        raise AssertionError(
            f"Sphere count {actual} differs from expected {expected} "
            f"by {diff} (tolerance: {effective_tolerance:.1f} = {tolerance_percent*100:.0f}%). {msg}"
        )


def assert_tightness_above_minimum(
    tightness: float,
    min_tightness: float = MIN_TIGHTNESS,
    msg: str = "",
) -> None:
    """Assert that tightness exceeds minimum threshold."""
    if tightness < min_tightness:
        raise AssertionError(
            f"Tightness {tightness:.4f} below minimum {min_tightness}. "
            f"This indicates excessive over-approximation (spheres too large). {msg}"
        )


def assert_volume_overhead_below_maximum(
    overhead: float,
    max_overhead: float = MAX_VOLUME_OVERHEAD,
    msg: str = "",
) -> None:
    """Assert that volume overhead is below maximum threshold."""
    if overhead > max_overhead:
        raise AssertionError(
            f"Volume overhead {overhead:.2f}x exceeds maximum {max_overhead}x. "
            f"Spheres occupy too much volume relative to mesh. {msg}"
        )
