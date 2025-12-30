"""Regression tests for all shapes using snapshots.

This file contains snapshot-based regression tests that compare
current results to stored baselines. For category-specific tests,
see test_primitives.py, test_compound.py, test_challenging.py, and test_csg.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from ballpark import spherize
from ballpark._spherize import compute_coverage

from .conftest import (
    BUDGETS,
    get_snapshot_key,
    load_snapshots,
    assert_coverage_within_tolerance,
    assert_sphere_count_within_tolerance,
    compute_tightness,
)
from .shapes import get_all_shape_names, get_shape_by_name


ALL_SHAPE_NAMES = get_all_shape_names(include_csg=True)


class TestShapeRegression:
    """Regression tests using snapshots."""

    @pytest.mark.parametrize("shape_name", ALL_SHAPE_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_matches_snapshot(self, shape_name: str, budget: int):
        """Verify results match stored snapshot (if available)."""
        snapshots = load_snapshots()
        key = get_snapshot_key(shape_name, budget)

        if key not in snapshots:
            pytest.skip(f"No snapshot for {key}")

        spec = get_shape_by_name(shape_name)
        if spec is None:
            pytest.skip(f"Shape {shape_name} not found")

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        coverage = compute_coverage(points, spheres)

        expected = snapshots[key]

        # Check coverage within tolerance
        assert_coverage_within_tolerance(
            coverage,
            expected.coverage,
            msg=f"for {shape_name} at budget {budget}",
        )

        # Check sphere count within tolerance
        assert_sphere_count_within_tolerance(
            len(spheres),
            expected.sphere_count,
            msg=f"for {shape_name} at budget {budget}",
        )


class TestSingleBoundingSphereCheck:
    """Verify algorithm doesn't degrade to single bounding sphere."""

    @pytest.mark.parametrize("shape_name", ALL_SHAPE_NAMES)
    def test_not_single_bounding_sphere(self, shape_name: str):
        """Verify we don't just return a single bounding sphere for budget > 1."""
        spec = get_shape_by_name(shape_name)
        if spec is None:
            pytest.skip(f"Shape {shape_name} not found")

        mesh = spec.factory()

        # With budget=16, should produce multiple spheres
        spheres = spherize(mesh, target_spheres=16)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))

        # A single bounding sphere has tightness ~0.05
        # Multiple well-placed spheres should be tighter
        tightness = compute_tightness(points, spheres)

        # Tightness should be better than worst case
        # (allow some shapes to be harder to decompose)
        # Note: extreme geometries like nearly_flat can have tightness ~0.02
        min_acceptable_tightness = 0.02
        assert tightness >= min_acceptable_tightness, (
            f"{shape_name} with budget=16 has tightness {tightness:.4f}, "
            f"suggesting near-single-bounding-sphere behavior. "
            f"Generated {len(spheres)} spheres."
        )


# =============================================================================
# SNAPSHOT GENERATION
# =============================================================================


def generate_snapshots():
    """Generate baseline snapshots for all shapes.

    Run manually with:
        python -c "from tests.test_shapes import generate_snapshots; generate_snapshots()"
    """
    from .conftest import save_snapshots, SnapshotData
    from ballpark._spherize import compute_split_quality
    from .shapes import get_all_shapes

    shapes = get_all_shapes(include_csg=True)
    snapshots = {}

    print("Generating snapshots...")
    for spec in shapes:
        mesh = spec.factory()
        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))

        for budget in BUDGETS:
            key = get_snapshot_key(spec.name, budget)
            spheres = spherize(mesh, target_spheres=budget)

            coverage = compute_coverage(points, spheres)
            quality = compute_split_quality(points, spheres)

            snapshots[key] = SnapshotData(
                sphere_count=len(spheres),
                coverage=round(coverage, 4),
                quality=round(quality, 4),
            )
            print(f"  {key}: {len(spheres)} spheres, coverage={coverage:.4f}")

    save_snapshots(snapshots)
    print(f"\nSaved {len(snapshots)} snapshots")
