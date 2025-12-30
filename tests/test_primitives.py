"""Tests for primitive shapes (EASY difficulty).

Primitives: cube, elongated_box, flat_box, sphere, cylinder,
short_cylinder, capsule, cone, torus
"""

from __future__ import annotations

import numpy as np
import pytest

from ballpark import spherize
from ballpark._spherize import compute_coverage

from .conftest import (
    BUDGETS,
    MIN_TIGHTNESS,
    MAX_VOLUME_OVERHEAD,
    compute_tightness,
    compute_volume_overhead,
    assert_tightness_above_minimum,
    assert_volume_overhead_below_maximum,
)
from .shapes import get_primitive_shape_names, get_shape_by_name


PRIMITIVE_NAMES = get_primitive_shape_names()


class TestPrimitiveValidity:
    """Test that primitive shapes produce valid spherization results."""

    @pytest.mark.parametrize("shape_name", PRIMITIVE_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_spheres_are_valid(self, shape_name: str, budget: int):
        """Verify all spheres have finite centers and positive radii."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        assert len(spheres) > 0, f"No spheres generated for {shape_name}"

        for i, s in enumerate(spheres):
            center = np.asarray(s.center)
            radius = float(s.radius)

            assert np.all(np.isfinite(center)), (
                f"Sphere {i} has non-finite center: {center}"
            )
            assert np.isfinite(radius), (
                f"Sphere {i} has non-finite radius: {radius}"
            )
            assert radius > 0, (
                f"Sphere {i} has non-positive radius: {radius}"
            )


class TestPrimitiveCoverage:
    """Test coverage for primitive shapes."""

    @pytest.mark.parametrize("shape_name", PRIMITIVE_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_high_coverage(self, shape_name: str, budget: int):
        """Primitive shapes should achieve high coverage (>90%)."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        coverage = compute_coverage(points, spheres)

        # Primitives should achieve at least 90% coverage
        min_coverage = 0.90
        assert coverage >= min_coverage, (
            f"{shape_name} coverage {coverage:.3f} below {min_coverage} at budget {budget}"
        )


class TestPrimitiveTightness:
    """Test tightness metrics for primitive shapes."""

    @pytest.mark.parametrize("shape_name", PRIMITIVE_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_tightness_above_minimum(self, shape_name: str, budget: int):
        """Verify spheres are not over-approximating excessively."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        tightness = compute_tightness(points, spheres)

        assert_tightness_above_minimum(
            tightness,
            min_tightness=MIN_TIGHTNESS,
            msg=f"for {shape_name} at budget {budget}",
        )

    @pytest.mark.parametrize("shape_name", PRIMITIVE_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_volume_overhead_bounded(self, shape_name: str, budget: int):
        """Verify total sphere volume is reasonable."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        overhead = compute_volume_overhead(points, spheres)

        assert_volume_overhead_below_maximum(
            overhead,
            max_overhead=MAX_VOLUME_OVERHEAD,
            msg=f"for {shape_name} at budget {budget}",
        )


class TestPrimitiveQuality:
    """Test combined quality metrics."""

    @pytest.mark.parametrize("shape_name", PRIMITIVE_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_quality_score(self, shape_name: str, budget: int):
        """Test combined quality = coverage * tightness."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))

        coverage = compute_coverage(points, spheres)
        tightness = compute_tightness(points, spheres)
        quality = coverage * tightness

        # Primitives should have quality > 0.14
        min_quality = 0.14
        assert quality >= min_quality, (
            f"{shape_name} quality {quality:.3f} (coverage={coverage:.3f}, "
            f"tightness={tightness:.3f}) below {min_quality} at budget {budget}"
        )

    @pytest.mark.parametrize("shape_name", PRIMITIVE_NAMES)
    def test_quality_at_high_budget(self, shape_name: str):
        """Verify quality at highest budget is acceptable."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))

        # Test at highest budget
        spheres = spherize(mesh, target_spheres=BUDGETS[-1])
        coverage = compute_coverage(points, spheres)
        tightness = compute_tightness(points, spheres)
        quality = coverage * tightness

        # Quality at highest budget should be good
        min_quality = 0.20
        assert quality >= min_quality, (
            f"{shape_name} quality {quality:.3f} at budget {BUDGETS[-1]} "
            f"below minimum {min_quality}"
        )
