"""Tests for compound shapes (MEDIUM difficulty).

Compound: l_shape, t_shape, cross_shape, dumbbell, gripper_fingers,
stacked_boxes, u_bracket
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
    MAX_OVER_EXTENSION_RATIO,
    compute_tightness,
    compute_volume_overhead,
    compute_over_extension,
    assert_tightness_above_minimum,
    assert_volume_overhead_below_maximum,
    assert_over_extension_below_maximum,
)
from .shapes import get_compound_shape_names, get_shape_by_name


COMPOUND_NAMES = get_compound_shape_names()


class TestCompoundValidity:
    """Test that compound shapes produce valid spherization results."""

    @pytest.mark.parametrize("shape_name", COMPOUND_NAMES)
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


class TestCompoundCoverage:
    """Test coverage for compound shapes."""

    @pytest.mark.parametrize("shape_name", COMPOUND_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_reasonable_coverage(self, shape_name: str, budget: int):
        """Compound shapes should achieve reasonable coverage (>85%)."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        coverage = compute_coverage(points, spheres)

        # Compound shapes should achieve at least 85% coverage
        min_coverage = 0.85
        assert coverage >= min_coverage, (
            f"{shape_name} coverage {coverage:.3f} below {min_coverage} at budget {budget}"
        )


class TestCompoundTightness:
    """Test tightness metrics for compound shapes."""

    @pytest.mark.parametrize("shape_name", COMPOUND_NAMES)
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

        # Compound shapes may have slightly lower tightness due to geometry
        min_tightness = MIN_TIGHTNESS * 0.8  # Allow 20% lower threshold
        assert_tightness_above_minimum(
            tightness,
            min_tightness=min_tightness,
            msg=f"for {shape_name} at budget {budget}",
        )

    @pytest.mark.parametrize("shape_name", COMPOUND_NAMES)
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

        # Compound shapes allowed slightly higher overhead
        max_overhead = MAX_VOLUME_OVERHEAD * 1.5
        assert_volume_overhead_below_maximum(
            overhead,
            max_overhead=max_overhead,
            msg=f"for {shape_name} at budget {budget}",
        )


class TestCompoundQuality:
    """Test combined quality metrics."""

    @pytest.mark.parametrize("shape_name", COMPOUND_NAMES)
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

        # Compound shapes should have quality > 0.10
        min_quality = 0.10
        assert quality >= min_quality, (
            f"{shape_name} quality {quality:.3f} (coverage={coverage:.3f}, "
            f"tightness={tightness:.3f}) below {min_quality} at budget {budget}"
        )


class TestCompoundOverExtension:
    """Test over-extension metric for compound shapes."""

    @pytest.mark.parametrize("shape_name", COMPOUND_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_over_extension_bounded(self, shape_name: str, budget: int):
        """Verify spheres don't extend too far beyond mesh surface."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        over_ext = compute_over_extension(mesh, spheres)

        # Compound shapes allowed slightly higher over-extension
        max_ratio = MAX_OVER_EXTENSION_RATIO * 1.5
        assert_over_extension_below_maximum(
            over_ext["over_extension_ratio"],
            max_ratio=max_ratio,
            msg=f"for {shape_name} at budget {budget}",
        )
