"""Tests for challenging shapes (HARD difficulty, non-CSG).

Challenging shapes have moderate difficulty with aspect ratios < 15:1:
thin_plate (10:1), star (6.7:1), thick_torus (2.4:1), tiny_tetrahedron (1.2:1),
dense_sphere (1.0:1), asymmetric_blob (1.4:1)

Shapes with extreme aspect ratios (>= 15:1) have been moved to test_pathological.py:
- needle (20:1)
- nearly_flat (100:1)
- elongated_star (15:1)
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
    assert_over_extension_below_maximum,
)
from .shapes import get_challenging_shape_names, get_shape_by_name


CHALLENGING_NAMES = get_challenging_shape_names()


class TestChallengingValidity:
    """Test that challenging shapes produce valid spherization results."""

    @pytest.mark.parametrize("shape_name", CHALLENGING_NAMES)
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


class TestChallengingCoverage:
    """Test coverage for challenging shapes."""

    @pytest.mark.parametrize("shape_name", CHALLENGING_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_minimum_coverage(self, shape_name: str, budget: int):
        """Challenging shapes should achieve minimum acceptable coverage."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        coverage = compute_coverage(points, spheres)

        # Challenging shapes may have lower coverage than primitives
        # but should still achieve at least 75% (pathological shapes tested separately)
        min_coverage = 0.75
        assert coverage >= min_coverage, (
            f"{shape_name} coverage {coverage:.3f} below {min_coverage} at budget {budget}"
        )


class TestChallengingTightness:
    """Test tightness metrics for challenging shapes."""

    @pytest.mark.parametrize("shape_name", CHALLENGING_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_tightness_reasonable(self, shape_name: str, budget: int):
        """Verify spheres have some reasonable tightness."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        tightness = compute_tightness(points, spheres)

        # Challenging shapes have lower tightness threshold than primitives (0.15)
        # Now that pathological shapes are tested separately, we can be more strict
        min_tightness = 0.05  # Permissive but reasonable for moderate aspect ratios
        assert tightness >= min_tightness, (
            f"{shape_name} tightness {tightness:.4f} below minimum {min_tightness}. "
            f"Spheres are excessively over-approximating at budget {budget}"
        )

    @pytest.mark.parametrize("shape_name", CHALLENGING_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_volume_overhead_bounded(self, shape_name: str, budget: int):
        """Verify total sphere volume is not excessive."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        overhead = compute_volume_overhead(points, spheres)

        # Challenging shapes allowed higher overhead than primitives (10x)
        # Now that pathological shapes are tested separately, we can be more strict
        max_overhead = MAX_VOLUME_OVERHEAD * 2  # 20x hull volume max (down from 100x)
        assert overhead <= max_overhead, (
            f"{shape_name} volume overhead {overhead:.2f}x exceeds max {max_overhead}x "
            f"at budget {budget}"
        )


class TestChallengingImprovement:
    """Test that more spheres improve approximation quality."""

    @pytest.mark.parametrize("shape_name", CHALLENGING_NAMES)
    def test_coverage_improves_with_budget(self, shape_name: str):
        """Coverage should generally improve with more spheres."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))

        coverages = []
        for budget in BUDGETS:
            spheres = spherize(mesh, target_spheres=budget)
            coverage = compute_coverage(points, spheres)
            coverages.append(coverage)

        # Coverage at highest budget should be >= coverage at lowest
        # (allowing small variance)
        assert coverages[-1] >= coverages[0] * 0.95, (
            f"{shape_name} coverage did not improve from "
            f"{coverages[0]:.3f} (budget={BUDGETS[0]}) to "
            f"{coverages[-1]:.3f} (budget={BUDGETS[-1]})"
        )


class TestChallengingOverExtension:
    """Test over-extension metric for challenging shapes."""

    @pytest.mark.parametrize("shape_name", CHALLENGING_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_over_extension_bounded(self, shape_name: str, budget: int):
        """Verify spheres don't extend too far beyond mesh surface."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        over_ext = compute_over_extension(mesh, spheres)

        # Challenging shapes allowed higher over-extension
        max_ratio = MAX_OVER_EXTENSION_RATIO * 2.0
        assert_over_extension_below_maximum(
            over_ext["over_extension_ratio"],
            max_ratio=max_ratio,
            msg=f"for {shape_name} at budget {budget}",
        )
