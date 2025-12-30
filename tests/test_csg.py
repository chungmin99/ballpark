"""Tests for CSG shapes (require boolean operations).

CSG: c_shape, horseshoe, frame, swiss_cheese
"""

from __future__ import annotations

import numpy as np
import pytest

from ballpark import spherize
from ballpark._spherize import compute_coverage

from .conftest import (
    BUDGETS,
    MAX_VOLUME_OVERHEAD,
    compute_tightness,
    compute_volume_overhead,
)
from .shapes import get_csg_shape_names, get_shape_by_name


CSG_NAMES = get_csg_shape_names()


class TestCSGValidity:
    """Test that CSG shapes produce valid spherization results."""

    @pytest.mark.parametrize("shape_name", CSG_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_spheres_are_valid(self, shape_name: str, budget: int):
        """Verify all spheres have finite centers and positive radii."""
        spec = get_shape_by_name(shape_name)
        if spec is None:
            pytest.skip(f"CSG shape {shape_name} not available")

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


class TestCSGCoverage:
    """Test coverage for CSG shapes."""

    @pytest.mark.parametrize("shape_name", CSG_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_minimum_coverage(self, shape_name: str, budget: int):
        """CSG shapes should achieve minimum acceptable coverage."""
        spec = get_shape_by_name(shape_name)
        if spec is None:
            pytest.skip(f"CSG shape {shape_name} not available")

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        coverage = compute_coverage(points, spheres)

        # CSG shapes should achieve at least 80% coverage
        min_coverage = 0.80
        assert coverage >= min_coverage, (
            f"{shape_name} coverage {coverage:.3f} below {min_coverage} at budget {budget}"
        )


class TestCSGTightness:
    """Test tightness metrics for CSG shapes."""

    @pytest.mark.parametrize("shape_name", CSG_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_tightness_reasonable(self, shape_name: str, budget: int):
        """Verify spheres have reasonable tightness for concave shapes."""
        spec = get_shape_by_name(shape_name)
        if spec is None:
            pytest.skip(f"CSG shape {shape_name} not available")

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        tightness = compute_tightness(points, spheres)

        # CSG shapes (often concave) have lower tightness threshold
        min_tightness = 0.05
        assert tightness >= min_tightness, (
            f"{shape_name} tightness {tightness:.4f} below minimum {min_tightness}. "
            f"Spheres are extremely over-approximating at budget {budget}"
        )

    @pytest.mark.parametrize("shape_name", CSG_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_volume_overhead_bounded(self, shape_name: str, budget: int):
        """Verify total sphere volume is not excessive."""
        spec = get_shape_by_name(shape_name)
        if spec is None:
            pytest.skip(f"CSG shape {shape_name} not available")

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        overhead = compute_volume_overhead(points, spheres)

        # CSG shapes allowed higher overhead due to concavity
        max_overhead = MAX_VOLUME_OVERHEAD * 2  # 20x hull volume max
        assert overhead <= max_overhead, (
            f"{shape_name} volume overhead {overhead:.2f}x exceeds max {max_overhead}x "
            f"at budget {budget}"
        )


class TestCSGQuality:
    """Test quality metrics for CSG shapes."""

    @pytest.mark.parametrize("shape_name", CSG_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_quality_score(self, shape_name: str, budget: int):
        """Test combined quality = coverage * tightness."""
        spec = get_shape_by_name(shape_name)
        if spec is None:
            pytest.skip(f"CSG shape {shape_name} not available")

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))

        coverage = compute_coverage(points, spheres)
        tightness = compute_tightness(points, spheres)
        quality = coverage * tightness

        # CSG shapes should have quality > 0.05
        min_quality = 0.05
        assert quality >= min_quality, (
            f"{shape_name} quality {quality:.3f} (coverage={coverage:.3f}, "
            f"tightness={tightness:.3f}) below {min_quality} at budget {budget}"
        )
