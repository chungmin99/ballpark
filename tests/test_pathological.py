"""Tests for pathological shapes (degenerate geometries).

Pathological shapes have extreme aspect ratios (>= 15:1) that create
degenerate geometry for sphere decomposition. These shapes:
- needle: 20:1 aspect ratio (very thin cylinder)
- nearly_flat: 100:1 aspect ratio (nearly 2D box)
- elongated_star: 15:1 aspect ratio (very flat star)

These shapes are tested separately because:
1. They require very permissive quality thresholds (tightness ~0.01, overhead ~100x)
2. The extreme geometry makes tight sphere approximation essentially impossible
3. They represent edge cases that would distort metrics for normal challenging shapes

Tests are marked with xfail or use extremely permissive thresholds to document
expected limitations rather than failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from ballpark import spherize
from ballpark._spherize import compute_coverage

from .conftest import (
    BUDGETS,
    compute_tightness,
    compute_volume_overhead,
    compute_over_extension,
)
from .shapes import get_pathological_shape_names, get_shape_by_name


PATHOLOGICAL_NAMES = get_pathological_shape_names()


class TestPathologicalValidity:
    """Test that pathological shapes produce valid spherization results."""

    @pytest.mark.parametrize("shape_name", PATHOLOGICAL_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_spheres_are_valid(self, shape_name: str, budget: int):
        """Verify all spheres have finite centers and positive radii.

        Even with degenerate geometry, spheres should still be mathematically valid.
        """
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


class TestPathologicalCoverage:
    """Test coverage for pathological shapes (very permissive)."""

    @pytest.mark.parametrize("shape_name", PATHOLOGICAL_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_minimum_coverage(self, shape_name: str, budget: int):
        """Pathological shapes should achieve at least some coverage.

        With extreme aspect ratios, even basic coverage is challenging.
        We only verify that some reasonable coverage is achieved.
        """
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        coverage = compute_coverage(points, spheres)

        # Very permissive: just verify we cover most of the shape
        min_coverage = 0.60  # 60% coverage is acceptable for pathological cases
        assert coverage >= min_coverage, (
            f"{shape_name} coverage {coverage:.3f} below {min_coverage} at budget {budget}. "
            f"Even with degenerate geometry, basic coverage should be maintained."
        )


class TestPathologicalTightness:
    """Test tightness metrics for pathological shapes (extremely permissive)."""

    @pytest.mark.parametrize("shape_name", PATHOLOGICAL_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_tightness_minimal(self, shape_name: str, budget: int):
        """Verify spheres have minimal tightness (extremely permissive).

        Pathological geometries with 15:1 to 100:1 aspect ratios cannot be
        tightly approximated by spheres. Tightness values as low as 0.001
        (0.1%) are acceptable for these degenerate cases.
        """
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        tightness = compute_tightness(points, spheres)

        # Extremely permissive: nearly_flat can have tightness ~0.001
        min_tightness = 0.001  # 0.1% tightness acceptable for extreme geometries
        assert tightness >= min_tightness, (
            f"{shape_name} tightness {tightness:.6f} below minimum {min_tightness}. "
            f"This indicates catastrophic over-approximation even for pathological geometry."
        )

    @pytest.mark.parametrize("shape_name", PATHOLOGICAL_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_volume_overhead_extreme(self, shape_name: str, budget: int):
        """Verify total sphere volume is not catastrophically excessive.

        Pathological shapes can have overhead of 100x-1000x hull volume.
        This test only checks that overhead stays within documented bounds.
        """
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        n_samples = 5000
        points = np.asarray(mesh.sample(n_samples))
        overhead = compute_volume_overhead(points, spheres)

        # Extremely permissive: allow up to 1000x overhead for pathological cases
        max_overhead = 1000.0
        assert overhead <= max_overhead, (
            f"{shape_name} volume overhead {overhead:.2f}x exceeds max {max_overhead}x "
            f"at budget {budget}. This indicates catastrophic over-approximation."
        )


class TestPathologicalImprovement:
    """Test that more spheres improve approximation quality (best effort)."""

    @pytest.mark.parametrize("shape_name", PATHOLOGICAL_NAMES)
    def test_coverage_trend(self, shape_name: str):
        """Coverage should not significantly degrade with more spheres.

        For pathological geometry, we don't expect strong improvement,
        but coverage should at least remain stable or improve slightly.
        """
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

        # Very permissive: just verify coverage doesn't significantly degrade
        # Allow for significant variance due to extreme geometry
        assert coverages[-1] >= coverages[0] * 0.90, (
            f"{shape_name} coverage degraded from "
            f"{coverages[0]:.3f} (budget={BUDGETS[0]}) to "
            f"{coverages[-1]:.3f} (budget={BUDGETS[-1]})"
        )


class TestPathologicalDocumentation:
    """Document the extreme behavior of pathological shapes."""

    @pytest.mark.parametrize("shape_name", PATHOLOGICAL_NAMES)
    def test_aspect_ratio_is_extreme(self, shape_name: str):
        """Document that pathological shapes have aspect ratios >= 15:1."""
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        extents = mesh.extents
        aspect = max(extents) / min(extents) if min(extents) > 1e-10 else float('inf')

        # Document the extreme aspect ratio
        assert aspect >= 15.0, (
            f"{shape_name} has aspect ratio {aspect:.1f}x. "
            f"Expected >= 15:1 for pathological category."
        )

        # Print for documentation
        print(f"\n{shape_name}: extents={extents}, aspect={aspect:.1f}:1")


class TestPathologicalOverExtension:
    """Test over-extension metric for pathological shapes (extremely permissive)."""

    @pytest.mark.parametrize("shape_name", PATHOLOGICAL_NAMES)
    @pytest.mark.parametrize("budget", BUDGETS)
    def test_over_extension_extreme(self, shape_name: str, budget: int):
        """Verify over-extension stays within extreme but bounded limits.

        Pathological shapes with 15:1 to 100:1 aspect ratios will have
        very high over-extension. This test documents the expected behavior.
        """
        spec = get_shape_by_name(shape_name)
        assert spec is not None

        mesh = spec.factory()
        spheres = spherize(mesh, target_spheres=budget)

        over_ext = compute_over_extension(mesh, spheres)

        # Extremely permissive: allow up to 100x over-extension for pathological cases
        max_ratio = 100.0
        assert over_ext["over_extension_ratio"] <= max_ratio, (
            f"{shape_name} over-extension {over_ext['over_extension_ratio']:.2f}x "
            f"exceeds max {max_ratio}x at budget {budget}. "
            f"This indicates catastrophic over-approximation."
        )
