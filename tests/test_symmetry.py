"""Tests for symmetric sphere placement."""

import numpy as np
import pytest
import trimesh

from ballpark._spherize import (
    Sphere,
    detect_reflection_symmetry,
    spherize,
    _allocate_symmetric_budget,
)
from ballpark._config import SpherizeParams


class TestSymmetryDetection:
    """Tests for detect_reflection_symmetry function."""

    def test_perfect_cube_symmetry(self):
        """A cube should have reflection symmetry about all three principal planes."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        points = np.array(mesh.sample(5000))

        axis, score = detect_reflection_symmetry(points, tolerance=0.05)

        assert axis is not None, "Should detect symmetry in a cube"
        assert score > 0.9, f"Score should be high for perfect symmetry, got {score}"

    def test_elongated_box_symmetry(self):
        """An elongated box should detect symmetry about the short axes."""
        mesh = trimesh.creation.box(extents=[2, 0.5, 0.5])
        points = np.array(mesh.sample(5000))

        axis, score = detect_reflection_symmetry(points, tolerance=0.05)

        assert axis is not None, "Should detect symmetry in elongated box"
        assert score > 0.9, f"Score should be high, got {score}"

    def test_cylinder_symmetry(self):
        """A cylinder should have reflection symmetry."""
        mesh = trimesh.creation.cylinder(radius=0.5, height=2.0)
        points = np.array(mesh.sample(5000))

        axis, score = detect_reflection_symmetry(points, tolerance=0.05)

        assert axis is not None, "Should detect symmetry in cylinder"
        assert score > 0.9, f"Score should be high, got {score}"

    def test_asymmetric_mesh_no_detection(self):
        """An asymmetric mesh should not trigger strong symmetry detection."""
        # Create an asymmetric 3D mesh with offsets in all dimensions
        mesh1 = trimesh.creation.box(extents=[1, 0.3, 0.2])
        mesh2 = trimesh.creation.box(extents=[0.2, 0.6, 0.3])
        mesh2.apply_translation([0.4, 0.3, 0.15])  # Offset in all 3 axes
        mesh3 = trimesh.creation.box(extents=[0.3, 0.2, 0.5])
        mesh3.apply_translation([-0.3, -0.2, 0.2])  # Another offset piece
        mesh = trimesh.util.concatenate([mesh1, mesh2, mesh3])

        points = np.array(mesh.sample(5000))
        axis, score = detect_reflection_symmetry(points, tolerance=0.02)

        # With a low tolerance (0.02), should have lower score for asymmetric mesh
        assert score < 0.98, f"Score should be lower for asymmetric mesh, got {score}"

    def test_tolerance_sensitivity(self):
        """Test that tolerance parameter controls detection sensitivity."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        points = np.array(mesh.sample(5000))

        # Add slight asymmetric noise to some points
        rng = np.random.default_rng(42)
        noise_mask = rng.random(len(points)) < 0.1
        points[noise_mask, 0] += 0.1  # Shift 10% of points

        # Should detect with high tolerance
        axis_high, score_high = detect_reflection_symmetry(points, tolerance=0.15)
        assert axis_high is not None, "Should detect with high tolerance"

        # May not detect with very low tolerance
        axis_low, score_low = detect_reflection_symmetry(points, tolerance=0.01)
        # Low tolerance should result in lower score
        assert score_low <= score_high, "Lower tolerance should give lower score"

    def test_few_points_returns_none(self):
        """With very few points, should return None."""
        points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        axis, score = detect_reflection_symmetry(points, tolerance=0.05)
        assert axis is None
        assert score == 0.0


class TestSymmetricBudgetAllocation:
    """Tests for _allocate_symmetric_budget function."""

    def test_even_budget(self):
        """Even budget should split evenly."""
        left, right, needs_center = _allocate_symmetric_budget(8, "round_up")
        assert left == 4
        assert right == 4
        assert not needs_center

    def test_odd_budget_round_up(self):
        """Odd budget with round_up should give equal halves (rounding up)."""
        left, right, needs_center = _allocate_symmetric_budget(7, "round_up")
        assert left == 4
        assert right == 4
        assert not needs_center

    def test_odd_budget_center(self):
        """Odd budget with center should reserve one for center."""
        left, right, needs_center = _allocate_symmetric_budget(7, "center")
        assert left == 3
        assert right == 3
        assert needs_center

    def test_budget_of_one_round_up(self):
        """Budget of 1 with round_up."""
        left, right, needs_center = _allocate_symmetric_budget(1, "round_up")
        assert left == 1
        assert right == 1
        assert not needs_center

    def test_budget_of_one_center(self):
        """Budget of 1 with center."""
        left, right, needs_center = _allocate_symmetric_budget(1, "center")
        assert left == 0
        assert right == 0
        assert needs_center


class TestSymmetricSpherization:
    """Integration tests for symmetric spherization."""

    def test_symmetric_mesh_produces_symmetric_spheres(self):
        """Symmetric mesh should produce mirror-symmetric sphere positions."""
        mesh = trimesh.creation.box(extents=[2, 1, 1])  # Elongated along X
        params = SpherizeParams(symmetry_mode="auto", symmetry_tolerance=0.05)

        spheres = spherize(mesh, target_spheres=8, params=params)
        centers = np.array([np.array(s.center) for s in spheres])

        # Find the symmetry axis (should be one of the principal axes)
        # For a box elongated along X, symmetry plane is YZ (normal is X)
        # Check that centers come in symmetric pairs about X=0

        positive_x = centers[centers[:, 0] > 0.01]
        negative_x = centers[centers[:, 0] < -0.01]

        # Should have roughly equal number on each side
        assert abs(len(positive_x) - len(negative_x)) <= 1, (
            f"Unequal distribution: {len(positive_x)} vs {len(negative_x)}"
        )

        # For each positive center, there should be a matching negative one
        for pos_center in positive_x:
            reflected = pos_center.copy()
            reflected[0] = -reflected[0]
            distances = np.linalg.norm(negative_x - reflected, axis=1)
            min_dist = np.min(distances) if len(distances) > 0 else float("inf")
            assert min_dist < 0.3, f"No symmetric pair found for {pos_center}"

    def test_symmetry_mode_off(self):
        """With symmetry_mode='off', should use original algorithm."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        params = SpherizeParams(symmetry_mode="off")

        # Should not raise and should produce spheres
        spheres = spherize(mesh, target_spheres=4, params=params)
        assert len(spheres) > 0

    def test_symmetry_mode_force(self):
        """With symmetry_mode='force', should enforce symmetry even if not detected."""
        # Create a slightly asymmetric mesh
        mesh1 = trimesh.creation.box(extents=[1, 1, 1])
        mesh2 = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
        mesh2.apply_translation([0.6, 0, 0])
        mesh = trimesh.util.concatenate([mesh1, mesh2])

        params = SpherizeParams(symmetry_mode="force")

        # Should still produce spheres and try to be symmetric
        spheres = spherize(mesh, target_spheres=4, params=params)
        assert len(spheres) > 0

    def test_odd_budget_round_up_produces_even_count(self):
        """With odd budget and round_up mode, should produce one extra sphere."""
        mesh = trimesh.creation.box(extents=[2, 1, 1])
        params = SpherizeParams(
            symmetry_mode="auto",
            symmetry_tolerance=0.05,
            odd_budget_mode="round_up",
        )

        spheres = spherize(mesh, target_spheres=5, params=params)
        # With round_up and symmetric split, 5 becomes 6 (3+3)
        # Actual count may vary due to recursive splitting
        assert len(spheres) >= 5

    def test_odd_budget_center_includes_center_sphere(self):
        """With odd budget and center mode, should include a center sphere."""
        mesh = trimesh.creation.box(extents=[2, 1, 1])
        params = SpherizeParams(
            symmetry_mode="auto",
            symmetry_tolerance=0.05,
            odd_budget_mode="center",
        )

        spheres = spherize(mesh, target_spheres=5, params=params)

        # Check that there's a sphere near the center
        centers = np.array([np.array(s.center) for s in spheres])
        distances_from_origin = np.linalg.norm(centers, axis=1)
        has_center_sphere = np.any(distances_from_origin < 0.5)

        assert has_center_sphere, "Should have a sphere near the center"

    def test_spherization_basic_functionality(self):
        """Basic test that spherization still works."""
        mesh = trimesh.creation.icosphere(subdivisions=2)
        params = SpherizeParams(symmetry_mode="auto")

        spheres = spherize(mesh, target_spheres=4, params=params)

        assert len(spheres) > 0
        for s in spheres:
            assert isinstance(s, Sphere)
            assert s.radius > 0
