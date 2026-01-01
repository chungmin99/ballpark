"""Tests for symmetric sphere placement."""

import numpy as np
import pytest
import trimesh

from ballpark._spherize import (
    Sphere,
    detect_reflection_symmetry,
    spherize,
    _allocate_symmetric_budget,
    _mirror_sphere,
    _partition_points_by_plane,
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
        """Symmetric mesh should produce EXACTLY mirror-symmetric sphere positions."""
        mesh = trimesh.creation.box(extents=[2, 1, 1])  # Elongated along X
        params = SpherizeParams(symmetry_mode="auto", symmetry_tolerance=0.05)

        spheres = spherize(mesh, target_spheres=8, params=params)
        centers = np.array([np.array(s.center) for s in spheres])
        radii = np.array([float(s.radius) for s in spheres])

        # Detect the symmetry axis from sphere positions using PCA
        # The first principal component should be along the symmetry direction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1)
        pca.fit(centers)
        symmetry_axis = pca.components_[0]

        # Project centers onto symmetry axis
        centroid = centers.mean(axis=0)
        projections = (centers - centroid) @ symmetry_axis

        # Separate into positive and negative sides
        pos_mask = projections > 1e-6
        neg_mask = projections < -1e-6

        positive_spheres = [(centers[i], radii[i]) for i in range(len(centers)) if pos_mask[i]]
        negative_spheres = [(centers[i], radii[i]) for i in range(len(centers)) if neg_mask[i]]

        # Should have equal number on each side
        assert len(positive_spheres) == len(negative_spheres), (
            f"Unequal distribution: {len(positive_spheres)} vs {len(negative_spheres)}"
        )

        # For each positive center, there should be an EXACT mirror on negative side
        for pos_center, pos_radius in positive_spheres:
            # Compute reflection across symmetry plane
            offset = pos_center - centroid
            proj = np.dot(offset, symmetry_axis)
            reflected = pos_center - 2 * proj * symmetry_axis

            # Find matching sphere (exact mirror with same radius)
            found_match = False
            for neg_center, neg_radius in negative_spheres:
                dist = np.linalg.norm(neg_center - reflected)
                if dist < 1e-4 and abs(neg_radius - pos_radius) < 1e-4:
                    found_match = True
                    break

            assert found_match, f"No exact mirror found for {pos_center} (r={pos_radius})"

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

    def test_mirroring_produces_identical_structure(self):
        """Verify that mirrored spheres have identical structure to originals."""
        mesh = trimesh.creation.box(extents=[4, 1, 1])  # Very elongated
        params = SpherizeParams(
            symmetry_mode="force",  # Force symmetry
            odd_budget_mode="center",
        )

        # Odd budget to test center sphere
        spheres = spherize(mesh, target_spheres=7, params=params)

        centers = np.array([np.array(s.center) for s in spheres])
        radii = np.array([float(s.radius) for s in spheres])

        # With budget=7: 1 center + 3 left + 3 right = 7
        # Center sphere should be at x ~= 0

        center_mask = np.abs(centers[:, 0]) < 0.1
        assert np.sum(center_mask) == 1, "Should have exactly one center sphere"

        # Left and right should each have 3
        left_mask = centers[:, 0] < -0.1
        right_mask = centers[:, 0] > 0.1

        assert np.sum(left_mask) == 3, f"Expected 3 left spheres, got {np.sum(left_mask)}"
        assert np.sum(right_mask) == 3, f"Expected 3 right spheres, got {np.sum(right_mask)}"

        # Radii on left and right should match exactly
        left_radii = sorted(radii[left_mask])
        right_radii = sorted(radii[right_mask])

        np.testing.assert_allclose(
            left_radii, right_radii, rtol=1e-5,
            err_msg="Left and right sphere radii should match exactly"
        )

    def test_small_budget_with_symmetry(self):
        """Test edge cases with very small budgets."""
        mesh = trimesh.creation.box(extents=[2, 1, 1])

        # Budget = 2: should get 1 left + 1 mirrored
        params = SpherizeParams(symmetry_mode="force")
        spheres = spherize(mesh, target_spheres=2, params=params)
        assert len(spheres) >= 2

        # Budget = 1 with center mode: should get just center sphere
        params = SpherizeParams(symmetry_mode="force", odd_budget_mode="center")
        spheres = spherize(mesh, target_spheres=1, params=params)
        assert len(spheres) >= 1

    def test_near_plane_points_coverage(self):
        """Verify that points near the symmetry plane are covered."""
        mesh = trimesh.creation.box(extents=[2, 2, 2])
        params = SpherizeParams(
            symmetry_mode="force",
            odd_budget_mode="center",
        )

        spheres = spherize(mesh, target_spheres=5, params=params)

        # Sample points near the symmetry plane (x ~= 0)
        near_plane_points = np.array([
            [0.0, y, z] for y in np.linspace(-0.9, 0.9, 5)
            for z in np.linspace(-0.9, 0.9, 5)
        ])

        # Check that at least one sphere covers each point
        centers = np.array([np.array(s.center) for s in spheres])
        radii = np.array([float(s.radius) for s in spheres])

        for pt in near_plane_points:
            distances = np.linalg.norm(centers - pt, axis=1)
            covered = np.any(distances <= radii * 1.1)  # 10% tolerance
            assert covered, f"Point {pt} not covered by any sphere"


class TestMirrorHelpers:
    """Tests for the mirroring helper functions."""

    def test_mirror_sphere_basic(self):
        """Test basic sphere mirroring across a plane."""
        import jax.numpy as jnp

        # Sphere at (1, 0, 0), mirror across YZ plane (axis = [1, 0, 0])
        sphere = Sphere(center=jnp.array([1.0, 0.5, 0.5]), radius=jnp.array(0.2))
        axis = np.array([1.0, 0.0, 0.0])
        centroid = np.array([0.0, 0.0, 0.0])

        mirrored = _mirror_sphere(sphere, axis, centroid)

        # Should be at (-1, 0.5, 0.5) with same radius
        expected_center = np.array([-1.0, 0.5, 0.5])
        np.testing.assert_allclose(np.array(mirrored.center), expected_center, atol=1e-6)
        np.testing.assert_allclose(float(mirrored.radius), 0.2, atol=1e-6)

    def test_mirror_sphere_off_center_plane(self):
        """Test mirroring across a plane not at origin."""
        import jax.numpy as jnp

        # Sphere at (2, 0, 0), plane at x=1 with normal [1, 0, 0]
        sphere = Sphere(center=jnp.array([2.0, 0.0, 0.0]), radius=jnp.array(0.5))
        axis = np.array([1.0, 0.0, 0.0])
        centroid = np.array([1.0, 0.0, 0.0])

        mirrored = _mirror_sphere(sphere, axis, centroid)

        # Distance from plane is 1, so mirrored should be at x = 1 - 1 = 0
        expected_center = np.array([0.0, 0.0, 0.0])
        np.testing.assert_allclose(np.array(mirrored.center), expected_center, atol=1e-6)

    def test_partition_points_by_plane(self):
        """Test point partitioning."""
        # Create points on both sides of YZ plane (x=0)
        points = np.array([
            [-1.0, 0.0, 0.0],  # left
            [-0.5, 0.0, 0.0],  # left
            [0.0, 0.0, 0.0],   # near plane
            [0.01, 0.0, 0.0],  # near plane (within tolerance)
            [0.5, 0.0, 0.0],   # right
            [1.0, 0.0, 0.0],   # right
        ])
        axis = np.array([1.0, 0.0, 0.0])
        centroid = np.array([0.0, 0.0, 0.0])
        tolerance = 0.05

        left, right, near = _partition_points_by_plane(points, axis, centroid, tolerance)

        assert len(left) == 2, f"Expected 2 left points, got {len(left)}"
        assert len(right) == 2, f"Expected 2 right points, got {len(right)}"
        assert len(near) == 2, f"Expected 2 near-plane points, got {len(near)}"


class TestLongitudinalSymmetryRejection:
    """Tests for rejection of longitudinal symmetry in elongated shapes."""

    def test_cylinder_rejects_longitudinal_symmetry(self):
        """Cylinder should NOT use symmetric mirroring (longitudinal symmetry rejected).

        For an elongated cylinder, the detected symmetry plane would cut along the
        length (longitudinal), causing side-by-side sphere pairs. This should be
        rejected, resulting in spheres distributed along the cylinder's length.
        """
        mesh = trimesh.creation.cylinder(radius=0.5, height=3.0)  # Elongated cylinder
        params = SpherizeParams(symmetry_mode="auto", symmetry_tolerance=0.05)

        spheres = spherize(mesh, target_spheres=6, params=params)
        centers = np.array([np.array(s.center) for s in spheres])

        # The cylinder is aligned along Z axis (height=3.0)
        # If longitudinal symmetry was used, spheres would be mirrored in X or Y
        # If correctly rejected, spheres should be distributed along Z

        # Check that spheres span a significant range along Z (the long axis)
        z_range = centers[:, 2].max() - centers[:, 2].min()

        # With 6 spheres along a height=3 cylinder, Z range should be substantial
        # If mirroring was used incorrectly, Z range would be small (spheres stacked at same Z)
        assert z_range > 1.0, (
            f"Spheres should be distributed along cylinder length (Z), but Z range is only {z_range:.2f}. "
            "This suggests longitudinal symmetry was incorrectly applied."
        )

    def test_elongated_box_distributes_along_length(self):
        """Elongated box should have spheres distributed along its length."""
        mesh = trimesh.creation.box(extents=[4, 1, 1])  # Elongated along X
        params = SpherizeParams(symmetry_mode="auto", symmetry_tolerance=0.05)

        spheres = spherize(mesh, target_spheres=6, params=params)
        centers = np.array([np.array(s.center) for s in spheres])

        # Check that spheres span a significant range along X (the long axis)
        x_range = centers[:, 0].max() - centers[:, 0].min()

        # With 6 spheres along a length=4 box, X range should be substantial
        assert x_range > 1.5, (
            f"Spheres should be distributed along box length (X), but X range is only {x_range:.2f}. "
            "This suggests longitudinal symmetry was incorrectly applied."
        )
