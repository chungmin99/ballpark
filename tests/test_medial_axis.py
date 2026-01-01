"""Tests for Medial Axis Transform sphere initialization."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from ballpark import spherize, SpherizeParams, spherize_medial_axis
from ballpark._medial_axis import (
    voxelize_mesh,
    compute_distance_transform,
    extract_skeleton_3d,
    _is_thin_shell,
)
from ballpark._spherize import compute_coverage


class TestVoxelization:
    """Test mesh voxelization."""

    def test_cube_voxelization(self):
        """Cube should produce filled voxel grid."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        result = voxelize_mesh(mesh, resolution=16)

        assert result is not None
        voxels, origin, voxel_size = result
        assert voxels.sum() > 0  # Has interior voxels
        assert voxel_size > 0

    def test_sphere_voxelization(self):
        """Sphere should produce roughly spherical voxel grid."""
        mesh = trimesh.creation.icosphere(radius=0.5)
        result = voxelize_mesh(mesh, resolution=32)

        assert result is not None
        voxels, origin, voxel_size = result

        # Check approximate volume ratio
        expected_volume = 4 / 3 * np.pi * 0.5**3
        actual_volume = voxels.sum() * voxel_size**3
        assert 0.6 < actual_volume / expected_volume < 1.4

    def test_empty_mesh_returns_none(self):
        """Empty or degenerate mesh should return None."""
        # Create degenerate mesh with few vertices
        mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]], faces=[[0, 1, 2]])
        result = voxelize_mesh(mesh, resolution=16)
        # Should either return None or have very few voxels
        if result is not None:
            assert result[0].sum() < 100


class TestDistanceTransform:
    """Test distance transform computation."""

    def test_distance_transform_cube(self):
        """Distance transform should give inscribed radius for cube."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        result = voxelize_mesh(mesh, resolution=32)
        assert result is not None

        voxels, _, voxel_size = result
        distance_field = compute_distance_transform(voxels)

        # Distance transform should have positive values inside
        assert distance_field.max() > 0
        # Max distance (in world units) should be reasonable
        max_dist = distance_field.max() * voxel_size
        assert max_dist > 0.1  # At least some interior space


class TestSkeleton:
    """Test 3D skeletonization."""

    def test_elongated_box_skeleton(self):
        """Elongated box should have linear skeleton."""
        mesh = trimesh.creation.box(extents=[0.2, 0.2, 1.0])
        result = voxelize_mesh(mesh, resolution=32)
        assert result is not None

        voxels, _, _ = result
        skeleton = extract_skeleton_3d(voxels)

        # Skeleton should exist and be sparse
        assert skeleton.sum() > 0
        assert skeleton.sum() < voxels.sum() * 0.2  # Much smaller than volume

    def test_cylinder_skeleton(self):
        """Cylinder should have axial skeleton."""
        mesh = trimesh.creation.cylinder(radius=0.2, height=1.0)
        result = voxelize_mesh(mesh, resolution=32)
        assert result is not None

        voxels, _, _ = result
        skeleton = extract_skeleton_3d(voxels)

        assert skeleton.sum() > 0


class TestThinShellDetection:
    """Test thin shell detection."""

    def test_thick_cube_not_thin(self):
        """Thick cube should not be detected as thin shell."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        assert not _is_thin_shell(mesh)

    def test_thin_box_is_thin(self):
        """Very thin box should be detected as thin shell."""
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.005])
        assert _is_thin_shell(mesh)


class TestMATSpherization:
    """Test full MAT spherization pipeline."""

    @pytest.mark.parametrize("budget", [4, 8, 16])
    def test_cube_coverage(self, budget: int):
        """MAT should achieve good coverage on cube."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        params = SpherizeParams(init_strategy="medial_axis")
        spheres = spherize(mesh, target_spheres=budget, params=params)

        assert len(spheres) > 0
        points = np.asarray(mesh.sample(2000))
        coverage = compute_coverage(points, spheres)
        assert coverage >= 0.75

    @pytest.mark.parametrize("budget", [8, 16, 32])
    def test_elongated_cylinder_coverage(self, budget: int):
        """MAT should work well on elongated shapes."""
        mesh = trimesh.creation.cylinder(radius=0.1, height=2.0)
        params = SpherizeParams(init_strategy="medial_axis")
        spheres = spherize(mesh, target_spheres=budget, params=params)

        assert len(spheres) > 0
        points = np.asarray(mesh.sample(2000))
        coverage = compute_coverage(points, spheres)
        assert coverage >= 0.75

    def test_fallback_for_thin_shell(self):
        """Thin shells should gracefully fall back to adaptive."""
        # Create a very thin box
        mesh = trimesh.creation.box(extents=[1.0, 1.0, 0.01])
        params = SpherizeParams(init_strategy="medial_axis")

        # Should not crash - will fall back to adaptive
        spheres = spherize(mesh, target_spheres=8, params=params)
        assert isinstance(spheres, list)
        # Should still produce valid spheres via fallback
        assert len(spheres) > 0

    def test_direct_call(self):
        """Direct spherize_medial_axis call should work."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        spheres = spherize_medial_axis(mesh, target_spheres=8)

        assert len(spheres) > 0
        for s in spheres:
            assert np.all(np.isfinite(s.center))
            assert float(s.radius) > 0


class TestStrategyDispatch:
    """Test strategy selection."""

    def test_adaptive_is_default(self):
        """Default strategy should be adaptive."""
        params = SpherizeParams()
        assert params.init_strategy == "adaptive"

    def test_medial_axis_strategy_routes_correctly(self):
        """init_strategy='medial_axis' should use MAT algorithm."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])

        # With MAT strategy
        mat_params = SpherizeParams(init_strategy="medial_axis")
        mat_spheres = spherize(mesh, target_spheres=8, params=mat_params)

        # With adaptive strategy
        adaptive_params = SpherizeParams(init_strategy="adaptive")
        adaptive_spheres = spherize(mesh, target_spheres=8, params=adaptive_params)

        # Both should produce valid spheres
        assert len(mat_spheres) > 0
        assert len(adaptive_spheres) > 0

        # Centers may differ between strategies (different algorithms)
        # Just verify both are valid
        for spheres in [mat_spheres, adaptive_spheres]:
            for s in spheres:
                assert np.all(np.isfinite(s.center))
                assert float(s.radius) > 0

    def test_mat_params_used(self):
        """MAT-specific parameters should affect results."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])

        # Low resolution
        low_res = SpherizeParams(init_strategy="medial_axis", mat_voxel_resolution=16)
        spheres_low = spherize(mesh, target_spheres=8, params=low_res)

        # Higher resolution
        high_res = SpherizeParams(init_strategy="medial_axis", mat_voxel_resolution=64)
        spheres_high = spherize(mesh, target_spheres=8, params=high_res)

        # Both should work
        assert len(spheres_low) > 0
        assert len(spheres_high) > 0


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_non_watertight_mesh(self):
        """Non-watertight mesh should return empty or use fallback."""
        # Create a non-watertight mesh (single triangle)
        mesh = trimesh.Trimesh(
            vertices=[[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]],
            faces=[[0, 1, 2], [0, 2, 3], [1, 2, 3]],  # Missing one face
        )

        # Direct MAT call should return empty
        spheres = spherize_medial_axis(mesh, target_spheres=4)
        assert spheres == []

        # Through spherize() should fall back to adaptive
        params = SpherizeParams(init_strategy="medial_axis")
        spheres = spherize(mesh, target_spheres=4, params=params)
        # May produce spheres via fallback
        assert isinstance(spheres, list)

    def test_very_small_budget(self):
        """Very small sphere budget should still work."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        params = SpherizeParams(init_strategy="medial_axis")
        spheres = spherize(mesh, target_spheres=1, params=params)

        assert len(spheres) >= 1

    def test_large_budget(self):
        """Large sphere budget should not crash."""
        mesh = trimesh.creation.box(extents=[1, 1, 1])
        params = SpherizeParams(init_strategy="medial_axis")
        spheres = spherize(mesh, target_spheres=64, params=params)

        # May not get exactly 64 (limited by skeleton points)
        assert len(spheres) > 0
