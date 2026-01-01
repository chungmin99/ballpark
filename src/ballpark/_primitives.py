"""Primitive shape detection and optimal sphere decomposition.

This module provides automatic detection of primitive shapes (box, cylinder, capsule)
and hardcoded optimal sphere decompositions for each primitive type.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from sklearn.decomposition import PCA
import trimesh

if TYPE_CHECKING:
    from ._spherize import Sphere


class PrimitiveType(Enum):
    """Detected primitive shape type."""
    BOX = auto()
    CYLINDER = auto()
    CAPSULE = auto()
    SPHERE = auto()
    UNKNOWN = auto()


@dataclass
class PrimitiveInfo:
    """Result of primitive detection."""
    primitive_type: PrimitiveType
    confidence: float  # 0-1 confidence score
    axes: np.ndarray  # (3, 3) principal axes (rows are axes)
    extents: np.ndarray  # (3,) extents along each axis
    center: np.ndarray  # (3,) centroid


def detect_primitive(mesh: trimesh.Trimesh, tolerance: float = 0.05) -> PrimitiveInfo:
    """
    Detect if mesh is a primitive shape (box, cylinder, capsule, sphere).

    Uses geometry fingerprinting based on:
    - Face count and topology
    - Bounding box vs mesh volume ratio
    - Surface curvature analysis
    - Vertex distribution analysis

    Args:
        mesh: The mesh to analyze
        tolerance: Relative tolerance for detection (0.05 = 5%)

    Returns:
        PrimitiveInfo with detected type and geometry info
    """
    # Sample points for analysis
    n_samples = min(2000, len(mesh.vertices) * 10)
    points = np.asarray(mesh.sample(n_samples))

    # Get principal axes and extents
    center = points.mean(axis=0)
    centered = points - center

    pca = PCA(n_components=3)
    pca.fit(centered)
    axes = pca.components_  # (3, 3) rows are principal axes

    # Project onto principal axes to get extents
    projected = centered @ axes.T  # (N, 3)
    extents = projected.max(axis=0) - projected.min(axis=0)

    # Sort axes by extent (largest first)
    sort_idx = np.argsort(-extents)
    axes = axes[sort_idx]
    extents = extents[sort_idx]

    # Create base info
    info = PrimitiveInfo(
        primitive_type=PrimitiveType.UNKNOWN,
        confidence=0.0,
        axes=axes,
        extents=extents,
        center=center,
    )

    # Test each primitive type
    sphere_conf = _test_sphere(mesh, points, extents, tolerance)
    capsule_conf = _test_capsule(mesh, points, axes, extents, tolerance)
    cylinder_conf = _test_cylinder(mesh, points, axes, extents, tolerance)
    box_conf = _test_box(mesh, points, axes, extents, tolerance)

    # Select best match
    candidates = [
        (PrimitiveType.SPHERE, sphere_conf),
        (PrimitiveType.CAPSULE, capsule_conf),
        (PrimitiveType.CYLINDER, cylinder_conf),
        (PrimitiveType.BOX, box_conf),
    ]

    best_type, best_conf = max(candidates, key=lambda x: x[1])

    if best_conf >= 0.7:  # Minimum confidence threshold
        info.primitive_type = best_type
        info.confidence = best_conf

    return info


def _test_sphere(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    extents: np.ndarray,
    tolerance: float,
) -> float:
    """Test if mesh is a sphere. Returns confidence 0-1."""
    # Spheres have equal extents in all directions
    aspect_ratios = extents / extents.max()
    if aspect_ratios.min() < 0.85:
        return 0.0

    # All points should be equidistant from center
    center = points.mean(axis=0)
    dists = np.linalg.norm(points - center, axis=1)

    mean_dist = dists.mean()
    dist_variance = np.std(dists) / mean_dist

    # Low variance in distances indicates sphere
    if dist_variance < tolerance * 2:
        return 1.0 - dist_variance / (tolerance * 2)

    return 0.0


def _test_capsule(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    axes: np.ndarray,
    extents: np.ndarray,
    tolerance: float,
) -> float:
    """Test if mesh is a capsule. Returns confidence 0-1."""
    # Capsule: one long axis, two equal shorter axes
    # extents[0] >> extents[1] ~ extents[2]

    if extents[0] < extents[1] * 1.3:
        return 0.0  # Not elongated enough

    cross_ratio = extents[2] / extents[1] if extents[1] > 1e-10 else 0
    if cross_ratio < 0.85:
        return 0.0  # Cross-section not circular

    # Project points onto main axis
    center = points.mean(axis=0)
    centered = points - center
    main_axis = axes[0]

    proj_main = centered @ main_axis
    proj_radial = centered - np.outer(proj_main, main_axis)
    radial_dists = np.linalg.norm(proj_radial, axis=1)

    # In middle section (cylinder part), radial distance should be constant
    # At ends (cap parts), radial distance decreases
    half_height = extents[0] / 2
    radius_estimate = extents[1] / 2

    # Check middle 50% for constant radius
    middle_mask = np.abs(proj_main) < half_height * 0.5
    if middle_mask.sum() < 10:
        return 0.0

    middle_radii = radial_dists[middle_mask]
    radius_variance = np.std(middle_radii) / (np.mean(middle_radii) + 1e-10)

    if radius_variance > tolerance * 2:
        return 0.0

    # Check that ends are hemispherical (points at ends closer to axis)
    end_mask = np.abs(proj_main) > half_height * 0.7
    if end_mask.sum() > 10:
        end_radii = radial_dists[end_mask]
        # End radii should be smaller than middle radii on average
        if np.mean(end_radii) >= np.mean(middle_radii):
            return 0.3  # Has constant middle but ends aren't hemispherical

    # Check volume ratio (capsule volume vs OBB volume)
    # Capsule: pi*r^2*h + (4/3)*pi*r^3 where h is cylinder height
    r = radius_estimate
    h = extents[0] - 2 * r  # Cylinder height (total - 2 hemispheres)
    if h < 0:
        h = 0
    capsule_vol = np.pi * r**2 * h + (4/3) * np.pi * r**3
    obb_vol = extents[0] * extents[1] * extents[2]

    expected_ratio = capsule_vol / obb_vol if obb_vol > 1e-10 else 0

    if mesh.is_volume:
        actual_ratio = mesh.volume / obb_vol if obb_vol > 1e-10 else 0
        ratio_error = abs(actual_ratio - expected_ratio) / (expected_ratio + 1e-10)
        if ratio_error < tolerance:
            return 0.95 - radius_variance

    return 0.7 - radius_variance


def _test_cylinder(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    axes: np.ndarray,
    extents: np.ndarray,
    tolerance: float,
) -> float:
    """Test if mesh is a cylinder. Returns confidence 0-1."""
    # Cylinder: one axis with flat ends, circular cross-section
    # Can be tall (extents[0] > extents[1]) or short (extents[0] ~ extents[1,2])

    # Cross-section should be circular
    cross_ratio = extents[2] / extents[1] if extents[1] > 1e-10 else 0
    if cross_ratio < 0.85:
        return 0.0  # Cross-section not circular

    # Project points onto main axis
    center = points.mean(axis=0)
    centered = points - center
    main_axis = axes[0]

    proj_main = centered @ main_axis
    proj_radial = centered - np.outer(proj_main, main_axis)
    radial_dists = np.linalg.norm(proj_radial, axis=1)

    # For cylinder surface samples, variance is higher because:
    # - Curved surface points are at fixed radius
    # - Flat cap points are at varying radii (0 to cylinder_radius)
    # So we check the main body (middle 70%) for constant radius
    half_height = extents[0] / 2
    middle_mask = np.abs(proj_main) < half_height * 0.7

    if middle_mask.sum() > 20:
        middle_radii = radial_dists[middle_mask]
        radius_variance = np.std(middle_radii) / (np.mean(middle_radii) + 1e-10)
    else:
        radius_variance = np.std(radial_dists) / (np.mean(radial_dists) + 1e-10)

    # Be more lenient with variance (0.25 instead of 0.15)
    if radius_variance > 0.25:
        return 0.0

    # Volume check - most reliable for cylinders
    r = extents[1] / 2
    h = extents[0]
    cylinder_vol = np.pi * r**2 * h
    obb_vol = extents[0] * extents[1] * extents[2]

    expected_ratio = cylinder_vol / obb_vol if obb_vol > 1e-10 else 0

    if mesh.is_volume:
        actual_ratio = mesh.volume / obb_vol if obb_vol > 1e-10 else 0
        ratio_error = abs(actual_ratio - expected_ratio) / (expected_ratio + 1e-10)
        if ratio_error < tolerance:
            return 0.95 - radius_variance * 0.5  # High confidence if volume matches
        elif ratio_error < tolerance * 2:
            return 0.85 - radius_variance * 0.5

    return max(0.0, 0.7 - radius_variance * 2)


def _test_box(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    axes: np.ndarray,
    extents: np.ndarray,
    tolerance: float,
) -> float:
    """Test if mesh is a box. Returns confidence 0-1."""
    # Box: rectangular shape with 6 faces, 8 vertices, 12 edges

    # Check vertex count (box has 8 vertices, subdivided boxes have more)
    n_verts = len(mesh.vertices)
    if n_verts < 8:
        return 0.0

    # Check face count (box has 6 faces * 2 triangles = 12 triangles minimum)
    n_faces = len(mesh.faces)
    if n_faces < 12:
        return 0.0

    # Project points onto each axis and check for bimodal distribution
    center = points.mean(axis=0)
    centered = points - center

    bimodal_scores = []
    for i in range(3):
        proj = centered @ axes[i]
        half_extent = extents[i] / 2

        # Points should cluster near +/- half_extent (the faces)
        # Count points near faces vs in middle
        face_threshold = half_extent * 0.3
        near_faces = (np.abs(proj) > half_extent - face_threshold).sum()
        in_middle = (np.abs(proj) < half_extent * 0.5).sum()

        # For a box, most surface points are near faces
        if len(points) > 0:
            face_ratio = near_faces / len(points)
            bimodal_scores.append(face_ratio)

    avg_bimodal = np.mean(bimodal_scores) if bimodal_scores else 0

    # Volume check (box volume should match OBB volume closely)
    obb_vol = extents[0] * extents[1] * extents[2]

    if mesh.is_volume and obb_vol > 1e-10:
        vol_ratio = mesh.volume / obb_vol
        if vol_ratio > 0.95:  # Very box-like
            return 0.9 + 0.1 * avg_bimodal
        elif vol_ratio > 0.85:
            return 0.7 + 0.2 * avg_bimodal

    # Fallback: use bimodal score
    if avg_bimodal > 0.5:
        return 0.6 + 0.3 * avg_bimodal

    return 0.0


def spherize_primitive(
    info: PrimitiveInfo,
    target_spheres: int,
    padding: float = 1.02,
) -> list["Sphere"]:
    """
    Generate optimal sphere decomposition for a detected primitive.

    Uses hardcoded placement strategies based on primitive type and aspect ratio.

    Args:
        info: Detected primitive information
        target_spheres: Target number of spheres
        padding: Radius padding multiplier

    Returns:
        List of optimally placed spheres
    """
    from ._spherize import Sphere

    if info.primitive_type == PrimitiveType.BOX:
        return _spherize_box(info, target_spheres, padding)
    elif info.primitive_type == PrimitiveType.CYLINDER:
        return _spherize_cylinder(info, target_spheres, padding)
    elif info.primitive_type == PrimitiveType.CAPSULE:
        return _spherize_capsule(info, target_spheres, padding)
    elif info.primitive_type == PrimitiveType.SPHERE:
        return _spherize_sphere(info, padding)
    else:
        return []  # Unknown primitive, fallback to general algorithm


def _spherize_sphere(info: PrimitiveInfo, padding: float) -> list["Sphere"]:
    """Optimal decomposition for sphere: single centered sphere."""
    from ._spherize import Sphere

    # Use max extent to ensure coverage, with padding
    # Add extra padding for spheres since surface sampling adds variance
    radius = info.extents.max() / 2 * padding * 1.05
    return [Sphere(center=jnp.asarray(info.center), radius=jnp.asarray(radius))]


def _spherize_box(
    info: PrimitiveInfo,
    target_spheres: int,
    padding: float,
) -> list["Sphere"]:
    """
    Optimal decomposition for box using grid-based placement.

    Adapts grid to box aspect ratio:
    - Cube-ish: 2x2x2 grid
    - Elongated: 1xNx1 or similar
    """
    from ._spherize import Sphere

    extents = info.extents
    axes = info.axes
    center = info.center

    # Determine grid dimensions based on aspect ratio and budget
    grid = _compute_box_grid(extents, target_spheres)
    nx, ny, nz = grid

    # Compute sphere radius (fit within grid cells with overlap)
    cell_extents = extents / np.array([nx, ny, nz])
    # Use largest inscribed sphere in cell, with padding
    radius = cell_extents.min() / 2 * padding

    # Also ensure spheres overlap enough for coverage
    # Sphere centers at grid intersections, radius should reach neighbors
    min_spacing = cell_extents.min()
    coverage_radius = min_spacing * 0.55 * padding  # 55% of spacing for overlap
    radius = max(radius, coverage_radius)

    spheres = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # Compute position in local coordinates
                # Centers at midpoints of grid cells
                local_pos = np.array([
                    (i + 0.5) / nx - 0.5,
                    (j + 0.5) / ny - 0.5,
                    (k + 0.5) / nz - 0.5,
                ]) * extents

                # Transform to world coordinates
                world_pos = center + local_pos @ axes

                spheres.append(Sphere(
                    center=jnp.asarray(world_pos),
                    radius=jnp.asarray(radius),
                ))

    return spheres


def _compute_box_grid(extents: np.ndarray, target_spheres: int) -> tuple[int, int, int]:
    """
    Compute optimal grid dimensions for box spherization.

    Returns (nx, ny, nz) where nx >= ny >= nz and nx*ny*nz <= target_spheres.
    """
    if target_spheres <= 1:
        return (1, 1, 1)

    # Normalize extents (largest = 1)
    norm_extents = extents / extents.max()

    # Find best grid that respects aspect ratio and budget
    best_grid = (1, 1, 1)
    best_score = float('inf')

    # Search over possible grid dimensions
    max_per_axis = int(np.ceil(target_spheres ** (1/3))) + 2

    for nx in range(1, max_per_axis + 1):
        for ny in range(1, nx + 1):
            for nz in range(1, ny + 1):
                count = nx * ny * nz
                if count > target_spheres:
                    continue

                # Score: prefer grids that match aspect ratio
                grid_ratios = np.array([nx, ny, nz], dtype=float)
                grid_ratios /= grid_ratios.max()

                # How well does grid ratio match extent ratio?
                ratio_error = np.sum((grid_ratios - norm_extents) ** 2)

                # Penalize under-using budget
                usage_penalty = (target_spheres - count) / target_spheres

                score = ratio_error + usage_penalty * 0.5

                if score < best_score:
                    best_score = score
                    best_grid = (nx, ny, nz)

    return best_grid


def _spherize_cylinder(
    info: PrimitiveInfo,
    target_spheres: int,
    padding: float,
) -> list["Sphere"]:
    """
    Optimal decomposition for cylinder using axial placement.

    For cylinders, spheres are placed along the axis. Each sphere needs
    radius large enough to cover both the circular cross-section (cylinder
    radius) and its portion of the length.
    """
    from ._spherize import Sphere

    extents = info.extents
    axes = info.axes
    center = info.center

    # Cylinder axis is the principal axis with largest extent
    height = extents[0]
    # Cross-section extents are the full diameter, so radius = extent/2
    radius_cyl = (extents[1] + extents[2]) / 4  # Average radius (extent/2 averaged)

    main_axis = axes[0]

    if target_spheres == 1:
        # Single sphere at center - use full bounding sphere
        bounding_radius = np.sqrt((height/2)**2 + radius_cyl**2)
        return [Sphere(
            center=jnp.asarray(center),
            radius=jnp.asarray(bounding_radius * padding),
        )]

    # For n spheres along axis, each covers height/n of the cylinder
    # Sphere must reach: radius_cyl laterally, and (height/n)/2 axially
    n_axial = target_spheres
    segment_half_height = height / (2 * n_axial)

    # Sphere radius must be at least the cylinder radius to cover cross-section
    # AND large enough to overlap with neighbors for axial coverage
    # Use pythagorean: need to reach (segment_half_height, radius_cyl) from center
    min_radius_for_coverage = np.sqrt(segment_half_height**2 + radius_cyl**2)

    # Also need overlap - with n spheres evenly distributed, adjacent sphere
    # centers are height/(n-1) apart if n>1
    if n_axial > 1:
        center_spacing = height / (n_axial - 1)
        # Need radius >= center_spacing/2 to just touch neighbors
        # For good coverage, use slightly more
        min_radius_for_overlap = center_spacing * 0.55
    else:
        min_radius_for_overlap = radius_cyl

    sphere_radius = max(min_radius_for_coverage, min_radius_for_overlap, radius_cyl) * padding

    spheres = []
    for i in range(n_axial):
        # Distribute evenly along axis
        if n_axial == 1:
            t = 0.0
        else:
            t = (i / (n_axial - 1)) - 0.5  # -0.5 to 0.5
        # Position spans from -height/2 to +height/2
        axial_pos = t * height
        pos = center + axial_pos * main_axis

        spheres.append(Sphere(
            center=jnp.asarray(pos),
            radius=jnp.asarray(sphere_radius),
        ))

    return spheres


def _compute_cylinder_layout(height: float, radius: float, target: int) -> tuple[int, int]:
    """
    Compute optimal (n_axial, n_radial) for cylinder spherization.

    Returns:
        (n_axial, n_radial) where n_radial is 1 for thin cylinders or >1 for wide ones.
    """
    if target <= 1:
        return (1, 1)

    aspect = float(height / (2 * radius)) if radius > 1e-10 else 10.0

    if aspect > 2:
        # Tall cylinder: mostly axial
        n_axial = min(target, int(np.ceil(aspect)))
        n_radial = 1
    elif aspect > 0.5:
        # Medium cylinder: balance axial and radial
        n_axial = max(1, int(np.sqrt(target * aspect)))
        n_radial = max(1, target // n_axial)
    else:
        # Disc-like: more radial
        n_radial = min(target, int(np.ceil(1.0 / aspect)))
        n_axial = max(1, target // n_radial)

    # Ensure we don't exceed budget
    while n_axial * n_radial > target and n_axial > 1:
        n_axial -= 1
    while n_axial * n_radial > target and n_radial > 1:
        n_radial -= 1

    return (max(1, n_axial), max(1, n_radial))


def _spherize_capsule(
    info: PrimitiveInfo,
    target_spheres: int,
    padding: float,
) -> list["Sphere"]:
    """
    Optimal decomposition for capsule: spheres along axis.

    A capsule is like a cylinder but with hemispherical caps. The optimal
    sphere placement is along the axis with radius = capsule_radius so each
    sphere matches the cross-section.
    """
    from ._spherize import Sphere

    extents = info.extents
    axes = info.axes
    center = info.center

    # Capsule: main axis is longest extent
    # The cross-section extents are the full diameter
    total_length = extents[0]
    capsule_radius = (extents[1] + extents[2]) / 4  # extent/2 averaged

    main_axis = axes[0]

    if target_spheres == 1:
        # Single sphere at center - use bounding sphere
        bounding_radius = np.sqrt((total_length/2)**2 + capsule_radius**2)
        return [Sphere(
            center=jnp.asarray(center),
            radius=jnp.asarray(bounding_radius * padding),
        )]

    # For n spheres along axis, use same logic as cylinder
    n_axial = target_spheres
    segment_half_height = total_length / (2 * n_axial)

    # Sphere radius covers both cross-section and segment height
    min_radius_for_coverage = np.sqrt(segment_half_height**2 + capsule_radius**2)

    # Overlap radius for adjacent spheres
    if n_axial > 1:
        center_spacing = total_length / (n_axial - 1)
        min_radius_for_overlap = center_spacing * 0.55
    else:
        min_radius_for_overlap = capsule_radius

    sphere_radius = max(min_radius_for_coverage, min_radius_for_overlap, capsule_radius) * padding

    spheres = []
    for i in range(n_axial):
        if n_axial == 1:
            t = 0.0
        else:
            t = (i / (n_axial - 1)) - 0.5  # -0.5 to 0.5
        axial_pos = t * total_length
        pos = center + axial_pos * main_axis

        spheres.append(Sphere(
            center=jnp.asarray(pos),
            radius=jnp.asarray(sphere_radius),
        ))

    return spheres
