"""Adaptive tight sphere fitting algorithm."""

from __future__ import annotations

from typing import cast, TYPE_CHECKING

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
import trimesh
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, QhullError

if TYPE_CHECKING:
    from ._config import SpherizeParams


@jdc.pytree_dataclass
class Sphere:
    """A sphere defined by center and radius.

    Can represent a single sphere (center: (3,), radius: scalar) or
    a batch of spheres (center: (N, 3), radius: (N,)).
    """

    center: jnp.ndarray
    radius: jnp.ndarray


def spherize(
    mesh: trimesh.Trimesh,
    target_spheres: int,
    params: "SpherizeParams | None" = None,
) -> list[Sphere]:
    """
    Adaptive splitting with tight sphere fitting.

    Uses a hybrid approach:
    1. Detect if mesh is a primitive (box, cylinder, capsule, sphere)
    2. For primitives: use optimal hardcoded decomposition
    3. For general meshes: use adaptive splitting with tight sphere fitting

    The actual number of spheres may slightly exceed target_spheres due to
    minimum allocation constraints during recursive splitting.

    Args:
        mesh: The mesh to spherize
        target_spheres: Target number of spheres to generate
        params: Algorithm parameters. If None, uses defaults.

    Returns:
        List of Sphere objects covering the mesh
    """
    # Import here to avoid circular import
    from ._config import SpherizeParams
    from ._symmetry import get_primitive_symmetry_planes, SymmetryInfo
    from ._primitives import detect_primitive, spherize_primitive, PrimitiveType

    p = params or SpherizeParams()

    # Strategy dispatch - MAT initialization
    if p.init_strategy == "medial_axis":
        from ._medial_axis import spherize_medial_axis

        spheres = spherize_medial_axis(mesh, target_spheres, p)
        if spheres:
            # Apply existing post-processing (containment, uniform radius)
            if p.containment_samples > 0 and mesh.is_watertight:
                spheres = [
                    cap_radius_by_containment(
                        s, mesh, p.containment_samples, p.min_containment_fraction
                    )
                    for s in spheres
                ]

            if p.uniform_radius and len(spheres) > 1:
                radii = np.array([s.radius for s in spheres])
                median_radius = np.median(radii)
                spheres = [
                    jdc.replace(
                        s, radius=np.clip(s.radius, median_radius * 0.4, median_radius * 2.5)
                    )
                    for s in spheres
                ]

            return spheres
        # Fall through to adaptive if MAT returns empty

    # Try primitive detection first (unless geometry_type is explicitly set)
    # Only use for small-medium budgets where primitive decomposition is effective
    if p.geometry_type is None and target_spheres <= 32:
        primitive_info = detect_primitive(mesh)
        if primitive_info.primitive_type != PrimitiveType.UNKNOWN and primitive_info.confidence >= 0.7:
            # Use optimal hardcoded decomposition for detected primitive
            spheres = spherize_primitive(primitive_info, target_spheres, p.padding)
            if spheres:
                # Quick quality check - ensure reasonable coverage
                test_points = cast(np.ndarray, mesh.sample(1000))
                coverage = compute_coverage(test_points, spheres)
                if coverage >= 0.85:  # Primitive spherization is good enough
                    return spheres
                # Otherwise fall through to general algorithm

    # Try convex decomposition for concave meshes (if enabled)
    if p.use_decomposition and mesh.is_watertight:
        from ._decompose import needs_decomposition, spherize_decomposed

        if needs_decomposition(mesh, threshold=p.decomposition_threshold):
            decomposed_spheres = spherize_decomposed(
                mesh,
                target_spheres,
                params=params,
                max_parts=p.max_decomposition_parts,
            )
            if decomposed_spheres:
                return decomposed_spheres

    # Unpack params
    target_tightness = p.target_tightness
    aspect_threshold = p.aspect_threshold
    n_samples = p.n_samples
    padding = p.padding
    percentile = p.percentile
    max_radius_ratio = p.max_radius_ratio
    uniform_radius = p.uniform_radius
    detect_symmetry = p.detect_symmetry
    geometry_type = p.geometry_type
    prefer_symmetric_splits = p.prefer_symmetric_splits
    n_volume_samples = p.n_volume_samples
    thickness_radius_scale = p.thickness_radius_scale
    containment_samples = p.containment_samples
    min_containment_fraction = p.min_containment_fraction

    points = cast(np.ndarray, mesh.sample(n_samples))

    # Sample interior/volume points for thickness estimation
    volume_points: np.ndarray | None = None
    if n_volume_samples > 0:
        try:
            if mesh.is_watertight:
                volume_points = trimesh.sample.volume_mesh(mesh, n_volume_samples)
                if len(volume_points) < 20:
                    # Too sparse (very thin mesh), fall back to surface-only
                    volume_points = None
        except Exception:
            # volume_mesh can fail for non-watertight or degenerate meshes
            volume_points = None

    # Auto-enable symmetry detection for known primitives
    symmetry_info: SymmetryInfo | None = None
    if geometry_type in ("box", "cylinder") or (detect_symmetry and geometry_type != "sphere"):
        if geometry_type in ("box", "cylinder"):
            # Use known symmetry planes for primitives
            symmetry_planes = get_primitive_symmetry_planes(points, geometry_type)
            if symmetry_planes:
                symmetry_info = SymmetryInfo(
                    reflection_planes=symmetry_planes,
                    rotation_axes=[],
                    symmetry_score=1.0,
                )
        elif detect_symmetry:
            # Fall back to general symmetry detection
            from ._symmetry import detect_symmetry as detect_symmetry_fn
            symmetry_info = detect_symmetry_fn(points, tolerance=p.symmetry_tolerance)

    # Compute max allowed radius
    bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0)).item()
    max_radius = bbox_diag * max_radius_ratio

    def should_split(pts, sphere, budget):
        if len(pts) < 20 or budget <= 1:
            return False

        aspect = get_aspect_ratio(pts)
        tightness = _compute_sphere_bloat(pts, sphere)

        # Force split if sphere exceeds max radius
        if sphere.radius > max_radius:
            return True

        # Split if elongated OR loose
        if aspect > aspect_threshold:
            return True
        if tightness > target_tightness:
            return True

        return False

    def split(pts, budget, vol_pts, target_mesh):
        if len(pts) < 15 or budget <= 0:
            if len(pts) > 0 and budget > 0:
                s = fit_sphere_thickness_aware(
                    pts, vol_pts, padding, percentile, thickness_radius_scale
                )
                s = jdc.replace(s, radius=min(s.radius, max_radius))  # cap radius
                # Apply containment check to prevent over-extension
                if containment_samples > 0 and target_mesh is not None:
                    s = cap_radius_by_containment(
                        s, target_mesh, containment_samples, min_containment_fraction
                    )
                return [s]
            return []

        sphere = fit_sphere_thickness_aware(
            pts, vol_pts, padding, percentile, thickness_radius_scale
        )

        if not should_split(pts, sphere, budget):
            # Apply containment check only to final (leaf) spheres
            if containment_samples > 0 and target_mesh is not None:
                sphere = cap_radius_by_containment(
                    sphere, target_mesh, containment_samples, min_containment_fraction
                )
            return [sphere]

        # Choose split axis (prefer symmetry planes if available)
        split_axis = None
        if symmetry_info is not None and prefer_symmetric_splits and symmetry_info.preferred_split_axes:
            # Use PCA to find principal direction
            pca = PCA(n_components=1)
            pca.fit(pts)
            principal_axis = pca.components_[0]

            # Find symmetry plane most aligned with principal direction
            best_alignment = 0.0
            for plane_normal in symmetry_info.preferred_split_axes:
                alignment = abs(np.dot(principal_axis, plane_normal))
                if alignment > best_alignment:
                    best_alignment = alignment
                    split_axis = plane_normal

            # Only use symmetry plane if well-aligned (>70% alignment)
            if best_alignment < 0.7:
                split_axis = principal_axis
        else:
            # Standard PCA split
            pca = PCA(n_components=1)
            pca.fit(pts)
            split_axis = pca.components_[0]

        proj = pts @ split_axis
        # Blend median with geometric midpoint for more symmetric splits
        proj_mid = (proj.min() + proj.max()) / 2
        split_point = 0.5 * np.median(proj) + 0.5 * proj_mid

        left = pts[proj <= split_point]
        right = pts[proj > split_point]

        # Allocate budget proportionally to point count
        left_frac = len(left) / len(pts)
        left_budget = max(1, int(round(budget * left_frac)))
        right_budget = max(1, budget - left_budget)

        result = []
        if len(left) >= 10:
            result.extend(split(left, left_budget, vol_pts, target_mesh))
        if len(right) >= 10:
            result.extend(split(right, right_budget, vol_pts, target_mesh))

        if not result:
            return [sphere]

        # Backtracking: check if split improves quality
        # Only perform this check if backtrack_threshold > 1.0 to avoid overhead
        # Quality = coverage_fraction * (hull_volume / total_sphere_volume)
        # Higher quality means better coverage with tighter fit
        if p.backtrack_threshold > 1.0:
            parent_quality = compute_split_quality(pts, [sphere])
            children_quality = compute_split_quality(pts, result)

            # If children don't improve enough, keep parent instead
            # Example: threshold=1.05 requires 5% quality improvement to keep split
            if children_quality < parent_quality * p.backtrack_threshold:
                return [sphere]

        return result

    spheres = split(points, target_spheres, volume_points, mesh)

    # Post-process: cap radius variance for more uniformity (may cause under-approximation)
    if uniform_radius and len(spheres) > 1:
        radii = np.array([s.radius for s in spheres])
        median_radius = np.median(radii)
        spheres = [
            jdc.replace(s, radius=np.clip(s.radius, median_radius * 0.4, median_radius * 2.5))
            for s in spheres
        ]

    return spheres


def fit_sphere_thickness_aware(
    surface_points: np.ndarray,
    volume_points: np.ndarray | None,
    padding: float = 1.0,
    percentile: float = 98.0,
    thickness_scale: float = 1.2,
) -> Sphere:
    """
    Fit a sphere with thickness-aware radius capping.

    Uses interior/volume points to estimate local mesh thickness and caps
    the sphere radius to avoid over-extension beyond the mesh boundary.

    Args:
        surface_points: (N, 3) array of surface points to enclose
        volume_points: Optional (M, 3) array of interior points for thickness estimation
        padding: Radius multiplier for safety margin
        percentile: Percentile of distances to use for radius
        thickness_scale: Allow radius up to this factor of thickness estimate

    Returns:
        Sphere that encloses the surface points, with radius capped by thickness
    """
    # Fit sphere to surface points using existing logic
    base_sphere = fit_sphere_minmax(surface_points, padding, percentile)

    if volume_points is None or len(volume_points) < 10:
        return base_sphere

    # Find volume points near this sphere's region
    center = np.asarray(base_sphere.center)
    base_radius = float(base_sphere.radius)
    vol_dists = np.linalg.norm(volume_points - center, axis=1)

    # Look for interior points within 2x the sphere radius
    nearby_mask = vol_dists < base_radius * 2.0
    n_nearby = nearby_mask.sum()

    if n_nearby < 5:
        return base_sphere  # No nearby interior points, can't estimate thickness

    # Estimate thickness as 95th percentile distance to nearby interior points
    # This represents how far solid material extends from the center
    thickness_estimate = np.percentile(vol_dists[nearby_mask], 95)

    # Cap radius based on thickness estimate
    # thickness_scale allows some margin (default 1.2 = 20% larger than thickness)
    max_radius = thickness_estimate * thickness_scale * padding
    capped_radius = min(base_radius, max_radius)

    # Ensure we don't shrink too aggressively (at least 50% of original)
    capped_radius = max(capped_radius, base_radius * 0.5)

    return Sphere(center=base_sphere.center, radius=jnp.asarray(capped_radius))


def fit_sphere_minmax(
    points: np.ndarray, padding: float = 1.0, percentile: float = 98.0
) -> Sphere:
    """
    Tighter sphere fitting using iterative refinement.

    Args:
        points: (N, 3) array of points to enclose
        padding: Radius multiplier for safety margin (1.02 = 2% larger)
        percentile: Use this percentile of distances instead of max (handles outliers)

    Returns:
        Sphere that encloses the points
    """
    if len(points) < 4:
        c = points.mean(axis=0) if len(points) > 0 else np.zeros(3)
        if len(points) > 0:
            r = np.linalg.norm(points - c, axis=1).max()
            r = max(r, 1e-4)  # Ensure minimum radius for degenerate cases
        else:
            r = 0.01
        return Sphere(center=jnp.asarray(c), radius=jnp.asarray(r * padding))

    center = points.mean(axis=0)

    # Iterative refinement: move center to reduce max distance
    for _ in range(5):
        dists = np.linalg.norm(points - center, axis=1)
        farthest = points[np.argmax(dists)]
        # Move center slightly toward farthest point
        center = center + 0.1 * (farthest - center)

    dists = np.linalg.norm(points - center, axis=1)
    # Blend percentile with median for more uniform radii (keep mostly percentile to avoid under-approx)
    p_high = np.percentile(dists, percentile)
    p_med = np.median(dists)
    radius = (0.85 * p_high + 0.15 * p_med) * padding

    return Sphere(center=jnp.asarray(center), radius=jnp.asarray(radius))


def get_aspect_ratio(points: np.ndarray) -> float:
    """Compute aspect ratio from PCA eigenvalues."""
    if len(points) < 10:
        return 1.0
    pca = PCA(n_components=min(3, len(points)))
    pca.fit(points)
    var = pca.explained_variance_
    if len(var) < 2 or var[0] < 1e-10 or var[1] < 1e-10:
        return 1.0
    # Cap the ratio to avoid extreme values for degenerate cases
    ratio = np.sqrt(var[0] / var[1])
    return min(ratio, 100.0)


def _compute_sphere_bloat(points: np.ndarray, sphere: Sphere) -> float:
    """Compute sphere volume / convex hull volume ratio (internal).

    Higher values indicate the sphere is loose compared to the point cloud.
    Used internally to decide when to split regions.
    """
    if len(points) < 4:
        return 1.0
    try:
        hull_vol = ConvexHull(points).volume
        sphere_vol = 4 / 3 * np.pi * float(sphere.radius) ** 3
        return float(sphere_vol / (hull_vol + 1e-10))
    except (QhullError, ValueError):
        return 1.0


def _fibonacci_sphere(n: int) -> np.ndarray:
    """Generate n approximately uniform points on unit sphere.

    Uses Fibonacci lattice for uniform distribution.

    Args:
        n: Number of points to generate

    Returns:
        (n, 3) array of unit vectors
    """
    if n <= 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])

    indices = np.arange(n, dtype=float)
    phi = np.pi * (3.0 - np.sqrt(5.0))  # golden angle

    y = 1 - (indices / (n - 1)) * 2  # y goes from 1 to -1
    radius_at_y = np.sqrt(1 - y * y)

    theta = phi * indices
    x = np.cos(theta) * radius_at_y
    z = np.sin(theta) * radius_at_y

    return np.column_stack([x, y, z])


def _generate_center_candidates(
    original_center: np.ndarray,
    mesh: trimesh.Trimesh,
    original_radius: float,
    n_candidates: int = 7,
) -> np.ndarray:
    """Generate candidate center positions for multi-candidate containment search.

    Generates candidates by:
    1. Original center (baseline)
    2. Shift toward mesh centroid (moves center inward)
    3. Shift along local PCA axes (±directions)

    Args:
        original_center: Original sphere center (3,)
        mesh: Mesh to check containment against
        original_radius: Original sphere radius
        n_candidates: Maximum number of candidates to generate

    Returns:
        (N, 3) array of candidate center positions
    """
    candidates = [original_center.copy()]

    # 1. Shift toward mesh centroid (moves center inward for corner cases)
    mesh_centroid = mesh.centroid
    direction_to_centroid = mesh_centroid - original_center
    dist_to_centroid = np.linalg.norm(direction_to_centroid)

    if dist_to_centroid > 1e-6:
        # Shift 30% toward centroid, capped at 0.3 * radius
        shift_dist = min(0.3 * dist_to_centroid, 0.3 * original_radius)
        shift_direction = direction_to_centroid / dist_to_centroid
        candidates.append(original_center + shift_direction * shift_dist)

    # 2. Shift along local PCA axes (both directions)
    # Use mesh vertices near the sphere to compute local geometry
    vertex_dists = np.linalg.norm(mesh.vertices - original_center, axis=1)
    nearby_mask = vertex_dists < original_radius * 2.0
    nearby_verts = mesh.vertices[nearby_mask]

    if len(nearby_verts) >= 10:
        pca = PCA(n_components=min(3, len(nearby_verts)))
        pca.fit(nearby_verts - nearby_verts.mean(axis=0))

        shift_dist = 0.2 * original_radius
        for axis in pca.components_:
            if len(candidates) >= n_candidates:
                break
            axis = axis / (np.linalg.norm(axis) + 1e-10)
            candidates.append(original_center + axis * shift_dist)
            if len(candidates) < n_candidates:
                candidates.append(original_center - axis * shift_dist)

    return np.array(candidates[:n_candidates])


def _find_best_radius_at_center(
    center: np.ndarray,
    mesh: trimesh.Trimesh,
    unit_sphere_points: np.ndarray,
    max_radius: float,
    min_fraction: float,
    min_radius_ratio: float,
) -> float:
    """Binary search for largest valid radius at given center.

    Args:
        center: Center position to evaluate (3,)
        mesh: Mesh to check containment against
        unit_sphere_points: Pre-computed unit sphere sampling points (N, 3)
        max_radius: Maximum radius to consider
        min_fraction: Minimum fraction of points that must be inside mesh
        min_radius_ratio: Minimum radius as fraction of max_radius

    Returns:
        Best radius found (largest that satisfies containment constraint)
    """
    lo = max_radius * min_radius_ratio
    hi = max_radius
    best_radius = lo

    for _ in range(8):  # 8 iterations for good precision
        mid = (lo + hi) / 2
        test_points = center + unit_sphere_points * mid
        inside = mesh.contains(test_points)
        fraction_inside = inside.sum() / len(inside)

        if fraction_inside >= min_fraction:
            best_radius = mid
            lo = mid
        else:
            hi = mid

    return best_radius


def cap_radius_by_containment(
    sphere: Sphere,
    mesh: trimesh.Trimesh,
    n_samples: int = 50,
    min_fraction: float = 0.85,
    min_radius_ratio: float = 0.3,
) -> Sphere:
    """Cap sphere to stay within mesh bounds, adjusting center if beneficial.

    Uses multi-candidate search to find the best (center, radius) pair that
    maximizes coverage while satisfying containment constraints. This improves
    coverage at mesh corners by allowing the center to shift inward.

    Samples points uniformly on the sphere's surface and checks if they're
    inside the mesh using trimesh.contains(). For each candidate center,
    uses binary search to find the largest valid radius.

    Args:
        sphere: Sphere to potentially cap
        mesh: Mesh to check containment against (must be watertight)
        n_samples: Number of points to sample on sphere surface
        min_fraction: Minimum fraction of points that must be inside mesh
        min_radius_ratio: Minimum radius as fraction of original (prevents over-shrinking)

    Returns:
        Sphere with potentially adjusted center and radius
    """
    if not mesh.is_watertight or n_samples <= 0:
        return sphere

    original_center = np.asarray(sphere.center)
    original_radius = float(sphere.radius)

    # Sample points on unit sphere (Fibonacci lattice for uniformity)
    unit_sphere_points = _fibonacci_sphere(n_samples)  # (N, 3) unit vectors

    # First check if current radius is already valid
    test_points = original_center + unit_sphere_points * original_radius
    inside = mesh.contains(test_points)
    fraction_inside = inside.sum() / len(inside)

    if fraction_inside >= min_fraction:
        return sphere  # Already valid, no need to adjust

    # Generate candidate centers
    candidates = _generate_center_candidates(
        original_center, mesh, original_radius, n_candidates=7
    )

    best_center = original_center
    best_radius = original_radius * min_radius_ratio
    best_score = 0.0

    for candidate_center in candidates:
        # Find best radius at this center
        radius = _find_best_radius_at_center(
            candidate_center,
            mesh,
            unit_sphere_points,
            original_radius,
            min_fraction,
            min_radius_ratio,
        )

        # Score by radius cubed (proxy for volume/coverage)
        score = radius**3

        if score > best_score:
            best_center = candidate_center
            best_radius = radius
            best_score = score

            # Early termination if we found a radius close to original
            if best_radius > original_radius * 0.95:
                break

    return Sphere(center=jnp.asarray(best_center), radius=jnp.asarray(best_radius))


def compute_coverage(points: np.ndarray, spheres: list[Sphere]) -> float:
    """Compute fraction of points inside at least one sphere.

    Args:
        points: (N, 3) array of points
        spheres: List of spheres to check coverage

    Returns:
        Fraction of points covered (0.0 to 1.0)
    """
    if len(points) == 0 or len(spheres) == 0:
        return 0.0

    covered = np.zeros(len(points), dtype=bool)
    for sphere in spheres:
        center = np.asarray(sphere.center)
        radius = float(sphere.radius)
        dists = np.linalg.norm(points - center, axis=1)
        covered |= dists <= radius

    return float(covered.sum() / len(points))


def compute_split_quality(points: np.ndarray, spheres: list[Sphere]) -> float:
    """Compute combined quality metric for sphere decomposition.

    Quality combines coverage and tightness:
        quality = coverage_fraction * (hull_volume / total_sphere_volume)

    Higher quality is better (more coverage, tighter fit).

    Args:
        points: (N, 3) array of points
        spheres: List of spheres covering the points

    Returns:
        Quality score (higher is better). Returns 0.0 for degenerate cases.
    """
    if len(points) < 4 or len(spheres) == 0:
        return 0.0

    # Compute coverage fraction
    coverage = compute_coverage(points, spheres)

    # Compute hull volume (with error handling)
    try:
        hull_vol = ConvexHull(points).volume
    except (QhullError, ValueError):
        # Degenerate hull - can't compute meaningful quality
        return 0.0

    # Compute total sphere volume
    total_sphere_vol = sum(
        4 / 3 * np.pi * float(sphere.radius) ** 3 for sphere in spheres
    )

    # Avoid division by zero
    if total_sphere_vol < 1e-10:
        return 0.0

    # Quality = coverage * inverse_tightness
    # Higher coverage and tighter fit (hull/sphere closer to 1) = higher quality
    quality = coverage * (hull_vol / total_sphere_vol)

    return float(quality)
