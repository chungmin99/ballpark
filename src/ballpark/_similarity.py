"""Geometry similarity detection for robot links.

Detects duplicate collision meshes across robot links to enable
consistent sphere decompositions for similar parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

import numpy as np
from loguru import logger


@dataclass
class SimilarityResult:
    """Result from similarity detection, cacheable for reuse.

    Attributes:
        groups: List of link name groups. Each group contains links with
            identical/similar collision meshes.
        transforms: Dict mapping (link_a, link_b) to 4x4 transform matrix
            that aligns link_a's mesh to link_b's mesh frame.
    """

    groups: list[list[str]] = field(default_factory=list)
    transforms: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    def get_group(self, link_name: str) -> list[str] | None:
        """Get the similarity group containing a given link.

        Args:
            link_name: Name of the link to find

        Returns:
            List of link names in the same group, or None if not in any group.
        """
        for group in self.groups:
            if link_name in group:
                return group
        return None


def _get_single_collision_fingerprint(geom) -> tuple | None:
    """Extract a fingerprint for a single collision geometry.

    The fingerprint uniquely identifies the geometry shape, independent of
    the link's pose in the robot.

    Args:
        geom: A collision geometry object from yourdfpy

    Returns:
        Tuple (mesh_file, scale) if from file, or
        Tuple (geom_type, params_hash) for primitives, or
        Tuple ("mesh", n_verts, n_faces, shape_hash) for inline meshes, or
        None if no geometry.
    """
    if geom.box is not None:
        # Box: fingerprint is the sorted extents (rotation-invariant)
        extents = tuple(sorted(geom.box.size))
        return ("box", extents)

    elif geom.cylinder is not None:
        return ("cylinder", geom.cylinder.radius, geom.cylinder.length)

    elif geom.sphere is not None:
        return ("sphere", geom.sphere.radius)

    elif geom.mesh is not None:
        # Mesh from file: use filename + scale as fingerprint
        mesh_file = geom.mesh.filename
        scale = tuple(geom.mesh.scale) if geom.mesh.scale is not None else (1.0, 1.0, 1.0)
        return ("mesh_file", mesh_file, scale)

    return None


def _get_collision_fingerprint(urdf, link_name: str) -> tuple | None:
    """Extract a fingerprint for a link's collision geometry.

    The fingerprint uniquely identifies the geometry shape, independent of
    the link's pose in the robot. For links with multiple collision geometries,
    all geometries are included in a composite fingerprint.

    Returns:
        Tuple of fingerprints for all collision geometries (sorted for consistency),
        or None if no collision geometry.
    """
    if link_name not in urdf.link_map:
        return None

    link = urdf.link_map[link_name]
    if not link.collisions:
        return None

    # Generate fingerprint for each collision and combine them
    fingerprints = []
    for collision in link.collisions:
        fp = _get_single_collision_fingerprint(collision.geometry)
        if fp is not None:
            fingerprints.append(fp)

    return tuple(sorted(fingerprints)) if fingerprints else None


def _get_geometry_fingerprint(mesh, tolerance: float = 0.001) -> tuple | None:
    """Geometry-based fingerprint: sorted extents + volume (mirror-invariant)."""
    if mesh.is_empty:
        return None
    extents = tuple(round(e / tolerance) for e in sorted(mesh.extents))
    try:
        vol = mesh.convex_hull.volume
        # Round volume to 3 significant figures to handle mesh discretization noise
        volume = round(vol, -int(np.floor(np.log10(abs(vol) + 1e-10))) + 2) if vol > 0 else 0
    except Exception:
        volume = 0
    return ("geometry", extents, volume)


def _compute_mesh_shape_hash(mesh) -> str:
    """Compute a hash representing the mesh shape (position-invariant).

    Uses sorted distances from centroid to create a rotation/translation
    invariant signature, combined with mesh topology information.
    """
    if mesh.is_empty:
        return ""

    # Center the mesh
    centroid = mesh.centroid
    centered_verts = mesh.vertices - centroid

    # Compute distances from centroid
    distances = np.linalg.norm(centered_verts, axis=1)
    sorted_distances = np.sort(distances)

    # Quantize to finer resolution (0.1mm) to avoid false positives
    quantized = np.round(sorted_distances * 10000).astype(np.int32)

    # Include mesh topology info in the hash to distinguish different meshes
    # Use first 100 sorted distances for efficiency while preserving uniqueness
    hash_input = np.concatenate([
        [len(mesh.vertices), len(mesh.faces)],
        quantized[:100]  # Use first 100 sorted distances for efficiency
    ])

    # Hash the combined topology + shape signature
    return hashlib.md5(hash_input.tobytes()).hexdigest()[:16]


def _compute_alignment_transform(mesh_a, mesh_b) -> np.ndarray:
    """Compute 4x4 transform that aligns mesh_a to mesh_b.

    Uses centroid alignment + PCA-based rotation alignment.

    Returns:
        4x4 homogeneous transform matrix T such that T @ point_in_a ≈ point_in_b
    """
    if mesh_a.is_empty or mesh_b.is_empty:
        return np.eye(4)

    # Centroid translation
    centroid_a = mesh_a.centroid
    centroid_b = mesh_b.centroid

    # PCA for rotation alignment
    centered_a = mesh_a.vertices - centroid_a
    centered_b = mesh_b.vertices - centroid_b

    # Compute principal axes via SVD
    _, _, Vt_a = np.linalg.svd(centered_a, full_matrices=False)
    _, _, Vt_b = np.linalg.svd(centered_b, full_matrices=False)

    # Rotation from A's frame to B's frame
    # V_a.T @ point_centered_a = point_in_pca_frame
    # V_b @ point_in_pca_frame = point_centered_b
    # So: rotation = V_b @ V_a.T
    rotation_matrix = Vt_b.T @ Vt_a

    # Handle reflection (ensure proper rotation, det = 1)
    if np.linalg.det(rotation_matrix) < 0:
        # Flip the last axis to make it a proper rotation
        Vt_a_fixed = Vt_a.copy()
        Vt_a_fixed[2] = -Vt_a_fixed[2]
        rotation_matrix = Vt_b.T @ Vt_a_fixed

    # Build 4x4 transform: translate to origin, rotate, translate to B
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = centroid_b - rotation_matrix @ centroid_a

    return T


def detect_similar_links(
    urdf,
    link_names: list[str] | None = None,
    verbose: bool = True,
) -> SimilarityResult:
    """Detect links with duplicate/similar collision meshes.

    Groups links that share the same underlying collision geometry (same mesh
    file, same primitive type/dimensions, or same mesh shape).

    This function is fast (~100ms for typical robots) but the result can be
    cached and passed to compute_spheres_for_robot() to avoid recomputation.

    Args:
        urdf: yourdfpy URDF object with collision meshes loaded
        link_names: Optional list of links to consider. If None, uses all links.
        verbose: If True, print detected similarity groups.

    Returns:
        SimilarityResult containing groups of similar links and transforms
        between them.

    Example:
        >>> similarity = detect_similar_links(urdf)
        Detected 1 similarity group(s):
          Group 1: finger_link_1, finger_link_2, finger_link_3
        >>> print(similarity.groups)
        [['finger_link_1', 'finger_link_2', 'finger_link_3']]
    """
    if link_names is None:
        link_names = list(urdf.link_map.keys())

    # Step 1: Group links by fingerprint
    fingerprint_to_links: dict[tuple, list[str]] = defaultdict(list)

    for link_name in link_names:
        fp = _get_collision_fingerprint(urdf, link_name)
        if fp is not None:
            fingerprint_to_links[fp].append(link_name)

    # Step 2: Build groups (only include groups with 2+ links)
    groups = []
    grouped_links = set()
    for fp, links in fingerprint_to_links.items():
        if len(links) >= 2:
            groups.append(sorted(links))
            grouped_links.update(links)

    # Step 2b: Geometry-based grouping for mesh-file links not yet grouped
    # Lazy import to avoid circular dependency
    from ._robot import get_collision_mesh_for_link

    ungrouped_mesh_links = [
        link for link, fp in ((l, _get_collision_fingerprint(urdf, l)) for l in link_names)
        if fp and link not in grouped_links and any(f[0] == "mesh_file" for f in fp)
    ]

    if ungrouped_mesh_links:
        geom_fp_to_links: dict[tuple, list[str]] = defaultdict(list)
        for link_name in ungrouped_mesh_links:
            mesh = get_collision_mesh_for_link(urdf, link_name)
            gfp = _get_geometry_fingerprint(mesh)
            if gfp:
                geom_fp_to_links[gfp].append(link_name)

        for gfp, links in geom_fp_to_links.items():
            if len(links) >= 2:
                groups.append(sorted(links))

    # Log detected groups
    if verbose:
        if groups:
            logger.info(f"Detected {len(groups)} similarity group(s):")
            for i, group in enumerate(groups, 1):
                logger.info(f"  Group {i}: {', '.join(group)}")
        else:
            logger.info("No similar links detected.")

    # Step 3: Compute transforms between links in each group
    transforms: dict[tuple[str, str], np.ndarray] = {}

    for group in groups:
        # Load meshes for this group
        meshes = {}
        for link_name in group:
            mesh = get_collision_mesh_for_link(urdf, link_name)
            if not mesh.is_empty:
                meshes[link_name] = mesh

        # Compute pairwise transforms (only need from first link to others)
        if len(meshes) >= 2:
            first_link = group[0]
            mesh_first = meshes.get(first_link)

            if mesh_first is not None:
                for other_link in group[1:]:
                    mesh_other = meshes.get(other_link)
                    if mesh_other is not None:
                        T = _compute_alignment_transform(mesh_first, mesh_other)
                        transforms[(first_link, other_link)] = T
                        # Also store inverse for convenience
                        transforms[(other_link, first_link)] = np.linalg.inv(T)

    return SimilarityResult(groups=groups, transforms=transforms)


def get_group_for_link(
    similarity_result: SimilarityResult,
    link_name: str,
) -> list[str] | None:
    """Get the similarity group containing a given link.

    Args:
        similarity_result: Result from detect_similar_links()
        link_name: Name of the link to find

    Returns:
        List of link names in the same group, or None if link is not in any group.
    """
    for group in similarity_result.groups:
        if link_name in group:
            return group
    return None
