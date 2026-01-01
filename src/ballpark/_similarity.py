"""Geometry similarity detection for robot links.

Detects duplicate collision meshes across robot links to enable
consistent sphere decompositions for similar parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import trimesh
from loguru import logger

from .utils._hash_geometry import _get_geometry_fingerprint


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


def _find_mirror_axis(centered_a: np.ndarray, centered_b: np.ndarray) -> int | None:
    """Check if meshes are mirrors of each other along a single axis.

    Args:
        centered_a: Vertices of mesh A centered at origin
        centered_b: Vertices of mesh B centered at origin

    Returns:
        Axis index (0=X, 1=Y, 2=Z) if mirrored, None otherwise
    """
    # Sort vertices for comparison (order-independent matching)
    sorted_a = np.sort(centered_a, axis=0)

    for axis in range(3):
        # Flip this axis in B
        flipped_b = centered_b.copy()
        flipped_b[:, axis] *= -1
        sorted_flipped_b = np.sort(flipped_b, axis=0)

        if np.allclose(sorted_a, sorted_flipped_b, atol=1e-6):
            return axis

    return None


def _compute_alignment_transform(mesh_a, mesh_b) -> tuple[np.ndarray, str]:
    """Compute 4x4 transform that aligns mesh_a to mesh_b.

    Handles three cases:
    1. Identical meshes: translation only
    2. Mirrored meshes: reflection + translation
    3. Rotated meshes: PCA-based rotation + translation

    Returns:
        Tuple of (transform, similarity_type) where:
        - transform: 4x4 homogeneous transform matrix T such that T @ point_in_a ≈ point_in_b
        - similarity_type: One of "identical", "mirror_X", "mirror_Y", "mirror_Z", or "rotational"
    """
    if mesh_a.is_empty or mesh_b.is_empty:
        return np.eye(4), "unknown"

    centroid_a = mesh_a.centroid
    centroid_b = mesh_b.centroid

    centered_a = mesh_a.vertices - centroid_a
    centered_b = mesh_b.vertices - centroid_b

    # Only do vertex-by-vertex comparison if meshes have same vertex count
    same_vertex_count = len(mesh_a.vertices) == len(mesh_b.vertices)

    # Case 1: Check if meshes are identical (just translated)
    if same_vertex_count:
        sorted_a = np.sort(centered_a, axis=0)
        sorted_b = np.sort(centered_b, axis=0)
        if np.allclose(sorted_a, sorted_b, atol=1e-6):
            T = np.eye(4)
            T[:3, 3] = centroid_b - centroid_a
            return T, "identical"

    # Case 2: Check if meshes are mirrored along a single axis
    mirror_axis = _find_mirror_axis(centered_a, centered_b) if same_vertex_count else None
    if mirror_axis is not None:
        # Build reflection matrix: flip the mirror axis
        T = np.eye(4)
        T[mirror_axis, mirror_axis] = -1.0
        # Translation: account for reflection of centroid_a
        reflected_centroid_a = centroid_a.copy()
        reflected_centroid_a[mirror_axis] *= -1.0
        T[:3, 3] = centroid_b - reflected_centroid_a
        axis_name = ["X", "Y", "Z"][mirror_axis]
        return T, f"mirror_{axis_name}"

    # Case 3: Fall back to PCA-based rotation alignment
    _, _, Vt_a = np.linalg.svd(centered_a, full_matrices=False)
    _, _, Vt_b = np.linalg.svd(centered_b, full_matrices=False)

    # Rotation from A's frame to B's frame
    rotation_matrix = Vt_b.T @ Vt_a

    # Handle reflection (ensure proper rotation, det = 1)
    if np.linalg.det(rotation_matrix) < 0:
        Vt_a_fixed = Vt_a.copy()
        Vt_a_fixed[2] = -Vt_a_fixed[2]
        rotation_matrix = Vt_b.T @ Vt_a_fixed

    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = centroid_b - rotation_matrix @ centroid_a

    return T, "rotational"


def detect_similar_links(
    link_meshes: dict[str, trimesh.Trimesh],
    link_fingerprints: dict[str, tuple],
) -> SimilarityResult:
    """Detect links with duplicate/similar collision meshes.

    Groups links that share the same underlying collision geometry (same mesh
    file, same primitive type/dimensions, or same mesh shape).

    This function is fast (~100ms for typical robots) but the result can be
    cached and passed to compute_spheres_for_robot() to avoid recomputation.

    Args:
        link_meshes: Dict mapping link names to their collision meshes.
        link_fingerprints: Dict mapping link names to collision fingerprints
            (pre-computed from URDF collision geometry).

    Returns:
        SimilarityResult containing groups of similar links and transforms
        between them.

    Example:
        >>> similarity = detect_similar_links(link_meshes, link_fingerprints)
        Similarity: 1 group(s), 3 links share geometry
          Group 1: finger_link_1 -> finger_link_2, finger_link_3
        >>> print(similarity.groups)
        [['finger_link_1', 'finger_link_2', 'finger_link_3']]
    """
    link_names = list(link_meshes.keys())

    # Step 1: Group links by fingerprint
    fingerprint_to_links: dict[tuple, list[str]] = defaultdict(list)

    for link_name in link_names:
        fp = link_fingerprints.get(link_name)
        if fp:
            fingerprint_to_links[fp].append(link_name)

    # Step 2: Build groups (only include groups with 2+ links)
    groups = []
    grouped_links = set()
    for fp, links in fingerprint_to_links.items():
        if len(links) >= 2:
            groups.append(sorted(links))
            grouped_links.update(links)

    # Step 2b: Geometry-based grouping for mesh-file links not yet grouped
    ungrouped_mesh_links = [
        link
        for link in link_names
        if link not in grouped_links
        and link_fingerprints.get(link)
        and any(f[0] == "mesh_file" for f in link_fingerprints[link])
    ]

    if ungrouped_mesh_links:
        geom_fp_to_links: dict[tuple, list[str]] = defaultdict(list)
        for link_name in ungrouped_mesh_links:
            mesh = link_meshes.get(link_name)
            if mesh is not None and not mesh.is_empty:
                gfp = _get_geometry_fingerprint(mesh)
                if gfp:
                    geom_fp_to_links[gfp].append(link_name)

        for gfp, links in geom_fp_to_links.items():
            if len(links) >= 2:
                groups.append(sorted(links))

    # Step 3: Compute transforms between links in each group
    transforms: dict[tuple[str, str], np.ndarray] = {}
    group_similarity_types: dict[int, list[str]] = {}

    for i, group in enumerate(groups):
        # Get meshes for this group
        meshes = {
            link_name: link_meshes[link_name]
            for link_name in group
            if link_name in link_meshes and not link_meshes[link_name].is_empty
        }

        # Compute pairwise transforms (only need from first link to others)
        if len(meshes) >= 2:
            first_link = group[0]
            mesh_first = meshes.get(first_link)

            # Track similarity types for this group
            similarity_types = []

            if mesh_first is not None:
                for other_link in group[1:]:
                    mesh_other = meshes.get(other_link)
                    if mesh_other is not None:
                        T, sim_type = _compute_alignment_transform(mesh_first, mesh_other)
                        transforms[(first_link, other_link)] = T
                        # Also store inverse for convenience
                        transforms[(other_link, first_link)] = np.linalg.inv(T)
                        similarity_types.append(sim_type)

            group_similarity_types[i] = similarity_types

    # Log detected groups with similarity types
    if groups:
        total_similar = sum(len(g) for g in groups)
        logger.info(f"Similarity: {len(groups)} group(s), {total_similar} links share geometry")
        for i, group in enumerate(groups):
            primary = group[0]
            secondaries = group[1:]

            # Format similarity type info
            similarity_types = group_similarity_types.get(i, [])
            if similarity_types:
                unique_types = set(similarity_types)
                if len(unique_types) == 1:
                    type_str = f" ({similarity_types[0]})"
                else:
                    # Multiple types - show per-link
                    types_list = [f"{secondaries[j]}: {similarity_types[j]}" for j in range(len(similarity_types))]
                    type_str = f" ({', '.join(types_list)})"
            else:
                type_str = ""

            logger.info(f"  Group {i+1}: {primary} -> {', '.join(secondaries)}{type_str}")
    else:
        logger.info("Similarity: No duplicate geometries found")

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
