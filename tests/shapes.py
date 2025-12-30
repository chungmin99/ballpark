"""Procedural shape generators for testing ballpark spherization.

This module provides a comprehensive collection of procedurally generated
3D meshes for regression testing and performance benchmarking.

Shape Categories:
    - Primitives: Basic geometric shapes (box, sphere, cylinder, etc.)
    - Compound: Robot-relevant composite shapes (L-shape, T-shape, etc.)
    - Challenging: Shapes that stress the algorithm (thin, concave, holes)
    - Pathological: Degenerate geometries with extreme aspect ratios (>= 15:1)
    - CSG: Shapes requiring constructive solid geometry operations
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable

import numpy as np
import trimesh


class ShapeDifficulty(Enum):
    """Expected difficulty for spherization algorithm."""

    EASY = auto()  # Primitives, convex shapes
    MEDIUM = auto()  # Compound shapes, moderate aspect ratios
    HARD = auto()  # Concave, thin, or shapes with holes


@dataclass
class ShapeSpec:
    """Specification for a test shape."""

    name: str
    factory: Callable[[], trimesh.Trimesh]
    difficulty: ShapeDifficulty
    min_coverage_4: float  # Expected min coverage with 4 spheres
    min_coverage_16: float  # Expected min coverage with 16 spheres
    min_coverage_64: float  # Expected min coverage with 64 spheres
    uses_csg: bool = False  # Whether shape requires CSG operations
    description: str = ""


# =============================================================================
# PRIMITIVE SHAPES
# =============================================================================


def make_cube() -> trimesh.Trimesh:
    """Unit cube centered at origin."""
    return trimesh.creation.box([1.0, 1.0, 1.0])


def make_elongated_box() -> trimesh.Trimesh:
    """Elongated box (2:1:1 aspect ratio)."""
    return trimesh.creation.box([2.0, 1.0, 1.0])


def make_flat_box() -> trimesh.Trimesh:
    """Flat box (4:4:1 pancake)."""
    return trimesh.creation.box([2.0, 2.0, 0.5])


def make_sphere() -> trimesh.Trimesh:
    """Unit sphere (icosphere subdivision 2)."""
    return trimesh.creation.icosphere(subdivisions=2, radius=0.5)


def make_cylinder() -> trimesh.Trimesh:
    """Standard cylinder (height=2, radius=0.5)."""
    return trimesh.creation.cylinder(radius=0.5, height=2.0)


def make_short_cylinder() -> trimesh.Trimesh:
    """Short/wide cylinder (disc-like)."""
    return trimesh.creation.cylinder(radius=1.0, height=0.5)


def make_capsule() -> trimesh.Trimesh:
    """Capsule shape (cylinder with hemispherical caps)."""
    return trimesh.creation.capsule(radius=0.3, height=1.5)


def make_cone() -> trimesh.Trimesh:
    """Cone pointing up."""
    return trimesh.creation.cone(radius=0.5, height=1.5)


def make_torus() -> trimesh.Trimesh:
    """Torus (donut shape) - parametric mesh."""
    major_radius = 0.7
    minor_radius = 0.25
    n_major = 32
    n_minor = 24

    theta = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)

    x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
    y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
    z = minor_radius * np.sin(phi)

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    # Create faces
    faces = []
    for i in range(n_minor):
        for j in range(n_major):
            i0 = i * n_major + j
            i1 = i * n_major + (j + 1) % n_major
            i2 = ((i + 1) % n_minor) * n_major + (j + 1) % n_major
            i3 = ((i + 1) % n_minor) * n_major + j
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


# =============================================================================
# COMPOUND SHAPES (Robot-relevant)
# =============================================================================


def make_l_shape() -> trimesh.Trimesh:
    """L-shaped bracket (two perpendicular boxes)."""
    box1 = trimesh.creation.box([2.0, 0.5, 0.5])
    box2 = trimesh.creation.box([0.5, 1.5, 0.5])

    # Position box2 at end of box1
    box2.apply_translation([0.75, 1.0, 0.0])

    return trimesh.util.concatenate([box1, box2])


def make_t_shape() -> trimesh.Trimesh:
    """T-shaped beam."""
    vertical = trimesh.creation.box([0.4, 1.5, 0.4])
    horizontal = trimesh.creation.box([1.5, 0.4, 0.4])

    horizontal.apply_translation([0.0, 0.75 + 0.2, 0.0])

    return trimesh.util.concatenate([vertical, horizontal])


def make_cross_shape() -> trimesh.Trimesh:
    """3D cross/plus shape."""
    x_bar = trimesh.creation.box([2.0, 0.4, 0.4])
    y_bar = trimesh.creation.box([0.4, 2.0, 0.4])
    z_bar = trimesh.creation.box([0.4, 0.4, 2.0])

    return trimesh.util.concatenate([x_bar, y_bar, z_bar])


def make_dumbbell() -> trimesh.Trimesh:
    """Dumbbell: two spheres connected by a cylinder."""
    sphere1 = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    sphere2 = trimesh.creation.icosphere(subdivisions=2, radius=0.4)
    connector = trimesh.creation.cylinder(radius=0.15, height=1.0)

    sphere1.apply_translation([0.0, -0.7, 0.0])
    sphere2.apply_translation([0.0, 0.7, 0.0])

    return trimesh.util.concatenate([sphere1, sphere2, connector])


def make_gripper_fingers() -> trimesh.Trimesh:
    """Parallel gripper fingers (two parallel boxes)."""
    finger1 = trimesh.creation.box([0.2, 1.0, 0.3])
    finger2 = trimesh.creation.box([0.2, 1.0, 0.3])

    finger1.apply_translation([-0.4, 0.0, 0.0])
    finger2.apply_translation([0.4, 0.0, 0.0])

    # Add connecting base
    base = trimesh.creation.box([1.0, 0.2, 0.3])
    base.apply_translation([0.0, -0.6, 0.0])

    return trimesh.util.concatenate([finger1, finger2, base])


def make_stacked_boxes() -> trimesh.Trimesh:
    """Three stacked boxes of different sizes."""
    box1 = trimesh.creation.box([1.0, 0.3, 1.0])
    box2 = trimesh.creation.box([0.8, 0.3, 0.8])
    box3 = trimesh.creation.box([0.6, 0.3, 0.6])

    box2.apply_translation([0.0, 0.3, 0.0])
    box3.apply_translation([0.0, 0.6, 0.0])

    return trimesh.util.concatenate([box1, box2, box3])


def make_u_bracket() -> trimesh.Trimesh:
    """U-shaped bracket (three connected boxes)."""
    left = trimesh.creation.box([0.3, 1.0, 0.3])
    right = trimesh.creation.box([0.3, 1.0, 0.3])
    bottom = trimesh.creation.box([1.0, 0.3, 0.3])

    left.apply_translation([-0.35, 0.35, 0.0])
    right.apply_translation([0.35, 0.35, 0.0])
    bottom.apply_translation([0.0, -0.35, 0.0])

    return trimesh.util.concatenate([left, right, bottom])


# =============================================================================
# CHALLENGING SHAPES (Non-CSG)
# =============================================================================


def make_thin_plate() -> trimesh.Trimesh:
    """Very thin plate (high aspect ratio, 10:10:1)."""
    return trimesh.creation.box([2.0, 2.0, 0.2])


def make_needle() -> trimesh.Trimesh:
    """Very elongated cylinder (needle-like, 20:1 aspect)."""
    return trimesh.creation.cylinder(radius=0.05, height=2.0)


def make_star() -> trimesh.Trimesh:
    """6-pointed star (spiky shape) - using concatenation."""
    box1 = trimesh.creation.box([2.0, 0.3, 0.3])
    box2 = trimesh.creation.box([2.0, 0.3, 0.3])
    box3 = trimesh.creation.box([2.0, 0.3, 0.3])

    # Rotate boxes
    rot_matrix_60 = trimesh.transformations.rotation_matrix(
        np.pi / 3, [0, 0, 1]
    )
    rot_matrix_120 = trimesh.transformations.rotation_matrix(
        2 * np.pi / 3, [0, 0, 1]
    )

    box2.apply_transform(rot_matrix_60)
    box3.apply_transform(rot_matrix_120)

    return trimesh.util.concatenate([box1, box2, box3])


def make_thick_torus() -> trimesh.Trimesh:
    """Thick torus (large minor radius) - parametric mesh."""
    major_radius = 0.5
    minor_radius = 0.35
    n_major = 32
    n_minor = 24

    theta = np.linspace(0, 2 * np.pi, n_major, endpoint=False)
    phi = np.linspace(0, 2 * np.pi, n_minor, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)

    x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
    y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
    z = minor_radius * np.sin(phi)

    vertices = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

    faces = []
    for i in range(n_minor):
        for j in range(n_major):
            i0 = i * n_major + j
            i1 = i * n_major + (j + 1) % n_major
            i2 = ((i + 1) % n_minor) * n_major + (j + 1) % n_major
            i3 = ((i + 1) % n_minor) * n_major + j
            faces.append([i0, i1, i2])
            faces.append([i0, i2, i3])

    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces))


def make_tiny_tetrahedron() -> trimesh.Trimesh:
    """Very small tetrahedron (few vertices)."""
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)],
    ], dtype=float) * 0.5

    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ])

    return trimesh.Trimesh(vertices=vertices, faces=faces)


def make_dense_sphere() -> trimesh.Trimesh:
    """High-resolution sphere (many vertices)."""
    return trimesh.creation.icosphere(subdivisions=4, radius=0.5)


def make_nearly_flat() -> trimesh.Trimesh:
    """Nearly degenerate flat shape (very thin)."""
    return trimesh.creation.box([2.0, 2.0, 0.02])


def make_asymmetric_blob() -> trimesh.Trimesh:
    """Asymmetric irregular shape."""
    # Start with sphere and apply random vertex perturbation
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    np.random.seed(42)  # Reproducible
    noise = np.random.randn(len(mesh.vertices), 3) * 0.1
    # Apply asymmetric scaling
    noise[:, 0] *= 1.5
    noise[:, 1] *= 0.8
    mesh.vertices += noise
    return mesh


def make_elongated_star() -> trimesh.Trimesh:
    """Elongated 4-pointed star (challenging aspect ratio)."""
    arm1 = trimesh.creation.box([3.0, 0.2, 0.2])
    arm2 = trimesh.creation.box([0.2, 3.0, 0.2])

    return trimesh.util.concatenate([arm1, arm2])


# =============================================================================
# CSG SHAPES (may fail - tested separately)
# =============================================================================


def make_c_shape() -> trimesh.Trimesh:
    """C-shaped concave bracket using CSG."""
    outer = trimesh.creation.box([1.5, 1.5, 0.4])
    inner = trimesh.creation.box([1.0, 1.0, 0.6])
    inner.apply_translation([0.4, 0.0, 0.0])

    result = outer.difference(inner)
    if result.is_empty:
        raise RuntimeError("CSG operation failed for c_shape")
    return result


def make_horseshoe() -> trimesh.Trimesh:
    """Horseshoe/arch shape using CSG."""
    outer = make_torus()
    # Cut in half
    cutter = trimesh.creation.box([3.0, 3.0, 3.0])
    cutter.apply_translation([0.0, -1.5, 0.0])

    result = outer.difference(cutter)
    if result.is_empty:
        raise RuntimeError("CSG operation failed for horseshoe")
    return result


def make_frame() -> trimesh.Trimesh:
    """Hollow rectangular frame using CSG."""
    outer = trimesh.creation.box([2.0, 1.5, 0.2])
    inner = trimesh.creation.box([1.6, 1.1, 0.4])

    result = outer.difference(inner)
    if result.is_empty:
        raise RuntimeError("CSG operation failed for frame")
    return result


def make_swiss_cheese() -> trimesh.Trimesh:
    """Box with cylindrical holes using CSG."""
    base = trimesh.creation.box([2.0, 2.0, 0.5])

    # Create holes
    hole1 = trimesh.creation.cylinder(radius=0.2, height=1.0)
    hole2 = trimesh.creation.cylinder(radius=0.2, height=1.0)
    hole3 = trimesh.creation.cylinder(radius=0.2, height=1.0)

    hole1.apply_translation([-0.5, -0.5, 0.0])
    hole2.apply_translation([0.5, -0.5, 0.0])
    hole3.apply_translation([0.0, 0.5, 0.0])

    result = base.difference(hole1)
    result = result.difference(hole2)
    result = result.difference(hole3)

    if result.is_empty:
        raise RuntimeError("CSG operation failed for swiss_cheese")
    return result


# =============================================================================
# SHAPE REGISTRY
# =============================================================================

# Coverage thresholds are based on expected algorithm behavior:
# - Easy shapes should achieve high coverage even with few spheres
# - Hard shapes may need more spheres for good coverage
# These are initial estimates and will be calibrated based on actual results

# Non-CSG shapes (primitives, compound, and challenging - but NOT pathological)
SHAPES: list[ShapeSpec] = [
    # Primitives (EASY) - indices 0-8
    ShapeSpec(
        name="cube",
        factory=make_cube,
        difficulty=ShapeDifficulty.EASY,
        min_coverage_4=0.85,
        min_coverage_16=0.92,
        min_coverage_64=0.96,
        description="Unit cube, baseline primitive",
    ),
    ShapeSpec(
        name="elongated_box",
        factory=make_elongated_box,
        difficulty=ShapeDifficulty.EASY,
        min_coverage_4=0.80,
        min_coverage_16=0.90,
        min_coverage_64=0.95,
        description="2:1:1 aspect ratio box",
    ),
    ShapeSpec(
        name="flat_box",
        factory=make_flat_box,
        difficulty=ShapeDifficulty.EASY,
        min_coverage_4=0.80,
        min_coverage_16=0.90,
        min_coverage_64=0.95,
        description="4:4:1 pancake box",
    ),
    ShapeSpec(
        name="sphere",
        factory=make_sphere,
        difficulty=ShapeDifficulty.EASY,
        min_coverage_4=0.90,
        min_coverage_16=0.95,
        min_coverage_64=0.98,
        description="Icosphere, ideal for single sphere",
    ),
    ShapeSpec(
        name="cylinder",
        factory=make_cylinder,
        difficulty=ShapeDifficulty.EASY,
        min_coverage_4=0.80,
        min_coverage_16=0.90,
        min_coverage_64=0.95,
        description="Standard cylinder",
    ),
    ShapeSpec(
        name="short_cylinder",
        factory=make_short_cylinder,
        difficulty=ShapeDifficulty.EASY,
        min_coverage_4=0.85,
        min_coverage_16=0.92,
        min_coverage_64=0.96,
        description="Disc-like cylinder",
    ),
    ShapeSpec(
        name="capsule",
        factory=make_capsule,
        difficulty=ShapeDifficulty.EASY,
        min_coverage_4=0.85,
        min_coverage_16=0.92,
        min_coverage_64=0.96,
        description="Cylinder with hemispherical caps",
    ),
    ShapeSpec(
        name="cone",
        factory=make_cone,
        difficulty=ShapeDifficulty.EASY,
        min_coverage_4=0.80,
        min_coverage_16=0.88,
        min_coverage_64=0.94,
        description="Cone pointing up",
    ),
    ShapeSpec(
        name="torus",
        factory=make_torus,
        difficulty=ShapeDifficulty.MEDIUM,
        min_coverage_4=0.70,
        min_coverage_16=0.85,
        min_coverage_64=0.92,
        description="Donut shape with hole",
    ),
    # Compound shapes (MEDIUM) - indices 9-15
    ShapeSpec(
        name="l_shape",
        factory=make_l_shape,
        difficulty=ShapeDifficulty.MEDIUM,
        min_coverage_4=0.75,
        min_coverage_16=0.88,
        min_coverage_64=0.94,
        description="L-bracket, two perpendicular boxes",
    ),
    ShapeSpec(
        name="t_shape",
        factory=make_t_shape,
        difficulty=ShapeDifficulty.MEDIUM,
        min_coverage_4=0.75,
        min_coverage_16=0.88,
        min_coverage_64=0.94,
        description="T-beam",
    ),
    ShapeSpec(
        name="cross_shape",
        factory=make_cross_shape,
        difficulty=ShapeDifficulty.MEDIUM,
        min_coverage_4=0.70,
        min_coverage_16=0.85,
        min_coverage_64=0.92,
        description="3D cross/plus",
    ),
    ShapeSpec(
        name="dumbbell",
        factory=make_dumbbell,
        difficulty=ShapeDifficulty.MEDIUM,
        min_coverage_4=0.75,
        min_coverage_16=0.88,
        min_coverage_64=0.94,
        description="Two spheres connected by cylinder",
    ),
    ShapeSpec(
        name="gripper_fingers",
        factory=make_gripper_fingers,
        difficulty=ShapeDifficulty.MEDIUM,
        min_coverage_4=0.70,
        min_coverage_16=0.85,
        min_coverage_64=0.92,
        description="Parallel gripper with base",
    ),
    ShapeSpec(
        name="stacked_boxes",
        factory=make_stacked_boxes,
        difficulty=ShapeDifficulty.MEDIUM,
        min_coverage_4=0.75,
        min_coverage_16=0.88,
        min_coverage_64=0.94,
        description="Three stacked boxes",
    ),
    ShapeSpec(
        name="u_bracket",
        factory=make_u_bracket,
        difficulty=ShapeDifficulty.MEDIUM,
        min_coverage_4=0.70,
        min_coverage_16=0.85,
        min_coverage_64=0.92,
        description="U-shaped bracket",
    ),
    # Challenging shapes (HARD) - indices 16-21
    # Note: Pathological shapes (needle, nearly_flat, elongated_star) are in separate list
    ShapeSpec(
        name="thin_plate",
        factory=make_thin_plate,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.60,
        min_coverage_16=0.80,
        min_coverage_64=0.90,
        description="High aspect ratio plate (10:1)",
    ),
    ShapeSpec(
        name="star",
        factory=make_star,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.55,
        min_coverage_16=0.78,
        min_coverage_64=0.88,
        description="6-pointed star",
    ),
    ShapeSpec(
        name="thick_torus",
        factory=make_thick_torus,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.65,
        min_coverage_16=0.82,
        min_coverage_64=0.90,
        description="Torus with large minor radius",
    ),
    ShapeSpec(
        name="tiny_tetrahedron",
        factory=make_tiny_tetrahedron,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.80,
        min_coverage_16=0.90,
        min_coverage_64=0.95,
        description="Minimal vertex count",
    ),
    ShapeSpec(
        name="dense_sphere",
        factory=make_dense_sphere,
        difficulty=ShapeDifficulty.EASY,
        min_coverage_4=0.90,
        min_coverage_16=0.95,
        min_coverage_64=0.98,
        description="High vertex count sphere",
    ),
    ShapeSpec(
        name="asymmetric_blob",
        factory=make_asymmetric_blob,
        difficulty=ShapeDifficulty.MEDIUM,
        min_coverage_4=0.75,
        min_coverage_16=0.88,
        min_coverage_64=0.94,
        description="Irregular perturbed sphere",
    ),
]

# CSG shapes (may be excluded if CSG fails)
CSG_SHAPES: list[ShapeSpec] = [
    ShapeSpec(
        name="c_shape",
        factory=make_c_shape,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.65,
        min_coverage_16=0.82,
        min_coverage_64=0.90,
        uses_csg=True,
        description="Concave C-bracket (CSG)",
    ),
    ShapeSpec(
        name="horseshoe",
        factory=make_horseshoe,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.60,
        min_coverage_16=0.80,
        min_coverage_64=0.88,
        uses_csg=True,
        description="Horseshoe/arch shape (CSG)",
    ),
    ShapeSpec(
        name="frame",
        factory=make_frame,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.60,
        min_coverage_16=0.80,
        min_coverage_64=0.88,
        uses_csg=True,
        description="Hollow rectangular frame (CSG)",
    ),
    ShapeSpec(
        name="swiss_cheese",
        factory=make_swiss_cheese,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.70,
        min_coverage_16=0.85,
        min_coverage_64=0.92,
        uses_csg=True,
        description="Box with cylindrical holes (CSG)",
    ),
]


def test_csg_shapes() -> list[ShapeSpec]:
    """Test which CSG shapes work and return only working ones."""
    working = []
    for spec in CSG_SHAPES:
        try:
            mesh = spec.factory()
            if not mesh.is_empty and len(mesh.vertices) > 0:
                working.append(spec)
        except Exception:
            pass  # CSG failed, exclude this shape
    return working


def get_all_shapes(include_csg: bool = True, include_pathological: bool = True) -> list[ShapeSpec]:
    """Get all available shapes.

    Args:
        include_csg: If True, test and include working CSG shapes.
        include_pathological: If True, include pathological/degenerate shapes.

    Returns:
        List of ShapeSpec for all available shapes.
    """
    all_shapes = list(SHAPES)
    if include_pathological:
        all_shapes.extend(PATHOLOGICAL_SHAPES)
    if include_csg:
        all_shapes.extend(test_csg_shapes())
    return all_shapes


def get_shapes_by_difficulty(
    difficulty: ShapeDifficulty,
    include_csg: bool = True,
) -> list[ShapeSpec]:
    """Get all shapes of a given difficulty level."""
    return [s for s in get_all_shapes(include_csg) if s.difficulty == difficulty]


def get_shape_by_name(name: str, include_csg: bool = True) -> ShapeSpec | None:
    """Get a shape specification by name."""
    for s in get_all_shapes(include_csg):
        if s.name == name:
            return s
    return None


def get_all_shape_names(include_csg: bool = True) -> list[str]:
    """Get all registered shape names."""
    return [s.name for s in get_all_shapes(include_csg)]


def get_min_coverage(spec: ShapeSpec, budget: int) -> float:
    """Get minimum expected coverage for a shape at a given budget."""
    if budget <= 4:
        return spec.min_coverage_4
    elif budget <= 16:
        return spec.min_coverage_16
    else:
        return spec.min_coverage_64


# =============================================================================
# CATEGORY-SPECIFIC GETTERS
# =============================================================================

# Define category boundaries in SHAPES list
# Primitives: indices 0-8 (cube through torus)
# Compound: indices 9-15 (l_shape through u_bracket)
# Challenging: indices 16-21 (thin_plate, star, thick_torus, tiny_tetrahedron, dense_sphere, asymmetric_blob)

PRIMITIVE_SHAPES = SHAPES[0:9]  # cube, elongated_box, flat_box, sphere, cylinder, short_cylinder, capsule, cone, torus
COMPOUND_SHAPES = SHAPES[9:16]  # l_shape, t_shape, cross_shape, dumbbell, gripper_fingers, stacked_boxes, u_bracket
CHALLENGING_SHAPES = SHAPES[16:22]  # thin_plate, star, thick_torus, tiny_tetrahedron, dense_sphere, asymmetric_blob

# Pathological shapes - degenerate/extreme aspect ratios >= 15:1
# These are defined separately (not in SHAPES) to avoid double-counting
PATHOLOGICAL_SHAPES: list[ShapeSpec] = [
    ShapeSpec(
        name="needle",
        factory=make_needle,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.50,
        min_coverage_16=0.75,
        min_coverage_64=0.88,
        description="Extremely elongated cylinder (20:1 aspect ratio)",
    ),
    ShapeSpec(
        name="nearly_flat",
        factory=make_nearly_flat,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.50,
        min_coverage_16=0.75,
        min_coverage_64=0.85,
        description="Nearly degenerate thin box (100:1 aspect ratio)",
    ),
    ShapeSpec(
        name="elongated_star",
        factory=make_elongated_star,
        difficulty=ShapeDifficulty.HARD,
        min_coverage_4=0.50,
        min_coverage_16=0.75,
        min_coverage_64=0.88,
        description="4-pointed star with high aspect ratio (15:1)",
    ),
]


def get_primitive_shapes() -> list[ShapeSpec]:
    """Get primitive shapes (EASY difficulty, basic geometries)."""
    return list(PRIMITIVE_SHAPES)


def get_compound_shapes() -> list[ShapeSpec]:
    """Get compound shapes (MEDIUM difficulty, robot-relevant)."""
    return list(COMPOUND_SHAPES)


def get_challenging_shapes() -> list[ShapeSpec]:
    """Get challenging shapes (HARD difficulty, moderate stress tests)."""
    return list(CHALLENGING_SHAPES)


def get_pathological_shapes() -> list[ShapeSpec]:
    """Get pathological shapes (degenerate geometries with extreme aspect ratios)."""
    return list(PATHOLOGICAL_SHAPES)


def get_csg_shapes() -> list[ShapeSpec]:
    """Get CSG shapes (only those that work)."""
    return test_csg_shapes()


def get_primitive_shape_names() -> list[str]:
    """Get names of primitive shapes."""
    return [s.name for s in PRIMITIVE_SHAPES]


def get_compound_shape_names() -> list[str]:
    """Get names of compound shapes."""
    return [s.name for s in COMPOUND_SHAPES]


def get_challenging_shape_names() -> list[str]:
    """Get names of challenging shapes."""
    return [s.name for s in CHALLENGING_SHAPES]


def get_pathological_shape_names() -> list[str]:
    """Get names of pathological shapes."""
    return [s.name for s in PATHOLOGICAL_SHAPES]


def get_csg_shape_names() -> list[str]:
    """Get names of working CSG shapes."""
    return [s.name for s in test_csg_shapes()]
