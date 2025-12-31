#!/usr/bin/env python3
"""Visualize procedural test shapes with sphere decomposition.

This script provides an interactive GUI for exploring all test shapes
defined in tests/shapes.py, with real-time spherization and quality metrics.

Usage:
    python -m tests.visualize_shapes
    # or from tests/ directory:
    python visualize_shapes.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser

# Add parent to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from ballpark import spherize, Sphere, SPHERE_COLORS
from ballpark._config import SpherizeParams
from ballpark.metrics import (
    compute_coverage,
    compute_over_extension,
    compute_quality,
    compute_tightness,
    compute_volume_overhead,
)

from shapes import (
    get_primitive_shapes,
    get_compound_shapes,
    get_challenging_shapes,
    get_pathological_shapes,
    get_csg_shapes,
    get_shape_by_name,
    ShapeSpec,
)


class _ShapesGui:
    """GUI controls for shape visualization."""

    def __init__(self, server: viser.ViserServer):
        self._server = server

        # Organize shapes by category
        self._shapes_by_category: dict[str, list[ShapeSpec]] = {
            "Primitives": get_primitive_shapes(),
            "Compound": get_compound_shapes(),
            "Challenging": get_challenging_shapes(),
            "Pathological": get_pathological_shapes(),
            "CSG": get_csg_shapes(),
        }

        # State tracking
        self._last_category: str = "Primitives"
        self._last_shape: str = "cube"
        self._last_budget: int = 16
        self._last_containment: bool = False
        self._last_opacity: float = 0.7
        self._last_show: bool = True
        self._needs_recompute = True
        self._needs_visual_update = True

        # Build GUI
        with server.gui.add_folder("Shape Selection"):
            self._category = server.gui.add_dropdown(
                "Category",
                options=list(self._shapes_by_category.keys()),
                initial_value="Primitives",
            )
            initial_shapes = self._get_shape_names_for_category("Primitives")
            self._shape = server.gui.add_dropdown(
                "Shape",
                options=initial_shapes,
                initial_value="cube",
            )

        with server.gui.add_folder("Spherization"):
            self._budget = server.gui.add_slider(
                "Sphere Budget",
                min=1,
                max=128,
                step=1,
                initial_value=16,
            )
            self._containment = server.gui.add_checkbox(
                "Containment Check",
                initial_value=False,
                hint="Cap sphere radii to stay inside mesh (reduces over-extension but may reduce coverage)",
            )
            self._sphere_count = server.gui.add_number(
                "Actual Spheres",
                initial_value=0.0001,
                disabled=True,
            )

        with server.gui.add_folder("Visualization"):
            self._show_spheres = server.gui.add_checkbox(
                "Show Spheres",
                initial_value=True,
            )
            self._opacity = server.gui.add_slider(
                "Sphere Opacity",
                min=0.1,
                max=1.0,
                step=0.1,
                initial_value=0.7,
            )

        with server.gui.add_folder("Quality Metrics"):
            self._coverage = server.gui.add_number(
                "Coverage ↑",
                initial_value=0.0001,
                disabled=True,
                hint="Fraction of mesh surface points inside spheres (0-1). Higher is better.",
            )
            self._tightness = server.gui.add_number(
                "Tightness ↑",
                initial_value=0.0001,
                disabled=True,
                hint="hull_volume / sphere_volume (0-1). Higher means tighter fit. A single bounding sphere ~0.05.",
            )
            self._quality = server.gui.add_number(
                "Quality ↑",
                initial_value=0.0001,
                disabled=True,
                hint="coverage × tightness. Combined score rewarding both high coverage and tight fit.",
            )
            self._volume_overhead = server.gui.add_number(
                "Volume Overhead ↓",
                initial_value=0.0001,
                disabled=True,
                hint="sphere_volume / hull_volume. Lower is better (1.0 = perfect, >1 = over-approximation).",
            )
            self._over_extension = server.gui.add_number(
                "Over Extension ↓",
                initial_value=0.0001,
                disabled=True,
                hint="Volume inside spheres but outside mesh, as ratio of mesh volume. Lower is better (0 = perfect).",
            )

    def _get_shape_names_for_category(self, category: str) -> list[str]:
        """Get shape names for a category."""
        shapes = self._shapes_by_category.get(category, [])
        return [s.name for s in shapes]

    def poll(self) -> None:
        """Check for GUI changes and update internal state."""
        # Category change - update shape dropdown options
        if self._category.value != self._last_category:
            self._last_category = self._category.value
            new_options = self._get_shape_names_for_category(self._category.value)
            self._shape.options = new_options
            # Select first shape in new category if current not available
            if self._shape.value not in new_options and new_options:
                self._shape.value = new_options[0]
            self._needs_recompute = True

        # Shape change
        if self._shape.value != self._last_shape:
            self._last_shape = self._shape.value
            self._needs_recompute = True

        # Budget change
        if int(self._budget.value) != self._last_budget:
            self._last_budget = int(self._budget.value)
            self._needs_recompute = True

        # Containment change
        if self._containment.value != self._last_containment:
            self._last_containment = self._containment.value
            self._needs_recompute = True

        # Visibility/opacity change (visual only)
        if (
            self._show_spheres.value != self._last_show
            or self._opacity.value != self._last_opacity
        ):
            self._last_show = self._show_spheres.value
            self._last_opacity = self._opacity.value
            self._needs_visual_update = True

    @property
    def current_shape_spec(self) -> ShapeSpec | None:
        """Get the currently selected shape spec."""
        return get_shape_by_name(self._shape.value, include_csg=True)

    @property
    def budget(self) -> int:
        return int(self._budget.value)

    @property
    def opacity(self) -> float:
        return float(self._opacity.value)

    @property
    def show_spheres(self) -> bool:
        return bool(self._show_spheres.value)

    @property
    def containment(self) -> bool:
        return bool(self._containment.value)

    @property
    def needs_recompute(self) -> bool:
        return self._needs_recompute

    @property
    def needs_visual_update(self) -> bool:
        return self._needs_visual_update

    def mark_computed(self) -> None:
        self._needs_recompute = False

    def mark_visuals_updated(self) -> None:
        self._needs_visual_update = False

    def update_metrics(
        self,
        coverage: float,
        tightness: float,
        quality: float,
        overhead: float,
        over_extension: float,
        count: int,
    ) -> None:
        """Update the metrics display."""
        self._coverage.value = round(coverage, 4)
        self._tightness.value = round(tightness, 4)
        self._quality.value = round(quality, 4)
        self._volume_overhead.value = round(overhead, 2)
        self._over_extension.value = round(over_extension, 2)
        self._sphere_count.value = count


class _ShapeVisuals:
    """Manages mesh and sphere visualization in viser."""

    def __init__(self, server: viser.ViserServer):
        self._server = server
        self._mesh_handle = None
        self._sphere_handles: list = []

    def update_mesh(self, mesh: trimesh.Trimesh) -> None:
        """Update displayed mesh."""
        # Remove old mesh
        if self._mesh_handle is not None:
            self._mesh_handle.remove()

        # Center mesh at origin for better viewing
        centered_mesh = mesh.copy()
        centered_mesh.vertices -= centered_mesh.centroid

        # Add new mesh
        self._mesh_handle = self._server.scene.add_mesh_trimesh(
            "/mesh",
            centered_mesh,
        )

    def update_spheres(
        self,
        spheres: list[Sphere],
        opacity: float,
        visible: bool,
        mesh_centroid: np.ndarray,
    ) -> None:
        """Update sphere visualization."""
        # Clear existing spheres
        for h in self._sphere_handles:
            h.remove()
        self._sphere_handles.clear()

        if not visible or not spheres:
            return

        for i, sphere in enumerate(spheres):
            color = SPHERE_COLORS[i % len(SPHERE_COLORS)]
            rgb = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

            # Offset center to match centered mesh
            center = np.asarray(sphere.center) - mesh_centroid

            handle = self._server.scene.add_icosphere(
                f"/spheres/{i}",
                radius=float(sphere.radius),
                position=(float(center[0]), float(center[1]), float(center[2])),
                color=rgb,
                opacity=opacity,
            )
            self._sphere_handles.append(handle)


def main() -> None:
    """Visualize test shapes with interactive sphere decomposition."""
    print("Loading shapes...")

    # Initialize viser server
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

    # Create GUI and visualization managers
    gui = _ShapesGui(server)
    viz = _ShapeVisuals(server)

    # Current state
    current_mesh: trimesh.Trimesh | None = None
    current_spheres: list[Sphere] = []
    current_centroid: np.ndarray = np.zeros(3)

    print("Starting visualization (open browser to view)...")
    while True:
        gui.poll()

        # Recompute when shape or budget changes
        if gui.needs_recompute:
            spec = gui.current_shape_spec
            if spec is not None:
                # Load shape
                try:
                    current_mesh = spec.factory()
                except RuntimeError as e:
                    print(f"Failed to create shape {spec.name}: {e}")
                    gui.mark_computed()
                    continue

                current_centroid = current_mesh.centroid.copy()
                points = np.asarray(current_mesh.sample(5000))

                # Spherize with optional containment check
                params = None
                if gui.containment:
                    params = SpherizeParams(
                        containment_samples=50,
                        min_containment_fraction=0.50,
                    )
                t0 = time.perf_counter()
                current_spheres = spherize(
                    current_mesh, target_spheres=gui.budget, params=params
                )
                elapsed = (time.perf_counter() - t0) * 1000
                containment_str = " [containment]" if gui.containment else ""
                print(
                    f"[{spec.name}] Generated {len(current_spheres)} spheres "
                    f"in {elapsed:.1f}ms{containment_str}"
                )

                # Compute metrics
                coverage = compute_coverage(points, current_spheres)
                tightness = compute_tightness(points, current_spheres)
                overhead = compute_volume_overhead(points, current_spheres)
                quality = compute_quality(coverage, tightness)
                over_ext = compute_over_extension(current_mesh, current_spheres)

                gui.update_metrics(
                    coverage,
                    tightness,
                    quality,
                    overhead,
                    over_ext["over_extension_ratio"],
                    len(current_spheres),
                )

                # Update visuals
                viz.update_mesh(current_mesh)
                viz.update_spheres(
                    current_spheres, gui.opacity, gui.show_spheres, current_centroid
                )

            gui.mark_computed()
            gui.mark_visuals_updated()

        # Update visuals only (opacity/visibility change)
        if gui.needs_visual_update:
            viz.update_spheres(
                current_spheres, gui.opacity, gui.show_spheres, current_centroid
            )
            gui.mark_visuals_updated()

        time.sleep(0.05)


if __name__ == "__main__":
    main()
