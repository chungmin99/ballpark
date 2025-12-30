"""Benchmark sphere decomposition on a simple cube."""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from ballpark import spherize, SpherizeParams


def compute_metrics(mesh, spheres, n_samples: int = 5000):
    """Compute coverage and efficiency metrics."""
    points = mesh.sample(n_samples)

    # Coverage
    covered = np.zeros(len(points), dtype=bool)
    for s in spheres:
        center = np.asarray(s.center)
        radius = float(s.radius)
        dists = np.linalg.norm(points - center, axis=1)
        covered |= dists <= radius
    coverage = float(covered.sum() / len(points))

    # Volume
    total_sphere_vol = sum(4/3 * np.pi * float(s.radius)**3 for s in spheres)
    hull_vol = ConvexHull(points).volume

    return {
        "coverage": coverage,
        "efficiency": hull_vol / total_sphere_vol,
        "vol_ratio": total_sphere_vol / hull_vol,
        "n_spheres": len(spheres),
    }


def main():
    """Run benchmark on cube with different sphere counts."""
    cube = trimesh.creation.box(extents=[1, 1, 1])

    target_counts = [1, 2, 4, 8, 16, 32, 64]

    print("\n" + "="*80)
    print("CUBE BENCHMARK (1x1x1 unit cube)")
    print("="*80)
    print(f"{'Target':>8} | {'Actual':>8} | {'Coverage':>10} | {'Efficiency':>10} | {'Vol Ratio':>10}")
    print("-"*80)

    for target in target_counts:
        spheres = spherize(cube, target_spheres=target)
        metrics = compute_metrics(cube, spheres)
        print(f"{target:>8} | {metrics['n_spheres']:>8} | {metrics['coverage']:>10.4f} | {metrics['efficiency']:>10.4f} | {metrics['vol_ratio']:>10.2f}x")

    print("="*80)

    # Also test with geometry_type=box (symmetry-aware)
    print("\nWith geometry_type='box' (symmetry-aware):")
    print("-"*80)
    print(f"{'Target':>8} | {'Actual':>8} | {'Coverage':>10} | {'Efficiency':>10} | {'Vol Ratio':>10}")
    print("-"*80)

    for target in [1, 2, 4, 8, 16, 32]:
        params = SpherizeParams(geometry_type="box")
        spheres = spherize(cube, target_spheres=target, params=params)
        metrics = compute_metrics(cube, spheres)
        print(f"{target:>8} | {metrics['n_spheres']:>8} | {metrics['coverage']:>10.4f} | {metrics['efficiency']:>10.4f} | {metrics['vol_ratio']:>10.2f}x")

    print("="*80)


if __name__ == "__main__":
    main()
