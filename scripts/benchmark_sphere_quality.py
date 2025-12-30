"""Benchmark sphere decomposition quality across different target counts."""

from __future__ import annotations

import numpy as np
import yourdfpy
from loguru import logger
from robot_descriptions.loaders.yourdfpy import load_robot_description
from scipy.spatial import ConvexHull, QhullError

from ballpark import Robot


def compute_coverage(points: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> float:
    """Compute fraction of points inside at least one sphere."""
    if len(points) == 0 or len(centers) == 0:
        return 0.0

    covered = np.zeros(len(points), dtype=bool)
    for center, radius in zip(centers, radii):
        dists = np.linalg.norm(points - center, axis=1)
        covered |= dists <= radius

    return float(covered.sum() / len(points))


def compute_quality_metrics(mesh, centers: np.ndarray, radii: np.ndarray, n_samples: int = 5000):
    """Compute quality metrics for sphere decomposition.

    Returns dict with:
        - coverage: fraction of surface points inside spheres
        - total_sphere_vol: sum of sphere volumes
        - hull_vol: convex hull volume of mesh
        - efficiency: hull_vol / sphere_vol (higher = tighter fit)
        - quality: coverage * efficiency (combined metric)
    """
    # Sample points from mesh surface
    points = mesh.sample(n_samples)

    # Compute coverage
    coverage = compute_coverage(points, centers, radii)

    # Compute total sphere volume
    total_sphere_vol = sum(4/3 * np.pi * r**3 for r in radii)

    # Compute hull volume
    try:
        hull_vol = ConvexHull(points).volume
    except (QhullError, ValueError):
        hull_vol = mesh.bounding_box.volume

    # Efficiency = how tight the spheres are (higher = better)
    efficiency = hull_vol / total_sphere_vol if total_sphere_vol > 1e-10 else 0.0

    # Combined quality metric
    quality = coverage * efficiency

    return {
        "coverage": coverage,
        "total_sphere_vol": total_sphere_vol,
        "hull_vol": hull_vol,
        "efficiency": efficiency,
        "quality": quality,
    }


def main():
    """Run benchmark with different target sphere counts."""
    # Load robot
    logger.info("Loading Panda robot...")
    urdf_obj = load_robot_description("panda_description")
    urdf_obj = yourdfpy.URDF(
        robot=urdf_obj.robot,
        filename_handler=urdf_obj._filename_handler,
        load_collision_meshes=True,
    )
    robot = Robot(urdf_obj)

    # Get combined mesh for quality computation
    logger.info("Extracting collision meshes...")
    all_meshes = []
    for link_name in robot.collision_links:
        result = robot._get_collision_mesh_for_link(link_name)
        if result is not None:
            m, _ = result  # Unpack (mesh, geom_type) tuple
            all_meshes.append(m)

    import trimesh
    combined_mesh = trimesh.util.concatenate(all_meshes)

    # Test different target counts
    target_counts = [10, 20, 40, 80, 120, 160, 200, 300]

    print("\n" + "="*90)
    print(f"{'Target':>8} | {'Actual':>8} | {'Coverage':>10} | {'Efficiency':>10} | {'Quality':>10} | {'Vol Ratio':>10}")
    print("="*90)

    results = []
    for target in target_counts:
        logger.info(f"Testing target_spheres={target}...")
        result = robot.spherize(target_spheres=target)

        # Collect all spheres
        all_centers = []
        all_radii = []
        for link_name, spheres in result.link_spheres.items():
            for s in spheres:
                all_centers.append(np.asarray(s.center).tolist())
                all_radii.append(float(s.radius))

        centers = np.array(all_centers)
        radii = np.array(all_radii)

        # Compute metrics
        metrics = compute_quality_metrics(combined_mesh, centers, radii)
        metrics["target"] = target
        metrics["actual"] = len(radii)
        results.append(metrics)

        vol_ratio = metrics['total_sphere_vol'] / metrics['hull_vol'] if metrics['hull_vol'] > 0 else 0
        print(f"{target:>8} | {len(radii):>8} | {metrics['coverage']:>10.4f} | {metrics['efficiency']:>10.4f} | {metrics['quality']:>10.4f} | {vol_ratio:>10.2f}x")

    print("="*90)

    # Check for quality degradation
    print("\nQuality trend analysis:")
    qualities = [r["quality"] for r in results]
    coverages = [r["coverage"] for r in results]
    efficiencies = [r["efficiency"] for r in results]

    # Coverage should generally increase with more spheres
    coverage_trend = np.polyfit(range(len(coverages)), coverages, 1)[0]
    print(f"  Coverage trend: {'increasing' if coverage_trend > 0 else 'DECREASING'} (slope: {coverage_trend:.6f})")

    # Efficiency trend
    efficiency_trend = np.polyfit(range(len(efficiencies)), efficiencies, 1)[0]
    print(f"  Efficiency trend: {'increasing' if efficiency_trend > 0 else 'decreasing'} (slope: {efficiency_trend:.6f})")

    # Quality might decrease slightly due to sphere volume overhead
    quality_diffs = [qualities[i+1] - qualities[i] for i in range(len(qualities)-1)]
    significant_drops = [d for d in quality_diffs if d < -0.01]

    if significant_drops:
        print(f"  Note: {len(significant_drops)} efficiency drops when increasing spheres")
        print("  This is expected behavior - more spheres don't always improve quality")
    else:
        print("  No significant quality degradation detected")

    # Summary
    print("\nSummary:")
    print(f"  Coverage range: {min(coverages):.4f} - {max(coverages):.4f} (target: >0.95)")
    print(f"  Efficiency range: {min(efficiencies):.4f} - {max(efficiencies):.4f}")
    print(f"  Best efficiency at target={target_counts[efficiencies.index(max(efficiencies))]} spheres")

    return results


if __name__ == "__main__":
    main()
