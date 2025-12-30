# Ballpark Test Suite

## Test Files

| File | Description |
|------|-------------|
| `test_primitives.py` | 9 basic shapes (cube, sphere, cylinder, capsule, etc.) |
| `test_compound.py` | 7 multi-part shapes (L-shape, T-shape, dumbbell, etc.) |
| `test_challenging.py` | 6 difficult geometries (thin plate, star, thick torus, etc.) |
| `test_pathological.py` | 3 degenerate shapes with extreme aspect ratios (>= 15:1) |
| `test_csg.py` | 4 CSG shapes requiring boolean operations |
| `test_shapes.py` | Regression tests comparing against stored snapshots |
| `test_robots.py` | Robot URDF spherization tests |

## Metrics

| Metric | Formula | Meaning |
|--------|---------|---------|
| **Coverage** | `points_inside_spheres / total_points` | Fraction of surface covered (higher = better) |
| **Tightness** | `hull_volume / sphere_volume` | How tightly spheres fit (higher = better) |
| **Volume Overhead** | `sphere_volume / hull_volume` | Wasted volume ratio (lower = better) |
| **Quality** | `coverage * tightness` | Combined score (higher = better) |

A single bounding sphere achieves ~100% coverage but only ~5% tightness. Good sphere decomposition maintains high coverage while improving tightness.

## Shape Categories

### Primitives (Easy)
Basic geometric shapes: cube, sphere, cylinder, capsule, etc.
- Expected quality: high coverage (>85%) and tightness (>0.3)

### Compound (Medium)
Multi-part robot-relevant shapes: L-shape, T-shape, dumbbell, etc.
- Expected quality: good coverage (>75%) and moderate tightness (>0.2)

### Challenging (Hard)
Difficult but non-degenerate geometries: thin_plate (10:1), star, thick_torus, etc.
- Expected quality: acceptable coverage (>75%) and lower tightness (>0.05)
- Volume overhead: <= 20x
- Aspect ratios: < 15:1

### Pathological (Degenerate)
Extreme aspect ratio shapes that cannot be tightly approximated:
- **needle** (20:1 aspect ratio): Very thin cylinder
- **nearly_flat** (100:1 aspect ratio): Nearly 2D box, effectively a paper-thin plate
- **elongated_star** (15:1 aspect ratio): Very flat 4-pointed star

These shapes require extremely permissive thresholds:
- Coverage: >= 60% acceptable
- Tightness: >= 0.001 (0.1%) acceptable
- Volume overhead: <= 1000x acceptable

**Why separate?** These shapes represent edge cases where sphere approximation is fundamentally limited by geometry. Testing them separately prevents them from distorting quality metrics for normal shapes.

## Running Tests

```bash
pytest tests/                    # Run all tests
pytest tests/test_primitives.py  # Run specific category
pytest tests/test_pathological.py # Run pathological/degenerate shapes
pytest tests/ -k "cube"          # Run tests matching pattern
```

## Regenerating Snapshots

```bash
python -c "from tests.test_shapes import generate_snapshots; generate_snapshots()"
```
