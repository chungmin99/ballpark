# Ballpark

Given a 3D mesh or a robot URDF, create a "ballpark" estimate of its spherical collision geometry.

![Sphere decompositions for various robots](assets/splash.png)

Features include:
- Fast mesh-to-sphere decomposition via recursive PCA-based splitting.
- Sphere set optimization with volume/surface/overlap losses, and more.
- Different presets for conservative, balanced, or surface-fitting sphere sets.

For robot URDFs, we also include:
- Automatic sphere distribution across robot links, proportional to their geometry complexity.
- Spheres are optimized on a robot-level to have minimal self-collision distance at rest pose.
- Similar links are detected and share sphere parameters for visual and geometric consistency.
- JSON export with sphere parameters for each link, and an ignore-list of link pairs for collision checking.

We also include an interactive visualization and parameter adjusting GUI using [viser](https://viser.studio).


## Installation

```bash
pip install -e .
pip install -e ".[examples]"
pip install -e ".[dev]"  # with development tools (linting, testing)
```

## Quick Start

### Mesh Spherization

```python
import trimesh
from ballpark import spherize

# Load mesh
mesh = trimesh.load("object.stl")

# Generate spheres with adaptive fitting
spheres = spherize(mesh, target_spheres=32)

for s in spheres:
    print(f"center={s.center}, radius={s.radius}")
```

### Robot URDF Spherization

```python
import yourdfpy
from robot_descriptions.loaders.yourdfpy import load_robot_description
from ballpark import Robot, BallparkConfig, SpherePreset

# Load robot URDF with collision meshes
urdf = load_robot_description("panda_description")
urdf_coll = yourdfpy.URDF(
    robot=urdf.robot,
    load_collision_meshes=True,
)

# Create robot and generate spheres
robot = Robot(urdf_coll)
result = robot.spherize(target_spheres=100)

# Optional: refine with link- and robot-level costs
config = BallparkConfig.from_preset(SpherePreset.BALANCED)
result = robot.refine(result, config=config)

for link_name, spheres in result.link_spheres.items():
    print(f"{link_name}: {len(spheres)} spheres")
```

## Acknowledgments

This project builds on ideas from:
- [foam](https://github.com/CoMMALab/foam)
- [MorphIt](https://github.com/HIRO-group/MorphIt-1)

