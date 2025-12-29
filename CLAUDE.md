# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Worktree Instructions

When working in a git worktree, always check if `../../CLAUDE.md` exists and read it first. That file may contain additional project-wide instructions that apply across all worktrees. Do not commit or modify the parent CLAUDE.md file from within a worktree.

## Project Overview

Ballpark is a Python library for sphere decomposition - converting 3D meshes into collections of bounding spheres for collision approximation. Primary use cases are mesh spherization and robot collision geometry generation from URDFs.

## Common Commands

```bash
# Install
pip install -e .                    # Base install
pip install -e ".[robot]"           # With robot utilities, visualization, CLI
pip install -e ".[dev]"             # With dev tools

# Lint
ruff check src/

# Type check
pyright

# Test
pytest

# Run robot visualization demo
python scripts/spherize_robot_interactive.py --robot_name panda
```

## Architecture

### Core Algorithm Pipeline
```
Mesh → Sample Points → Adaptive Fitting → Optional Refinement → Sphere List
```

### Key Modules

**`_sphere.py`**: Simple `Sphere(center, radius)` dataclass.

**`_adaptive_tight.py`**: Core algorithm - `spherize_adaptive_tight()`. Recursively splits point clouds using PCA, fits bounding spheres with iterative refinement, respects budget limits. Key helpers: `fit_sphere_minmax()`, `get_aspect_ratio()`, `compute_tightness()`.

**`_nlls_refine.py`**: JAX-based optimization - `refine_spheres_nlls()`. Multi-objective loss (under-approximation, over-approximation, overlap, uniformity) using Adam optimizer.

**`_robot.py`**: Robot integration - `get_collision_mesh_for_link()` extracts URDF collision geometry, `compute_spheres_for_robot()` allocates sphere budget across links proportionally to link complexity, `load_robot_from_urdf()` wraps pyroki import for kinematics.

### Data Flow (Robot Use Case)
```
URDF → get_collision_mesh_for_link() → compute_spheres_for_robot()
     → spherize_adaptive_tight() [per-link] → refine_spheres_nlls() [optional]
     → dict[link_name → List[Sphere]]
```

### Key Hyperparameters
- `target_tightness`: Sphere vol / hull vol threshold (default 1.2)
- `aspect_threshold`: Max aspect ratio before splitting (default 1.3)
- `percentile`: Use percentile distances for outlier robustness (default 98.0)
- `padding`: Safety margin multiplier (default 1.02)
- Refinement lambdas: `lambda_under`, `lambda_over`, `lambda_overlap`, `lambda_uniform`
