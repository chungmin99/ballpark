# Spherization Improvement Specification

## Problem Statement

The current spherification algorithm produces "blobby/messy" results that fail to meet quality standards:

### Core Issues
1. **Uneven sphere sizes** - Mix of large and tiny spheres creating irregular appearance
2. **Misaligned splits** - PCA-based splits don't follow shape symmetry (e.g., cube splits diagonally)
3. **Over-splitting** - Too many small spheres when fewer larger ones would suffice
4. **Gaps/overlaps** - Visible coverage issues
5. **No interior spheres** - All spheres placed near surface, none in mesh interior for thick objects
6. **Symmetry violations** - Symmetric shapes get asymmetric decompositions
7. **Over-extension** - Spheres extend beyond mesh surface

### Tuning Problems
- Budget changes are unpredictable (20→40 spheres changes everything, not just adds more)
- Settings that work for one robot fail on another
- Non-deterministic feel - hard to predict where splits happen

### Root Cause
The recursive PCA-based splitting algorithm is fundamentally geometry-unaware. It makes splitting decisions based on local statistics (aspect ratio, tightness) rather than understanding shape structure.

---

## Goals

### Primary Goal: Robustness
The spherification process should work reliably across diverse meshes without manual tuning.

### Quality Targets
1. **Regular appearance** - Spheres should look intentional, not random
2. **Primitive excellence** - Box, cylinder, capsule must have clean symmetric decompositions
3. **Interior coverage** - Spheres placed throughout volume, not just near surface
4. **Minimal over-extension** - Slight (1-5%) extension OK for coverage, but no excessive protrusion
5. **Balanced overlap** - Some overlap for coverage, but avoid excessive redundancy

### Constraints
- **Performance**: <100ms per link
- **Dependencies**: NumPy/SciPy, current stack (trimesh, JAX), Open3D OK, vhacdx OK (optional)
- **Mesh quality**: Can assume watertight meshes
- **API**: Keep same `spherize()` interface
- **Sphere budget**: Typically 5-20 spheres per mesh

---

## Technical Approach

### Algorithm Architecture: Hybrid Pipeline

```
Mesh → Primitive Detection → [Primitive Handler OR General Handler] → Quality Metrics → Output
                                         │
                                         ├── For primitives: Hardcoded optimal decomposition
                                         │
                                         └── For general meshes:
                                               Convex Decomposition → Per-Part Spherization → Merge
```

### 1. Primitive Detection & Handling

**Supported Primitives**: Box, Cylinder, Capsule

**Detection**: Automatic detection (not user-specified) using:
- Geometry fingerprinting (face count, edge ratios, symmetry analysis)
- Bounding box vs mesh volume ratio
- Surface curvature analysis

**Hardcoded Decompositions**:
- Adaptive to aspect ratio (e.g., elongated box: 1×2×4 grid, cube-ish: 2×2×2 grid)
- Guaranteed symmetric, regular sphere placement
- Sphere count matched to budget

### 2. Convex Decomposition (for non-primitives)

**Approach**: Decompose concave meshes into convex parts before spherization

**Implementation**:
- Use `vhacdx` (optional dependency) via `trimesh.decomposition`
- If vhacdx unavailable, fallback to convex hull or grid-based approach

**Per-Part Spherization**: Each convex part spherized independently, then merged

### 3. Interior Sphere Placement

**Center Candidate Generation**:
- Sample interior points (volume sampling for watertight meshes)
- Place sphere centers **inside** mesh, not just near surface
- More spheres in thick regions, fewer in thin regions

**Radius Determination**:
- Radius = distance to nearest mesh surface
- Allow slight extension (1-5% configurable padding) for better coverage
- Never exceed mesh boundary significantly

**Fallback if medial axis infeasible**: Grid-based interior sampling
- Create 3D grid inside mesh bounding box
- Filter to points inside mesh
- Use as sphere center candidates
- Select subset via greedy coverage algorithm

### 4. Medial Axis Research (Feasibility Investigation)

**Question to Answer**: Can medial axis computation work within constraints?

**Constraints**:
- <100ms per link
- Minimal dependencies (Open3D OK)
- Robustness across diverse meshes

**Research Tasks**:
1. Evaluate Open3D's medial axis / skeleton capabilities
2. Test performance on Panda robot links
3. Compare quality vs grid-based approach
4. Document findings for go/no-go decision

**If Feasible**: Use medial axis points as sphere center candidates
**If Not Feasible**: Use grid-based interior sampling as primary approach

### 5. Sphere Fitting Strategy

**Center-first approach**:
1. Generate candidate center positions (interior points)
2. For each center, radius = distance to nearest surface (with small padding)
3. Select subset of spheres via greedy coverage optimization

**Optimization objectives** (for selection):
- Maximize volume coverage
- Minimize total sphere volume (tightness)
- Minimize over-extension
- Prefer uniform sphere sizes within regions

### 6. Quality Metrics & Output

**Always return quality metrics with sphere output**:

**Current metrics to keep**:
- Coverage (fraction of points inside spheres)
- Tightness (hull volume / sphere volume)
- Over-extension (volume outside mesh)

**New metrics to add**:
- **Regularity/uniformity score** - Measure how organized vs chaotic the placement is
- **Symmetry score** - How well spherization respects mesh symmetry

**Stricter test thresholds**: Current thresholds are too permissive - tighten them based on new algorithm performance

### 7. Failure Detection & Fallback

**"Detect and fallback" strategy**:
1. Run primary algorithm
2. Check quality metrics against thresholds
3. If quality insufficient:
   - Try alternative approach (e.g., grid-based if medial axis failed)
   - Log warning with diagnostics
4. Return best result with quality scores

**Best effort on budget**: If sphere budget too small for good coverage, do best possible (no auto-adjustment)

---

## Refinement Integration

**Keep separate**: The JAX-based refinement (`_nlls_refine.py`) remains available as optional second pass

**Better initial placement reduces refinement burden**: Good starting positions should require less optimization

---

## Files to Modify

### Core Algorithm
- `src/ballpark/_spherize.py` - Major refactor: new spherization pipeline
- `src/ballpark/_adaptive_tight.py` - May be deprecated or heavily modified

### New Modules
- `src/ballpark/_primitives.py` - Primitive detection and hardcoded decompositions
- `src/ballpark/_interior.py` - Interior point sampling and sphere placement

### Metrics
- `src/ballpark/metrics.py` - Add regularity and symmetry metrics

### Configuration
- `src/ballpark/_config.py` - Update presets for new algorithm

### Tests
- `tests/test_shapes.py` - Tighten thresholds
- `tests/test_primitives.py` - Add primitive-specific tests
- `tests/test_robots.py` - Add Panda robot validation

### Dependencies
- `pyproject.toml` - Add `vhacdx` as optional dependency

---

## Implementation Phases

### Phase 1: Primitive Handling
- Implement automatic primitive detection
- Create hardcoded optimal decompositions for box, cylinder, capsule
- Add primitive-specific tests with strict thresholds

### Phase 2: Interior Sphere Placement
- Implement grid-based interior sampling
- Add greedy sphere selection algorithm
- Update radius computation (distance to surface)

### Phase 3: Convex Decomposition Integration
- Add vhacdx as optional dependency
- Implement per-part spherization pipeline
- Add fallback for when vhacdx unavailable

### Phase 4: Quality Metrics
- Add regularity/uniformity metric
- Add symmetry score
- Return metrics with all spherization results
- Tighten test thresholds

### Phase 5: Medial Axis Research
- Spike/prototype medial axis approach
- Benchmark against grid-based
- Make go/no-go decision
- Document findings

### Phase 6: Failure Detection & Polish
- Implement quality-based fallback logic
- Validate on Panda robot
- Performance optimization (<100ms per link)

---

## Test Validation

### Primary Test Robot: Panda (Franka Panda 7DOF arm)

### Test Categories
1. **Primitives**: Box, cylinder, capsule with various aspect ratios
2. **Panda links**: All collision geometries from Panda URDF
3. **Regression**: Existing test meshes shouldn't regress

### Quality Assertions
- Primitives: Must achieve symmetric, regular decompositions
- General meshes: Coverage >90%, over-extension <5%
- Performance: <100ms per link

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Interior spheres? | Always place interior spheres |
| Primitive detection | Automatic, not user-specified |
| Convex decomposition | Use vhacdx (optional dep) |
| Medial axis | Research feasibility, grid-based fallback |
| API changes | Keep same interface |
| Timeline | Production-ready implementation |

---

## References

- [vhacdx - V-HACD Python bindings](https://github.com/trimesh/vhacdx)
- [trimesh.decomposition](https://trimesh.org/trimesh.decomposition.html)
- [foam - CoMMALab sphere decomposition](https://github.com/CoMMALab/foam)
- [MorphIt - HIRO group](https://github.com/HIRO-group/MorphIt-1)
