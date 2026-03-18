# Push Selection Pipeline

Automatic selection of the optimal **tipping edge** and **push contact point** for a robotic tipping maneuver, given an object mesh and an estimated 2D center-of-mass projection.

---

## Purpose

Before the robot can perform a CoM estimation via tipping, it must decide:

1. **Which edge** of the object's support polygon to tip over
2. **Where on the object surface** to place the pusher finger

This module answers both questions by scoring all valid `(tip_edge, push_face)` pairs from the object mesh geometry and ranking them.

---

## Files

| File | Description |
|------|-------------|
| `push_selection_pipeline.py` | Core pipeline: support polygon extraction, tip edge computation, push face search, scoring, and visualization |
| `run_push_selection.py` | Multi-mesh runner: calls the pipeline on all experiment objects and displays results |

---

## Pipeline Overview

```
Mesh (STL/OBJ) + 2D CoM estimate (xc, yc)
        │
        ▼
1. Extract support polygon
   └─ Convex hull of base vertices (z ≈ 0)
        │
        ▼
2. Compute tip edge metrics for each polygon edge
   ├─ edge direction (tipping axis)
   ├─ inward normal (push direction)
   └─ d_perp_to_com (perpendicular CoM–edge distance)
        │
        ▼
3. Select top-k easiest edges (smallest d_perp_to_com)
   For each: find valid push contact faces on opposing side
   └─ face normal must oppose push direction (within 60°)
   └─ face must be above min height (z > 0.02 m)
   └─ estimate local surface curvature (convex vs. concave)
        │
        ▼
4. Score every (tip_edge, push_face) pair
   ├─ Hard constraints (reject if violated)
   └─ Weighted soft scores → ranked list
        │
        ▼
Output: ranked list of ScoredPair objects
```

---

## Scoring Details

### Hard Constraints

Both must pass or the pair is rejected entirely:

| Constraint | Default Threshold | Rationale |
|---|---|---|
| Line-of-action offset | 1 cm (`loa_epsilon`) | Push direction must pass near the CoM in top-down view — otherwise the object rotates sideways rather than tipping cleanly |
| Surface curvature | `curvature_sign < −0.05` rejected | Concave push surfaces cause unstable/slipping finger contact |

### Soft Scores (weighted sum)

| Term | Default Weight | Formula | What It Captures |
|---|:---:|---|---|
| `tipping_ease` | **4.0** | `1 / (1 + d_perp × 20)` | Sigmoid of CoM-to-edge distance. Smaller distance = easier to tip. Dominates the ranking. |
| `push_alignment` | **2.0** | `|face_normal · (−push_dir)|` | How squarely the push face confronts the robot. 1.0 = flat face facing directly at robot, 0 = grazing angle. |
| `leverage` | **1.5** | `push_height / object_height` | Higher contact point → more torque about the tipping edge → easier topple. |
| `edge_stability` | **1.0** | `edge_length / max_edge_length` | Longer tipping axis = more stable rotation (less lateral drift). |

**Total score** = Σ (weight × normalized_term)

`tipping_ease` has 2× the weight of the next term, so the pipeline primarily picks the edge the CoM is closest to, then uses push quality and leverage to break ties.

### Customizing Weights

Pass a custom `weights` dict to `select_push_config()`:

```python
results = select_push_config(
    mesh, com_2d,
    weights={
        'tipping_ease':   4.0,   # increase to prioritize easy-to-tip edges
        'push_alignment': 2.0,   # increase to prioritize flat push surfaces
        'leverage':       1.5,   # increase to prefer higher push points
        'edge_stability': 1.0,   # increase to prefer long, stable tip axes
    }
)
```

---

## Usage

### Quick Start

```python
import trimesh
import numpy as np
from push_selection_pipeline import load_object_mesh, select_push_config, visualize_ranked_pairs

mesh   = load_object_mesh("path/to/object.stl")   # auto-shifts base to z=0
com_2d = np.array([xc, yc])                        # 2D CoM estimate in object frame

ranked = select_push_config(mesh, com_2d, verbose=True)
visualize_ranked_pairs(mesh, ranked, com_2d, top_n=5)
```

### Run on All Experiment Objects

```bash
cd push_selection
python run_push_selection.py
```

Optional flags:

```bash
python run_push_selection.py --objects box heart flashlight   # subset of objects
python run_push_selection.py --backend matplotlib             # use matplotlib instead of trimesh viewer
python run_push_selection.py --no-viz                         # scores only, no visualization
python run_push_selection.py --loa-epsilon 0.02               # relax line-of-action tolerance to 2 cm
python run_push_selection.py --top-k 3                        # evaluate only top 3 tip edges per object
```

### Built-in Demos

```bash
python push_selection_pipeline.py
```

Runs three demos in sequence:
1. Symmetric box (CoM at center)
2. Box with offset CoM (+6 cm toward one edge)
3. Heart STL from `assets/my_objects/`

---

## API Reference

### `load_object_mesh(stl_path)`
Loads an STL/OBJ and translates it so the lowest point is at z=0.

### `select_push_config(mesh, com_2d, loa_epsilon, weights, top_k_edges, verbose)`
Full pipeline. Returns a list of `ScoredPair` objects sorted by score (descending).

### `visualize_ranked_pairs(mesh, ranked_pairs, com_2d, top_n, show, backend)`
Renders the top-n configurations on the mesh.
- `backend='trimesh'` — interactive 3D viewer (default)
- `backend='matplotlib'` — static 3D plot (fallback for headless environments)

### Data Structures

```python
TipEdge
  .v0, .v1          # 3D edge vertices on z=0
  .edge_vec          # unit vector along tipping axis
  .inward_normal     # push direction (unit, horizontal)
  .length            # edge length [m]
  .d_perp_to_com     # CoM–edge perpendicular distance [m]
  .midpoint          # edge midpoint

PushCandidate
  .point             # 3D contact point (face centroid)
  .normal            # outward face normal
  .height            # z-coordinate [m]
  .curvature_sign    # >0 convex, ~0 flat, <0 concave
  .face_index        # index into mesh.faces

ScoredPair
  .tip_edge          # TipEdge
  .push              # PushCandidate
  .score             # total weighted score (−inf if rejected)
  .score_breakdown   # dict with per-term scores
```

---

## Dependencies

```bash
pip install trimesh numpy scipy matplotlib
```

| Package | Use |
|---------|-----|
| `trimesh` | Mesh loading, convex hull, face adjacency, visualization |
| `numpy` | Array math, rotation, dot/cross products |
| `scipy` | ConvexHull for support polygon extraction |
| `matplotlib` | Fallback 3D visualization |
