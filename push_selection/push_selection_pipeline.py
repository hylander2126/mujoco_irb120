"""
Push Face / Tip Edge Selection Pipeline
========================================
Given:
  - Object mesh (STL or OBJ)
  - 2D CoM projection (xc, yc) in object frame
  - Support plane at z = 0

Produces:
  - Ranked list of (tip_edge, push_face_contact) pairs
  - Scoring based on tipping feasibility, push surface quality, and leverage

Dependencies: trimesh, numpy, scipy
"""

import trimesh
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TipEdge:
    """A candidate tipping edge on the support polygon."""
    v0: np.ndarray           # vertex 0 (3D, on z=0 plane)
    v1: np.ndarray           # vertex 1 (3D, on z=0 plane)
    edge_vec: np.ndarray     # unit vector along edge
    inward_normal: np.ndarray  # unit normal pointing into the object (horizontal)
    length: float
    d_perp_to_com: float     # perpendicular distance from edge to 2D CoM
    midpoint: np.ndarray

    @property
    def direction(self):
        """Tipping axis direction (unit vector along edge)."""
        return self.edge_vec


@dataclass
class PushCandidate:
    """A candidate push contact point on the object surface."""
    point: np.ndarray        # 3D contact point
    normal: np.ndarray       # outward surface normal at contact
    height: float            # z-coordinate (push height)
    curvature_sign: float    # >0 convex, ~0 flat, <0 concave
    face_index: int          # mesh face index


@dataclass
class ScoredPair:
    """A scored (tip_edge, push_contact) pair."""
    tip_edge: TipEdge
    push: PushCandidate
    score: float
    score_breakdown: dict = field(default_factory=dict)


# =============================================================================
# Step 1: Extract Support Polygon and Tip Edges
# =============================================================================

def extract_support_polygon(mesh: trimesh.Trimesh, z_tol: float = 1e-3) -> np.ndarray:
    """
    Extract the convex hull of vertices on the support plane (z ≈ 0).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Object mesh with support surface at z ≈ 0.
    z_tol : float
        Tolerance for identifying base vertices.

    Returns
    -------
    hull_vertices : (N, 2) array
        Ordered 2D vertices of the convex support polygon.
    """
    # Find vertices near z = 0
    base_mask = np.abs(mesh.vertices[:, 2]) < z_tol
    if base_mask.sum() < 3:
        # Fallback: use lowest z-slice
        z_min = mesh.vertices[:, 2].min()
        base_mask = mesh.vertices[:, 2] < (z_min + z_tol * 10)

    base_verts_2d = mesh.vertices[base_mask][:, :2]

    # Convex hull of base projection
    from scipy.spatial import ConvexHull
    hull = ConvexHull(base_verts_2d)
    hull_vertices = base_verts_2d[hull.vertices]

    return hull_vertices


def compute_tip_edges(hull_vertices_2d: np.ndarray,
                      com_2d: np.ndarray) -> List[TipEdge]:
    """
    Compute candidate tip edges from the support polygon.

    Parameters
    ----------
    hull_vertices_2d : (N, 2) array
        Ordered convex hull vertices (CCW).
    com_2d : (2,) array
        2D CoM projection (xc, yc).

    Returns
    -------
    edges : list of TipEdge
        All edges of the support polygon with computed metrics.
    """
    n = len(hull_vertices_2d)
    edges = []

    for i in range(n):
        v0_2d = hull_vertices_2d[i]
        v1_2d = hull_vertices_2d[(i + 1) % n]

        # 3D versions (on z=0 plane)
        v0 = np.array([v0_2d[0], v0_2d[1], 0.0])
        v1 = np.array([v1_2d[0], v1_2d[1], 0.0])

        edge_vec_raw = v1 - v0
        length = np.linalg.norm(edge_vec_raw)
        if length < 1e-8:
            continue
        edge_vec = edge_vec_raw / length

        # Inward normal (pointing toward interior of polygon)
        # For CCW-ordered hull, inward normal is: rotate edge_vec by -90 deg
        outward_normal_2d = np.array([edge_vec[1], -edge_vec[0], 0.0])
        # Check if it points toward CoM (inward)
        midpoint = (v0 + v1) / 2.0
        to_com = np.array([com_2d[0], com_2d[1], 0.0]) - midpoint
        if np.dot(outward_normal_2d[:2], to_com[:2]) < 0:
            inward_normal = -outward_normal_2d
        else:
            inward_normal = outward_normal_2d

        # Perpendicular distance from CoM to edge line (2D scalar cross product).
        # Avoid np.cross on 2D vectors (deprecated in newer NumPy).
        edge_2d = v1_2d - v0_2d
        v0_to_com = com_2d - v0_2d
        cross_2d = edge_2d[0] * v0_to_com[1] - edge_2d[1] * v0_to_com[0]
        d_perp = abs(cross_2d) / length

        edges.append(TipEdge(
            v0=v0, v1=v1,
            edge_vec=edge_vec,
            inward_normal=inward_normal,
            length=length,
            d_perp_to_com=d_perp,
            midpoint=midpoint,
        ))

    return edges


# =============================================================================
# Step 2: Evaluate Push Surface Quality
# =============================================================================

def find_push_candidates(mesh: trimesh.Trimesh,
                         tip_edge: TipEdge,
                         top_band_frac: float = 0.20,
                         normal_alignment_thresh: float = 0.5,
                         n_samples: int = 50) -> List[PushCandidate]:
    """
    For a given tip edge, find candidate push contact points near the TOP
    of the object on the opposing side.

    Only faces whose centroid falls within the uppermost `top_band_frac`
    of the object's height range are considered.  This mirrors the bottom
    support-polygon approach but for the push surface: we want to push near
    the top to maximise torque about the tipping edge, and we rely on the
    2nd-order surface curvature (already computed per-face) rather than a
    simple height threshold, so the full curvature-based scoring still applies
    within this top band.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Object mesh.
    tip_edge : TipEdge
        The candidate tipping edge.
    top_band_frac : float
        Fraction of total object height defining the "top band".
        0.20 = top 20 % of the object height (default).
    normal_alignment_thresh : float
        Minimum |cos(angle)| between push direction and face outward normal.
        0.5 = 60° tolerance. 0.7 = 45° tolerance.
    n_samples : int
        Maximum number of candidate points to return.

    Returns
    -------
    candidates : list of PushCandidate
        Viable push contact points within the top band, sorted by height (descending).
    """
    push_dir = tip_edge.inward_normal  # direction robot pushes (toward tip edge)

    face_normals = mesh.face_normals
    face_centroids = mesh.triangles_center

    z_min = mesh.bounds[0, 2]
    z_max = mesh.bounds[1, 2]
    z_band_floor = z_max - top_band_frac * (z_max - z_min)

    # Filter faces:
    # 1. Normal opposes push direction (face confronts the approaching robot)
    alignment = np.dot(face_normals, push_dir)
    opposing_mask = alignment < -normal_alignment_thresh

    # 2. Centroid must be in the top band of the object.
    height_mask = face_centroids[:, 2] >= z_band_floor

    # 3. Centroid must be on the FAR side of the object from the tip edge.
    #    push_dir points TOWARD the tip edge (inward), so the robot approaches
    #    from the OPPOSITE direction.  Faces the robot can reach project
    #    GREATER than the mesh center along push_dir.
    mesh_center_proj = np.dot(mesh.centroid, push_dir)
    face_projs       = face_centroids @ push_dir
    far_side_mask    = face_projs > mesh_center_proj

    valid_mask = opposing_mask & height_mask & far_side_mask
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) == 0:
        return []

    candidates = []
    for fi in valid_indices:
        centroid = face_centroids[fi]
        normal = face_normals[fi]
        height = centroid[2]

        # Estimate local curvature sign from neighboring face normals
        # Positive = convex (good), negative = concave (bad)
        curvature_sign = _estimate_local_curvature(mesh, fi, push_dir)

        candidates.append(PushCandidate(
            point=centroid,
            normal=normal,
            height=height,
            curvature_sign=curvature_sign,
            face_index=fi,
        ))

    # Sort by height descending (prefer higher push points for leverage)
    candidates.sort(key=lambda c: c.height, reverse=True)

    # Return top n_samples
    return candidates[:n_samples]


def _estimate_local_curvature(mesh: trimesh.Trimesh,
                              face_idx: int,
                              push_dir: np.ndarray) -> float:
    """
    Estimate local curvature sign at a face by comparing its normal
    to neighboring face normals. Convex surfaces have normals that
    fan outward; concave surfaces have normals that converge.

    Returns positive for convex, negative for concave, ~0 for flat.
    """
    # Get face adjacency
    neighbors = mesh.face_adjacency
    # Find faces adjacent to face_idx
    adj_mask = (neighbors[:, 0] == face_idx) | (neighbors[:, 1] == face_idx)
    adj_faces = neighbors[adj_mask].flatten()
    adj_faces = adj_faces[adj_faces != face_idx]

    if len(adj_faces) == 0:
        return 0.0

    center_normal = mesh.face_normals[face_idx]
    neighbor_normals = mesh.face_normals[adj_faces]

    # Only consider neighbors that are roughly coplanar (same face of object)
    # Sharp edges (like box corners) connect different faces and should be ignored
    dots = np.dot(neighbor_normals, center_normal)
    coplanar_mask = dots > 0.7  # within ~45 degrees

    if not np.any(coplanar_mask):
        # All neighbors are on different faces → this is a flat face with sharp edges
        return 0.0  # flat

    coplanar_normals = neighbor_normals[coplanar_mask]
    # For coplanar neighbors, check if normals converge (concave) or diverge (convex)
    dn = coplanar_normals - center_normal
    curvature_sign = -np.mean(np.dot(dn, push_dir))

    return curvature_sign


# =============================================================================
# Step 3: Line of Action Check
# =============================================================================

def line_of_action_offset(push_point: np.ndarray,
                          push_dir: np.ndarray,
                          com_2d: np.ndarray) -> float:
    """
    Compute the perpendicular distance between the line of action
    (ray from push_point along push_dir) and the 2D CoM.

    All computed in the XY plane (top-down view).

    Returns
    -------
    d : float
        Perpendicular distance in XY plane.
    """
    # Project to 2D
    p = push_point[:2]
    d = push_dir[:2]
    d_norm = np.linalg.norm(d)
    if d_norm < 1e-8:
        return np.inf
    d_hat = d / d_norm

    # Distance from CoM to the line through p along d_hat
    v = com_2d - p
    # Perpendicular component
    offset = abs(v[0] * (-d_hat[1]) + v[1] * d_hat[0])

    return offset


# =============================================================================
# Step 4: Scoring Function
# =============================================================================

def score_pair(tip_edge: TipEdge,
               push: PushCandidate,
               com_2d: np.ndarray,
               object_height: float,
               max_edge_length: float,
               loa_epsilon: float = 0.01,
               weights: Optional[dict] = None) -> ScoredPair:
    """
    Score a (tip_edge, push_contact) pair.

    Parameters
    ----------
    tip_edge : TipEdge
    push : PushCandidate
    com_2d : (2,) array
    object_height : float
        Total object height for normalizing push leverage.
    max_edge_length : float
        Longest edge length for normalization.
    loa_epsilon : float
        Maximum allowable line-of-action offset from CoM (meters).
    weights : dict, optional
        Keys: 'tipping_ease', 'push_alignment', 'leverage', 'edge_stability'

    Returns
    -------
    scored : ScoredPair
    """
    if weights is None:
        weights = {
            'tipping_ease': 4.0,      # Primary: CoM proximity to edge
            'push_alignment': 2.0,    # Secondary: push face normal quality
            'leverage': 1.5,          # Push height
            'edge_stability': 1.0,    # Edge length (anti-swirl)
        }

    # --- Hard constraints ---
    push_dir = tip_edge.inward_normal
    loa_offset = line_of_action_offset(push.point, push_dir, com_2d)
    if loa_offset > loa_epsilon:
        return ScoredPair(tip_edge, push, score=-np.inf,
                          score_breakdown={
                              'rejected': f'LoA offset {loa_offset:.4f}m > tol {loa_epsilon:.4f}m — '
                                          f'push line misses CoM; object would spin sideways',
                              'loa_offset': loa_offset,
                          })

    # Reject concave push surfaces
    if push.curvature_sign < -0.05:
        return ScoredPair(tip_edge, push, score=-np.inf,
                          score_breakdown={
                              'rejected': f'concave push surface (curvature={push.curvature_sign:.3f}) — '
                                          f'robot finger would slip off inward curve',
                              'loa_offset': loa_offset,
                          })

    # --- Soft scores ---
    # 1. Tipping ease: inversely proportional to CoM-edge distance
    #    Closer CoM to edge = easier to tip
    if tip_edge.d_perp_to_com > 1e-6:
        s_tip = 1.0 / tip_edge.d_perp_to_com
    else:
        s_tip = 1e6  # CoM right on edge — trivial to tip
    # Normalize: use max possible distance (half the object extent)
    # We'll leave this un-normalized for now; relative ranking still works

    # 2. Push face alignment: cos(angle between -push_dir and face outward normal)
    #    Want face normal ≈ -push_dir (face faces the robot)
    s_align = abs(np.dot(push.normal, -push_dir))  # ∈ [0, 1]

    # 3. Leverage: push height normalized by object height
    s_leverage = push.height / max(object_height, 1e-3)  # ∈ [0, 1]

    # 4. Edge stability: longer edge = more stable tipping axis
    s_edge = tip_edge.length / max(max_edge_length, 1e-3)  # ∈ [0, 1]

    # Weighted sum (s_tip is on different scale; normalize it)
    # Normalize s_tip to [0, 1] range using sigmoid-like mapping
    s_tip_norm = 1.0 / (1.0 + tip_edge.d_perp_to_com * 20)  # tunable scale

    total = (weights['tipping_ease'] * s_tip_norm
             + weights['push_alignment'] * s_align
             + weights['leverage'] * s_leverage
             + weights['edge_stability'] * s_edge)

    breakdown = {
        'tipping_ease': s_tip_norm,
        'push_alignment': s_align,
        'leverage': s_leverage,
        'edge_stability': s_edge,
        'loa_offset': loa_offset,
    }

    return ScoredPair(tip_edge, push, score=total, score_breakdown=breakdown)


# =============================================================================
# Main Pipeline
# =============================================================================

def select_push_config(mesh: trimesh.Trimesh,
                       com_2d: np.ndarray,
                       loa_epsilon: float = 0.03,
                       weights: Optional[dict] = None,
                       top_k_edges: int = 5,
                       verbose: bool = True) -> List[ScoredPair]:
    """
    Full pipeline: given mesh and 2D CoM, return ranked push configurations.

    ALL candidates are returned — valid ones ranked by score first, then
    rejected ones (score == -inf) appended after, each with a rejection
    reason in score_breakdown['rejected'].  Visualization will show all
    of them with distinct color coding so you can diagnose filtering issues.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Object mesh, oriented with support surface at z ≈ 0.
    com_2d : (2,) array
        Estimated 2D CoM projection (xc, yc).
    loa_epsilon : float
        Line of action tolerance (meters).  Default 3 cm.
    weights : dict, optional
        Scoring weights.
    top_k_edges : int
        Only evaluate push faces for the top-k closest edges.
    verbose : bool
        Print progress info.

    Returns
    -------
    ranked_pairs : list of ScoredPair
        Valid pairs (score > -inf) sorted descending, followed by rejected
        pairs sorted by loa_offset ascending so the 'least bad' rejected
        candidates appear first.
    """
    com_2d = np.asarray(com_2d, dtype=float)

    # --- Step 1: Support polygon and tip edges ---
    hull_verts = extract_support_polygon(mesh)
    edges = compute_tip_edges(hull_verts, com_2d)

    if verbose:
        print(f"Support polygon: {len(hull_verts)} vertices, {len(edges)} edges")

    # Sort edges by CoM proximity (ascending distance = best first)
    edges.sort(key=lambda e: e.d_perp_to_com)

    if verbose:
        print(f"\nTip edge ranking (by CoM proximity):")
        for i, e in enumerate(edges):
            print(f"  {i}: d_perp={e.d_perp_to_com:.4f}m, length={e.length:.4f}m, "
                  f"midpoint=({e.midpoint[0]:.3f}, {e.midpoint[1]:.3f})")

    # --- Step 2: For top-k edges, find push candidates ---
    object_height = mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min()
    max_edge_length = max(e.length for e in edges)

    candidate_edges = edges[:top_k_edges]

    valid_pairs = []
    rejected_pairs = []

    for edge in candidate_edges:
        push_candidates = find_push_candidates(mesh, edge)
        if verbose:
            print(f"\n  Edge (d={edge.d_perp_to_com:.4f}): "
                  f"{len(push_candidates)} push candidates found")

        for push in push_candidates:
            pair = score_pair(
                edge, push, com_2d,
                object_height, max_edge_length,
                loa_epsilon, weights
            )
            if pair.score > -np.inf:
                valid_pairs.append(pair)
            else:
                rejected_pairs.append(pair)

    # --- Step 3: Rank valid pairs; sort rejected by LoA offset ascending ---
    valid_pairs.sort(key=lambda p: p.score, reverse=True)
    rejected_pairs.sort(
        key=lambda p: p.score_breakdown.get('loa_offset', np.inf)
    )

    all_pairs = valid_pairs + rejected_pairs

    if verbose:
        print(f"\n{'='*60}")
        if valid_pairs:
            print(f"Top {min(5, len(valid_pairs))} VALID configurations:")
            print(f"{'='*60}")
            for i, pair in enumerate(valid_pairs[:5]):
                e = pair.tip_edge
                p = pair.push
                b = pair.score_breakdown
                print(f"\n  #{i+1} | Score: {pair.score:.3f}")
                print(f"    Tip edge: d_perp={e.d_perp_to_com:.4f}m, "
                      f"len={e.length:.4f}m")
                print(f"    Push at:  ({p.point[0]:.3f}, {p.point[1]:.3f}, "
                      f"{p.point[2]:.3f}), height={p.height:.3f}m")
                print(f"    Push dir: ({e.inward_normal[0]:.3f}, "
                      f"{e.inward_normal[1]:.3f}, {e.inward_normal[2]:.3f})")
                print(f"    Breakdown: tip_ease={b['tipping_ease']:.3f}, "
                      f"align={b['push_alignment']:.3f}, "
                      f"leverage={b['leverage']:.3f}, "
                      f"edge_stab={b['edge_stability']:.3f}, "
                      f"LoA_offset={b['loa_offset']:.4f}m")
        else:
            print("WARNING: No VALID push configurations found.")
            print(f"  loa_epsilon={loa_epsilon:.3f}m — consider increasing it.")

        if rejected_pairs:
            print(f"\n  {len(rejected_pairs)} candidates REJECTED. "
                  f"Top rejected (by LoA offset):")
            for i, pair in enumerate(rejected_pairs[:3]):
                b = pair.score_breakdown
                reason = b.get('rejected', 'unknown')
                loa = b.get('loa_offset', float('nan'))
                print(f"    [{i}] reason='{reason}', LoA={loa:.4f}m, "
                      f"push_z={pair.push.height:.3f}m")

    return all_pairs


# =============================================================================
# Visualization Helpers
# =============================================================================

def _apply_color(mesh_obj: trimesh.Trimesh,
                 color: Tuple[int, int, int, int]) -> trimesh.Trimesh:
    """Assign a uniform RGBA color to every face of a mesh, replacing any existing visual."""
    rgba = np.array(color, dtype=np.uint8)
    mesh_obj.visual = trimesh.visual.ColorVisuals(
        mesh=mesh_obj,
        face_colors=np.tile(rgba, (len(mesh_obj.faces), 1)),
    )
    return mesh_obj


def _segment_mesh(p0: np.ndarray,
                  p1: np.ndarray,
                  radius: float,
                  color: Tuple[int, int, int, int]) -> trimesh.Trimesh:
    """Create a colored cylinder segment between two 3D points."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length < 1e-10:
        return _point_marker(p0, radius, color)

    z_hat = np.array([0.0, 0.0, 1.0])
    axis = vec / length
    cross = np.cross(z_hat, axis)
    cross_norm = np.linalg.norm(cross)
    dot_val = float(np.dot(z_hat, axis))
    if cross_norm < 1e-8:
        rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]) if dot_val < 0 else np.eye(4)
    else:
        rot = trimesh.transformations.rotation_matrix(
            np.arctan2(cross_norm, dot_val), cross / cross_norm
        )

    mid = (p0 + p1) / 2.0
    T = trimesh.transformations.translation_matrix(mid)

    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
    cyl.apply_transform(T @ rot)
    return _apply_color(cyl, color)


def _point_marker(center: np.ndarray,
                  radius: float,
                  color: Tuple[int, int, int, int]) -> trimesh.Trimesh:
    """Create a colored sphere marker at a given 3D point."""
    marker = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    marker.apply_translation(center)
    return _apply_color(marker, color)


def visualize_ranked_pairs(mesh: trimesh.Trimesh,
                           ranked_pairs: List[ScoredPair],
                           com_2d: np.ndarray,
                           top_n: int = 5,
                           show: bool = True,
                           loa_epsilon: float = 0.03) -> trimesh.Scene:
    """
    Visualize push/tip configurations on top of the object mesh (trimesh viewer).

    ALL candidates are shown — valid ones are color-ranked, rejected ones
    are shown in grey with their rejection reason printed to stdout.

    Color scheme
    ------------
    Valid candidates (ranked best → worst):
      #1  bright green   (60, 190, 80)
      #2  yellow-orange  (255, 180, 60)
      #3  orange         (255, 120, 60)
      #4  red            (255, 80, 80)
      #5  magenta        (200, 60, 140)
    Rejected candidates:
      semi-transparent grey (150, 150, 150, 130)
    CoM marker: bright red dot
    LoA epsilon: transparent grey sphere of radius loa_epsilon around CoM
    Push direction arrow: cyan
    """
    scene = trimesh.Scene()

    # Keep original mesh visible but slightly transparent.
    mesh_vis = mesh.copy()
    mesh_vis.visual.face_colors = np.array([180, 180, 190, 200], dtype=np.uint8)
    scene.add_geometry(mesh_vis, node_name="object_mesh")

    if len(ranked_pairs) == 0:
        if show:
            scene.show()
        return scene

    z_min = mesh.bounds[0, 2]
    z_max = mesh.bounds[1, 2]
    diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    marker_r = max(diag * 0.008, 0.0015)
    line_r   = max(diag * 0.003, 0.0008)
    arrow_len = max(diag * 0.12, 0.03)

    # CoM: place at mid-height so it sits visibly on the object rather than the floor.
    com_z = z_min + (z_max - z_min) * 0.5
    com_pt = np.array([com_2d[0], com_2d[1], com_z])

    # LoA epsilon sphere: semi-transparent cyan shell around CoM.
    # Added before the CoM dot so the dot renders on top.
    eps_sphere = trimesh.creation.icosphere(subdivisions=3, radius=loa_epsilon)
    eps_sphere.apply_translation(com_pt)
    eps_sphere.visual.face_colors = np.array([80, 200, 220, 90], dtype=np.uint8)
    scene.add_geometry(eps_sphere, node_name="loa_epsilon_sphere")

    # CoM dot: pinpoint marker, always smaller than the epsilon sphere.
    com_dot_r = loa_epsilon * 0.08
    # com_dot_r = 0.1
    scene.add_geometry(
        _point_marker(com_pt, com_dot_r, (220, 30, 30, 255)),
        node_name="com",
    )

    # # Bright red CoM dot.
    # scene.add_geometry(
    #     _point_marker(com_pt, marker_r * 1.8, (220, 30, 30, 255)),
    #     node_name="com",
    # )

    # Colors for valid ranked candidates (best → worst).
    valid_colors = [
        (60, 190, 80, 255),
        (255, 180, 60, 255),
        (255, 120, 60, 255),
        (255, 80, 80, 255),
        (200, 60, 140, 255),
    ]
    rejected_color = (150, 150, 150, 130)

    # Split into valid / rejected.
    valid_pairs    = [p for p in ranked_pairs if p.score > -np.inf]
    rejected_pairs = [p for p in ranked_pairs if p.score <= -np.inf]

    # ---- Render valid top-N ----
    # Each tip edge is drawn only once, using the color of its best-ranked pair.
    rendered_edges: set = set()
    for i, pair in enumerate(valid_pairs[:top_n]):
        color = valid_colors[min(i, len(valid_colors) - 1)]
        e = pair.tip_edge

        edge_key = (tuple(np.round(e.v0, 6)), tuple(np.round(e.v1, 6)))
        if edge_key not in rendered_edges:
            scene.add_geometry(
                _segment_mesh(e.v0, e.v1, line_r, color),
                node_name=f"tip_edge_{i}",
            )
            rendered_edges.add(edge_key)

        p = pair.push.point
        scene.add_geometry(
            _point_marker(p, marker_r, color),
            node_name=f"push_point_{i}",
        )
        d = -e.inward_normal / (np.linalg.norm(e.inward_normal) + 1e-9)
        arrow_end = p + d * arrow_len
        scene.add_geometry(
            _segment_mesh(p, arrow_end, line_r * 0.8, (60, 230, 230, 255)),
            node_name=f"push_dir_{i}",
        )

    # ---- Render rejected candidates (grey, smaller markers) ----
    # Deduplicate: one representative per unique push point (cap at 20 to avoid clutter).
    seen_points: set = set()
    rej_shown = 0
    for pair in rejected_pairs:
        if rej_shown >= 20:
            break
        pt_key = tuple(np.round(pair.push.point, 4))
        if pt_key in seen_points:
            continue
        seen_points.add(pt_key)

        e = pair.tip_edge
        p = pair.push.point

        scene.add_geometry(
            _point_marker(p, marker_r * 0.6, rejected_color),
            node_name=f"rej_point_{rej_shown}",
        )
        d = -e.inward_normal / (np.linalg.norm(e.inward_normal) + 1e-9)
        arrow_end = p + d * arrow_len * 0.7
        scene.add_geometry(
            _segment_mesh(p, arrow_end, line_r * 0.5, rejected_color),
            node_name=f"rej_dir_{rej_shown}",
        )
        rej_shown += 1

    # ---- Print rejection reasons to stdout ----
    if rejected_pairs:
        print("\n--- Rejected candidates (unique reasons) ---")
        reason_counts: dict = {}
        for pair in rejected_pairs:
            reason = pair.score_breakdown.get('rejected', 'unknown')
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  x{count:3d}  {reason}")
        print()

    if show:
        scene.show()

    return scene


def load_object_mesh(stl_path: str) -> trimesh.Trimesh:
    """
    Load an object mesh and shift it so the lowest point lies on z=0.
    """
    mesh = trimesh.load_mesh(stl_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a single Trimesh from '{stl_path}', got {type(mesh)}")

    # Sanity-check units: everyday objects should fit within 1 m in every axis.
    extents = mesh.bounding_box.extents  # (dx, dy, dz)
    if np.any(extents > 1.0):
        RED = "\033[1;31m"
        RESET = "\033[0m"
        raise SystemExit(
            f"{RED}\n"
            f"  *** UNIT ERROR: mesh is too large to be in meters ***\n"
            f"  Extents: {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} "
            f"(units assumed meters)\n"
            f"  The file was likely exported in millimeters instead of meters.\n"
            f"  Fix: re-export with meter units, or scale the STL by 0.001.\n"
            f"{RESET}"
        )

    mesh = mesh.copy()
    z_min = mesh.vertices[:, 2].min()
    mesh.vertices[:, 2] -= z_min
    return mesh


if __name__ == "__main__":
    import argparse

    _default_stl = (
        Path(__file__).resolve().parents[1]
        / "assets" / "my_objects" / "box" / "box_m.stl" #"heart" / "heart_exp.stl"
    )

    parser = argparse.ArgumentParser(
        description="Run the push-selection pipeline on an object mesh."
    )
    parser.add_argument(
        "stl", nargs="?", default=str(_default_stl),
        help="Path to STL (or OBJ) file. Defaults to the heart asset.",
    )
    parser.add_argument(
        "--loa-epsilon", type=float, default=0.03,
        help="Line-of-action tolerance in metres (default: 0.03).",
    )
    parser.add_argument(
        "--top-k-edges", type=int, default=6,
        help="Number of tip-edge candidates to evaluate (default: 6).",
    )
    parser.add_argument(
        "--top-n", type=int, default=5,
        help="Number of results to highlight in the viewer (default: 5).",
    )
    args = parser.parse_args()

    mesh = load_object_mesh(args.stl)
    com = mesh.center_mass
    com_2d = np.array([com[0], com[1]])

    print(f"Mesh: {args.stl}")
    print(f"  Vertices: {len(mesh.vertices)}")
    print(f"  Faces:    {len(mesh.faces)}")
    print(f"  Bounds:   {mesh.bounds}")
    print(f"  CoM 2D:   {com_2d}")
    print()

    ranked = select_push_config(
        mesh, com_2d,
        loa_epsilon=args.loa_epsilon,
        top_k_edges=args.top_k_edges,
        verbose=True,
    )

    visualize_ranked_pairs(
        mesh, ranked, com_2d,
        top_n=args.top_n,
        show=True,
        loa_epsilon=args.loa_epsilon,
    )
