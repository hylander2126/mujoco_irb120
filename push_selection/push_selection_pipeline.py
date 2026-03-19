"""
Push Face / Tip Edge Selection Pipeline
========================================
Given:
  - Object mesh (STL or OBJ)
  - 2D CoM projection (xc, yc) in object frame
  - Support plane at z = 0

Pipeline
--------
1. Extract tip edges from the support polygon (bottom face convex hull).
2. Extract push *faces* from the top band of the mesh (top 10% by default).
3. Pair tip edges with push faces whose horizontal normals are PARALLEL
   (tip.inward_normal ≈ push.outward_normal), meaning the push face is
   on the opposite side of the object from the tip edge.
4. LoA check: the line through the push face centroid along the push direction,
   projected to z=0, must pass within loa_epsilon of the CoM.
5. Score and rank valid pairs.

Dependencies: trimesh, numpy, scipy
"""

import trimesh
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


MAX_DISPLAY_SELECTIONS = 3


class _TeeStream:
    """Mirror writes to multiple streams (console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class TipEdge:
    """A candidate tipping edge on the support polygon."""
    v0: np.ndarray           # vertex 0 (3D, on z=0 plane)
    v1: np.ndarray           # vertex 1 (3D, on z=0 plane)
    edge_vec: np.ndarray     # unit vector along edge
    inward_normal: np.ndarray  # unit horizontal normal pointing INTO the object
    length: float
    d_perp_to_com: float     # perpendicular distance from edge to 2D CoM
    midpoint: np.ndarray


@dataclass
class PushFace:
    """A candidate push face on the top band of the object surface."""
    face_index: int          # index into mesh.faces
    vertices: np.ndarray     # (3, 3) triangle vertices
    centroid: np.ndarray     # centroid of the face (3D)
    outward_normal: np.ndarray  # unit horizontal outward normal of the face
    height: float            # average z of the face (used for leverage scoring)
    area: float              # area of the face


@dataclass
class ScoredPair:
    """A scored (tip_edge, push_face) pair."""
    tip_edge: TipEdge
    push_face: PushFace
    push_dir: np.ndarray     # actual push direction used (= -push_face.outward_normal)
    score: float
    score_breakdown: dict = field(default_factory=dict)


# =============================================================================
# Step 1: Extract Support Polygon and Tip Edges
# =============================================================================

def extract_support_polygon(mesh: trimesh.Trimesh, z_tol: float = 1e-3) -> np.ndarray:
    """
    Extract the convex hull of vertices on the support plane (z ~ 0).

    Returns
    -------
    hull_vertices : (N, 2) array
        Ordered 2D vertices of the convex support polygon.
    """
    base_mask = np.abs(mesh.vertices[:, 2]) < z_tol
    if base_mask.sum() < 3:
        z_min = mesh.vertices[:, 2].min()
        base_mask = mesh.vertices[:, 2] < (z_min + z_tol * 10)

    base_verts_2d = mesh.vertices[base_mask][:, :2]

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
    """
    n = len(hull_vertices_2d)
    edges = []

    for i in range(n):
        v0_2d = hull_vertices_2d[i]
        v1_2d = hull_vertices_2d[(i + 1) % n]

        v0 = np.array([v0_2d[0], v0_2d[1], 0.0])
        v1 = np.array([v1_2d[0], v1_2d[1], 0.0])

        edge_vec_raw = v1 - v0
        length = np.linalg.norm(edge_vec_raw)
        if length < 1e-8:
            continue
        edge_vec = edge_vec_raw / length

        # Candidate outward normal (rotate edge_vec 90 deg in XY)
        normal_candidate = np.array([edge_vec[1], -edge_vec[0], 0.0])
        midpoint = (v0 + v1) / 2.0
        to_com = np.array([com_2d[0], com_2d[1], 0.0]) - midpoint

        # Inward normal points toward CoM
        if np.dot(normal_candidate[:2], to_com[:2]) < 0:
            inward_normal = -normal_candidate
        else:
            inward_normal = normal_candidate

        # Perpendicular distance from CoM to edge line
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
# Step 2: Extract Push Faces from Top Band
# =============================================================================

def extract_push_faces(mesh: trimesh.Trimesh,
                       top_band_frac: float = 0.10,
                       min_horiz_component: float = 0.3,
                       normal_cluster_thresh: float = 0.98) -> List[PushFace]:
    """
    Extract candidate push faces from the top band of the mesh.

    A push face is a mesh triangle where:
      - Its centroid is in the top band (top `top_band_frac` of object height).
      - The face has a meaningful horizontal outward normal (i.e. it is a
        near-vertical face, not the top cap).

    Faces are clustered by their horizontal outward normal direction so that
    each distinct side of the object produces exactly ONE representative
    PushFace whose centroid is the area-weighted average of all constituent
    triangles on that side.  This places the push contact point at the
    geometric centre of each face rather than at an arbitrary triangle centroid.

    Parameters
    ----------
    mesh : trimesh.Trimesh
    top_band_frac : float
        Fraction of total height defining the top band (default 0.10 = top 10%).
    min_horiz_component : float
        Minimum magnitude of the horizontal component of the face normal for
        a face to be considered pushable (filters out top/bottom cap faces).
    normal_cluster_thresh : float
        Minimum dot product between two face normals to be merged into the same
        cluster (default 0.98 ≈ 11°).

    Returns
    -------
    push_faces : list of PushFace
        One entry per distinct side direction, centroid at the area-weighted
        centre of that side's top-band triangles.
    """
    z_min = mesh.bounds[0, 2]
    z_max = mesh.bounds[1, 2]
    z_band_floor = z_max - top_band_frac * (z_max - z_min)

    verts = mesh.vertices
    faces = mesh.faces
    face_normals = mesh.face_normals

    # Precompute centroids for all faces
    centroids = verts[faces].mean(axis=1)  # (F, 3)

    # --- collect raw top-band side triangles ---
    raw: List[dict] = []
    for fi in range(len(faces)):
        centroid = centroids[fi]
        if centroid[2] < z_band_floor:
            continue

        fn = face_normals[fi]
        horiz = np.array([fn[0], fn[1], 0.0])
        horiz_mag = np.linalg.norm(horiz)
        if horiz_mag < min_horiz_component:
            continue

        outward_normal = horiz / horiz_mag
        face_verts = verts[faces[fi]]
        e1 = face_verts[1] - face_verts[0]
        e2 = face_verts[2] - face_verts[0]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2))

        raw.append({
            'face_index': fi,
            'centroid': centroid.copy(),
            'outward_normal': outward_normal,
            'area': area,
            'vertices': face_verts.copy(),
        })

    if not raw:
        return []

    # --- cluster by outward normal direction ---
    # Greedy: assign each triangle to the first cluster whose representative
    # normal has dot >= normal_cluster_thresh with this triangle's normal.
    clusters: List[dict] = []  # list of {normal, faces: [dict]}

    for item in raw:
        n = item['outward_normal']
        assigned = False
        for cluster in clusters:
            if np.dot(cluster['normal'], n) >= normal_cluster_thresh:
                cluster['faces'].append(item)
                assigned = True
                break
        if not assigned:
            clusters.append({'normal': n.copy(), 'faces': [item]})

    # --- build one PushFace per cluster (area-weighted centroid) ---
    push_faces: List[PushFace] = []
    for cluster in clusters:
        cluster_faces = cluster['faces']
        total_area = sum(f['area'] for f in cluster_faces)
        if total_area < 1e-12:
            continue

        # Area-weighted centroid gives the geometric centre of the side
        weighted_centroid = sum(
            f['centroid'] * f['area'] for f in cluster_faces
        ) / total_area

        # Representative normal: area-weighted average (re-normalised)
        avg_normal = sum(
            f['outward_normal'] * f['area'] for f in cluster_faces
        ) / total_area
        avg_normal_mag = np.linalg.norm(avg_normal)
        if avg_normal_mag < 1e-8:
            continue
        avg_normal /= avg_normal_mag

        # Representative face index: the triangle whose centroid is closest
        # to the cluster centroid (used only for face highlighting in viz)
        best_face = min(
            cluster_faces,
            key=lambda f: np.linalg.norm(f['centroid'] - weighted_centroid),
        )
        best_fi = best_face['face_index']

        # Surface contact point: project the weighted centroid onto the mesh
        # surface along the outward normal so the arrow starts ON the surface,
        # not inside it (the avg of curved-face centroids pulls inward).
        ray_origin = weighted_centroid + avg_normal * 0.5  # start outside
        ray_dir    = -avg_normal                           # shoot inward
        locs, _, _ = mesh.ray.intersects_location(
            ray_origins=ray_origin[np.newaxis],
            ray_directions=ray_dir[np.newaxis],
        )
        if len(locs) > 0:
            # Pick the intersection closest to weighted_centroid
            dists = np.linalg.norm(locs - weighted_centroid, axis=1)
            surface_pt = locs[np.argmin(dists)]
        else:
            # Fallback: use the centroid of the nearest triangle
            surface_pt = best_face['centroid']

        push_faces.append(PushFace(
            face_index=best_fi,
            vertices=best_face['vertices'],
            centroid=surface_pt,
            outward_normal=avg_normal,
            height=surface_pt[2],
            area=total_area,
        ))

    return push_faces


# =============================================================================
# Step 3: Pair by Normal Alignment
# =============================================================================

def pair_faces_by_normal(tip_edges: List[TipEdge],
                         push_faces: List[PushFace],
                         alignment_thresh: float = 0.90) -> List[Tuple[TipEdge, PushFace, float]]:
    """
    Pair tip edges with push faces satisfying two hard geometric constraints,
    then annotate each pair with its orthogonality score for ranking.

    Hard constraints (both must hold in the XY / overhead view):
      1. The push face's outward normal LINE must intersect the tip edge SEGMENT.
         i.e. shooting a ray from push_centroid along -outward_normal hits [v0,v1].
      2. The push face must lie on the far side of the object from the tip edge
         (push centroid projection along inward_normal > tip midpoint projection).

    Ranking metric (higher = better):
      orthogonality = 1 - |dot(push.outward_normal_2d, tip.edge_vec_2d)|
      A value of 1.0 means the push direction is perfectly perpendicular to the
      tip edge (maximum tipping moment); 0.0 means it's parallel (no moment).

    Parameters
    ----------
    alignment_thresh : float
        Kept for API compatibility; not used (constraint 1 is strictly geometric).

    Returns
    -------
    pairs : list of (TipEdge, PushFace, orthogonality)
        Sorted by orthogonality descending (best first).
    """
    pairs = []
    for tip in tip_edges:
        tip_proj = np.dot(tip.midpoint[:2], tip.inward_normal[:2])
        A = tip.v0[:2]
        B = tip.v1[:2]
        AB = B - A

        for push in push_faces:
            # --- Constraint 2: push face on far side ---
            push_proj = np.dot(push.centroid[:2], tip.inward_normal[:2])
            if push_proj <= tip_proj:
                continue

            # --- Constraint 1: -outward_normal ray hits the tip edge segment ---
            # Ray: P + t * d,  d = -outward_normal_2d,  t >= 0
            # Edge: A + s * AB,  s in [0, 1]
            # Solve: P + t*d = A + s*AB
            #   => t*d - s*AB = A - P
            P = push.centroid[:2]
            d = -push.outward_normal[:2]   # push direction (inward)
            AP = A - P                     # A - P

            # 2x2 system: [d | -AB] * [t, s]^T = AP
            # denom = det([d, -AB]) = d[0]*(-AB[1]) - d[1]*(-AB[0])
            #                       = -d[0]*AB[1] + d[1]*AB[0]
            denom = d[0] * (-AB[1]) - d[1] * (-AB[0])
            if abs(denom) < 1e-10:
                # Ray is parallel to the tip edge — skip
                continue

            t = (AP[0] * (-AB[1]) - AP[1] * (-AB[0])) / denom
            s = (AP[0] * d[1]     - AP[1] * d[0])     / denom  # noqa: E221

            if t < -1e-6:
                # Intersection is behind the push face (wrong direction)
                continue
            if not (-1e-4 <= s <= 1.0 + 1e-4):
                # Ray misses the tip edge segment
                continue

            # --- Orthogonality: how perpendicular is push_dir to tip edge? ---
            orth = 1.0 - abs(np.dot(push.outward_normal[:2], tip.edge_vec[:2]))

            pairs.append((tip, push, orth))

    # Sort best-first so callers can take top-N directly
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


# =============================================================================
# Step 4: Line of Action Check
# =============================================================================

def line_of_action_offset(push_centroid_2d: np.ndarray,
                          tip_v0_2d: np.ndarray,
                          tip_v1_2d: np.ndarray,
                          com_2d: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Find the minimum perpendicular distance from the CoM to any line
    drawn from the push centroid to a point on the tip edge.

    The robot pushes from push_centroid toward some contact point T on
    the tip edge [v0, v1].  The Line of Action (LoA) is the line through
    push_centroid and T.  We sweep T along the tip edge and return the
    minimum distance from the CoM to any such LoA.

    Parameters
    ----------
    push_centroid_2d : (2,) push face centroid projected to XY
    tip_v0_2d, tip_v1_2d : (2,) tip edge endpoints in XY
    com_2d : (2,) center of mass in XY

    Returns
    -------
    min_offset : float
        Minimum perpendicular distance from CoM to any LoA line.
    best_push_dir : (3,) ndarray
        Unit push direction (3D, z=0) toward the optimal tip-edge point.
    """
    P = push_centroid_2d
    A = tip_v0_2d
    B = tip_v1_2d
    C = com_2d

    # Parameterise tip edge: T(t) = A + t*(B - A),  t in [0, 1].
    # LoA line from P through T(t).
    # Signed distance from C to line P-T:
    #   d(t) = cross(T-P, C-P) / |T-P|
    # We want to minimise |d(t)| over t in [0, 1].
    #
    # The numerator cross(T-P, C-P) is linear in t, so the zero crossing
    # (if it exists in [0,1]) gives the exact minimum.  Otherwise the
    # minimum is at one of the endpoints.

    AB = B - A
    CP = C - P
    AP = A - P

    # cross(T-P, C-P) = cross(AP + t*AB, CP) = cross(AP,CP) + t*cross(AB,CP)
    cross_ap_cp = AP[0] * CP[1] - AP[1] * CP[0]
    cross_ab_cp = AB[0] * CP[1] - AB[1] * CP[0]

    candidates = [0.0, 1.0]

    # Zero of numerator: t* = -cross_ap_cp / cross_ab_cp
    if abs(cross_ab_cp) > 1e-12:
        t_zero = -cross_ap_cp / cross_ab_cp
        if 0.0 <= t_zero <= 1.0:
            candidates.append(t_zero)

    best_offset = np.inf
    best_t = 0.0
    for t in candidates:
        T = A + t * AB
        TP = T - P
        tp_len = np.linalg.norm(TP)
        if tp_len < 1e-10:
            continue
        cross_val = TP[0] * CP[1] - TP[1] * CP[0]
        offset = abs(cross_val) / tp_len
        if offset < best_offset:
            best_offset = offset
            best_t = t

    # Compute the actual push direction (3D, z=0)
    T_best = A + best_t * AB
    push_dir_2d = T_best - P
    d_norm = np.linalg.norm(push_dir_2d)
    if d_norm < 1e-10:
        best_push_dir = np.array([0.0, 0.0, 0.0])
    else:
        best_push_dir = np.array([push_dir_2d[0] / d_norm,
                                   push_dir_2d[1] / d_norm,
                                   0.0])

    return best_offset, best_push_dir


# =============================================================================
# Step 5: Scoring Function
# =============================================================================

def score_pair(tip_edge: TipEdge,
               push_face: PushFace,
               com_2d: np.ndarray,
               object_height: float,
               max_tip_edge_length: float,
               loa_epsilon: float = 0.03,
               enforce_loa: bool = False,
               weights: Optional[dict] = None,
               orthogonality: float = 0.0) -> ScoredPair:
    """
    Score a (tip_edge, push_face) pair.

    All candidates reaching this function have already passed the hard geometric
    filter in pair_faces_by_normal (face normal line intersects tip edge segment).

    Primary ranking key  — orthogonality (passed in from the pairing step):
      1 - |dot(push.outward_normal_2d, tip.edge_vec_2d)|
      = 1.0  =>  push is perfectly perpendicular to tip edge (best tipping moment)
      = 0.0  =>  push is parallel to tip edge (no moment)

    Secondary soft scores (all in [0, 1]):
      - tipping_ease:   CoM proximity to tip edge
      - leverage:       push height
      - edge_stability: tip edge length
      - loa_closeness:  (enforce_loa only) LoA proximity to CoM

    Parameters
    ----------
    enforce_loa : bool
        If True, pairs whose LoA offset exceeds loa_epsilon are hard-rejected
        and loa_closeness contributes to the score.
        If False (default), LoA constraint is skipped; push_dir = -outward_normal.
    orthogonality : float
        Pre-computed value from pair_faces_by_normal (in [0, 1]).
    """
    if weights is None:
        weights = {
            'orthogonality':  5.0,   # primary: normal ⊥ tip edge
            'tipping_ease':   4.0,
            'leverage':       1.5,
            'edge_stability': 1.0,
            'loa_closeness':  3.0,
        }

    if enforce_loa:
        loa_offset, push_dir = line_of_action_offset(
            push_face.centroid[:2],
            tip_edge.v0[:2],
            tip_edge.v1[:2],
            com_2d,
        )

        if loa_offset > loa_epsilon:
            return ScoredPair(tip_edge, push_face, push_dir=push_dir, score=-np.inf,
                              score_breakdown={
                                  'rejected': (f'LoA offset {loa_offset:.4f}m > '
                                               f'tol {loa_epsilon:.4f}m'),
                                  'loa_offset': loa_offset,
                              })

        s_loa = 1.0 - (loa_offset / max(loa_epsilon, 1e-8))
    else:
        push_dir = -push_face.outward_normal
        loa_offset = None
        s_loa = None

    s_tip      = 1.0 / (1.0 + tip_edge.d_perp_to_com * 20)
    s_leverage = push_face.height / max(object_height, 1e-3)
    s_edge     = tip_edge.length / max(max_tip_edge_length, 1e-3)

    total = (weights['orthogonality']  * orthogonality
             + weights['tipping_ease']  * s_tip
             + weights['leverage']      * s_leverage
             + weights['edge_stability']* s_edge
             + (weights['loa_closeness'] * s_loa if s_loa is not None else 0.0))

    breakdown = {
        'orthogonality':  orthogonality,
        'tipping_ease':   s_tip,
        'leverage':       s_leverage,
        'edge_stability': s_edge,
        'loa_offset':     loa_offset,
    }
    if s_loa is not None:
        breakdown['loa_closeness'] = s_loa

    return ScoredPair(tip_edge, push_face, push_dir=push_dir, score=total,
                      score_breakdown=breakdown)


# =============================================================================
# Main Pipeline
# =============================================================================

def select_push_config(mesh: trimesh.Trimesh,
                       com_2d: np.ndarray,
                       loa_epsilon: float = 0.03,
                       enforce_loa: bool = False,
                       weights: Optional[dict] = None,
                       top_k_edges: int = 5,
                       top_band_frac: float = 0.10,
                       normal_alignment_thresh: float = 0.90,
                       verbose: bool = True) -> List[ScoredPair]:
    """
    Full pipeline: given mesh and 2D CoM, return ranked push configurations.

    Steps
    -----
    1. Extract tip edges from the support polygon.
    2. Extract push faces from the top band of the mesh.
    3. Pair by PARALLEL normal alignment.
    4. LoA check + scoring.

    Returns
    -------
    ranked_pairs : list of ScoredPair
        Valid pairs sorted by score descending, then rejected pairs sorted
        by loa_offset ascending.
    """
    com_2d = np.asarray(com_2d, dtype=float)

    # --- Step 1: Tip edges ---
    hull_verts = extract_support_polygon(mesh)
    tip_edges = compute_tip_edges(hull_verts, com_2d)
    tip_edges.sort(key=lambda e: e.d_perp_to_com)

    if verbose:
        print(f"Support polygon: {len(hull_verts)} vertices, {len(tip_edges)} tip edges")
        print(f"\nTip edge ranking (by CoM proximity):")
        for i, e in enumerate(tip_edges):
            print(f"  {i}: d_perp={e.d_perp_to_com:.4f}m, length={e.length:.4f}m, "
                  f"midpoint=({e.midpoint[0]:.3f}, {e.midpoint[1]:.3f}), "
                  f"inward_normal=({e.inward_normal[0]:.2f}, {e.inward_normal[1]:.2f})")

    candidate_tip_edges = tip_edges[:top_k_edges]

    # --- Step 2: Push faces ---
    push_faces = extract_push_faces(mesh, top_band_frac=top_band_frac)

    if verbose:
        print(f"\nPush faces in top {top_band_frac*100:.0f}% band: {len(push_faces)}")

    # --- Step 3: Pair by normal alignment ---
    pairs = pair_faces_by_normal(candidate_tip_edges, push_faces,
                                 alignment_thresh=normal_alignment_thresh)

    if verbose:
        print(f"Face pairs after normal alignment: {len(pairs)}")

    # --- Step 4 & 5: LoA check + scoring ---
    object_height = mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min()
    max_tip_len = max(e.length for e in tip_edges)

    valid_pairs = []
    rejected_pairs = []

    for tip, push, orth in pairs:
        scored = score_pair(tip, push, com_2d, object_height, max_tip_len,
                            loa_epsilon, enforce_loa, weights, orthogonality=orth)
        if scored.score > -np.inf:
            valid_pairs.append(scored)
        else:
            rejected_pairs.append(scored)

    # De-duplicate: keep only the best-scored pair per unique
    # (tip_edge_midpoint, push_outward_normal) combination so that
    # multiple mesh triangles on the same side don't inflate the top-N list.
    def _pair_key(p: ScoredPair) -> Tuple:
        tip_mid = tuple(np.round(p.tip_edge.midpoint[:2], 4))
        push_n  = tuple(np.round(p.push_face.outward_normal[:2], 2))
        return (tip_mid, push_n)

    seen_keys: dict = {}
    for p in valid_pairs:
        k = _pair_key(p)
        if k not in seen_keys or p.score > seen_keys[k].score:
            seen_keys[k] = p
    valid_pairs = sorted(seen_keys.values(), key=lambda p: p.score, reverse=True)

    rejected_pairs.sort(key=lambda p: p.score_breakdown.get('loa_offset', np.inf))

    all_pairs = valid_pairs + rejected_pairs

    display_limit = MAX_DISPLAY_SELECTIONS

    if verbose:
        print(f"\n{'='*60}")
        if valid_pairs:
            print(f"Top {min(display_limit, len(valid_pairs))} VALID configurations:")
            print(f"{'='*60}")
            for i, pair in enumerate(valid_pairs[:display_limit]):
                e = pair.tip_edge
                pf = pair.push_face
                b = pair.score_breakdown
                print(f"\n  #{i+1} | Score: {pair.score:.3f}")
                print(f"    Tip edge:  d_perp={e.d_perp_to_com:.4f}m, len={e.length:.4f}m, "
                      f"midpoint=({e.midpoint[0]:.3f}, {e.midpoint[1]:.3f})")
                print(f"    Push face: centroid=({pf.centroid[0]:.3f}, {pf.centroid[1]:.3f}, "
                      f"{pf.centroid[2]:.3f}), height={pf.height:.3f}m, area={pf.area:.6f}m2")
                print(f"    Push dir:  ({pair.push_dir[0]:.3f}, {pair.push_dir[1]:.3f})")
                loa_str = (f", loa_close={b['loa_closeness']:.3f}"
                           f", LoA_offset={b['loa_offset']:.4f}m"
                           if 'loa_closeness' in b else
                           ", LoA_offset=n/a (relaxed)")
                print(f"    Breakdown: orth={b['orthogonality']:.3f}, "
                      f"tip_ease={b['tipping_ease']:.3f}, "
                      f"leverage={b['leverage']:.3f}, "
                      f"edge_stab={b['edge_stability']:.3f}"
                      f"{loa_str}")
        else:
            print("WARNING: No VALID push configurations found.")
            print(f"  loa_epsilon={loa_epsilon:.3f}m -- consider increasing it.")
            print(f"  normal_alignment_thresh={normal_alignment_thresh:.2f} "
                  f"-- consider decreasing it.")

        if rejected_pairs:
            print(f"\n  {len(rejected_pairs)} candidates REJECTED. "
                  f"Top rejected (by LoA offset):")
            for i, pair in enumerate(rejected_pairs[:3]):
                b = pair.score_breakdown
                print(f"    [{i}] {b.get('rejected','?')}, "
                      f"push_z={pair.push_face.height:.3f}m")

    return all_pairs


# =============================================================================
# Visualization Helpers
# =============================================================================

def _apply_color(mesh_obj: trimesh.Trimesh,
                 color: Tuple[int, int, int, int]) -> trimesh.Trimesh:
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
    """Create a colored cylinder between two 3D points."""
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
    marker = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    marker.apply_translation(center)
    return _apply_color(marker, color)


def _face_highlight(mesh: trimesh.Trimesh,
                    face_index: int,
                    color: Tuple[int, int, int, int],
                    offset: float = 0.0005) -> trimesh.Trimesh:
    """
    Create a thin highlighted triangle slightly offset from the mesh surface
    to visually mark a push face.
    """
    verts = mesh.vertices[mesh.faces[face_index]].copy()
    fn = mesh.face_normals[face_index]
    # Offset slightly along the face normal to avoid z-fighting
    verts += fn * offset
    face_mesh = trimesh.Trimesh(
        vertices=verts,
        faces=np.array([[0, 1, 2]]),
    )
    return _apply_color(face_mesh, color)


def visualize_ranked_pairs(mesh: trimesh.Trimesh,
                           ranked_pairs: List[ScoredPair],
                           com_2d: np.ndarray,
                           top_n: int = MAX_DISPLAY_SELECTIONS,
                           show: bool = True,
                           save_png_path: Optional[Path] = None,
                           loa_epsilon: float = 0.03) -> trimesh.Scene:
    """
    Visualize push/tip configurations on top of the object mesh.

    Color scheme (valid pairs, best -> worst):
      #1  bright green   (60, 190, 80)
      #2  yellow-orange  (255, 180, 60)
      #3  orange         (255, 120, 60)
      #4  red            (255, 80, 80)
      #5  magenta        (200, 60, 140)
    Rejected: semi-transparent grey push faces.
    CoM: small red dot inside a cyan epsilon sphere.
    Push direction arrow + tip edge + push face all share the rank color.
    Line thickness is scaled by rank: best pair is thickest, worst is thinnest.
    """
    scene = trimesh.Scene()

    mesh_vis = mesh.copy()
    mesh_vis.visual.face_colors = np.array([180, 180, 190, 200], dtype=np.uint8)
    scene.add_geometry(mesh_vis, node_name="object_mesh")

    if len(ranked_pairs) == 0:
        if save_png_path is not None:
            save_png_path = Path(save_png_path)
            try:
                png_bytes = scene.save_image(resolution=(1600, 1000), visible=True)
                if png_bytes is None:
                    raise RuntimeError("scene.save_image returned no image bytes")
                save_png_path.write_bytes(png_bytes)
                print(f"Saved visualization PNG: {save_png_path}")
            except Exception as exc:
                print(f"WARNING: Failed to save visualization PNG to {save_png_path}: {exc}")
        if show:
            scene.show()
        return scene

    # Never display more than MAX_DISPLAY_SELECTIONS ranked candidates.
    top_n = min(top_n, MAX_DISPLAY_SELECTIONS)

    z_min = mesh.bounds[0, 2]
    z_max = mesh.bounds[1, 2]
    diag = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    base_line_r = max(diag * 0.003, 0.0008)
    arrow_len   = max(diag * 0.12, 0.03)

    # CoM at mid-height
    com_z = z_min + (z_max - z_min) * 0.5
    com_pt = np.array([com_2d[0], com_2d[1], com_z])

    # LoA epsilon sphere (cyan, transparent)
    eps_sphere = trimesh.creation.icosphere(subdivisions=3, radius=loa_epsilon)
    eps_sphere.apply_translation(com_pt)
    eps_sphere.visual.face_colors = np.array([80, 200, 220, 90], dtype=np.uint8)
    scene.add_geometry(eps_sphere, node_name="loa_epsilon_sphere")

    # CoM dot
    com_dot_r = loa_epsilon * 0.08
    scene.add_geometry(
        _point_marker(com_pt, com_dot_r, (220, 30, 30, 255)),
        node_name="com",
    )

    # Rank colors
    rank_colors = [
        (60, 190, 80, 255),    # #1 bright green
        (255, 180, 60, 255),   # #2 yellow-orange
        (255, 120, 60, 255),   # #3 orange
        (255, 80, 80, 255),    # #4 red
        (200, 60, 140, 255),   # #5 magenta
    ]
    rank_colors_face = [
        (60, 190, 80, 180),
        (255, 180, 60, 180),
        (255, 120, 60, 180),
        (255, 80, 80, 180),
        (200, 60, 140, 180),
    ]
    rejected_color = (150, 150, 150, 100)

    valid_pairs    = [p for p in ranked_pairs if p.score > -np.inf]
    rejected_pairs = [p for p in ranked_pairs if p.score <= -np.inf]

    # ---- Valid top-N ----
    # Line thickness: rank 0 (best) gets 3x base, rank top_n-1 gets 1x base.
    rendered_tip_edges: set = set()
    rendered_push_faces: set = set()

    n_show = min(top_n, len(valid_pairs))
    for i, pair in enumerate(valid_pairs[:top_n]):
        color = rank_colors[min(i, len(rank_colors) - 1)]
        face_color = rank_colors_face[min(i, len(rank_colors_face) - 1)]
        e = pair.tip_edge
        pf = pair.push_face

        # Line radius: best=3x, worst=1x, linearly interpolated
        if n_show > 1:
            thickness_scale = 3.0 - 2.0 * (i / (n_show - 1))
        else:
            thickness_scale = 3.0
        line_r = base_line_r * thickness_scale

        # Tip edge -- draw once per unique edge, color & thickness of best rank
        tip_key = (tuple(np.round(e.v0, 6)), tuple(np.round(e.v1, 6)))
        if tip_key not in rendered_tip_edges:
            scene.add_geometry(
                _segment_mesh(e.v0, e.v1, line_r, color),
                node_name=f"tip_edge_{i}",
            )
            rendered_tip_edges.add(tip_key)

        # Push face -- highlight the triangle on the mesh surface
        face_key = pf.face_index
        if face_key not in rendered_push_faces:
            scene.add_geometry(
                _face_highlight(mesh, pf.face_index, face_color),
                node_name=f"push_face_{i}",
            )
            rendered_push_faces.add(face_key)

        # Arrow inward: push face centroid → tip edge (push direction)
        push_dir = pair.push_dir
        # arrow_end = pf.centroid + push_dir * arrow_len
        # scene.add_geometry(
        #     _segment_mesh(pf.centroid, arrow_end, line_r * 0.7, color),
        #     node_name=f"push_dir_{i}",
        # )

        # Arrow outward: centroid → outside the object (approach direction)
        outward_end = pf.centroid + pf.outward_normal * arrow_len
        scene.add_geometry(
            _segment_mesh(pf.centroid, outward_end, line_r * 0.7, color),
            node_name=f"approach_dir_{i}",
        )

    # ---- Rejected (grey, deduplicated by face index) ----
    seen_faces: set = set()
    rej_shown = 0
    for pair in rejected_pairs:
        if rej_shown >= 20:
            break
        fk = pair.push_face.face_index
        if fk in seen_faces:
            continue
        seen_faces.add(fk)

        scene.add_geometry(
            _face_highlight(mesh, fk, rejected_color),
            node_name=f"rej_push_{rej_shown}",
        )
        rej_shown += 1

    if rejected_pairs:
        print("\n--- Rejected candidates (unique reasons) ---")
        reason_counts: dict = {}
        for pair in rejected_pairs:
            reason = pair.score_breakdown.get('rejected', 'unknown')
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"  x{count:3d}  {reason}")
        print()

    if save_png_path is not None:
        save_png_path = Path(save_png_path)
        try:
            png_bytes = scene.save_image(resolution=(1600, 1000), visible=True)
            if png_bytes is None:
                raise RuntimeError("scene.save_image returned no image bytes")
            save_png_path.write_bytes(png_bytes)
            print(f"Saved visualization PNG: {save_png_path}")
        except Exception as exc:
            print(f"WARNING: Failed to save visualization PNG to {save_png_path}: {exc}")

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
    extents = mesh.bounding_box.extents
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

    output_dir = Path(__file__).resolve().parent
    log_path = output_dir / "debug.log"
    png_path = output_dir / "most_recent_selection.png"

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_file = log_path.open("a", encoding="utf-8")
    sys.stdout = _TeeStream(original_stdout, log_file)
    sys.stderr = _TeeStream(original_stderr, log_file)

    try:
        print(f"Streaming output to: {log_path.resolve()}")
        print(f"Will save visualization to: {png_path.resolve()}")

        _default_stl = (
            Path(__file__).resolve().parents[1]
            / "assets" / "my_objects" / "box" / "box_m.stl"
        )

        parser = argparse.ArgumentParser(
            description="Run the push-selection pipeline on an object mesh."
        )
        parser.add_argument(
            "stl", nargs="?", default=str(_default_stl),
            help="Path to STL (or OBJ) file.",
        )
        parser.add_argument(
            "--loa-epsilon", "--loa_epsilon", type=float, default=0.03,
            help="Line-of-action tolerance in metres (default: 0.03).",
        )
        parser.add_argument(
            "--top-k-edges", "--top_k_edges", type=int, default=6,
            help="Number of tip-edge candidates to evaluate (default: 6).",
        )
        parser.add_argument(
            "--top-n", "--top_n", type=int, default=MAX_DISPLAY_SELECTIONS,
            help=(f"Number of results to highlight in the viewer "
                  f"(default/max: {MAX_DISPLAY_SELECTIONS})."),
        )
        parser.add_argument(
            "--top-band-frac", "--top_band_frac", type=float, default=0.10,
            help="Fraction of object height defining the push-face top band (default: 0.10).",
        )
        parser.add_argument(
            "--normal-thresh", "--normal_thresh", type=float, default=0.90,
            help="Min |cos| for parallel normal pairing (default: 0.90).",
        )
        parser.add_argument(
            "--no-enforce-loa", action="store_true",
            help="Relax the LoA constraint: score by normals only, push straight inward.",
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
            enforce_loa=not args.no_enforce_loa,
            top_k_edges=args.top_k_edges,
            top_band_frac=args.top_band_frac,
            normal_alignment_thresh=args.normal_thresh,
            verbose=True,
        )

        if args.top_n > MAX_DISPLAY_SELECTIONS:
            print(f"Requested top_n={args.top_n}; capping display to {MAX_DISPLAY_SELECTIONS}.")

        visualize_ranked_pairs(
            mesh, ranked, com_2d,
            top_n=args.top_n,
            show=True,
            save_png_path=png_path,
            loa_epsilon=args.loa_epsilon,
        )
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()