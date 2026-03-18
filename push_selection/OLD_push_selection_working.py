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

        # Perpendicular distance from CoM to edge line
        # Using 2D signed distance: |(v1-v0) x (v0-com)| / |v1-v0|
        edge_2d = v1_2d - v0_2d
        v0_to_com = com_2d - v0_2d
        d_perp = abs(np.cross(edge_2d, v0_to_com)) / length

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
                         min_height: float = 0.02,
                         normal_alignment_thresh: float = 0.5,
                         n_samples: int = 50) -> List[PushCandidate]:
    """
    For a given tip edge, find candidate push contact points on the
    opposing side of the object.

    The push direction is along the tip edge's inward normal.
    We look for mesh faces on the opposite side whose outward normal
    is roughly anti-parallel to the push direction.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Object mesh.
    tip_edge : TipEdge
        The candidate tipping edge.
    min_height : float
        Minimum push height (m) above support plane.
    normal_alignment_thresh : float
        Minimum |cos(angle)| between push direction and face outward normal.
        0.5 = 60° tolerance. 0.7 = 45° tolerance.
    n_samples : int
        Number of candidate points to evaluate.

    Returns
    -------
    candidates : list of PushCandidate
        Viable push contact points, sorted by height (descending).
    """
    push_dir = tip_edge.inward_normal  # direction robot pushes
    # We want faces whose outward normal opposes push_dir
    # i.e., face_normal · push_dir < -threshold (face faces the robot)

    face_normals = mesh.face_normals
    face_centroids = mesh.triangles_center

    # Filter faces:
    # 1. Normal opposes push direction
    alignment = np.dot(face_normals, push_dir)
    opposing_mask = alignment < -normal_alignment_thresh

    # 2. Above minimum height
    height_mask = face_centroids[:, 2] > min_height

    # 3. On the "push side" of the object (behind the CoM relative to push dir)
    # Face centroid should be on the side the push comes FROM
    # i.e., face centroid projected onto push_dir should be less than CoM
    valid_mask = opposing_mask & height_mask
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
                          score_breakdown={'rejected': f'LoA offset {loa_offset:.4f} > {loa_epsilon}'})

    # Reject concave push surfaces
    if push.curvature_sign < -0.05:
        return ScoredPair(tip_edge, push, score=-np.inf,
                          score_breakdown={'rejected': 'concave push surface'})

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
                       loa_epsilon: float = 0.01,
                       weights: Optional[dict] = None,
                       top_k_edges: int = 5,
                       verbose: bool = True) -> List[ScoredPair]:
    """
    Full pipeline: given mesh and 2D CoM, return ranked push configurations.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Object mesh, oriented with support surface at z ≈ 0.
    com_2d : (2,) array
        Estimated 2D CoM projection (xc, yc).
    loa_epsilon : float
        Line of action tolerance (meters).
    weights : dict, optional
        Scoring weights.
    top_k_edges : int
        Only evaluate push faces for the top-k closest edges.
    verbose : bool
        Print progress info.

    Returns
    -------
    ranked_pairs : list of ScoredPair
        All valid pairs, sorted by score (descending).
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

    all_pairs = []
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
                all_pairs.append(pair)

    # --- Step 3: Rank ---
    all_pairs.sort(key=lambda p: p.score, reverse=True)

    if verbose and all_pairs:
        print(f"\n{'='*60}")
        print(f"Top 5 configurations:")
        print(f"{'='*60}")
        for i, pair in enumerate(all_pairs[:5]):
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

    if verbose and not all_pairs:
        print("\n  WARNING: No valid push configurations found!")
        print("  Try increasing loa_epsilon or checking mesh orientation.")

    return all_pairs


# =============================================================================
# Demo / Test with a simple box
# =============================================================================

def demo_box():
    """Test pipeline with a box (known geometry, known CoM)."""
    # Create a box: 0.2m x 0.15m x 0.3m, base centered at origin
    box = trimesh.creation.box(extents=[0.2, 0.15, 0.3])
    # Shift so base is at z=0 (box is centered at origin by default)
    box.vertices[:, 2] += 0.15  # half-height

    # Known CoM for uniform box: geometric center
    com_2d = np.array([0.0, 0.0])

    print("Box mesh:")
    print(f"  Vertices: {len(box.vertices)}")
    print(f"  Faces: {len(box.faces)}")
    print(f"  Bounds: {box.bounds}")
    print(f"  2D CoM: {com_2d}")
    print()

    results = select_push_config(
        box, com_2d,
        loa_epsilon=0.05,  # 5cm tolerance (triangle centroids offset from ideal)
        verbose=True
    )

    return results


def demo_offset_com():
    """
    Test with a box where CoM is NOT at geometric center.
    Simulates non-uniform mass distribution.
    """
    box = trimesh.creation.box(extents=[0.2, 0.15, 0.3])
    box.vertices[:, 2] += 0.15

    # CoM offset toward +x edge
    com_2d = np.array([0.06, 0.0])  # 6cm offset from center

    print("Box mesh with offset CoM:")
    print(f"  2D CoM: {com_2d}")
    print(f"  (CoM is 6cm from center toward +x edge)")
    print()

    results = select_push_config(
        box, com_2d,
        loa_epsilon=0.05,
        verbose=True
    )

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("DEMO 1: Symmetric Box (CoM at center)")
    print("=" * 60)
    demo_box()

    print("\n\n")
    print("=" * 60)
    print("DEMO 2: Box with Offset CoM")
    print("=" * 60)
    demo_offset_com()
