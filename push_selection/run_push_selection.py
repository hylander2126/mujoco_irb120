"""
run_push_selection.py
=====================
Run the push/tip selection pipeline on one or more experiment object meshes.

Usage
-----
    python run_push_selection.py                          # all objects
    python run_push_selection.py --objects box heart      # subset
    python run_push_selection.py --no-viz                 # scores only
    python run_push_selection.py --backend matplotlib     # static plots
    python run_push_selection.py --loa-epsilon 0.02       # relax LoA tolerance
    python run_push_selection.py --top-k 3                # fewer tip-edge candidates
"""

import argparse
from pathlib import Path
import numpy as np

from push_selection_pipeline import (
    load_object_mesh,
    select_push_config,
    visualize_ranked_pairs,
)

# ---------------------------------------------------------------------------
# Object registry
# Each entry: name → (stl_path_relative_to_repo_root, com_2d_estimate)
#
# com_2d is the (x, y) CoM projection in the object frame [meters].
# Use (0, 0) as a default for rotationally symmetric objects or when unknown.
# Replace with estimated values from analyze_experiment.py output as needed.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

OBJECTS = {
    "box": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "box"         / "box_exp.stl",
        "com_2d": np.array([0.0, 0.0]),   # symmetric box, CoM at center
        "loa_epsilon": 0.05,
    },
    "heart": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "heart"       / "heart_exp.stl",
        "com_2d": np.array([0.0, 0.0]),   # replace with estimate from analysis
        "loa_epsilon": 0.02,
    },
    "flashlight": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "flashlight"  / "flashlight_exp.stl",
        "com_2d": np.array([0.0, 0.0]),
        "loa_epsilon": 0.02,
    },
    "L": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "L"           / "L_exp.stl",
        "com_2d": np.array([0.0, 0.0]),   # asymmetric — update with real estimate
        "loa_epsilon": 0.02,
    },
    "monitor": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "monitor"     / "monitor_exp.stl",
        "com_2d": np.array([0.0, 0.0]),
        "loa_epsilon": 0.02,
    },
    "soda": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "soda"        / "soda_exp.stl",
        "com_2d": np.array([0.0, 0.0]),   # cylindrical, symmetric
        "loa_epsilon": 0.03,
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_object(name: str,
               config: dict,
               loa_epsilon_override: float | None,
               top_k: int,
               visualize: bool,
               backend: str) -> None:
    """Run the full pipeline for one object and print results."""

    stl_path = config["stl"]
    com_2d   = config["com_2d"]
    loa_eps  = loa_epsilon_override if loa_epsilon_override is not None else config["loa_epsilon"]

    print("\n" + "=" * 65)
    print(f"  Object : {name.upper()}")
    print(f"  Mesh   : {stl_path.relative_to(REPO_ROOT)}")
    print(f"  CoM 2D : {com_2d}")
    print(f"  LoA ε  : {loa_eps} m")
    print("=" * 65)

    if not stl_path.exists():
        print(f"  [SKIP] STL not found: {stl_path}")
        return

    mesh   = load_object_mesh(str(stl_path))
    ranked = select_push_config(
        mesh,
        com_2d,
        loa_epsilon=loa_eps,
        top_k_edges=top_k,
        verbose=True,
    )

    if visualize and ranked:
        visualize_ranked_pairs(
            mesh,
            ranked,
            com_2d,
            top_n=5,
            show=True,
            backend=backend,
        )
    elif visualize and not ranked:
        print("  [VIZ SKIPPED] No valid configurations found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run push/tip selection pipeline on experiment object meshes."
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        metavar="NAME",
        choices=list(OBJECTS.keys()),
        default=list(OBJECTS.keys()),
        help="Object(s) to process. Choices: %(choices)s. Default: all.",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization, print scores only.",
    )
    parser.add_argument(
        "--backend",
        choices=["trimesh", "matplotlib"],
        default="trimesh",
        help="Visualization backend. Default: trimesh (interactive).",
    )
    parser.add_argument(
        "--loa-epsilon",
        type=float,
        default=None,
        metavar="METERS",
        help="Override line-of-action tolerance for all objects (meters).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="K",
        help="Number of top tip-edge candidates to evaluate per object. Default: 5.",
    )
    args = parser.parse_args()

    for name in args.objects:
        run_object(
            name=name,
            config=OBJECTS[name],
            loa_epsilon_override=args.loa_epsilon,
            top_k=args.top_k,
            visualize=not args.no_viz,
            backend=args.backend,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
