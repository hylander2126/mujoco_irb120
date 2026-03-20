"""
run_push_selection.py
=====================
Run push/tip selection on the four experiment meshes and save one PNG each.

Usage
-----
    python run_push_selection.py
    python run_push_selection.py --output-dir figures/push_selection
    python run_push_selection.py --loa-epsilon 0.02
    python run_push_selection.py --top-k 3
"""

import argparse
from pathlib import Path
import numpy as np

try:
    # Works when launched from repo root (package import).
    from push_selection.push_selection_pipeline import (
        load_object_mesh,
        select_push_config,
        visualize_ranked_pairs,
    )
except ModuleNotFoundError:
    # Works when launched directly from the push_selection directory.
    from push_selection_pipeline import (
        load_object_mesh,
        select_push_config,
        visualize_ranked_pairs,
    )

# ---------------------------------------------------------------------------
# Object registry
# Each entry: name -> per-object settings and STL path.
#
# CoM is computed from mesh.center_mass so batch runs match standalone runs.
# The requested experiment meshes are processed in FIXED_OBJECT_ORDER.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

OBJECTS = {
    "box": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "box"         / "box_exp.stl",
        "loa_epsilon": 0.05,
    },
    "heart": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "heart"       / "heart_exp.stl",
        "loa_epsilon": 0.02,
    },
    "flashlight": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "flashlight"  / "flashlight_exp.stl",
        "loa_epsilon": 0.02,
    },
    "monitor": {
        "stl":    REPO_ROOT / "assets" / "my_objects" / "monitor"     / "monitor_exp.stl",
        "loa_epsilon": 0.02,
    },
}

FIXED_OBJECT_ORDER = ["box", "heart", "flashlight", "monitor"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_object(name: str,
               config: dict,
               loa_epsilon_override: float | None,
               top_k: int,
               output_dir: Path) -> None:
    """Run the full pipeline for one object and print results."""

    stl_path = config["stl"]
    loa_eps  = loa_epsilon_override if loa_epsilon_override is not None else config["loa_epsilon"]

    print("\n" + "=" * 65)
    print(f"  Object : {name.upper()}")
    print(f"  Mesh   : {stl_path.relative_to(REPO_ROOT)}")
    print(f"  LoA ε  : {loa_eps} m")
    print("=" * 65)

    if not stl_path.exists():
        print(f"  [SKIP] STL not found: {stl_path}")
        return

    mesh = load_object_mesh(str(stl_path))
    com = mesh.center_mass
    com_2d = np.array([com[0], com[1]], dtype=float)
    print(f"  CoM 2D : {com_2d}")
    ranked = select_push_config(
        mesh,
        com_2d,
        loa_epsilon=loa_eps,
        top_k_edges=top_k,
        verbose=True,
    )

    output_path = output_dir / f"{name}.png"
    visualize_ranked_pairs(
        mesh,
        ranked,
        com_2d,
        top_n=5,
        show=False,
        save_png_path=output_path,
        loa_epsilon=loa_eps,
    )
    if output_path.exists():
        try:
            disp = output_path.relative_to(REPO_ROOT)
        except ValueError:
            disp = output_path
        print(f"  [SAVED] {disp}")
    else:
        print(f"  [WARN] Expected image not found after render: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run push/tip selection for box, heart, flashlight, and monitor "
            "and save [object].png for each."
        )
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        metavar="DIR",
        help="Directory for output PNG files. Default: push_selection/",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Processing objects:", ", ".join(FIXED_OBJECT_ORDER))
    print(f"Output directory: {args.output_dir}")

    for name in FIXED_OBJECT_ORDER:
        run_object(
            name=name,
            config=OBJECTS[name],
            loa_epsilon_override=args.loa_epsilon,
            top_k=args.top_k,
            output_dir=args.output_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
