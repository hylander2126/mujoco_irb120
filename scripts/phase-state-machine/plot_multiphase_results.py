"""
plot_multiphase_results.py
--------------------------
Plot object rotation, measured wrench, and labeled phase transitions over time
from PhaseController simulation output.

Usage:
    cd scripts/
    python plot_multiphase_results.py
    python plot_multiphase_results.py --input simulation_data_multiphase.npz
    python plot_multiphase_results.py --save figures/multiphase_overview.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R


# Make package importable when run as a script from scripts/
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mujoco_irb120.common.phase_controller import PHASE_NAMES, Phase
from mujoco_irb120.common.plotting_helper import plot_wrench_and_tipping


DEFAULT_SKIP_PHASES = {
    int(Phase.IDLE),
    int(Phase.SCAN),
    int(Phase.APPROACH_PUSH),
}

ALWAYS_SKIP_PHASES = {
    int(Phase.RETREAT),
}


def _find_phase_transitions(time_s: np.ndarray, phase_hist: np.ndarray) -> list[tuple[float, int]]:
    """Return list of (time, phase_id) at each phase transition (including start)."""
    transitions: list[tuple[float, int]] = []
    if len(phase_hist) == 0:
        return transitions

    transitions.append((float(time_s[0]), int(phase_hist[0])))
    changed_idx = np.flatnonzero(np.diff(phase_hist) != 0) + 1
    for idx in changed_idx:
        transitions.append((float(time_s[idx]), int(phase_hist[idx])))
    return transitions


def _phase_name(phase_id: int) -> str:
    return PHASE_NAMES.get(phase_id, f"PHASE_{phase_id}")


def _phase_runs(phase_hist: np.ndarray) -> list[tuple[int, int, int]]:
    """Return contiguous runs as (start_idx, end_idx_exclusive, phase_id)."""
    runs: list[tuple[int, int, int]] = []
    n = len(phase_hist)
    if n == 0:
        return runs

    start = 0
    current = int(phase_hist[0])
    for i in range(1, n):
        p = int(phase_hist[i])
        if p != current:
            runs.append((start, i, current))
            start = i
            current = p
    runs.append((start, n, current))
    return runs


def _compact_time_by_phase_runs(
    t: np.ndarray,
    runs: list[tuple[int, int, int]],
    max_segment_seconds: float,
) -> np.ndarray:
    """Compress long phase durations while preserving order and transition boundaries."""
    t_compact = np.zeros_like(t)
    offset = 0.0

    for start, end, _phase_id in runs:
        t0 = float(t[start])
        dur = float(t[end - 1] - t0) if end - start > 1 else 0.0
        shown = min(dur, max_segment_seconds)
        scale = (shown / dur) if dur > 1e-12 else 0.0

        if end - start > 1:
            t_compact[start:end] = offset + (t[start:end] - t0) * scale
        else:
            t_compact[start:end] = offset

        offset += shown

    return t_compact


def _plot_with_helper(
    t: np.ndarray,
    w_hist: np.ndarray,
    pitch_rad: np.ndarray,
    transitions: list[tuple[float, int]],
    title: str,
    xlabel: str,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax1, ax2 = plot_wrench_and_tipping(
        t=t,
        force_xyz=w_hist[:, :3],
        torque_primary=w_hist[:, 4],
        pitch_rad=pitch_rad,
        torque_label="tau_y",
        force_labels=("F_x", "F_y", "F_z"),
        y_label="Wrench (N, Nm)",
        contact_time=0.0,
        title=title,
        show=False,
    )

    ax1.set_xlabel(xlabel)

    for tx, phase_id in transitions:
        phase_label = _phase_name(phase_id)
        ax1.axvline(tx, linestyle="--", linewidth=1.0, alpha=0.35, color="k")
        if ax2 is not None:
            ax2.axvline(tx, linestyle="--", linewidth=1.0, alpha=0.35, color="k")

        y_top = ax1.get_ylim()[1]
        ax1.text(
            tx,
            y_top,
            phase_label,
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            alpha=0.85,
            clip_on=True,
        )

    fig.tight_layout()
    return fig, ax1


def _save_figure(fig: plt.Figure, save_path: Path | None) -> None:
    if save_path is None:
        return
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")


def _segment_has_interaction(
    w_seg: np.ndarray,
    pitch_seg: np.ndarray,
    force_thresh_n: float = 0.75,
    torque_thresh_nm: float = 0.06,
    pitch_span_thresh_deg: float = 1.0,
) -> bool:
    """Return True if segment shows nontrivial physical interaction."""
    if w_seg.size == 0 or pitch_seg.size == 0:
        return False

    force_peak = float(np.max(np.linalg.norm(w_seg[:, :3], axis=1)))
    torque_peak = float(np.max(np.linalg.norm(w_seg[:, 3:], axis=1)))
    pitch_span = float(np.max(pitch_seg) - np.min(pitch_seg))

    return (
        force_peak >= force_thresh_n
        or torque_peak >= torque_thresh_nm
        or abs(pitch_span) >= pitch_span_thresh_deg
    )


def plot_multiphase(
    npz_path: Path,
    save_path: Path | None = None,
    show: bool = True,
    mode: str = "segments",
    compact_max_segment_s: float = 3.0,
    include_move_phases: bool = False,
) -> None:
    if not npz_path.exists():
        raise FileNotFoundError(f"Input file not found: {npz_path}")

    data = np.load(npz_path)

    required = ["t_hist", "w_hist", "quat_hist", "phase_hist"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing required fields in npz: {missing}")

    t_hist = np.asarray(data["t_hist"], dtype=float).reshape(-1)
    w_hist = np.asarray(data["w_hist"], dtype=float).reshape(-1, 6)
    quat_hist = np.asarray(data["quat_hist"], dtype=float)
    phase_hist = np.asarray(data["phase_hist"], dtype=int).reshape(-1)

    if not (len(t_hist) == len(w_hist) == len(quat_hist) == len(phase_hist)):
        raise ValueError(
            "Length mismatch among t_hist, w_hist, quat_hist, phase_hist: "
            f"{len(t_hist)}, {len(w_hist)}, {len(quat_hist)}, {len(phase_hist)}"
        )

    # Convert to relative time for readability.
    t = t_hist - t_hist[0]

    # quat is expected in scipy format [x, y, z, w].
    rpy_rad = R.from_quat(quat_hist).as_euler("xyz", degrees=False)
    pitch_rad = rpy_rad[:, 1]
    pitch_deg = np.rad2deg(pitch_rad)

    transitions = _find_phase_transitions(t, phase_hist)
    runs = _phase_runs(phase_hist)

    if mode == "overview":
        fig, _ax = _plot_with_helper(
            t=t,
            w_hist=w_hist,
            pitch_rad=pitch_rad,
            transitions=transitions,
            title="Measured Wrench + Pitch with Phase Transitions",
            xlabel="Time (s)",
        )
        _save_figure(fig, save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return

    if mode == "compact":
        t_compact = _compact_time_by_phase_runs(t, runs, compact_max_segment_s)
        transitions_compact = _find_phase_transitions(t_compact, phase_hist)
        fig, _ax = _plot_with_helper(
            t=t_compact,
            w_hist=w_hist,
            pitch_rad=pitch_rad,
            transitions=transitions_compact,
            title=(
                "Measured Wrench + Pitch with Compressed Phase Time "
                f"(max {compact_max_segment_s:.1f} s per phase run)"
            ),
            xlabel="Compressed Time (s)",
        )
        _save_figure(fig, save_path)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return

    # mode == "segments"
    base_save = Path(save_path) if save_path is not None else None
    figs: list[plt.Figure] = []

    shown = 0
    skipped = 0

    for run_idx, (start, end, phase_id) in enumerate(runs, start=1):
        t_seg = t[start:end] - t[start]
        w_seg = w_hist[start:end, :]
        pitch_seg_rad = pitch_rad[start:end]
        pitch_seg = pitch_deg[start:end]

        if phase_id in ALWAYS_SKIP_PHASES:
            skipped += 1
            continue

        if not include_move_phases and phase_id in DEFAULT_SKIP_PHASES:
            if not _segment_has_interaction(w_seg, pitch_seg):
                skipped += 1
                continue

        # In segment view, we always show transition at local t=0.
        seg_transitions = [(0.0, phase_id)]
        fig, _ax = _plot_with_helper(
            t=t_seg,
            w_hist=w_seg,
            pitch_rad=pitch_seg_rad,
            transitions=seg_transitions,
            title=(
                f"Phase Segment {run_idx}: {_phase_name(phase_id)} "
                f"(abs t={t[start]:.2f}s to {t[end - 1]:.2f}s, dur={t_seg[-1] if len(t_seg) else 0.0:.2f}s)"
            ),
            xlabel="Phase-Local Time (s)",
        )
        figs.append(fig)
        shown += 1

        if base_save is not None:
            phase_tag = _phase_name(phase_id).lower()
            stem = base_save.stem
            suffix = base_save.suffix if base_save.suffix else ".png"
            out = base_save.parent / f"{stem}_{run_idx:02d}_{phase_tag}{suffix}"
            _save_figure(fig, out)

    if not include_move_phases:
        print(f"Segments shown: {shown} | move-only segments skipped: {skipped}")

    if show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multiphase simulation output.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "simulation_data_multiphase.npz",
        help="Path to simulation_data_multiphase.npz",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help=(
            "Optional output image path. In segments mode this is treated as a base name "
            "and one file is saved per phase run."
        ),
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open interactive window (useful when only saving).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["overview", "compact", "segments"],
        default="segments",
        help=(
            "overview: original timeline; compact: compress each phase run duration; "
            "segments: one figure per phase run"
        ),
    )
    parser.add_argument(
        "--compact-max-segment-s",
        type=float,
        default=3.0,
        help="In compact mode, maximum visible seconds allocated to any phase run.",
    )
    parser.add_argument(
        "--include-move-phases",
        action="store_true",
        help="In segments mode, include move-only phases (IDLE, SCAN, APPROACH_PUSH).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    plot_multiphase(
        args.input,
        save_path=args.save,
        show=not args.no_show,
        mode=args.mode,
        compact_max_segment_s=args.compact_max_segment_s,
        include_move_phases=args.include_move_phases,
    )


if __name__ == "__main__":
    main()
