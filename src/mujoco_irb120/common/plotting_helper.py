"""Reusable plotting helpers for wrench and tipping-angle time histories."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .com_estimation import align_zeros


def plot_wrench_and_tipping(
    t: np.ndarray,
    force_xyz: np.ndarray,
    torque_primary: np.ndarray,
    pitch_rad: Optional[np.ndarray] = None,
    *,
    torque_label: str = "tau_y",
    force_labels: Sequence[str] = ("f_x", "f_y", "f_z"),
    y_label: str = "Force & Torque (N, Nm)",
    contact_time: float = 0.0,
    figsize: Tuple[float, float] = (12, 5),
    legend_fontsize: int = 14,
    line_width: float = 3.0,
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot wrench channels with optional pitch-angle overlay.

    Args:
        t: Time axis of shape (N,).
        force_xyz: Force channels of shape (N,3).
        torque_primary: Primary torque channel of shape (N,).
        pitch_rad: Optional tipping angle history in radians, shape (N,).
        torque_label: Legend label for the torque curve.
        force_labels: Legend labels for force x/y/z curves.
        y_label: Left-axis y-label text.
        contact_time: X-location (s) for the vertical contact marker.
        figsize: Matplotlib figure size.
        legend_fontsize: Combined legend font size.
        line_width: Shared line width for plotted curves.
        show: If True, call plt.show().
        title: Optional title for the plot.

    Returns:
        fig, ax1, ax2 where ax2 is None if pitch_rad is not provided.
    """
    t = np.asarray(t)
    force_xyz = np.asarray(force_xyz)
    torque_primary = np.asarray(torque_primary)

    if t.ndim != 1:
        raise ValueError(f"Expected t with shape (N,), got {t.shape}")
    if force_xyz.shape != (t.size, 3):
        raise ValueError(f"Expected force_xyz with shape ({t.size}, 3), got {force_xyz.shape}")
    if torque_primary.shape != (t.size,):
        raise ValueError(
            f"Expected torque_primary with shape ({t.size},), got {torque_primary.shape}"
        )
    if len(force_labels) != 3:
        raise ValueError("force_labels must contain exactly 3 entries")

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(t, torque_primary, color="tab:orange", linewidth=line_width, label=torque_label)
    ax1.plot(t, force_xyz[:, 0], color="tab:red", linewidth=line_width, label=force_labels[0])
    ax1.plot(t, force_xyz[:, 1], color="tab:green", linewidth=line_width, label=force_labels[1])
    ax1.plot(t, force_xyz[:, 2], color="tab:blue", linewidth=line_width, label=force_labels[2])

    ax1.axvline(contact_time, color="k", linestyle="-", linewidth=2, label="first contact")
    ax1.set_xlabel("Time from first contact (s)")
    ax1.set_ylabel(y_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)

    ax2 = None
    if pitch_rad is not None:
        pitch_rad = np.asarray(pitch_rad)
        if pitch_rad.shape != (t.size,):
            raise ValueError(f"Expected pitch_rad with shape ({t.size},), got {pitch_rad.shape}")

        ax2 = ax1.twinx()
        ax2.plot(
            t,
            np.rad2deg(pitch_rad),
            color="black",
            linewidth=line_width,
            linestyle="-.",
            label="pitch angle",
        )
        ax2.set_ylabel("Primary tipping angle (deg)", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=legend_fontsize)
        align_zeros([ax1, ax2])
    else:
        ax1.legend(loc="best", fontsize=legend_fontsize)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax1, ax2
