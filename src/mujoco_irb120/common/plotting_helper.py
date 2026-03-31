"""Reusable plotting helpers for wrench and tipping-angle time histories."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from .com_estimation import align_zeros


def _maybe_use_scientific_ticks(ax, values: np.ndarray) -> None:
    """Use scientific tick labels when magnitudes would produce long tick strings."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    vmax = float(np.max(np.abs(vals)))
    if vmax <= 0.0:
        return

    exponent = int(np.floor(np.log10(vmax)))
    if exponent >= 3 or exponent <= -3:
        fmt = ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))


def plot_4vec_vs_angle(
    vec_xyzw: np.ndarray,
    pitch_rad: Optional[np.ndarray] = None,
    *,
    vec_labels: Sequence[str] = ("f_x", "f_y", "f_z", "tau_y"),
    x_label: str = "Tipping angle ||$^\circ$||",
    y_label: str = "Force (N)",
    torque_y_label: str = "Torque (Nm)",
    figsize: Tuple[float, float] = (12, 5),
    legend_fontsize: int = 13,
    line_width: float = 3.0,
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot wrench channels with optional pitch-angle overlay.

    Args:
        vec_xyzw: Vector channels of shape (N,4).
        pitch_rad: Optional tipping angle history in radians, shape (N,).
        vec_labels: Legend labels for vector x/y/z curves.
        y_label: Left-axis y-label text.
        figsize: Matplotlib figure size.
        legend_fontsize: Combined legend font size.
        line_width: Shared line width for plotted curves.
        show: If True, call plt.show().
        title: Optional title for the plot.

    Returns:
        fig, ax1
    """
    vec_xyzw = np.asarray(vec_xyzw)
    if vec_xyzw.ndim != 2 or vec_xyzw.shape[1] != 4:
        raise ValueError(
            f"Expected vec_xyz with shape (N, 4), got {vec_xyzw.shape}"
        )
    if len(vec_labels) != 4:
        raise ValueError("vec_labels must contain exactly 4 entries")
    
    pitch_deg = abs(np.rad2deg(pitch_rad) if pitch_rad is not None else None)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(pitch_deg, vec_xyzw[:, 0], color="tab:red", linewidth=line_width, label=vec_labels[0])
    ax1.plot(pitch_deg, vec_xyzw[:, 1], color="tab:green", linewidth=line_width, label=vec_labels[1])
    ax1.plot(pitch_deg, vec_xyzw[:, 2], color="tab:blue", linewidth=line_width, label=vec_labels[2])

    ax_torque = ax1.twinx()
    ax_torque.plot(pitch_deg, vec_xyzw[:, 3], color="tab:orange", linewidth=line_width, label=vec_labels[3])

    # ax1.axvline(contact_time, color="k", linestyle="-", linewidth=2, label="first contact")
    ax1.set_xlabel(x_label, color="k")
    ax1.set_ylabel(y_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax_torque.set_ylabel(torque_y_label, color="tab:orange")
    ax_torque.tick_params(axis="y", labelcolor="tab:orange")
    _maybe_use_scientific_ticks(ax_torque, vec_xyzw[:, 3])
    ax1.grid(True)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines_t, labels_t = ax_torque.get_legend_handles_labels()
    ax1.legend(lines1 + lines_t, labels1 + labels_t, loc="best", fontsize=legend_fontsize)

    if title is not None:
        plt.title(title)

    plt.tight_layout()

    align_zeros([ax1, ax_torque])
    if show:
        plt.show()

    return fig, ax1

def plot_3vec_vs_angle(
    vec_xyz: np.ndarray,
    pitch_rad: Optional[np.ndarray] = None,
    *,
    vec_labels: Sequence[str] = ("f_x", "f_y", "f_z"),
    x_label: str = "Tipping angle ||$^\circ$||",
    y_label: str = "Force (N)",
    contact_time: float = 0.0,
    figsize: Tuple[float, float] = (12, 5),
    legend_fontsize: int = 13,
    line_width: float = 3.0,
    title: Optional[str] = None,
    show: bool = True,
):
    """Plot wrench channels with optional pitch-angle overlay.

    Args:
        vec_xyz: Vector channels of shape (N,3).
        pitch_rad: Optional tipping angle history in radians, shape (N,).
        vec_labels: Legend labels for vector x/y/z curves.
        y_label: Left-axis y-label text.
        contact_time: X-location (s) for the vertical contact marker.
        figsize: Matplotlib figure size.
        legend_fontsize: Combined legend font size.
        line_width: Shared line width for plotted curves.
        show: If True, call plt.show().
        title: Optional title for the plot.

    Returns:
        fig, ax1
    """
    vec_xyz = np.asarray(vec_xyz)
    if vec_xyz.ndim != 2 or vec_xyz.shape[1] != 3:
        raise ValueError(
            f"Expected vec_xyz with shape (N, 3), got {vec_xyz.shape}"
        )
    if len(vec_labels) != 3:
        raise ValueError("vec_labels must contain exactly 3 entries")
    
    pitch_deg = abs(np.rad2deg(pitch_rad) if pitch_rad is not None else None)

    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(pitch_deg, vec_xyz[:, 0], color="tab:pink", linewidth=line_width, label=vec_labels[0])
    ax1.plot(pitch_deg, vec_xyz[:, 1], color="tab:olive", linewidth=line_width, label=vec_labels[1])
    ax1.plot(pitch_deg, vec_xyz[:, 2], color="tab:cyan", linewidth=line_width, label=vec_labels[2])

    # ax1.axvline(contact_time, color="k", linestyle="-", linewidth=2, label="first contact")
    ax1.set_xlabel(x_label, color="k")
    ax1.set_ylabel(y_label, color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)

    ax1.legend(loc="best", fontsize=legend_fontsize)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax1


def plot_wrench_and_tipping(
    t: np.ndarray,
    force_xyz: np.ndarray,
    torque_primary: np.ndarray,
    pitch_rad: Optional[np.ndarray] = None,
    *,
    torque_label: str = "tau_y",
    force_labels: Sequence[str] = ("f_x", "f_y", "f_z"),
    y_label: str = "Force (N)",
    contact_time: float = 0.0,
    figsize: Tuple[float, float] = (12, 5),
    legend_fontsize: int = 13,
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
    ax1.plot(t, force_xyz[:, 0], color="tab:red", linewidth=line_width, label=force_labels[0])
    ax1.plot(t, force_xyz[:, 1], color="tab:green", linewidth=line_width, label=force_labels[1])
    ax1.plot(t, force_xyz[:, 2], color="tab:blue", linewidth=line_width, label=force_labels[2])
    ax1.plot(t, torque_primary, color="tab:orange", linewidth=line_width, label=torque_label)

    ax1.axvline(contact_time, color="k", linestyle="-", linewidth=2, label="first contact")
    ax1.set_xlabel("Time from first contact (s)")
    ax1.set_ylabel(f"{y_label}")
    ax1.tick_params(axis="y")
    # _maybe_use_scientific_ticks(ax1, np.concatenate([force_xyz.ravel(), torque_primary]))
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
        ax2.set_ylabel("Tipping angle ($^\circ$)", color="black")
        ax2.tick_params(axis="y", labelcolor="black")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="lower left",
            fontsize=legend_fontsize,
        )
        align_zeros([ax1, ax2])
    else:
        ax1.legend(loc="best", fontsize=legend_fontsize)

    if title is not None:
        plt.title(title)

    plt.tight_layout()
    if show:
        plt.show()

    return fig, ax1, ax2
