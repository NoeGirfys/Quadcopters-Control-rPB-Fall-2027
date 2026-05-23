#!/usr/bin/env python3
"""
Compact report figures for the trained (noisy) regulation policy.

Loads a trained ConcurrentPolicyMLP, evaluates it on the 27-point initial
cube used for training, and produces three lightweight figures (positions
only + NN outputs), replacing the dense 6x3 ``save_plots`` figure:

  * <prefix>_pos_xyz.png : x, y, z versus time (3 subplots, 27 trajectories)
  * <prefix>_pos_3d.png  : the same trajectories in 3-D
  * <prefix>_nn_out.png  : the 4 NN outputs (thrust, roll, pitch, yaw-rate)

Usage:
    python plot_report_figures.py --weights trained_cf_pid_T1_ch200_h64_ep50_noisy.pt
"""

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

from train_nn_cf_pid import (
    ConcurrentPolicyMLP, evaluate, generate_cube_points,
    NN_FREQ, HOVER_THRUST_U16, PID_VEL_ROLL_MAX, PID_VEL_PITCH_MAX, YAW_RATE_MAX,
)

RAD2DEG = 180.0 / np.pi


def load_policy(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy = ConcurrentPolicyMLP(T=ckpt["T"], hidden=ckpt["hidden"]).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    return policy, ckpt


def plot_positions_xyz(t, pos, colors, out_path):
    """x, y, z versus time — one column per axis, 27 trajectories."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    names = ["x", "y", "z"]
    for j, ax in enumerate(axes):
        for b in range(pos.shape[0]):
            ax.plot(t, pos[b, :, j], lw=0.8, alpha=0.7, color=colors[b])
        ax.axhline(0.0, color="k", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{names[j]} [m]")
        ax.set_title(f"Position {names[j]}")
        ax.set_xlim(t[0], t[-1])
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_positions_3d(pos, t, out_path):
    """27 position trajectories in 3-D, coloured by time (single colormap).

    Every trajectory uses the same colormap mapped to elapsed time, so the
    colour gradient (dark -> bright) reads as the flow of time as all 27
    drones converge to the origin target.
    """
    import matplotlib as mpl
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=float(t[0]), vmax=float(t[-1]))

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for b in range(pos.shape[0]):
        p = pos[b]                                   # (T, 3)
        pts = p.reshape(-1, 1, 3)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)   # (T-1, 2, 3)
        lc = Line3DCollection(segs, cmap=cmap, norm=norm, alpha=0.75)
        lc.set_array(t[:-1])
        lc.set_linewidth(0.9)
        ax.add_collection3d(lc)

    # Common target at the origin.
    ax.scatter([0], [0], [0], color="crimson", marker="*", s=150,
               depthshade=False, label="target (origin)")

    mn = pos.reshape(-1, 3).min(axis=0)
    mx = pos.reshape(-1, 3).max(axis=0)
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Regulation trajectories (27 initial points)")
    ax.legend(loc="upper right", fontsize=8)
    ax.view_init(elev=22, azim=-58)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.10, shrink=0.6)
    cbar.set_label("time [s]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_nn_outputs(t, A, colors, out_path):
    """The 4 NN outputs (thrust, roll, pitch, yaw-rate) for 27 trajectories."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    specs = [
        (0, "Thrust [uint16]", "NN thrust", HOVER_THRUST_U16, "hover", None),
        (1, "Roll setpoint [deg]", "NN roll", None, None, PID_VEL_ROLL_MAX),
        (2, "Pitch setpoint [deg]", "NN pitch", None, None, PID_VEL_PITCH_MAX),
        (3, "Yaw-rate setpoint [deg/s]", "NN yaw rate", None, None, None),
    ]
    for ax, (idx, ylabel, title, hline, hlabel, sat) in zip(axes.flat, specs):
        for b in range(A.shape[0]):
            ax.plot(t, A[b, :, idx], lw=0.8, alpha=0.7, color=colors[b])
        if hline is not None:
            ax.axhline(hline, color="gray", ls=":", lw=1.0,
                       label=f"{hlabel}≈{hline:.0f}")
            ax.legend(fontsize=8)
        if sat is not None:   # firmware saturation band
            ax.axhline(+sat, color="r", ls="--", lw=0.8, alpha=0.6)
            ax.axhline(-sat, color="r", ls="--", lw=0.8, alpha=0.6)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(t[0], t[-1])
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Compact report figures.")
    parser.add_argument("--weights",
                        default="trained_cf_pid_T1_ch200_h64_ep50_noisy.pt")
    parser.add_argument("--half_side", type=float, default=0.5)
    parser.add_argument("--out_prefix", default="nn_regulation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    ckpt_path = args.weights if os.path.isabs(args.weights) \
        else os.path.join(here, args.weights)
    policy, ckpt = load_policy(ckpt_path, args.device)
    n_chunks = ckpt["n_chunks"]

    cube = generate_cube_points(args.half_side)
    X, A, R = evaluate(policy, cube, n_chunks, device=args.device, verbose=True)

    X = X.cpu().numpy()
    A = A.cpu().numpy()
    pos = X[:, :, [0, 2, 4]]                       # (27, total, 3)
    t = np.arange(1, X.shape[1] + 1) / NN_FREQ
    colors = plt.cm.viridis(np.linspace(0, 1, X.shape[0]))

    prefix = os.path.join(here, args.out_prefix)
    # Position: 3-D only for the report (time-coloured); plot_positions_xyz is
    # kept available but not generated by default.
    plot_positions_3d(pos, t, prefix + "_pos_3d.png")
    plot_nn_outputs(t, A, colors, prefix + "_nn_out.png")


if __name__ == "__main__":
    main()
