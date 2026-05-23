#!/usr/bin/env python3
"""
Overlay simulation vs real-drone NN-regulation flights for the report.

Loads a sim log (flight_sim_*.npz) and a real log (flight_real_*.npz) produced
by fly_nn_cf_pid.py (run with identical arguments) and draws two figures:

  * <prefix>_state.png  : 3x3 overlay of position / velocity / attitude,
                          simulation vs real, with the setpoint references.
  * <prefix>_nnout.png  : 2x2 overlay of the four NN outputs (thrust, roll,
                          pitch, yaw rate), simulation vs real (NN phase only).

Both flights are aligned at t = 0 (takeoff start). Run from this directory.

Usage:
    python plot_sim_real_overlay.py            # latest sim + real in flights/
    python plot_sim_real_overlay.py --sim ... --real ...
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from train_nn_cf_pid import HOVER_THRUST_U16
from crazyflie_firmware.constants import PID_VEL_ROLL_MAX, PID_VEL_PITCH_MAX

HERE = os.path.dirname(os.path.abspath(__file__))
SIM_C, REAL_C, SP_C = "steelblue", "tomato", "gray"


def _latest(pattern):
    files = sorted(glob.glob(os.path.join(HERE, "flights", pattern)))
    if not files:
        raise FileNotFoundError(f"no file matching flights/{pattern}")
    return files[-1]


def _phase_times_sim(d):
    """(t_hover, t_nn, t_land) transition times from the sim phase log."""
    t, ph = d["t"], d["phase"]
    def first(v):
        idx = np.where(ph == v)[0]
        return float(t[idx[0]]) if len(idx) else None
    return first(int(d["PHASE_HOVER"])), first(int(d["PHASE_NN"])), first(int(d["PHASE_LAND"]))


def plot_state(sim, real, out_path):
    """3x3: position / velocity / attitude, sim vs real vs setpoint."""
    t_hover, t_nn, t_land = _phase_times_sim(sim)
    nn_tgt, pid_tgt = sim["nn_target"], sim["pid_target"]

    fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
    fig.suptitle("NN regulation — simulation vs real drone", fontsize=14)

    pos_lbl = ["x [m]", "y [m]", "z [m]"]
    vel_lbl = ["vx [m/s]", "vy [m/s]", "vz [m/s]"]
    att_lbl = ["roll [deg]", "pitch [deg]", "yaw [deg]"]

    def _phase_lines(ax):
        for tt, c in [(t_nn, "red"), (t_land, "purple")]:
            if tt is not None:
                ax.axvline(tt, color=c, ls=":", lw=1.0, alpha=0.5)

    for col in range(3):
        # Position
        ax = axes[0, col]
        ax.plot(sim["t"], sim["pos"][:, col], color=SIM_C, lw=1.5, label="sim")
        ax.plot(real["t"], real["pos"][:, col], color=REAL_C, lw=1.5, alpha=0.85, label="real")
        ax.axhline(nn_tgt[col], color=SP_C, ls="--", lw=1.0, alpha=0.8, label="NN setpoint")
        ax.set_ylabel(pos_lbl[col]); ax.set_title(f"Position {'xyz'[col]}")
        _phase_lines(ax); ax.grid(True, alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7)

        # Velocity
        ax = axes[1, col]
        ax.plot(sim["t"], sim["vel"][:, col], color=SIM_C, lw=1.5)
        ax.plot(real["t"], real["vel"][:, col], color=REAL_C, lw=1.5, alpha=0.85)
        ax.set_ylabel(vel_lbl[col]); ax.set_title(f"Velocity {['vx','vy','vz'][col]}")
        _phase_lines(ax); ax.grid(True, alpha=0.3)

        # Attitude
        ax = axes[2, col]
        ax.plot(sim["t"], sim["rpy"][:, col], color=SIM_C, lw=1.5)
        ax.plot(real["t"], real["rpy"][:, col], color=REAL_C, lw=1.5, alpha=0.85)
        ax.set_ylabel(att_lbl[col]); ax.set_title(["Roll", "Pitch", "Yaw"][col])
        ax.set_xlabel("Time [s]")
        _phase_lines(ax); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_sim_pipeline(sim, out_path):
    """Sim-only deployment pipeline: position vs time with the four phases.

    Tells the §5.3 story: the cascade PID takes off and hovers, the NN regulates
    to the commanded setpoint, and the cascade PID lands the drone.
    """
    t, pos, phase = sim["t"], sim["pos"], sim["phase"]
    t_hover, t_nn, t_land = _phase_times_sim(sim)
    nn_tgt, pid_tgt = sim["nn_target"], sim["pid_target"]
    t_end = float(t[-1])

    # Phase intervals (start, end, colour, label); skip absent phases.
    pid_end = t_hover if t_hover is not None else t_nn
    spans = [(0.0, pid_end, "tab:blue", "PID takeoff")]
    if t_hover is not None:
        spans.append((t_hover, t_nn, "tab:olive", "hover"))
    spans.append((t_nn, t_land if t_land is not None else t_end, "tab:green", "NN regulation"))
    if t_land is not None:
        spans.append((t_land, t_end, "tab:gray", "PID landing"))

    fig, ax = plt.subplots(figsize=(12, 5))
    for x0, x1, c, lbl in spans:
        if x0 is not None and x1 is not None:
            ax.axvspan(x0, x1, color=c, alpha=0.10, label=lbl)

    for col, (c, name) in enumerate(zip(["C0", "C1", "C2"], ["x", "y", "z"])):
        ax.plot(t, pos[:, col], color=c, lw=1.6, label=name)
        ax.axhline(nn_tgt[col], color=c, ls="--", lw=1.0, alpha=0.6)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    sp_txt = ", ".join(f"{float(v):.2f}" for v in nn_tgt)
    ax.set_title(f"Simulated deployment pipeline (NN setpoint = ({sp_txt}) m)")
    ax.set_xlim(0, t_end)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=4, loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def plot_nn_outputs(sim, real, out_path):
    """2x2: the four NN outputs, sim vs real, restricted to the NN phase."""
    # Sim: nn_cmd is logged every step; keep the NN phase only.
    nn_mask = sim["phase"] == int(sim["PHASE_NN"])
    ts, cs = sim["t"][nn_mask], sim["nn_cmd"][nn_mask]
    have_real = "nn_cmd" in real
    tr = real["nn_cmd_t"] if have_real else None
    cr = real["nn_cmd"] if have_real else None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    specs = [(0, "Thrust [uint16]", "NN thrust", HOVER_THRUST_U16, None),
             (1, "Roll setpoint [deg]", "NN roll", None, PID_VEL_ROLL_MAX),
             (2, "Pitch setpoint [deg]", "NN pitch", None, PID_VEL_PITCH_MAX),
             (3, "Yaw-rate setpoint [deg/s]", "NN yaw rate", None, None)]
    for ax, (idx, ylabel, title, hline, sat) in zip(axes.flat, specs):
        ax.plot(ts, cs[:, idx], color=SIM_C, lw=1.3, label="sim")
        if have_real:
            ax.plot(tr, cr[:, idx], color=REAL_C, lw=1.3, alpha=0.85, label="real")
        if hline is not None:
            ax.axhline(hline, color=SP_C, ls=":", lw=1.0, label=f"hover≈{hline:.0f}")
        if sat is not None:
            ax.axhline(+sat, color="r", ls="--", lw=0.8, alpha=0.5)
            ax.axhline(-sat, color="r", ls="--", lw=0.8, alpha=0.5)
        ax.set_ylabel(ylabel); ax.set_title(title); ax.set_xlabel("Time [s]")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle("NN outputs — simulation vs real drone (NN phase)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    p = argparse.ArgumentParser(description="Overlay sim vs real NN flights.")
    p.add_argument("--sim", default=None, help="sim npz (default: latest)")
    p.add_argument("--real", default=None, help="real npz (default: latest)")
    p.add_argument("--out_prefix", default="nn_sim_real")
    args = p.parse_args()

    sim_path = args.sim or _latest("flight_sim_*.npz")
    real_path = args.real or _latest("flight_real_*.npz")
    print(f"[sim ] {sim_path}")
    print(f"[real] {real_path}")
    sim = dict(np.load(sim_path, allow_pickle=True))
    real = dict(np.load(real_path, allow_pickle=True))

    prefix = os.path.join(HERE, args.out_prefix)
    plot_sim_pipeline(sim, prefix + "_pipeline.png")   # §5.3 (sim only)
    plot_state(sim, real, prefix + "_state.png")       # §5.4 overlay
    plot_nn_outputs(sim, real, prefix + "_nnout.png")  # §5.4 overlay


if __name__ == "__main__":
    main()
