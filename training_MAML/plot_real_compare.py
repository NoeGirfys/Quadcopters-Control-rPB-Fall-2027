"""Real-drone counterpart of the Figure-21 PyBullet comparison plot.

Takes four real flight logs from ``fly_maml_cf_pid.py`` (one per controller:
firmware PID, baseline NN, baseline NN adapted, MAML adapted) flown with the
same payload and the same NN target, and plots the x/y/z position traces of
the NN phase side by side, in the same style/colours as
``compare_pybullet.plot_compare``.

Time is aligned so that t = 0 is the start of the NN phase
(``elapsed_phases01`` in the log); the landing phase is cut off.

Usage:
    cd training_MAML
    python plot_real_compare.py \
        --pid        flights/maml_real_pid_8g_20260611_152800.npz \
        --base       flights/maml_real_base_8g_20260611_152902.npz \
        --base-adapt flights/maml_real_base-adapt_8g_20260611_153010.npz \
        --maml       flights/maml_real_maml_8g_20260611_152611.npz \
        --mass 8.21
"""
import argparse
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Same order/colours/styles as compare_pybullet.CTRL_STYLE.
CTRL_ORDER = ["pid", "base", "base_adapt", "maml_adapt"]
CTRL_STYLE = {
    "pid":        dict(color="C2",   ls="-.", label="firmware PID (no NN)"),
    "base":       dict(color="0.55", ls="--", label="baseline NN (no adapt)"),
    "base_adapt": dict(color="C0",   ls="-",  label="baseline NN adapted"),
    "maml_adapt": dict(color="C3",   ls="-",  label="MAML adapted"),
}


def load_nn_phase(path: str):
    """Load a real flight log and return (t_rel, pos) for the NN phase only.

    t_rel = 0 at the start of the NN phase; the trace stops at the end of the
    NN phase (last NN command for NN flights, --duration for the PID flight).
    """
    d = np.load(path, allow_pickle=True)
    t = np.asarray(d["t"], dtype=float)
    pos = np.asarray(d["pos"], dtype=float)
    t_start = float(d["elapsed_phases01"])
    if "nn_cmd_t" in d:
        t_end = float(np.asarray(d["nn_cmd_t"])[-1])
    else:
        # pure-PID reference: phase 2 runs until the requested total duration,
        # which matches the NN flights' command window (same --duration).
        t_end = t_start + 11.0
    m = (t >= t_start) & (t <= t_end)
    target = np.asarray(d["nn_target"], dtype=float)
    return t[m] - t_start, pos[m], target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", required=True, help="firmware PID flight (.npz)")
    p.add_argument("--base", required=True, help="baseline (no adapt) flight (.npz)")
    p.add_argument("--base-adapt", required=True, help="baseline adapted flight (.npz)")
    p.add_argument("--maml", required=True, help="MAML adapted flight (.npz)")
    p.add_argument("--mass", type=float, default=8.21, help="payload mass [g]")
    p.add_argument("--t-max", type=float, default=None,
                   help="cut all traces at this NN-phase time [s] "
                        "(default: shortest common window)")
    p.add_argument("--out", default=None, help="output figure path")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = {"pid": args.pid, "base": args.base,
             "base_adapt": args.base_adapt, "maml_adapt": args.maml}
    runs, targets = {}, []
    for key, path in files.items():
        t, pos, target = load_nn_phase(path)
        runs[key] = (t, pos)
        targets.append(target)
        print(f"[LOAD] {key:10s} {os.path.basename(path)}  "
              f"NN phase {t[-1]:.1f}s  target={target}")
    if not all(np.allclose(targets[0], tg) for tg in targets[1:]):
        print(f"[WARN] NN targets differ across flights: {targets}")
    target = targets[0]

    t_max = args.t_max or min(runs[k][0][-1] for k in CTRL_ORDER)

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.5), sharex=True)
    axis_names = ["x", "y", "z"]
    for j in range(3):
        ax = axes[j]
        for key in CTRL_ORDER:
            st = CTRL_STYLE[key]
            t, pos = runs[key]
            m = t <= t_max
            ax.plot(t[m], pos[m, j], color=st["color"], ls=st["ls"], lw=1.5,
                    label=(st["label"] if j == 0 else None))
        ax.axhline(target[j], color="k", ls=":", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{axis_names[j]}  (target {target[j]:.2f} m)")
        ax.set_xlabel("time since NN takeover [s]")
        if j == 0:
            ax.set_ylabel(f"{args.mass:.2f} g\nposition [m]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Real drone position: firmware PID vs baseline vs "
                 "baseline-adapted vs MAML-adapted", y=0.998, fontsize=13)
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.84])

    out = args.out or os.path.join(
        SCRIPT_DIR, f"maml_real_compare_{args.mass:g}g.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[PLOT] saved {out}")


if __name__ == "__main__":
    main()
