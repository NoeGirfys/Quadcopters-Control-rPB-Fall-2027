"""Regenerate diagnostic plots from a saved MAML / baseline checkpoint.

Training (``train_maml.py`` / ``train_baseline.py``) writes the
checkpoint only; this script rebuilds, for any checkpoint, three figures
saved next to the ``.pt`` file:

  * ``<base>_map.png``  : the composite task map — mass-attachment
    positions coloured by task with motor positions and σ circles, plus
    the drone start positions in the cube of octants. Optional target
    task is overlaid in red.
  * ``<base>_mass.png`` : the per-task payload-magnitude strip plot — one
    lane per task, every sampled extra mass as a dot (same per-task colour
    as the map), the per-task mean marked. Optional target task overlaid
    in red. This is the axis that differentiates tasks for MAML.
  * ``<base>_loss.png`` : the per-epoch loss curve. For MAML this shows
    inner (pre-adaptation) and outer (meta) losses; for the baseline,
    train and eval. In both cases, ``target_loss`` is overlaid if
    present.

Usage:
    cd training_MAML
    python plot_ckpt.py --ckpt maml_..._ep500.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from maml_lib import CompositeTaskSet, plot_fixed_task_map
from maml_lib import config as C


# ---------------------------------------------------------------------------
# Mass-magnitude strip plot
# ---------------------------------------------------------------------------

def plot_mass_values(out_path: str, *, task_set, target_set=None) -> None:
    """Per-task payload-magnitude strip plot (one lane per task).

    Mirrors the task-map conventions: one colour per training task (same
    ``tab20`` cmap as ``plot_fixed_task_map``), the held-out target task(s)
    overlaid in red. Each dot is one sampled extra mass (train + eval); the
    per-task mean is drawn as a short vertical bar and annotated in grams.
    """
    import matplotlib.pyplot as plt

    jitter_rng = np.random.default_rng(0)          # visual jitter only
    cmap = plt.get_cmap("tab20") if task_set.N <= 20 else plt.get_cmap("viridis")

    def _vals_g(ts, i):
        return np.concatenate([np.asarray(ts.m_extras_train[i]).reshape(-1),
                               np.asarray(ts.m_extras_eval[i]).reshape(-1)]) * 1e3

    lanes = [(task_set.names[i], _vals_g(task_set, i), cmap(i % cmap.N), False)
             for i in range(task_set.N)]
    if target_set is not None:
        lanes += [(target_set.names[i], _vals_g(target_set, i), "red", True)
                  for i in range(target_set.N)]

    fig, ax = plt.subplots(figsize=(9, 0.7 * len(lanes) + 2.0))
    yticks, ylabels = [], []
    for y, (name, vals, color, is_tgt) in enumerate(lanes):
        jit = (jitter_rng.random(vals.shape[0]) - 0.5) * 0.5
        ax.scatter(vals, np.full_like(vals, float(y)) + jit,
                   s=22, color=color, alpha=0.55,
                   marker=("x" if is_tgt else "o"),
                   linewidths=(1.3 if is_tgt else 0),
                   edgecolors="none")
        m = float(vals.mean())
        ax.plot([m, m], [y - 0.42, y + 0.42], color=color, lw=2.2)
        ax.text(m, y + 0.46, f"{m:.1f} g", ha="center", va="bottom",
                fontsize=8, color=color)
        yticks.append(y)
        ylabels.append(name + (" (target)" if is_tgt else ""))

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_ylim(-0.7, len(lanes) - 0.3)
    ax.set_xlabel("Extra (payload) mass [g]")
    ax.set_title(f"Per-task payload magnitude  "
                 f"(base drone = {C.M_BASE * 1e3:.0f} g; ○ = train, ✕ = target)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def _flatten_target_loss(target_loss):
    """``history['target_loss']`` is a list of lists (one inner list per
    epoch, with N_target entries). Return a (N_target, n_epochs) array,
    or None if empty."""
    if not target_loss:
        return None
    arr = np.asarray(target_loss, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]                                    # legacy float per epoch
    return arr.T                                              # (N_target, n_epochs)


def plot_loss_curve(history: dict, out_path: str, is_maml: bool,
                    show_target: bool = True) -> None:
    """Plot the per-epoch loss curve stored in ``history``.

    ``show_target=False`` omits the held-out ``target_loss`` overlay. The
    target metric is evaluated noise-free while the training curves are run
    with observation noise, so on a convergence plot they sit on different
    footings; hiding it keeps the convergence figure clean (the held-out
    comparison is then done separately on the final models, e.g. with
    ``eval_gap.py --obs-noise-scale``).
    """
    import matplotlib.pyplot as plt

    epochs = history.get("epoch", [])
    if not epochs:
        print("[Loss] checkpoint history is empty — skipping loss plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if is_maml:
        ax.plot(epochs, history["inner_pre"], color="C0",
                label="inner (pre-adaptation)")
        ax.plot(epochs, history["meta_loss"], color="C3",
                label="outer (post-adaptation / meta)")
        title = "MAML training loss"
    else:
        ax.plot(epochs, history["train_loss"], color="C0", label="train loss")
        ax.plot(epochs, history["eval_loss"],  color="C3", label="eval loss")
        title = "Baseline training loss"

    target_curves = _flatten_target_loss(history.get("target_loss", [])) if show_target else None
    if target_curves is not None:
        for k in range(target_curves.shape[0]):
            ax.plot(epochs[: target_curves.shape[1]], target_curves[k],
                    color="C2", linestyle="--", alpha=0.85,
                    label=(f"target loss [{k}]" if target_curves.shape[0] > 1
                           else "target loss (held-out)"))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Regenerate the task map and loss curve from a checkpoint.")
    p.add_argument("--ckpt", required=True,
                   help="MAML or baseline checkpoint (.pt)")
    p.add_argument("--no-target-loss", action="store_true",
                   help="omit the held-out target_loss overlay on the loss "
                        "curve (it is evaluated noise-free, unlike the noisy "
                        "training curves; hide it for a clean convergence plot)")
    p.add_argument("--no-target", action="store_true",
                   help="omit the held-out target from ALL three plots "
                        "(map, mass strip, loss curve). Implies --no-target-loss.")
    args = p.parse_args()

    # Headless-safe backend; must be set before pyplot is imported anywhere.
    import matplotlib
    matplotlib.use("Agg")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    history = ckpt.get("history", {})
    is_maml = "meta_loss" in history

    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    base    = os.path.splitext(os.path.basename(args.ckpt))[0]

    print(f"[Type]  {'MAML' if is_maml else 'baseline'} checkpoint  "
          f"({len(history.get('epoch', []))} epochs)")

    # ── Rebuild task sets once (shared by the map and mass plots) ──────
    task_set = target_set = None
    try:
        if "task_set" not in ckpt:
            raise KeyError(
                "Checkpoint has no 'task_set' field — was it produced by an "
                "older version of train_maml/baseline.py? Re-train with the "
                "current code.")
        task_set = CompositeTaskSet.from_dict(ckpt["task_set"])
        target_set = (CompositeTaskSet.from_dict(ckpt["target_set"])
                      if ckpt.get("target_set") is not None else None)
    except Exception as e:
        print(f"[Tasks] could not rebuild task set: {e}")

    target_for_overlay = None if args.no_target else target_set

    # ── Task map ───────────────────────────────────────────────────────
    if task_set is not None:
        map_path = os.path.join(out_dir, base + "_map.png")
        try:
            plot_fixed_task_map(map_path,
                                task_set=task_set, target_set=target_for_overlay)
            print(f"[Saved] {map_path}")
        except Exception as e:
            print(f"[Map] failed: {e}")

        # ── Mass-magnitude strip plot ──────────────────────────────────
        mass_path = os.path.join(out_dir, base + "_mass.png")
        try:
            plot_mass_values(mass_path,
                             task_set=task_set, target_set=target_for_overlay)
        except Exception as e:
            print(f"[Mass] failed: {e}")

    # ── Loss curve ─────────────────────────────────────────────────────
    loss_path = os.path.join(out_dir, base + "_loss.png")
    show_target_loss = not (args.no_target_loss or args.no_target)
    plot_loss_curve(history, loss_path, is_maml, show_target=show_target_loss)

    print("Done.")


if __name__ == "__main__":
    main()
