"""Regenerate diagnostic plots from a saved MAML / baseline checkpoint.

Training (``train_maml.py`` / ``train_baseline.py``) writes the
checkpoint only; this script rebuilds, for any checkpoint, two figures
saved next to the ``.pt`` file:

  * ``<base>_map.png``  : the composite task map — mass-attachment
    positions coloured by task with motor positions and σ circles, plus
    the drone start positions in the cube of octants. Optional target
    task is overlaid in red.
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


def plot_loss_curve(history: dict, out_path: str, is_maml: bool) -> None:
    """Plot the per-epoch loss curve stored in ``history``."""
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

    target_curves = _flatten_target_loss(history.get("target_loss", []))
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

    # ── Task map ───────────────────────────────────────────────────────
    map_path = os.path.join(out_dir, base + "_map.png")
    try:
        if "task_set" not in ckpt:
            raise KeyError(
                "Checkpoint has no 'task_set' field — was it produced by an "
                "older version of train_maml/baseline.py? Re-train with the "
                "current code.")
        task_set = CompositeTaskSet.from_dict(ckpt["task_set"])
        target_set = (CompositeTaskSet.from_dict(ckpt["target_set"])
                      if ckpt.get("target_set") is not None else None)
        plot_fixed_task_map(map_path,
                            task_set=task_set, target_set=target_set)
        print(f"[Saved] {map_path}")
    except Exception as e:
        print(f"[Map] failed: {e}")

    # ── Loss curve ─────────────────────────────────────────────────────
    loss_path = os.path.join(out_dir, base + "_loss.png")
    plot_loss_curve(history, loss_path, is_maml)

    print("Done.")


if __name__ == "__main__":
    main()
