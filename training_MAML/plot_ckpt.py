"""Regenerate diagnostic plots from a saved MAML / baseline checkpoint.

Training (``train_maml.py`` / ``train_baseline.py``) no longer emits any
plot — it only writes the checkpoint.  This script rebuilds, for any
checkpoint, two figures saved next to the ``.pt`` file:

  * ``<base>_map.png``  : the fixed task map (mass positions coloured by dz
    + the train/eval x0 sets).  Identical to what training used to emit;
    the map is fully determined by data stored in the checkpoint.
  * ``<base>_loss.png`` : the per-epoch loss curve.
        - MAML checkpoint     -> inner (pre-adaptation) and outer (meta) loss
        - baseline checkpoint -> train and eval loss

The checkpoint type is detected from its ``history`` keys, so both
``*_inprogress.pt`` and final ``*_epXXX.pt`` checkpoints work.

Usage:
    cd training_MAML
    python plot_ckpt.py --ckpt maml_linearized_h64_o1_n50_ep500.pt
    python plot_ckpt.py --ckpt baseline_linearized_h64_n50_ep500.pt
"""
import argparse
import os
import sys

import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from maml_lib import plot_fixed_task_map


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

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

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")          # quadratic costs span several decades
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
    history    = ckpt.get("history", {})
    saved_args = ckpt.get("args", {})
    is_maml    = "meta_loss" in history

    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    base    = os.path.splitext(os.path.basename(args.ckpt))[0]

    print(f"[Type]  {'MAML' if is_maml else 'baseline'} checkpoint  "
          f"({len(history.get('epoch', []))} epochs)")

    # ── Task map ───────────────────────────────────────────────────────
    map_path = os.path.join(out_dir, base + "_map.png")
    try:
        plot_fixed_task_map(
            map_path,
            positions=ckpt["task_positions"],
            x0_train=ckpt["x0_train"],
            x0_eval=ckpt["x0_eval"],
            half_side=float(saved_args["half_side"]),
            xy_min=float(saved_args["xy_min"]),
            xy_max=float(saved_args["xy_max"]),
            z_min=float(saved_args["z_min"]),
            z_max=float(saved_args["z_max"]),
        )
        print(f"[Saved] {map_path}")
    except Exception as e:
        print(f"[Map] failed: {e}")

    # ── Loss curve ─────────────────────────────────────────────────────
    loss_path = os.path.join(out_dir, base + "_loss.png")
    plot_loss_curve(history, loss_path, is_maml)

    print("Done.")


if __name__ == "__main__":
    main()
