#!/usr/bin/env python3
"""
Overlay the training-loss curves of several runs on a single figure.

Each checkpoint produced by train_nn_cf_pid.py stores a per-epoch ``history``
dict (keys ``epoch``, ``loss``, ``pos_T_mean``, ``obs_noise_scale``). This
script reads that history from one or more checkpoints and plots the loss
curves together — typically to compare the noiseless and noisy training runs
for the report.

Usage
-----
    python plot_loss_comparison.py \
        --ckpts trained_cf_pid_T1_ch200_h64_ep100_noiseless.pt \
                trained_cf_pid_T1_ch200_h64_ep100_noisy.pt \
        --labels "no noise" "noise (scale 2.0)" \
        --out loss_comparison.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _load_history(ckpt_path: str) -> dict:
    """Return the training history stored in a checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history")
    if not history or not history.get("epoch"):
        raise ValueError(
            f"{ckpt_path}: no per-epoch history found. Retrain with the "
            f"current train_nn_cf_pid.py to log the loss.")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay training-loss curves.")
    parser.add_argument("--ckpts", nargs="+", required=True,
                        help="Checkpoint files to compare.")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Legend labels (default: derived from obs_noise_scale).")
    parser.add_argument("--out", default="loss_comparison.png",
                        help="Output PNG path.")
    parser.add_argument("--metric", default="loss",
                        choices=["loss", "pos_T_mean"],
                        help="Which per-epoch quantity to plot (default: loss).")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.ckpts):
        parser.error("--labels must have the same number of entries as --ckpts")

    fig, ax = plt.subplots(figsize=(9, 5))
    log_y = (args.metric == "loss")

    for i, ckpt_path in enumerate(args.ckpts):
        history = _load_history(ckpt_path)
        epochs = history["epoch"]
        values = history[args.metric]

        if args.labels is not None:
            label = args.labels[i]
        else:
            scale = history.get("obs_noise_scale", "?")
            label = "no noise" if scale in (0, 0.0) else f"noise (scale {scale})"

        plot = ax.semilogy if log_y else ax.plot
        plot(epochs, values, lw=1.5, label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (log scale)" if log_y
                  else "Terminal position error [m]")
    ax.set_title("Training convergence: noiseless vs noisy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"[Saved] {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
