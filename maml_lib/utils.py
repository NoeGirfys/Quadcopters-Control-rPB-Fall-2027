"""Misc utilities shared across MAML training scripts."""
import itertools
import signal
import sys

import torch


def sample_hover_x0(n: int, half_side: float, gen: torch.Generator,
                    device: str = "cpu") -> torch.Tensor:
    """Draw n hover initial states (B, 12).

    Position (x, y, z) is uniform in the cube [-half_side, +half_side]^3.
    Velocity, attitude and rates are zero (hover). Sampling uses ``gen``
    so the result is reproducible from a single seed.

    Reproducibility caveat: ``gen`` is device-bound. A CPU and a CUDA
    ``torch.Generator`` seeded identically produce *different* sequences,
    so a standalone run on CPU will not reproduce the x0 set of a GPU run
    (the mass tasks themselves use NumPy and stay reproducible). To compare
    a baseline against a MAML run trained on another device, load the x0
    set from its checkpoint (``--from-maml-ckpt``) instead of re-sampling.
    """
    pos = (torch.rand(n, 3, generator=gen, device=device) * 2.0 - 1.0) * half_side
    X0 = torch.zeros(n, 12, dtype=torch.float32, device=device)
    X0[:, 0] = pos[:, 0]
    X0[:, 2] = pos[:, 1]
    X0[:, 4] = pos[:, 2]
    return X0


def plot_fixed_task_map(out_path: str, *,
                        positions,
                        x0_train: torch.Tensor,
                        x0_eval: torch.Tensor,
                        half_side: float,
                        xy_min: float, xy_max: float,
                        z_min: float, z_max: float):
    """Save a 2-panel diagnostic plot for the fixed uniform task set.

    * Left  : body-frame xy scatter of the N fixed mass positions,
              coloured by dz value; box shows the sampling range.
    * Right : xy projection of the fixed train/eval x0 sets.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    positions = np.asarray(positions)          # (N, 3)
    fig, axs = plt.subplots(1, 2, figsize=(13, 6))

    # ── Left: mass positions ──────────────────────────────────────────
    ax = axs[0]
    sc = ax.scatter(positions[:, 0] * 100, positions[:, 1] * 100,
                    c=positions[:, 2] * 100, cmap="coolwarm",
                    s=30, alpha=0.7, edgecolors="none")
    plt.colorbar(sc, ax=ax, label="dz [cm]")
    rect = plt.Rectangle((xy_min * 100, xy_min * 100),
                          (xy_max - xy_min) * 100, (xy_max - xy_min) * 100,
                          linewidth=1.5, edgecolor="k", facecolor="none",
                          linestyle="--", label="sampling box")
    ax.add_patch(rect)
    ax.set_aspect("equal"); ax.grid(True)
    ax.set_title(f"Fixed task set (N={len(positions)}) — body-frame xy")
    ax.set_xlabel("dx [cm]"); ax.set_ylabel("dy [cm]")
    ax.legend(fontsize=8)

    # ── Right: x0 sets ───────────────────────────────────────────────
    ax = axs[1]
    x0_t = x0_train.detach().cpu().numpy()
    x0_e = x0_eval.detach().cpu().numpy()
    ax.scatter(x0_t[:, 0], x0_t[:, 2], c="C0",
               label=f"train x0 (n={len(x0_t)})", s=25, alpha=0.8)
    ax.scatter(x0_e[:, 0], x0_e[:, 2], c="C3",
               label=f"eval x0 (n={len(x0_e)})",  s=25, alpha=0.8)
    ax.set_aspect("equal"); ax.grid(True)
    ax.set_title(f"Initial positions, xy projection (cube ±{half_side}m)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def make_x0_batch(xyz_list, device="cpu") -> torch.Tensor:
    """Initial states (B, 12) from a list of (x, y, z); rest = 0."""
    X0 = torch.zeros(len(xyz_list), 12, dtype=torch.float32, device=device)
    for b, (x, y, z) in enumerate(xyz_list):
        X0[b, 0] = x
        X0[b, 2] = y
        X0[b, 4] = z
    return X0


def generate_cube_points(half_side: float = 0.3):
    """27 points: vertices + edges + faces + center of a cube."""
    vals = [-half_side, 0.0, half_side]
    return list(itertools.product(vals, repeat=3))


class GracefulKiller:
    """Catch Ctrl+C so training stops cleanly at the end of an epoch."""

    def __init__(self):
        self.kill_now = False
        self._count = 0
        signal.signal(signal.SIGINT,  self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, *_):
        self._count += 1
        if self._count == 1:
            print("\n[Interruption] Arrêt à la fin de l'epoch courant. "
                  "Ctrl+C à nouveau pour forcer.")
            self.kill_now = True
        else:
            print("\n[Arrêt forcé]")
            sys.exit(1)
