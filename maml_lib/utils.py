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
    """
    pos = (torch.rand(n, 3, generator=gen, device=device) * 2.0 - 1.0) * half_side
    X0 = torch.zeros(n, 12, dtype=torch.float32, device=device)
    X0[:, 0] = pos[:, 0]
    X0[:, 2] = pos[:, 1]
    X0[:, 4] = pos[:, 2]
    return X0


def plot_training_map(out_path: str, *, sampled_positions, motor_centers,
                      target_motor: int, training_motor_indices,
                      sigma: float, x0_train: torch.Tensor,
                      x0_eval: torch.Tensor, half_side: float):
    """Save a 2-panel diagnostic plot:

    * Left  : body-frame top view of motor centres (target = star, others
              = circles), σ rings, and one scatter of sampled mass
              offsets per training task.
    * Right : xy projection of the fixed train/eval x0 sets, two colours.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    cmap = plt.get_cmap("tab10")

    ax = axs[0]
    for i, (mx, my) in enumerate(motor_centers):
        is_target = (i == target_motor)
        ax.scatter([mx], [my], s=300,
                   marker=("*" if is_target else "o"),
                   color=("k" if is_target else cmap(i)),
                   edgecolors="k", zorder=5,
                   label=f"motor {i}{' (target)' if is_target else ''}")
        ax.add_patch(plt.Circle((mx, my), sigma, fill=False,
                                 linestyle="--",
                                 edgecolor=("k" if is_target else cmap(i)),
                                 alpha=0.5))

    for k, motor_idx in enumerate(training_motor_indices):
        samples = sampled_positions[k]
        if not samples:
            continue
        xs = [p[0] for p in samples]
        ys = [p[1] for p in samples]
        ax.scatter(xs, ys, s=8, alpha=0.5, color=cmap(motor_idx))

    ax.set_aspect("equal"); ax.grid(True)
    ax.set_title("Body-frame mass offsets (top view)")
    ax.set_xlabel("dx [m]"); ax.set_ylabel("dy [m]")
    ax.legend(loc="upper right", fontsize=8)

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
