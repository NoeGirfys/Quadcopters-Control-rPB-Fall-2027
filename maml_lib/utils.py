"""Misc utilities shared across MAML training scripts."""
import itertools
import signal
import sys

import torch

from . import config as C


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


def plot_fixed_task_map(out_path: str, *, task_set, target_set=None):
    """2-panel diagnostic plot for a :class:`CompositeTaskSet`.

    * Left  : body-frame xy scatter of every sampled mass-attachment point,
              one colour per task. Motor positions are marked, and 1σ/3σ
              circles around each motor show the per-task Gaussian. Marker
              size encodes the extra-mass magnitude.
    * Right : xy projection of every sampled drone start point. Circle =
              z > 0 (octants 1-4), down-triangle = z < 0 (octants 5-8).
              Sign-plane lines and the cube ±half_side outline are drawn.

    The optional ``target_set`` is overlaid in red.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    N = task_set.N
    cmap = plt.get_cmap("tab20") if N <= 20 else plt.get_cmap("viridis")

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # ── Left: mass-attachment positions ───────────────────────────────
    ax = axs[0]

    motor_pos = np.asarray(C.MOTOR_POS, dtype=np.float32)            # (4, 3)
    for k in range(4):
        ax.plot(motor_pos[k, 0] * 100, motor_pos[k, 1] * 100,
                marker='+', color='k', markersize=14, mew=2)
        for r in (1, 3):
            circ = plt.Circle(
                (motor_pos[k, 0] * 100, motor_pos[k, 1] * 100),
                r * task_set.mass_pos_sigma * 100,
                fill=False, linestyle=':', color='gray',
                linewidth=0.6, alpha=0.6)
            ax.add_patch(circ)
        ax.text(motor_pos[k, 0] * 100 + 0.3, motor_pos[k, 1] * 100 + 0.3,
                f"M{k+1}", fontsize=8, color='k')

    s_lo, s_hi = 8.0, 60.0
    mspan = max(task_set.mass_max - task_set.mass_min, 1e-9)

    def _mass_to_size(m):
        return s_lo + (s_hi - s_lo) * (m - task_set.mass_min) / mspan

    for i in range(N):
        mp = np.concatenate(
            [task_set.mass_positions_train[i],
             task_set.mass_positions_eval[i]], axis=0)
        me = np.concatenate(
            [task_set.m_extras_train[i],
             task_set.m_extras_eval[i]], axis=0)
        ax.scatter(mp[:, 0] * 100, mp[:, 1] * 100,
                   s=_mass_to_size(me), color=cmap(i % cmap.N),
                   alpha=0.55, edgecolors='none', label=task_set.names[i])

    if target_set is not None:
        for i in range(target_set.N):
            mp = np.concatenate(
                [target_set.mass_positions_train[i],
                 target_set.mass_positions_eval[i]], axis=0)
            ax.scatter(mp[:, 0] * 100, mp[:, 1] * 100,
                       marker='x', color='red', s=35, linewidths=1.4,
                       alpha=0.85,
                       label=f"target: {target_set.names[i]}")

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Mass-attachment positions — body frame "
        f"(N={N}, σ={task_set.mass_pos_sigma*100:.1f} cm, "
        f"m∈[{task_set.mass_min*1e3:.1f},{task_set.mass_max*1e3:.1f}] g)")
    ax.set_xlabel("dx [cm]"); ax.set_ylabel("dy [cm]")
    ax.legend(fontsize=7, loc='best', ncol=2)

    # ── Right: drone start positions (xy projection) ──────────────────
    ax = axs[1]
    half = task_set.half_side
    ax.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='k', linewidth=0.8, alpha=0.5)
    rect = plt.Rectangle((-half, -half), 2 * half, 2 * half,
                         linewidth=1.5, edgecolor='k', facecolor='none',
                         linestyle='--', label=f"cube ±{half} m")
    ax.add_patch(rect)

    for i in range(N):
        x0 = torch.cat([task_set.x0_train[i], task_set.x0_eval[i]], dim=0
                       ).detach().cpu().numpy()
        z_pos = x0[:, 4] >= 0
        color = cmap(i % cmap.N)
        if z_pos.any():
            ax.scatter(x0[z_pos, 0], x0[z_pos, 2], marker='o', s=14,
                       color=color, alpha=0.55, edgecolors='none')
        if (~z_pos).any():
            ax.scatter(x0[~z_pos, 0], x0[~z_pos, 2], marker='v', s=14,
                       color=color, alpha=0.55, edgecolors='none')
        # one legend entry per task
        ax.plot([], [], 'o', color=color, alpha=0.55, label=task_set.names[i])

    if target_set is not None:
        for i in range(target_set.N):
            x0 = torch.cat([target_set.x0_train[i], target_set.x0_eval[i]],
                           dim=0).detach().cpu().numpy()
            z_pos = x0[:, 4] >= 0
            if z_pos.any():
                ax.scatter(x0[z_pos, 0], x0[z_pos, 2], marker='x',
                           color='red', s=30, linewidths=1.4, alpha=0.85)
            if (~z_pos).any():
                ax.scatter(x0[~z_pos, 0], x0[~z_pos, 2], marker='+',
                           color='red', s=40, linewidths=1.4, alpha=0.85)
            ax.plot([], [], 'x', color='red',
                    label=f"target: {target_set.names[i]}")

    ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
    ax.set_title("Drone start positions, xy projection  (○ = z≥0, ▽ = z<0)")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.legend(fontsize=7, loc='best', ncol=2)

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
