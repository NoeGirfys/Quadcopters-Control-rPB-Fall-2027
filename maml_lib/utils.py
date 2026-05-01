"""Misc utilities shared across MAML training scripts."""
import itertools
import signal
import sys

import torch


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
