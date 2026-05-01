"""Quadratic trajectory cost (per-trajectory, batched).

Same shape as ``train_nn_cf_pid.trajectory_cost`` but returns per-trajectory
losses (B,) so callers can ``.mean()`` themselves. This is convenient for
MAML because we sometimes need per-task statistics.
"""
import torch

from . import config as C


def trajectory_cost(X: torch.Tensor, terminal_weight: float = 50.0,
                    pos_weight: float = 10.0,
                    z_weight: float = 1.0) -> torch.Tensor:
    """Return per-trajectory cost (B,)."""
    Q = torch.as_tensor(C.Q_DIAG, dtype=X.dtype, device=X.device)
    Q_scaled = Q.clone()
    Q_scaled[[0, 2, 4]] *= pos_weight
    Q_scaled[4] *= z_weight

    cost = (X ** 2 * Q_scaled).sum(dim=2).sum(dim=1)             # (B,)
    if terminal_weight > 0:
        pos_T = X[:, -1, [0, 2, 4]]
        z_scale = torch.tensor([1.0, 1.0, z_weight],
                               dtype=X.dtype, device=X.device)
        cost = cost + terminal_weight * (pos_T ** 2 * z_scale).sum(dim=1)
    return cost
