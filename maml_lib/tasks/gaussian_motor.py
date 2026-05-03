"""Gaussian motor task: a Gaussian distribution over (extra mass, attachment
point) centred at a fixed body-frame position (typically a CF2X motor).

Each task instance is a *fixed* Gaussian; ``sample(rng)`` produces a fresh
``MassParams`` from that Gaussian. A pool of 4 such tasks (one per motor)
gives MAML a structured task family with clear semantics: each motor =
one mode of the task distribution.
"""
from typing import List, Tuple

import numpy as np

from .base import Task
from .. import config as C
from ..mass_params import MassParams, compute_mass_params


class GaussianMotorTask(Task):
    """Gaussian distribution over the extra mass *and* its body-frame offset.

    Sampling rule (independent Gaussians):
        m  ~ max(mass_min, N(mass_mean, mass_std))
        dx ~ N(center.x, pos_std)
        dy ~ N(center.y, pos_std)
        dz ~ N(center.z, pos_std)

    Args:
        center:    (3,) body-frame point the position Gaussian is centred on.
        pos_std:   scalar std on each of (dx, dy, dz) [m].
        mass_mean: mean of the extra mass distribution [kg].
        mass_std:  std of the extra mass distribution [kg].
        mass_min:  lower clip on the sampled mass [kg]; default 0 (physical).
        label:     human-readable label, used by ``describe`` and logging.
    """

    def __init__(
        self,
        center: Tuple[float, float, float],
        pos_std: float = 0.005,
        mass_mean: float = 0.010,
        mass_std: float = 0.003,
        mass_min: float = 0.0,
        label: str = "gaussian",
    ):
        self.center    = tuple(float(c) for c in center)
        self.pos_std   = float(pos_std)
        self.mass_mean = float(mass_mean)
        self.mass_std  = float(mass_std)
        self.mass_min  = float(mass_min)
        self.label     = label

    def sample(self, rng) -> MassParams:
        m  = float(rng.normal(self.mass_mean, self.mass_std))
        m  = max(self.mass_min, m)
        dx = float(rng.normal(self.center[0], self.pos_std))
        dy = float(rng.normal(self.center[1], self.pos_std))
        dz = float(rng.normal(self.center[2], self.pos_std))
        return compute_mass_params(m, (dx, dy, dz))

    def deterministic_set(self, n: int) -> List[MassParams]:
        rng = np.random.default_rng(123)
        return [self.sample(rng) for _ in range(n)]

    def describe(self, mass: MassParams) -> str:
        dx, dy, dz = mass.r_offset
        return (f"[{self.label}] m={mass.m_extra*1e3:+.1f}g  "
                f"r=({dx*100:+.1f},{dy*100:+.1f},{dz*100:+.1f})cm")


def make_motor_tasks(
    pos_std: float = 0.005,
    mass_mean: float = 0.010,
    mass_std: float = 0.003,
    mass_min: float = 0.0,
) -> List[GaussianMotorTask]:
    """Build a 4-task pool: one Gaussian per CF2X motor.

    Centers come from ``config.MOTOR_POS`` (CF2X X-shape):
        M1 front-right, M2 rear-right, M3 rear-left, M4 front-left.
    All four share the same scaling (``pos_std``, ``mass_mean``,
    ``mass_std``); only the centre differs.
    """
    motor_names = ("front-right", "rear-right", "rear-left", "front-left")
    return [
        GaussianMotorTask(
            center=tuple(float(v) for v in C.MOTOR_POS[i]),
            pos_std=pos_std,
            mass_mean=mass_mean,
            mass_std=mass_std,
            mass_min=mass_min,
            label=f"M{i+1}_{motor_names[i]}",
        )
        for i in range(4)
    ]
