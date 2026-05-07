"""Gaussian-motor task: a fixed-mass point attached near a given motor.

Each task is centred on the body-frame xy of a specific motor. ``sample``
draws (dx, dy) from an isotropic 2D Gaussian around that centre; dz is 0
and m_extra is fixed (no z-offset, no random mass any more).

This lets us train MAML on three motor neighbourhoods and hold out the
fourth motor as the target task for true out-of-distribution evaluation.
"""
from typing import List, Tuple

import numpy as np

from .base import Task
from ..mass_params import MassParams, compute_mass_params


class GaussianMotorTask(Task):
    def __init__(self,
                 motor_xy: Tuple[float, float],
                 sigma: float,
                 m_extra: float,
                 motor_idx: int = -1):
        self.motor_xy  = (float(motor_xy[0]), float(motor_xy[1]))
        self.sigma     = float(sigma)
        self.m_extra   = float(m_extra)
        self.motor_idx = int(motor_idx)

    def sample(self, rng) -> MassParams:
        dx = float(rng.normal(self.motor_xy[0], self.sigma))
        dy = float(rng.normal(self.motor_xy[1], self.sigma))
        return compute_mass_params(self.m_extra, (dx, dy, 0.0))

    def deterministic_set(self, n: int) -> List[MassParams]:
        rng = np.random.default_rng(123 + self.motor_idx)
        return [self.sample(rng) for _ in range(n)]

    def describe(self, mass: MassParams) -> str:
        dx, dy, _ = mass.r_offset
        return (f"motor#{self.motor_idx} "
                f"centre=({self.motor_xy[0]*100:+.2f},{self.motor_xy[1]*100:+.2f})cm "
                f"σ={self.sigma*100:.2f}cm  m={mass.m_extra*1e3:.0f}g  "
                f"sample=({dx*100:+.2f},{dy*100:+.2f})cm")
