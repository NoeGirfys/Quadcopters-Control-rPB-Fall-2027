"""Offset-mass task: an extra point mass attached at an arbitrary position.

This perturbs all three drone parameters (mass, CoM, inertia tensor).
Default sampling ranges (confirmed by the user):
    m_extra ∈ [0, 15] g
    dx, dy  ∈ ±3 cm
    dz      ∈ [-3, +5] cm
"""
from typing import List

import numpy as np

from .base import Task
from ..mass_params import MassParams, compute_mass_params


class OffsetMassTask(Task):
    def __init__(self,
                 m_min: float = 0.0,    m_max: float = 0.015,
                 dx_max: float = 0.03,  dy_max: float = 0.03,
                 dz_min: float = -0.03, dz_max: float = 0.05):
        self.m_min  = m_min;  self.m_max  = m_max
        self.dx_max = dx_max; self.dy_max = dy_max
        self.dz_min = dz_min; self.dz_max = dz_max

    def sample(self, rng) -> MassParams:
        m  = float(rng.uniform(self.m_min, self.m_max))
        dx = float(rng.uniform(-self.dx_max, self.dx_max))
        dy = float(rng.uniform(-self.dy_max, self.dy_max))
        dz = float(rng.uniform(self.dz_min, self.dz_max))
        return compute_mass_params(m, (dx, dy, dz))

    def deterministic_set(self, n: int) -> List[MassParams]:
        rng = np.random.default_rng(123)
        return [self.sample(rng) for _ in range(n)]

    def describe(self, mass: MassParams) -> str:
        dx, dy, dz = mass.r_offset
        return (f"m={mass.m_extra*1e3:+.1f}g  "
                f"r=({dx*100:+.1f},{dy*100:+.1f},{dz*100:+.1f})cm")
