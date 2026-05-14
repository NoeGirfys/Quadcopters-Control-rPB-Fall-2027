"""Fixed uniform mass task set: N positions sampled once at startup.

The entire set is fixed for the lifetime of a training run so that each
epoch is a full pass over the same N tasks.  This makes the meta-gradient
estimate much more stable than re-sampling tasks every epoch.

Positions are drawn uniformly from an axis-aligned box:
    dx ~ Uniform(xy_min, xy_max)
    dy ~ Uniform(xy_min, xy_max)
    dz ~ Uniform(z_min,  z_max)
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..mass_params import MassParams, compute_mass_params


class FixedUniformMassSet:
    """A fixed set of N_tasks mass configurations, sampled once at startup.

    Attributes:
        masses    : list of N MassParams (single-task tensors, shape (1,...))
        positions : (N, 3) float32 array of (dx, dy, dz) offsets [m]
        m_extra   : extra mass [kg] (same for all tasks)
    """

    def __init__(self, masses: List[MassParams],
                 positions: np.ndarray, m_extra: float):
        self.masses    = masses
        self.positions = positions          # (N, 3)
        self.m_extra   = float(m_extra)

    def __len__(self) -> int:
        return len(self.masses)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def sample(cls, n_tasks: int,
               xy_min: float, xy_max: float,
               z_min:  float, z_max:  float,
               m_extra: float,
               rng: np.random.Generator) -> FixedUniformMassSet:
        """Sample N_tasks positions uniformly and build the fixed set."""
        dx = rng.uniform(xy_min, xy_max, n_tasks).astype(np.float32)
        dy = rng.uniform(xy_min, xy_max, n_tasks).astype(np.float32)
        dz = rng.uniform(z_min,  z_max,  n_tasks).astype(np.float32)
        positions = np.stack([dx, dy, dz], axis=1)          # (N, 3)
        masses = [compute_mass_params(m_extra, tuple(pos))
                  for pos in positions]
        return cls(masses, positions, m_extra)

    @classmethod
    def from_positions(cls, positions: np.ndarray,
                       m_extra: float) -> FixedUniformMassSet:
        """Reconstruct from saved (N, 3) positions array (for resume)."""
        masses = [compute_mass_params(m_extra, tuple(pos.tolist()))
                  for pos in positions]
        return cls(masses, positions.astype(np.float32), m_extra)
