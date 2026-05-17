"""Fixed uniform mass task set: N tasks sampled once at startup.

The entire set is fixed for the lifetime of a training run so that each
epoch is a full pass over the same N tasks.  This makes the meta-gradient
estimate much more stable than re-sampling tasks every epoch.

Each task is an offset point mass, sampled uniformly:
    dx      ~ Uniform(xy_min,  xy_max)
    dy      ~ Uniform(xy_min,  xy_max)
    dz      ~ Uniform(z_min,   z_max)
    m_extra ~ Uniform(mass_min, mass_max)

Varying ``m_extra`` across tasks makes the per-task hover thrust differ,
which is what gives MAML's few-shot adaptation a genuine edge over a
single joint-trained baseline.  Set ``mass_min == mass_max`` for the
legacy constant-mass behaviour.
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
        m_extras  : (N,) float32 array of per-task extra masses [kg]
    """

    def __init__(self, masses: List[MassParams],
                 positions: np.ndarray, m_extras: np.ndarray):
        self.masses    = masses
        self.positions = np.asarray(positions, dtype=np.float32)        # (N, 3)
        self.m_extras  = np.asarray(m_extras, dtype=np.float32).reshape(-1)

    def __len__(self) -> int:
        return len(self.masses)

    @property
    def m_extra(self) -> float:
        """Mean extra mass — representative scalar (e.g. for the hover bias)."""
        return float(self.m_extras.mean())

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def sample(cls, n_tasks: int,
               xy_min: float, xy_max: float,
               z_min:  float, z_max:  float,
               mass_min: float, mass_max: float,
               rng: np.random.Generator) -> FixedUniformMassSet:
        """Sample N_tasks offset positions and extra masses uniformly."""
        dx = rng.uniform(xy_min, xy_max, n_tasks).astype(np.float32)
        dy = rng.uniform(xy_min, xy_max, n_tasks).astype(np.float32)
        dz = rng.uniform(z_min,  z_max,  n_tasks).astype(np.float32)
        positions = np.stack([dx, dy, dz], axis=1)                      # (N, 3)
        m_extras  = rng.uniform(mass_min, mass_max, n_tasks).astype(np.float32)
        masses = [compute_mass_params(float(m), tuple(pos))
                  for m, pos in zip(m_extras, positions)]
        return cls(masses, positions, m_extras)

    @classmethod
    def from_arrays(cls, positions: np.ndarray,
                    m_extras: np.ndarray) -> FixedUniformMassSet:
        """Reconstruct from saved (N, 3) positions and (N,) extra masses.

        Used to resume a run or to mirror another run's task set exactly.
        """
        positions = np.asarray(positions, dtype=np.float32)
        m_extras  = np.asarray(m_extras, dtype=np.float32).reshape(-1)
        masses = [compute_mass_params(float(m), tuple(pos.tolist()))
                  for m, pos in zip(m_extras, positions)]
        return cls(masses, positions, m_extras)
