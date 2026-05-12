"""Per-task mass parameters: total mass, CoM offset, and inertia tensor.

Realistic offset-mass model:
    (a) total mass change         M_total = M_base + m_extra
    (b) centre-of-mass shift      r_com   = m_extra * r_offset / M_total
    (c) inertia tensor change     full Steiner / parallel-axis on both
                                  the drone body and the point mass.
"""
from dataclasses import dataclass
import math

import torch

from . import config as C


@dataclass
class MassParams:
    """Mass parameters for a single MAML task.

    Each tensor has a leading task dim of 1 so it broadcasts cleanly with
    the (B, ...) batch dimension used inside the rollout.
    """
    M_total:   torch.Tensor   # (1,)
    I:         torch.Tensor   # (1, 3, 3)  inertia about new CoM (body frame)
    I_inv:     torch.Tensor   # (1, 3, 3)
    r_com:     torch.Tensor   # (1, 3)     CoM offset from geometric center
    hover_rpm: torch.Tensor   # (1,)
    m_extra:   float          # for logging
    r_offset:  tuple          # for logging  (raw offset of the point mass)

    def to(self, device):
        return MassParams(
            M_total=self.M_total.to(device),
            I=self.I.to(device),
            I_inv=self.I_inv.to(device),
            r_com=self.r_com.to(device),
            hover_rpm=self.hover_rpm.to(device),
            m_extra=self.m_extra,
            r_offset=self.r_offset,
        )


def compute_mass_params(m_extra: float, r_offset: tuple) -> MassParams:
    """Build the (full Steiner) MassParams for one task.

    Args:
        m_extra:  additional point mass [kg]
        r_offset: (dx, dy, dz) attachment point in the drone body frame [m]
    """
    M_t = C.M_BASE + m_extra
    r_off = torch.tensor(r_offset, dtype=torch.float32)        # (3,)
    r_com = m_extra * r_off / M_t                              # (3,)

    I3 = torch.eye(3, dtype=torch.float32)

    # Drone body inertia (about original CoM = geometric center) translated
    # to the new CoM. d = r_com (vector from original CoM to new CoM).
    I_drone = torch.tensor(C.I_DRONE, dtype=torch.float32)
    d = r_com
    norm_d2 = (d * d).sum()
    outer_d = d.unsqueeze(-1) @ d.unsqueeze(0)                 # (3,3)
    I_drone_new = I_drone + C.M_BASE * (norm_d2 * I3 - outer_d) # inertia of the basic drone (without extra mass) around the new CoM

    # Point mass m_extra at r_off, expressed about the new CoM.
    s = r_off - r_com
    norm_s2 = (s * s).sum()
    outer_s = s.unsqueeze(-1) @ s.unsqueeze(0)
    I_extra_new = m_extra * (norm_s2 * I3 - outer_s) #inertia of the extra mass around the new CoM

    I_total = I_drone_new + I_extra_new
    I_inv = torch.linalg.inv(I_total)

    hover_rpm = math.sqrt(M_t * C.G / (4 * C.KF))

    return MassParams(
        M_total=torch.tensor([M_t], dtype=torch.float32),
        I=I_total.unsqueeze(0),
        I_inv=I_inv.unsqueeze(0),
        r_com=r_com.unsqueeze(0),
        hover_rpm=torch.tensor([hover_rpm], dtype=torch.float32),
        m_extra=float(m_extra),
        r_offset=tuple(float(v) for v in r_offset),
    )


