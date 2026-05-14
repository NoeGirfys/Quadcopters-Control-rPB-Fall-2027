"""Per-task mass parameters: total mass, CoM offset, and inertia tensor.

Realistic offset-mass model:
    (a) total mass change         M_total = M_base + m_extra
    (b) centre-of-mass shift      r_com   = m_extra * r_offset / M_total
    (c) inertia tensor change     full Steiner / parallel-axis on both
                                  the drone body and the point mass.

Single-task tensors have a leading dim of 1 that broadcasts with (B, ...).
Batched tensors (from batch_mass_params) have leading dim N*B.
"""
from dataclasses import dataclass
from typing import List
import math

import torch

from . import config as C


@dataclass
class MassParams:
    """Mass parameters for one task (or a batched block of N*B tasks).

    Single-task shapes  : M_total (1,), I (1,3,3), I_inv (1,3,3),
                          r_com (1,3), hover_rpm (1,)
    Batched shapes      : M_total (N*B,), I (N*B,3,3), I_inv (N*B,3,3),
                          r_com (N*B,3), hover_rpm (N*B,)
    """
    M_total:   torch.Tensor   # (1,) or (N*B,)
    I:         torch.Tensor   # (1,3,3) or (N*B,3,3)
    I_inv:     torch.Tensor   # (1,3,3) or (N*B,3,3)
    r_com:     torch.Tensor   # (1,3) or (N*B,3)
    hover_rpm: torch.Tensor   # (1,) or (N*B,)
    m_extra:   float          # for logging (representative value)
    r_offset:  tuple          # for logging (representative value)

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


def batch_mass_params(masses: List[MassParams], B: int,
                      device: str) -> MassParams:
    """Tile N MassParams into one batched block of shape (N*B, ...).

    Row slice [i*B : (i+1)*B] corresponds to masses[i], which lets a
    single rollout on (N*B, 12) states serve N independent tasks.

    Args:
        masses : list of N single-task MassParams (shape (1,...))
        B      : number of rollout states per task
        device : target device
    """
    M_total   = torch.cat([m.M_total.expand(B).to(device)       for m in masses])
    I         = torch.cat([m.I.expand(B, 3, 3).to(device)       for m in masses])
    I_inv     = torch.cat([m.I_inv.expand(B, 3, 3).to(device)   for m in masses])
    r_com     = torch.cat([m.r_com.expand(B, 3).to(device)      for m in masses])
    hover_rpm = torch.cat([m.hover_rpm.expand(B).to(device)     for m in masses])
    return MassParams(
        M_total=M_total, I=I, I_inv=I_inv, r_com=r_com,
        hover_rpm=hover_rpm,
        m_extra=masses[0].m_extra,
        r_offset=masses[0].r_offset,
    )
