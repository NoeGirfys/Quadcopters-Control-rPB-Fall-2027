"""Per-task mass parameters: total mass, CoM offset, and inertia tensor.

Realistic offset-mass model:
    (a) total mass change         M_total = M_base + m_extra
    (b) centre-of-mass shift      r_com   = m_extra * r_offset / M_total
    (c) inertia tensor change     full Steiner / parallel-axis on both
                                  the drone body and the point mass.

Single-task tensors have a leading dim of 1 that broadcasts with (B, ...).
Batched tensors have leading dim equal to the batch size.
"""
from dataclasses import dataclass
from typing import List
import math

import numpy as np
import torch

from . import config as C


@dataclass
class MassParams:
    """Mass parameters for one or more attached-mass configurations.

    Single-task shapes  : M_total (1,), I (1,3,3), I_inv (1,3,3),
                          r_com (1,3), hover_rpm (1,)
    Batched shapes      : M_total (B,), I (B,3,3), I_inv (B,3,3),
                          r_com (B,3), hover_rpm (B,)
    """
    M_total:   torch.Tensor   # (B,)
    I:         torch.Tensor   # (B,3,3)
    I_inv:     torch.Tensor   # (B,3,3)
    r_com:     torch.Tensor   # (B,3)
    hover_rpm: torch.Tensor   # (B,)
    m_extra:   float          # representative value (for logging)
    r_offset:  tuple          # representative value (for logging)

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
    """Build the (full Steiner) MassParams for a single mass configuration."""
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
    I_drone_new = I_drone + C.M_BASE * (norm_d2 * I3 - outer_d)

    # Point mass m_extra at r_off, expressed about the new CoM.
    s = r_off - r_com
    norm_s2 = (s * s).sum()
    outer_s = s.unsqueeze(-1) @ s.unsqueeze(0)
    I_extra_new = m_extra * (norm_s2 * I3 - outer_s)

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


def compute_mass_params_batched(m_extras, positions) -> MassParams:
    """Vectorised :func:`compute_mass_params` for M points at once.

    Args:
        m_extras  : (M,) array-like of extra masses [kg]
        positions : (M, 3) array-like of attachment offsets in the body
                    frame [m]
    Returns:
        MassParams whose tensors have leading batch dim M.
    """
    m = torch.as_tensor(np.asarray(m_extras), dtype=torch.float32).reshape(-1)
    r_off = torch.as_tensor(np.asarray(positions), dtype=torch.float32)
    if r_off.dim() != 2 or r_off.shape[-1] != 3:
        raise ValueError(f"positions must be (M, 3); got {tuple(r_off.shape)}")
    if r_off.shape[0] != m.shape[0]:
        raise ValueError("m_extras and positions must share the leading "
                         f"dim M (got {m.shape[0]} vs {r_off.shape[0]})")

    M_t = C.M_BASE + m                                              # (M,)
    r_com = (m.unsqueeze(-1) * r_off) / M_t.unsqueeze(-1)            # (M, 3)

    I3 = torch.eye(3, dtype=torch.float32).unsqueeze(0)              # (1, 3, 3)
    I_drone = torch.as_tensor(C.I_DRONE, dtype=torch.float32).unsqueeze(0)

    d = r_com                                                        # (M, 3)
    norm_d2 = (d * d).sum(dim=-1)                                    # (M,)
    outer_d = d.unsqueeze(-1) @ d.unsqueeze(-2)                      # (M, 3, 3)
    I_drone_new = I_drone + C.M_BASE * (norm_d2[:, None, None] * I3 - outer_d)

    s = r_off - r_com                                                # (M, 3)
    norm_s2 = (s * s).sum(dim=-1)
    outer_s = s.unsqueeze(-1) @ s.unsqueeze(-2)
    I_extra_new = m[:, None, None] * (norm_s2[:, None, None] * I3 - outer_s)

    I_total = I_drone_new + I_extra_new                              # (M, 3, 3)
    I_inv = torch.linalg.inv(I_total)
    hover_rpm = torch.sqrt(M_t * C.G / (4.0 * C.KF))                  # (M,)

    return MassParams(
        M_total=M_t, I=I_total, I_inv=I_inv,
        r_com=r_com, hover_rpm=hover_rpm,
        m_extra=float(m.mean()),                                     # representative
        r_offset=tuple(float(v) for v in r_off.mean(dim=0)),
    )


def cat_mass_params(masses: List[MassParams], device=None) -> MassParams:
    """Concatenate already-batched MassParams along the batch dim.

    Each input ``m`` has tensors of shape ``(B_i, ...)``; the result has
    shape ``(sum_i B_i, ...)``. Use this to combine N per-task batched
    MassParams into one big batched MassParams for a single rollout.
    """
    if device is None:
        device = masses[0].M_total.device
    return MassParams(
        M_total=torch.cat([m.M_total.to(device) for m in masses]),
        I=torch.cat([m.I.to(device) for m in masses]),
        I_inv=torch.cat([m.I_inv.to(device) for m in masses]),
        r_com=torch.cat([m.r_com.to(device) for m in masses]),
        hover_rpm=torch.cat([m.hover_rpm.to(device) for m in masses]),
        m_extra=float(masses[0].m_extra),
        r_offset=tuple(masses[0].r_offset),
    )


def batch_mass_params(masses: List[MassParams], B: int,
                      device: str) -> MassParams:
    """Tile N single-task MassParams into one batched block of shape (N*B, ...).

    Row slice [i*B : (i+1)*B] corresponds to ``masses[i]``. Kept as a
    utility for tests and legacy code; the main training loop uses the
    more flexible :func:`cat_mass_params` (per-row masses).
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
