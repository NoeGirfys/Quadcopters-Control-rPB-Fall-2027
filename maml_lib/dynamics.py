"""Nonlinear and linearized rigid-body dynamics, mass-task aware.

Both step functions share the same signature so the rollout can swap them
freely::

    state_new = dynamics_step(state, rpm, dt, mass)

with shapes ``state (B,12)``, ``rpm (B,4)``, scalar ``dt``, and a
``MassParams`` whose tensors have a leading task dim of 1.

Mass effects (offset point mass) modelled exactly via ``MassParams``:
  * total mass M_total used in translational acceleration
  * full inertia matrix I (Steiner-corrected) used in torque -> ang accel
  * CoM offset r_com adds an extra arm to every per-motor force, which
    yields the gravity-induced torque bias when the drone is level.

The per-motor thrust mapping ``F_i = KF * rpm_i^2`` is kept exact in both
modes; only the rotational kinematics differ:

  nonlinear  : full ZYX rotation matrix and ω×Iω gyroscopic coupling
  linearized : small-angle  R z ≈ (theta, -phi, 1), no gyroscopic coupling
"""
import torch

from . import config as C
from .mass_params import MassParams

_MOTOR_POS_T = torch.tensor(C.MOTOR_POS)        # (4, 3)
_YAW_SIGNS_T = torch.tensor(C.YAW_SIGNS)        # (4,)

"""
_a = L / math.sqrt(2)
MOTOR_POS = np.array([
    [+_a, -_a, 0.0],   # M1 front-right
    [-_a, -_a, 0.0],   # M2 rear-right
    [-_a, +_a, 0.0],   # M3 rear-left
    [+_a, +_a, 0.0],   # M4 front-left
], dtype=np.float32)
"""
def _torques_body(F_i: torch.Tensor, M_yaw_i: torch.Tensor,
                  r_com: torch.Tensor) -> torch.Tensor:
    """Body torque about new CoM. F_i (B,4), r_com (1,3) -> (B,3)."""
    motor_pos = _MOTOR_POS_T.to(F_i.device)                            # (4, 3)
    # lever arm from new CoM to each motor:  (1,4,3) - (1,1,3) -> (1,4,3)
    lever = motor_pos.unsqueeze(0) - r_com.unsqueeze(1)                # (1, 4, 3)
    # p × F with F = (0, 0, F_z):   (py F_z, -px F_z, 0)
    tau_x = (lever[..., 1] * F_i).sum(dim=-1)                          # (B,)
    tau_y = (-lever[..., 0] * F_i).sum(dim=-1)
    tau_z = M_yaw_i.sum(dim=-1)
    return torch.stack([tau_x, tau_y, tau_z], dim=-1)                  # (B, 3)


def nonlinear_step(state: torch.Tensor, rpm: torch.Tensor,
                   dt: float, mass: MassParams) -> torch.Tensor:
    """Full nonlinear semi-implicit Euler step, mass-task aware."""
    x, vx       = state[:, 0], state[:, 1]
    y, vy       = state[:, 2], state[:, 3]
    z, vz       = state[:, 4], state[:, 5]
    phi, p      = state[:, 6], state[:, 7]
    theta, q    = state[:, 8], state[:, 9]
    psi, r      = state[:, 10], state[:, 11]

    yaw_signs = _YAW_SIGNS_T.to(rpm.device) # = (-1, +1, -1, +1)
    F_i      = (rpm ** 2) * C.KF
    M_yaw_i  = (rpm ** 2) * C.KM * yaw_signs
    F_total  = F_i.sum(dim=-1)                                         # (B,)

    # World-frame thrust: third column of R_zyx times F_total.
    cphi, sphi = torch.cos(phi), torch.sin(phi)
    cth,  sth  = torch.cos(theta), torch.sin(theta)
    cpsi, spsi = torch.cos(psi), torch.sin(psi)
    R_z = torch.stack([
        cphi * sth * cpsi + sphi * spsi,
        cphi * sth * spsi - sphi * cpsi,
        cphi * cth,
    ], dim=-1)                                                         # (B, 3)

    a_world = F_total.unsqueeze(-1) * R_z / mass.M_total.unsqueeze(-1) # (B,3)
    ax = a_world[:, 0]
    ay = a_world[:, 1]
    az = a_world[:, 2] - C.G

    # Rotational dynamics (full I matrix, gyroscopic coupling included).
    tau = _torques_body(F_i, M_yaw_i, mass.r_com)                      # (B, 3)
    omega = torch.stack([p, q, r], dim=-1)                             # (B, 3)
    Iomega = (mass.I @ omega.unsqueeze(-1)).squeeze(-1)                # (B, 3)
    gyro = torch.cross(omega, Iomega, dim=-1)
    omega_dot = (mass.I_inv @ (tau - gyro).unsqueeze(-1)).squeeze(-1)  # (B, 3)
    p_dot, q_dot, r_dot = omega_dot[:, 0], omega_dot[:, 1], omega_dot[:, 2]

    # Semi-implicit Euler integration.
    vx_n = vx + dt * ax;  vy_n = vy + dt * ay;  vz_n = vz + dt * az
    p_n  = p + dt * p_dot;  q_n = q + dt * q_dot;  r_n = r + dt * r_dot
    x_n     = x     + dt * vx_n;  y_n     = y     + dt * vy_n;  z_n   = z   + dt * vz_n
    phi_n   = phi   + dt * p_n;   theta_n = theta + dt * q_n;   psi_n = psi + dt * r_n

    return torch.stack([x_n, vx_n, y_n, vy_n, z_n, vz_n,
                        phi_n, p_n, theta_n, q_n, psi_n, r_n], dim=-1)


def linearized_step(state: torch.Tensor, rpm: torch.Tensor,
                    dt: float, mass: MassParams) -> torch.Tensor:
    """Hover-linearized semi-implicit Euler step, mass-task aware.

    Linearization only on the rotational kinematics:
      - small-angle rotation matrix:  R z ≈ (theta, -phi, 1)
      - no  ω × Iω  gyroscopic coupling

    The thrust mapping F_i = KF * rpm_i^2 stays exact, and torques are
    still computed about the (offset) new CoM, so the gravity-induced
    bias on a level drone with offset mass is preserved.
    """
    x, vx       = state[:, 0], state[:, 1]
    y, vy       = state[:, 2], state[:, 3]
    z, vz       = state[:, 4], state[:, 5]
    phi, p      = state[:, 6], state[:, 7]
    theta, q    = state[:, 8], state[:, 9]
    psi, r      = state[:, 10], state[:, 11]

    yaw_signs = _YAW_SIGNS_T.to(rpm.device)
    F_i      = (rpm ** 2) * C.KF
    M_yaw_i  = (rpm ** 2) * C.KM * yaw_signs
    F_total  = F_i.sum(dim=-1)

    M_t = mass.M_total                                                 # (1,)
    ax = F_total * theta / M_t
    ay = -F_total * phi / M_t
    az = F_total / M_t - C.G

    tau = _torques_body(F_i, M_yaw_i, mass.r_com)
    omega_dot = (mass.I_inv @ tau.unsqueeze(-1)).squeeze(-1)
    p_dot, q_dot, r_dot = omega_dot[:, 0], omega_dot[:, 1], omega_dot[:, 2]

    vx_n = vx + dt * ax;  vy_n = vy + dt * ay;  vz_n = vz + dt * az
    p_n  = p + dt * p_dot;  q_n = q + dt * q_dot;  r_n = r + dt * r_dot
    x_n     = x     + dt * vx_n;  y_n     = y     + dt * vy_n;  z_n   = z   + dt * vz_n
    phi_n   = phi   + dt * p_n;   theta_n = theta + dt * q_n;   psi_n = psi + dt * r_n

    return torch.stack([x_n, vx_n, y_n, vy_n, z_n, vz_n,
                        phi_n, p_n, theta_n, q_n, psi_n, r_n], dim=-1)


# Registry for CLI selection.
DYNAMICS = {
    "nonlinear":  nonlinear_step,
    "linearized": linearized_step,
}
