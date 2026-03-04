"""
Goffin Dynamics Model
=====================
Implementation of the quadcopter dynamics from:
    Cyril Goffin, "Meta-Learning for Quadcopter Control", DECODE Semester Project,
    EPFL, January 2026. (Equations 2.1 - 2.19)

Supports:
    - Plus (+) and Cross (x) motor configurations
    - Full nonlinear dynamics (eq 2.7-2.12)
    - Linearized dynamics around hover (eq 2.13-2.18)
    - Forward Euler integration (eq 2.21)

Motor numbering follows gym-pybullet-drones convention (0-indexed).
"""

import numpy as np


class GoffinDynamics:
    """
    Quadcopter dynamics from the Goffin DECODE report.

    State vector (12-dimensional):
        x = [x, xd, y, yd, z, zd, phi, phid, theta, thetad, psi, psid]^T
             0   1   2   3  4   5   6     7     8      9     10    11

    Input: internally converts RPMs -> [F, tau_x, tau_y, tau_z]
    """

    # State indices
    IX, IXD = 0, 1
    IY, IYD = 2, 3
    IZ, IZD = 4, 5
    IPHI, IPHID = 6, 7
    ITHETA, ITHETAD = 8, 9
    IPSI, IPSID = 10, 11

    def __init__(self, m, g, Ix, Iy, Iz, l, kf, km, config='cross', dt=1/240):
        self.m = m
        self.g = g
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.l = l
        self.kf = kf
        self.km = km
        self.config = config
        self.dt = dt
        self._l_over_sqrt2 = l / np.sqrt(2)

    # ==================================================================
    #  RPM -> Wrench (configuration-dependent)
    # ==================================================================

    def rpms_to_wrench_plus(self, rpms):
        """
        + CONFIGURATION (CF2P)

        Motor layout (gym-pybullet-drones, top view):

                Motor 2 (+xB)
                    |
            Motor 3 --*-- Motor 1
             (-yB)          (+yB)
                    |
                Motor 0 (-xB)

        Torques (from gym-pybullet-drones _dynamics, CF2P):
            tau_x = L * (F1 - F3)
            tau_y = L * (-F0 + F2)
            tau_z = -t0 + t1 - t2 + t3    where ti = km * RPMi^2
        """
        F0 = self.kf * rpms[0]**2
        F1 = self.kf * rpms[1]**2
        F2 = self.kf * rpms[2]**2
        F3 = self.kf * rpms[3]**2

        t0 = self.km * rpms[0]**2
        t1 = self.km * rpms[1]**2
        t2 = self.km * rpms[2]**2
        t3 = self.km * rpms[3]**2

        F_total = F0 + F1 + F2 + F3
        tau_x = self.l * (F1 - F3)
        tau_y = self.l * (-F0 + F2)
        tau_z = -t0 + t1 - t2 + t3

        return np.array([F_total, tau_x, tau_y, tau_z])

    def rpms_to_wrench_cross(self, rpms):
        """
        x CONFIGURATION (CF2X)

        Motor layout (gym-pybullet-drones, top view):

            Motor 0           Motor 1
          (front-left)     (front-right)
                \\             /
                 \\           /
                  *---------*
                 /           \\
                /             \\
            Motor 3           Motor 2
          (back-left)      (back-right)

        Effective lever arm: L/sqrt(2)

        Torques (from gym-pybullet-drones _dynamics, CF2X):
            tau_x = -(F0 + F1 - F2 - F3) * L/sqrt(2)
            tau_y = (-F0 + F1 + F2 - F3) * L/sqrt(2)
            tau_z = -t0 + t1 - t2 + t3
        """
        F0 = self.kf * rpms[0]**2
        F1 = self.kf * rpms[1]**2
        F2 = self.kf * rpms[2]**2
        F3 = self.kf * rpms[3]**2

        t0 = self.km * rpms[0]**2
        t1 = self.km * rpms[1]**2
        t2 = self.km * rpms[2]**2
        t3 = self.km * rpms[3]**2

        F_total = F0 + F1 + F2 + F3
        tau_x = -(F0 + F1 - F2 - F3) * self._l_over_sqrt2
        tau_y = (-F0 + F1 + F2 - F3) * self._l_over_sqrt2
        tau_z = -t0 + t1 - t2 + t3

        return np.array([F_total, tau_x, tau_y, tau_z])

    def rpms_to_wrench(self, rpms):
        if self.config == 'plus':
            return self.rpms_to_wrench_plus(rpms)
        elif self.config == 'cross':
            return self.rpms_to_wrench_cross(rpms)
        else:
            raise ValueError(f"Unknown config '{self.config}'")

    # ==================================================================
    #  Nonlinear continuous dynamics  (Goffin eq 2.7-2.12)
    # ==================================================================

    def nonlinear_continuous(self, state, rpms):
        """
        Full nonlinear dynamics from Goffin eq 2.7-2.12.

        Translational:
            m*x_dd = F*(cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))    (2.7)
            m*y_dd = F*(cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))    (2.8)
            m*z_dd = F*cos(phi)*cos(theta) - m*g                             (2.9)

        Rotational (Euler's equations with gyroscopic coupling):
            Ix*phi_dd   = tau_x + (Iy - Iz)*thetad*psid                     (2.10)
            Iy*theta_dd = tau_y + (Iz - Ix)*phid*psid                        (2.11)
            Iz*psi_dd   = tau_z + (Ix - Iy)*phid*thetad                      (2.12)
        """
        phi    = state[self.IPHI]
        phid   = state[self.IPHID]
        theta  = state[self.ITHETA]
        thetad = state[self.ITHETAD]
        psi    = state[self.IPSI]
        psid   = state[self.IPSID]

        F, tau_x, tau_y, tau_z = self.rpms_to_wrench(rpms)

        c_phi, s_phi = np.cos(phi), np.sin(phi)
        c_theta, s_theta = np.cos(theta), np.sin(theta)
        c_psi, s_psi = np.cos(psi), np.sin(psi)

        F_over_m = F / self.m

        # --- Translational (eq 2.7-2.9) ---
        x_dd = F_over_m * (c_phi * s_theta * c_psi + s_phi * s_psi)        # (2.7)
        y_dd = F_over_m * (c_phi * s_theta * s_psi - s_phi * c_psi)        # (2.8)
        z_dd = F_over_m * c_phi * c_theta - self.g                         # (2.9)

        # --- Rotational (eq 2.10-2.12) ---
        phi_dd   = (tau_x + (self.Iy - self.Iz) * thetad * psid) / self.Ix   # (2.10)
        theta_dd = (tau_y + (self.Iz - self.Ix) * phid   * psid) / self.Iy   # (2.11)
        psi_dd   = (tau_z + (self.Ix - self.Iy) * phid   * thetad) / self.Iz # (2.12)

        dxdt = np.zeros(12)
        dxdt[self.IX]      = state[self.IXD]
        dxdt[self.IXD]     = x_dd
        dxdt[self.IY]      = state[self.IYD]
        dxdt[self.IYD]     = y_dd
        dxdt[self.IZ]      = state[self.IZD]
        dxdt[self.IZD]     = z_dd
        dxdt[self.IPHI]    = phid
        dxdt[self.IPHID]   = phi_dd
        dxdt[self.ITHETA]  = thetad
        dxdt[self.ITHETAD] = theta_dd
        dxdt[self.IPSI]    = psid
        dxdt[self.IPSID]   = psi_dd
        return dxdt

    # ==================================================================
    #  Linearized continuous dynamics  (Goffin eq 2.13-2.18)
    # ==================================================================

    def linearized_continuous(self, state, rpms):
        """
        Linearized dynamics around hover (Goffin eq 2.13-2.18).

        Assumptions: small angles, psi=0, F/m ~ g at hover.

            x_dd     = g * theta                    (2.13)
            y_dd     = g * phi                      (2.14, Goffin/Ahmad convention)
            z_dd     = F/m - g                      (2.15)
            phi_dd   = tau_x / Ix                   (2.16)
            theta_dd = tau_y / Iy                   (2.17)
            psi_dd   = tau_z / Iz                   (2.18)
        """
        phi    = state[self.IPHI]
        phid   = state[self.IPHID]
        theta  = state[self.ITHETA]
        thetad = state[self.ITHETAD]
        psid   = state[self.IPSID]

        F, tau_x, tau_y, tau_z = self.rpms_to_wrench(rpms)

        # --- Linearized translational ---
        x_dd = self.g * theta                     # (2.13)
        y_dd = self.g * phi                       # (2.14)
        z_dd = F / self.m - self.g                # (2.15)

        # --- Linearized rotational ---
        phi_dd   = tau_x / self.Ix                # (2.16)
        theta_dd = tau_y / self.Iy                # (2.17)
        psi_dd   = tau_z / self.Iz                # (2.18)

        dxdt = np.zeros(12)
        dxdt[self.IX]      = state[self.IXD]
        dxdt[self.IXD]     = x_dd
        dxdt[self.IY]      = state[self.IYD]
        dxdt[self.IYD]     = y_dd
        dxdt[self.IZ]      = state[self.IZD]
        dxdt[self.IZD]     = z_dd
        dxdt[self.IPHI]    = phid
        dxdt[self.IPHID]   = phi_dd
        dxdt[self.ITHETA]  = thetad
        dxdt[self.ITHETAD] = theta_dd
        dxdt[self.IPSI]    = psid
        dxdt[self.IPSID]   = psi_dd
        return dxdt

    # ==================================================================
    #  Integration (Forward Euler, Goffin eq 2.21)
    # ==================================================================

    def step(self, state, rpms, model='nonlinear'):
        """
        x_{k+1} = x_k + dt * f(x_k, u_k)     (eq 2.21)
        """
        if model == 'nonlinear':
            dxdt = self.nonlinear_continuous(state, rpms)
        elif model == 'linearized':
            dxdt = self.linearized_continuous(state, rpms)
        else:
            raise ValueError(f"Unknown model '{model}'")
        return state + self.dt * dxdt


# ======================================================================
#  State conversion utilities
# ======================================================================

def gym_state_to_goffin(pos, rpy, vel, ang_v):
    """
    Convert gym-pybullet-drones state to Goffin's 12-dim state.

    ang_v from getBaseVelocity() is in WORLD frame.
    Goffin uses Euler angle rates (phi_dot, theta_dot, psi_dot).

    Conversion:
        1. world omega -> body omega:  omega_body = R^T @ ang_v
        2. body omega  -> Euler rates: via kinematic matrix
    """
    phi, theta, psi = rpy
    c_phi, s_phi = np.cos(phi), np.sin(phi)
    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_psi, s_psi = np.cos(psi), np.sin(psi)

    # Rotation matrix body->world (eq 2.1)
    R = np.array([
        [c_theta*c_psi,  s_phi*s_theta*c_psi - c_phi*s_psi,  c_phi*s_theta*c_psi + s_phi*s_psi],
        [c_theta*s_psi,  s_phi*s_theta*s_psi + c_phi*c_psi,  c_phi*s_theta*s_psi - s_phi*c_psi],
        [-s_theta,       s_phi*c_theta,                       c_phi*c_theta                    ]
    ])

    # World -> body angular velocity
    omega_body = R.T @ ang_v  # [p, q, r]

    # Body rates -> Euler rates
    #if np.abs(c_theta) > 1e-8:
    #    t_theta = s_theta / c_theta
    #    phi_dot   = omega_body[0] + s_phi * t_theta * omega_body[1] + c_phi * t_theta * omega_body[2]
    #    theta_dot = c_phi * omega_body[1] - s_phi * omega_body[2]
    #    psi_dot   = (s_phi / c_theta) * omega_body[1] + (c_phi / c_theta) * omega_body[2]
    #else:
    #    phi_dot, theta_dot, psi_dot = omega_body

    return np.array([
        pos[0], vel[0],
        pos[1], vel[1],
        pos[2], vel[2],
        phi,    omega_body[0],
        theta,  omega_body[1],
        psi,    omega_body[2]
    ])


def gym_20dim_to_goffin(state_20):
    """
    Convert _getDroneStateVector() 20-dim to Goffin 12-dim.

    Layout: [pos(3), quat(4), rpy(3), vel(3), ang_v(3), rpms(4)]
    """
    return gym_state_to_goffin(
        pos=state_20[0:3],
        rpy=state_20[7:10],
        vel=state_20[10:13],
        ang_v=state_20[13:16]
    )


def goffin_state_labels():
    return [
        'x (m)', 'x_dot (m/s)',
        'y (m)', 'y_dot (m/s)',
        'z (m)', 'z_dot (m/s)',
        'phi (rad)', 'phi_dot (rad/s)',
        'theta (rad)', 'theta_dot (rad/s)',
        'psi (rad)', 'psi_dot (rad/s)',
    ]