#!/usr/bin/env python3
"""
Train a neural network controller for the Crazyflie 2.x using the real
firmware PID as a differentiable inner loop.

The NN outputs an attitude rate setpoint and a thrust setpoint,
which is then fed through a fully differentiable PyTorch reimplementation of
the CF2.1+ firmware last PID.  End-to-end gradients flow: NN weights →
setpoint → PID states → RPMs → dynamics → trajectory cost.

NN outputs (p, q, r [deg/s], thrust [u16])
→ bypass pos+vel+att PIDs

Physics: nonlinear (default) - full rotation-matrix dynamics, semi-implicit Euler

Usage:
    python train_nn_cf_pid.py
    python train_nn_cf_pid.py --flowdeck --flowdeck_delay_ms 110
    python train_nn_cf_pid.py --epochs 1000 --lr 5e-4 --tag exp1
"""

import argparse
import itertools
import os

import numpy as np
import torch
import torch.nn as nn


# =====================================================================
# 1. Physical Parameters  (CF2X — matches gym-pybullet-drones URDF)
# =====================================================================

M       = 0.027          # mass [kg]
G       = 9.8            # gravity [m/s²]
I_X     = 1.4e-5         # Ixx [kg·m²]
I_Y     = 1.4e-5         # Iyy [kg·m²]
I_Z     = 2.17e-5        # Izz [kg·m²]
L       = 0.0397         # arm length [m]
KF      = 3.16e-10       # thrust coeff [N/RPM²]
KM      = 7.94e-12       # torque coeff [N·m/RPM²]
T2W     = 2.25           # thrust-to-weight ratio

GRAVITY    = M * G
HOVER_RPM  = np.sqrt(GRAVITY / (4 * KF))
MAX_RPM    = np.sqrt((T2W * GRAVITY) / (4 * KF))
F_MAX      = 4 * KF * MAX_RPM**2
TAU_XY_MAX = (2 * L * KF * MAX_RPM**2) / np.sqrt(2)
TAU_Z_MAX  = 2 * KM * MAX_RPM**2

U_MAX = np.array([F_MAX, TAU_XY_MAX, TAU_XY_MAX, TAU_Z_MAX], dtype=np.float32)


# =====================================================================
# 2. CF Firmware PID Gains  (platform_defaults_cf2.h)
# =====================================================================

# --- Rate PIDs ---
PID_ROLL_RATE_KP  = 250.0;  PID_ROLL_RATE_KI  = 500.0;  PID_ROLL_RATE_KD  = 2.5; PID_ROLL_RATE_INTEGRATION_LIMIT  = 33.3
PID_PITCH_RATE_KP = 250.0;  PID_PITCH_RATE_KI = 500.0;  PID_PITCH_RATE_KD = 2.5; PID_PITCH_RATE_INTEGRATION_LIMIT = 33.3
PID_YAW_RATE_KP   = 120.0;  PID_YAW_RATE_KI   = 16.7;   PID_YAW_RATE_KD   = 0.0; PID_YAW_RATE_INTEGRATION_LIMIT   = 166.7

# --- Limits ---
PID_VEL_THRUST_BASE  = 36000.0
PID_VEL_THRUST_MIN   = 20000.0
THRUST_SCALE         = 1000.0
PID_POS_VEL_X_MAX    = 1.0     # m/s
PID_POS_VEL_Y_MAX    = 1.0
PID_POS_VEL_Z_MAX    = 1.0
VEL_MAX_OVERHEAD     = 1.10
RP_LIMIT_OVERHEAD    = 1.10
UINT16_MAX           = 65535.0

# --- Motor conversion ---
CF2_THRUST_MAX_PER_MOTOR = 0.12  # N

# --- Firmware rates ---
ATTITUDE_RATE      = 500       # Hz
ATTITUDE_UPDATE_DT = 1.0 / ATTITUDE_RATE   # 0.002 s

# Motor dynamics filter
MOTOR_TAU = 0.02   # s  (brushed motor time constant)
MOTOR_ALPHA = ATTITUDE_UPDATE_DT / (MOTOR_TAU + ATTITUDE_UPDATE_DT)


# =====================================================================
# 3. State scaling and cost matrices
# =====================================================================

X_MAX    = 2.0;   X_DMAX    = 1.0
Y_MAX    = 2.0;   Y_DMAX    = 1.0
Z_MAX    = 2.0;   Z_DMAX    = 1.0
PHI_MAX  = np.deg2rad(30);  PHI_DMAX  = np.deg2rad(200)
THETA_MAX= np.deg2rad(30);  THETA_DMAX= np.deg2rad(200)
PSI_MAX  = np.deg2rad(45);  PSI_DMAX  = np.deg2rad(120)

X_SCALE = np.array([X_MAX, X_DMAX, Y_MAX, Y_DMAX, Z_MAX, Z_DMAX,
                    PHI_MAX, PHI_DMAX, THETA_MAX, THETA_DMAX,
                    PSI_MAX, PSI_DMAX], dtype=np.float32)

RATE_SCALE      = 720.0  # deg/s

Q_DIAG = np.array([
    1/X_MAX**2,     1/X_DMAX**2,
    1/Y_MAX**2,     1/Y_DMAX**2,
    1/Z_MAX**2,     1/Z_DMAX**2,
    1/PHI_MAX**2,   1/PHI_DMAX**2,
    1/THETA_MAX**2, 1/THETA_DMAX**2,
    1/PSI_MAX**2,   1/PSI_DMAX**2,
], dtype=np.float32)

THRUST_HOVER    = PID_VEL_THRUST_BASE  # = 36000

R_DIAG = np.array([
        1 / RATE_SCALE**2,
        1 / RATE_SCALE**2,
        1 / RATE_SCALE**2,
        1 / THRUST_HOVER**2
    ], dtype=np.float32)



# =====================================================================
# 4. Simulation timing
# =====================================================================
T_SIM      = 4.0                       # simulation duration [s]
INNER_STEPS = int(T_SIM * ATTITUDE_RATE)  # = 2000

print(f"[CF2X] m={M}, Ixx={I_X}, KF={KF}, KM={KM}")
print(f"[CF2X] HOVER_RPM={HOVER_RPM:.1f}, MAX_RPM={MAX_RPM:.1f}")
print(f"[Timing] dt={ATTITUDE_UPDATE_DT*1000:.1f}ms, inner_steps={INNER_STEPS}")


# =====================================================================
# 5. PID State Layout
# =====================================================================
#
# pid_state: (B, 8)
#
#  Indices  PID (integ, prevMeas)
#  [0:2]  pidRollRate
#  [2:4]  pidPitchRate
#  [4:6]  pidYawRate
#  [6]     attitude_desired_yaw  (deg)
#  [7]     actuator_thrust       (uint16)


IDX_PRR  = (0, 2)  # pidRollRate
IDX_PPR  = (2, 4)  # pidPitchRate
IDX_PYR  = (4, 6)  # pidYawRate
IDX_YAW_DES = 6
IDX_THRUST = 7

PID_STATE_DIM = 8

# motor_filter_state: (B, 4)


# =====================================================================
# 6. Differentiable PID Update
# =====================================================================

def pid_update_batch(
    integ: torch.Tensor,        # (B,)
    prev_meas: torch.Tensor,    # (B,)
    measured: torch.Tensor,     # (B,)
    desired: torch.Tensor,      # (B,) or scalar tensor
    kp: float, ki: float, kd: float,
    dt: float,
    i_limit: float,
    output_limit: float = 0.0,
    is_yaw: bool = False,
):
    """Batched single-axis PID update.  Pure function (no in-place ops).

    Returns (output, integ_new, measured_new).
    Derivative acts on measurement (not error) to avoid derivative kick.
    """
    error = desired - measured
    if is_yaw:
        # wrap to ±180 deg
        error = error - 360.0 * torch.round(error / 360.0)

    # P term
    out_p = kp * error

    # D term (on measurement)
    delta = -(measured - prev_meas)
    if is_yaw:
        delta = delta - 360.0 * torch.round(delta / 360.0)
    deriv = delta / dt
    out_d = kd * deriv

    # I term
    integ_new = torch.clamp(integ + error * dt, -i_limit, i_limit)
    out_i = ki * integ_new

    output = out_p + out_d + out_i

    if output_limit > 0.0:
        output = torch.clamp(output, -output_limit, output_limit)

    return output, integ_new, measured


# =====================================================================
# 7. PID State Helpers  (must precede the controller functions)
# =====================================================================

DEFAULT_ILIM = 5000.0  # default integral limit (same as firmware)


def _set_pid(ps: torch.Tensor, idx: tuple, integ_new: torch.Tensor,
             prev_meas_new: torch.Tensor) -> torch.Tensor:
    """Return a new pid_state tensor with updated (integ, prevMeas) for one PID.

    Uses torch.cat to avoid in-place ops that would break autograd.
    """
    i, j = idx
    parts = []
    if i > 0:
        parts.append(ps[:, :i])
    parts.append(integ_new.unsqueeze(1))
    parts.append(prev_meas_new.unsqueeze(1))
    if j < ps.shape[1]:
        parts.append(ps[:, j:])
    return torch.cat(parts, dim=1)


def _set_scalar(ps: torch.Tensor, idx: int,
                value: torch.Tensor) -> torch.Tensor:
    """Return a new pid_state tensor with ps[:, idx] replaced by value."""
    parts = []
    if idx > 0:
        parts.append(ps[:, :idx])
    parts.append(value.unsqueeze(1))
    if idx + 1 < ps.shape[1]:
        parts.append(ps[:, idx+1:])
    return torch.cat(parts, dim=1)


def rate_controller_update(
    pid_state: torch.Tensor,          # (B, 8)
    state_gyro_deg: torch.Tensor,     # (B, 3)  [p, q, r] deg/s
    desired_roll_rate: torch.Tensor,  # (B,)  deg/s
    desired_pitch_rate: torch.Tensor, # (B,)
    desired_yaw_rate: torch.Tensor,   # (B,)
) -> tuple:
    """Run only the rate PID (bypasses attitude PID).

    Returns (roll_cmd, pitch_cmd, yaw_cmd, pid_state_new).
    """
    ps = pid_state.clone()

    gyro_x = state_gyro_deg[:, 0]
    gyro_y = state_gyro_deg[:, 1]
    gyro_z = state_gyro_deg[:, 2]

    roll_cmd, integ_prr, _ = pid_update_batch(
        ps[:, IDX_PRR[0]], ps[:, IDX_PRR[0]+1],
        gyro_x, desired_roll_rate,
        PID_ROLL_RATE_KP, PID_ROLL_RATE_KI, PID_ROLL_RATE_KD,
        ATTITUDE_UPDATE_DT, PID_ROLL_RATE_INTEGRATION_LIMIT)
    ps = _set_pid(ps, IDX_PRR, integ_prr, gyro_x)
    roll_cmd = torch.clamp(roll_cmd, -32767.0, 32767.0)

    pitch_cmd, integ_ppr, _ = pid_update_batch(
        ps[:, IDX_PPR[0]], ps[:, IDX_PPR[0]+1],
        gyro_y, desired_pitch_rate,
        PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD,
        ATTITUDE_UPDATE_DT, PID_PITCH_RATE_INTEGRATION_LIMIT)
    ps = _set_pid(ps, IDX_PPR, integ_ppr, gyro_y)
    pitch_cmd = torch.clamp(pitch_cmd, -32767.0, 32767.0)

    yaw_cmd, integ_pyr, _ = pid_update_batch(
        ps[:, IDX_PYR[0]], ps[:, IDX_PYR[0]+1],
        gyro_z, desired_yaw_rate,
        PID_YAW_RATE_KP, PID_YAW_RATE_KI, PID_YAW_RATE_KD,
        ATTITUDE_UPDATE_DT, PID_YAW_RATE_INTEGRATION_LIMIT)
    ps = _set_pid(ps, IDX_PYR, integ_pyr, gyro_z)
    yaw_cmd = torch.clamp(yaw_cmd, -32767.0, 32767.0)

    yaw_cmd = -yaw_cmd
    return roll_cmd, pitch_cmd, yaw_cmd, ps


# =====================================================================
# 8. Power Distribution  (thrust + roll/pitch/yaw → 4 motor PWMs)
# =====================================================================

def power_distribute(
    thrust: torch.Tensor,     # (B,)  uint16
    roll_cmd: torch.Tensor,   # (B,)  int16
    pitch_cmd: torch.Tensor,  # (B,)
    yaw_cmd: torch.Tensor,    # (B,)
) -> torch.Tensor:            # (B, 4)  PWM in [0, 65535]
    """Differentiable motor mixing (CF2X X-config).

    Mirrors power_distribution_quadrotor.c:84-93 + cap logic.
    """
    r = roll_cmd  / 2.0
    p = pitch_cmd / 2.0

    m1 = thrust - r + p + yaw_cmd
    m2 = thrust - r - p - yaw_cmd
    m3 = thrust + r - p + yaw_cmd
    m4 = thrust + r + p - yaw_cmd

    motor_pwm = torch.stack([m1, m2, m3, m4], dim=-1)  # (B, 4)

    # Cap: shift all down if any exceeds 65535
    highest = motor_pwm.max(dim=-1, keepdim=True).values
    reduction = torch.clamp(highest - UINT16_MAX, min=0.0)
    motor_pwm = torch.clamp(motor_pwm - reduction, min=0.0, max=UINT16_MAX)

    return motor_pwm


# =====================================================================
# 9. PWM → RPM  (differentiable)
# =====================================================================

def pwm_to_rpm_torch(motor_pwm: torch.Tensor) -> torch.Tensor:
    """Convert uint16 PWM → RPM via thrust = (pwm/65535)*0.12 N.

    Skips 8-bit truncation (non-differentiable) for training.
    """
    thrust = (motor_pwm / UINT16_MAX) * CF2_THRUST_MAX_PER_MOTOR
    thrust = torch.clamp(thrust, min=1e-5)
    return torch.sqrt(thrust / KF)


# =====================================================================
# 10. Motor Dynamics Filter  (1st-order LP, differentiable)
# =====================================================================

def motor_filter_step(
    rpm_prev: torch.Tensor,  # (B, 4)
    rpm_cmd: torch.Tensor,   # (B, 4)
) -> torch.Tensor:           # (B, 4)
    return MOTOR_ALPHA * rpm_cmd + (1.0 - MOTOR_ALPHA) * rpm_prev


# =====================================================================
# 11. Nonlinear Dynamics  (reused from train_nn_cf2x.py)
# =====================================================================

def rotation_matrix_zyx(phi, theta, psi):
    """Intrinsic ZYX rotation matrix. Parameters: (B,) tensors."""
    cphi = torch.cos(phi);   sphi = torch.sin(phi)
    cth  = torch.cos(theta); sth  = torch.sin(theta)
    cpsi = torch.cos(psi);   spsi = torch.sin(psi)

    r00 = cth * cpsi
    r01 = sphi * sth * cpsi - cphi * spsi
    r02 = cphi * sth * cpsi + sphi * spsi
    r10 = cth * spsi
    r11 = sphi * sth * spsi + cphi * cpsi
    r12 = cphi * sth * spsi - sphi * cpsi
    r20 = -sth
    r21 = sphi * cth
    r22 = cphi * cth

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)
    return R


def dynamics_substep(state: torch.Tensor, rpm: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Intégration d'Euler semi-implicite.
    Prend directement les RPMs des moteurs en entrée (match exact avec PyBullet DYN).
    state: (B, 12), rpm: (B, 4)
    """
    x     = state[:, 0];  vx = state[:, 1]
    y     = state[:, 2];  vy = state[:, 3]
    z     = state[:, 4];  vz = state[:, 5]
    phi   = state[:, 6];  p  = state[:, 7]
    theta = state[:, 8];  q  = state[:, 9]
    psi   = state[:, 10]; r  = state[:, 11]

    # --- 1. Calcul des forces et couples à partir des RPMs ---
    # Équivalent de: forces = np.array(rpm**2) * self.KF
    forces = (rpm ** 2) * KF
    
    # Équivalent de: z_torques = np.array(rpm**2)*self.KM
    z_torques = (rpm ** 2) * KM

    # Poussée totale (Thrust) sur l'axe Z local
    F = forces.sum(dim=-1)

    # Configuration CF2X (Match exact avec les lignes de PyBullet)
    _a = L / np.sqrt(2)
    tau_x = -(forces[:, 0] + forces[:, 1] - forces[:, 2] - forces[:, 3]) * _a
    tau_y = (-forces[:, 0] + forces[:, 1] + forces[:, 2] - forces[:, 3]) * _a
    tau_z = -z_torques[:, 0] + z_torques[:, 1] - z_torques[:, 2] + z_torques[:, 3]

    # --- 2. Projection de la poussée dans le repère Monde ---
    R = rotation_matrix_zyx(phi, theta, psi)
    # thrust_world_frame = np.dot(rotation, thrust)
    thrust_world = F.unsqueeze(-1) * R[:, :, 2]

    # Accélérations linéaires (force_world_frame / self.M)
    ax = thrust_world[:, 0] / M
    ay = thrust_world[:, 1] / M
    az = thrust_world[:, 2] / M - G

    # --- 3. Couplage gyroscopique ---
    # torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
    gyro_x = (I_Z - I_Y) * q * r
    gyro_y = (I_X - I_Z) * p * r
    gyro_z = (I_Y - I_X) * p * q

    # rpy_rates_deriv = np.dot(self.J_INV, torques)
    p_dot = (tau_x - gyro_x) / I_X
    q_dot = (tau_y - gyro_y) / I_Y
    r_dot = (tau_z - gyro_z) / I_Z

    # --- 4. Intégration d'Euler ---
    # vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
    vx_new = vx + dt * ax;   vy_new = vy + dt * ay;   vz_new = vz + dt * az
    
    # rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
    p_new  = p  + dt * p_dot; q_new = q + dt * q_dot; r_new  = r + dt * r_dot

    # pos = pos + self.PYB_TIMESTEP * vel
    x_new     = x     + dt * vx_new
    y_new     = y     + dt * vy_new
    z_new     = z     + dt * vz_new
    
    # Intégration des angles (Différence PyBullet/PyTorch)
    phi_new   = phi   + dt * p_new
    theta_new = theta + dt * q_new
    psi_new   = psi   + dt * r_new

    return torch.stack([
        x_new, vx_new, y_new, vy_new, z_new, vz_new,
        phi_new, p_new, theta_new, q_new, psi_new, r_new
    ], dim=-1)



# =====================================================================
# 12. Neural Network  (setpoint-mode-specific head)
# =====================================================================

class PolicyMLPPID(nn.Module):
    """MLP controller outputting setpoints for the PID cascade.

    output : (roll_rate_deg_s, pitch_rate_deg_s, yaw_rate_deg_s, thrust_u16)
    """

    def __init__(self, x_scale: np.ndarray, hidden: int = 64):
        super().__init__()

        output_dim = 4

        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(12, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )
        self._init_hover()

    def _init_hover(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

        # Initialize thrust to hover: sigmoid(b) * 65535 = THRUST_HOVER
        ratio = self.THRUST_HOVER / UINT16_MAX
        last.bias.data[3] = float(np.log(ratio / (1.0 - ratio)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = x / self.x_scale
        raw = self.net(x_n)

        # rate_sp
        p_rate = torch.tanh(raw[..., [0]]) * RATE_SCALE
        q_rate = torch.tanh(raw[..., [1]]) * RATE_SCALE
        r_rate = torch.tanh(raw[..., [2]]) * RATE_SCALE
        thr    = torch.sigmoid(raw[..., [3]]) * UINT16_MAX
        return torch.cat([p_rate, q_rate, r_rate, thr], dim=-1)


# =====================================================================
# 13. FlowDeck Delay Buffer  (differentiable)
# =====================================================================

class FlowDeckDelay:
    """Circular delay buffer for position and velocity (runs at ATTITUDE_RATE).

    push() must be called every inner step (500 Hz).
    read_delayed() returns the state from ~delay_ms ms ago.
    Gradients flow through all stored tensors (no in-place ops on leaf data).
    """

    def __init__(self, delay_ms: float, ctrl_freq: int = ATTITUDE_RATE):
        # delay in inner steps (500 Hz)
        self.delay_steps = max(1, round(delay_ms * 1e-3 * ctrl_freq))
        self._buf: list = []   # list of (pos (B,3), vel (B,3)) tuples

    def reset(self, init_pos: torch.Tensor, init_vel: torch.Tensor):
        """Fill buffer with the starting position/velocity."""
        self._buf = [(init_pos.clone(), init_vel.clone())
                     for _ in range(self.delay_steps)]

    def push(self, pos: torch.Tensor, vel: torch.Tensor):
        """Push current (pos, vel) into the buffer (call every inner step)."""
        self._buf.append((pos, vel))
        self._buf.pop(0)   # drop oldest

    def read_delayed(self) -> tuple:
        """Return the delayed (pos, vel) — the oldest entry in the buffer."""
        return self._buf[0]


# =====================================================================
# 14. Yaw accumulator helper
# =====================================================================

def _wrap_deg(x: torch.Tensor) -> torch.Tensor:
    return x - 360.0 * torch.round(x / 360.0)


def _accumulate_yaw(pid_state: torch.Tensor,
                    yaw_rate_deg_s: torch.Tensor) -> torch.Tensor:
    """Update attitude_desired_yaw in pid_state from a yaw rate command."""
    yaw_des = pid_state[:, IDX_YAW_DES]
    yaw_des_new = _wrap_deg(yaw_des + yaw_rate_deg_s * ATTITUDE_UPDATE_DT)
    return _set_scalar(pid_state, IDX_YAW_DES, yaw_des_new)


# =====================================================================
# 15. Main Rollout Function
# =====================================================================

def rollout_pid(
    policy: PolicyMLPPID,
    x0: torch.Tensor,               # (B, 12)
    inner_steps: int = INNER_STEPS,
    use_flowdeck: bool = False,
    flowdeck_delay_ms: float = 110.0,
    use_motor_filter: bool = True
) -> tuple:
    """Roll out the policy through the differentiable PID cascade.

    Returns (X, cost_sum) where:
      X: (B, inner_steps, 12) state trajectory sampled at 500 Hz
      cost_sum: scalar (sum of quadratic costs)
    """
    B = x0.shape[0]
    dev = x0.device

    # --- Initialize PID state ---
    pid_state = torch.zeros(B, PID_STATE_DIM, device=dev, dtype=torch.float32)

    # Initial thrust = hover
    pid_state = _set_scalar(
        pid_state, IDX_THRUST,
        torch.full((B,), PID_VEL_THRUST_BASE, device=dev))

    # attitude_desired_yaw = initial yaw (in degrees)
    init_yaw_deg = torch.rad2deg(x0[:, 10])
    pid_state = _set_scalar(pid_state, IDX_YAW_DES, init_yaw_deg)

    # --- Motor filter state ---
    motor_rpm = torch.full((B, 4), HOVER_RPM, device=dev, dtype=torch.float32)

    # --- FlowDeck delay buffer ---
    flowdeck = None
    if use_flowdeck:
        flowdeck = FlowDeckDelay(flowdeck_delay_ms)
        init_pos = x0[:, [0, 2, 4]]   # (B, 3)
        init_vel = x0[:, [1, 3, 5]]   # (B, 3)
        flowdeck.reset(init_pos, init_vel)

    # --- matrices for cost ---
    Q = torch.tensor(Q_DIAG, dtype=torch.float32, device=dev)
    R = torch.tensor(R_DIAG, dtype=torch.float32, device=dev)

    state = x0.clone()
    cost_sum = torch.zeros(1, device=dev)

    # Storage for sampled trajectory
    n_samples = inner_steps
    X = torch.zeros(B, n_samples, 12, device=dev)
    sample_idx = 0

    # Cached setpoints (updated every step)
    # Initialized to values that produce hover
    thrust_cached   = torch.full((B,), PID_VEL_THRUST_BASE, device=dev)

    # For att_sp / rate_sp: yaw_rate and rates cached
    yaw_rate_cached  = torch.zeros(B, device=dev)
    rate_r_cached    = torch.zeros(B, device=dev)
    rate_p_cached    = torch.zeros(B, device=dev)

    for k in range(inner_steps):

        # Extract state components
        pos   = state[:, [0, 2, 4]]              # (B, 3)
        vel   = state[:, [1, 3, 5]]              # (B, 3)
        phi   = state[:, 6];   p_rate = state[:, 7]
        theta = state[:, 8];   q_rate = state[:, 9]
        psi   = state[:, 10];  r_rate = state[:, 11]

        gyro_deg = torch.stack([
            torch.rad2deg(p_rate), torch.rad2deg(q_rate), torch.rad2deg(r_rate)
        ], dim=-1)

        # ---- FlowDeck push (every inner step = 500 Hz) ----
        if use_flowdeck:
            flowdeck.push(pos, vel)

        # =========================================================
        # INFERENCE DU NN
        # =========================================================
        nn_out = policy(state) # Appelé à chaque itération k
        rate_r_cached   = nn_out[:, 0]
        rate_p_cached   = nn_out[:, 1]
        yaw_rate_cached = nn_out[:, 2]
        thrust_cached   = nn_out[:, 3]
        pid_state = _set_scalar(pid_state, IDX_THRUST, thrust_cached)

        # ---- Yaw accumulation (every inner step, from yaw_rate) ----
        pid_state = _accumulate_yaw(pid_state, yaw_rate_cached)

        # ---- Attitude + Rate PIDs (every inner step = 500 Hz) ----
        thrust = pid_state[:, IDX_THRUST]

        roll_cmd, pitch_cmd, yaw_cmd, pid_state = rate_controller_update(pid_state,
                                                                         gyro_deg,
                                                                         rate_r_cached, 
                                                                         rate_p_cached,
                                                                         yaw_rate_cached)

        # ---- Power distribution ----
        motor_pwm = power_distribute(thrust, roll_cmd, pitch_cmd, yaw_cmd)

        # ---- PWM → RPM ----
        rpm_cmd = pwm_to_rpm_torch(motor_pwm)

        # ---- Motor dynamics filter ----
        if use_motor_filter:
            motor_rpm = motor_filter_step(motor_rpm, rpm_cmd)
        else:
            motor_rpm = rpm_cmd

        # ---- Physics step ----
        state = dynamics_substep(state, motor_rpm, ATTITUDE_UPDATE_DT)

        X[:, sample_idx, :] = state
        cost_state = (state ** 2 * Q).sum(dim=1).mean()
        cost_input = (nn_out ** 2 * R).sum(dim=1).mean()
        cost_step = cost_state + cost_input
        cost_sum = cost_sum + cost_step
        sample_idx += 1

    return X, cost_sum


# =====================================================================
# 16. Training Utilities
# =====================================================================

def make_x0_batch(xyz_list, device="cpu"):
    X0 = torch.zeros(len(xyz_list), 12, dtype=torch.float32, device=device)
    for b, (x, y, z) in enumerate(xyz_list):
        X0[b, 0] = x
        X0[b, 2] = y
        X0[b, 4] = z
    return X0


def generate_cube_points(half_side=0.3):
    vals = [-half_side, 0.0, half_side]
    return list(itertools.product(vals, repeat=3))


def build_run_name(setpoint_mode, linearized, epochs, lr, hidden,
                   flowdeck, tag=""):
    dyn   = "linear" if linearized else "nonlinear"
    fd    = "_flowdeck" if flowdeck else ""
    lr_str = f"{lr:.0e}" if lr < 1e-2 else str(lr).replace(".", "p")
    name  = f"cf_pid_{setpoint_mode}_{dyn}_h{hidden}_ep{epochs}_lr{lr_str}{fd}"
    if tag:
        name += f"_{tag.replace(' ', '_')}"
    return name


# =====================================================================
# 17. Training Loop
# =====================================================================

def train(
    epochs: int = 500,
    lr: float = 1e-3,
    hidden: int = 64,
    use_flowdeck: bool = False,
    flowdeck_delay_ms: float = 110.0,
    use_motor_filter: bool = True,
    half_side: float = 0.3,
    t_sim: float = T_SIM,
    device: str = "cpu",
):
    inner_steps = int(t_sim * ATTITUDE_RATE)

    policy = PolicyMLPPID(X_SCALE, hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    cube_pts = generate_cube_points(half_side)
    x0_batch = make_x0_batch(cube_pts, device=device)

    # Verify hover init
    with torch.no_grad():
        nn_test = policy(torch.zeros(1, 12, device=device))
        print(f"\n[Init] NN output at x=0: {nn_test[0].cpu().numpy()}")

    fd_label  = f" + FlowDeck({flowdeck_delay_ms}ms)" if use_flowdeck else ""
    print(f"[Train] {len(cube_pts)} pts, epochs={epochs}, lr={lr},"
          f"t_sim={t_sim}s ({inner_steps} inner steps)")

    for ep in range(epochs):
        X, cost = rollout_pid(
            policy, x0_batch,
            inner_steps=inner_steps,
            use_flowdeck=use_flowdeck,
            flowdeck_delay_ms=flowdeck_delay_ms,
            use_motor_filter=use_motor_filter
        )

        opt.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        scheduler.step()

        with torch.no_grad():
            pos_end = X[:, -1, [0, 2, 4]].norm(dim=1)
            print(f"  [ep {ep+1:4d}/{epochs}]  cost={cost.item():.4e}  "
                    f"|x_T|_mean={pos_end.mean().item():.4f}  "
                    f"|x_T|_max={pos_end.max().item():.4f}")

    return policy


# =====================================================================
# 18. Main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--flowdeck", action="store_true",
                        help="Simulate FlowDeck delay during training")
    parser.add_argument("--flowdeck_delay_ms", type=float, default=110.0)
    parser.add_argument("--no_motor_filter", action="store_true",
                        help="Disable motor dynamics filter")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--t_sim", type=float, default=T_SIM)
    parser.add_argument("--half_side", type=float, default=0.3)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device}")

    policy = train(
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        use_flowdeck=args.flowdeck,
        flowdeck_delay_ms=args.flowdeck_delay_ms,
        use_motor_filter=not args.no_motor_filter,
        half_side=args.half_side,
        t_sim=args.t_sim,
        device=device,
    )

    # Save
    out_dir  = os.path.dirname(os.path.abspath(__file__))
    run_name = build_run_name(args.epochs, args.lr, args.hidden,
                              args.flowdeck, args.tag)
    path_full = os.path.join(out_dir, f"trained_policy_{run_name}.pt")
    path_dict = os.path.join(out_dir, f"trained_weights_{run_name}.pt")

    torch.save(policy.cpu(), path_full)
    torch.save({
        "state_dict":       policy.state_dict(),
        "hidden":           args.hidden,
        "x_scale":          X_SCALE,
        "ctrl_freq":        ATTITUDE_RATE,
        "Ts":               ATTITUDE_UPDATE_DT,
        "KF": KF, "KM": KM, "L": L, "M": M, "G": G,
        "MAX_RPM":          MAX_RPM,
        "use_flowdeck":     args.flowdeck,
        "flowdeck_delay_ms": args.flowdeck_delay_ms,
        "use_motor_filter": not args.no_motor_filter,
        "pid_gains": {
            "roll_rate":  (PID_ROLL_RATE_KP,  PID_ROLL_RATE_KI,  PID_ROLL_RATE_KD),
            "pitch_rate": (PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD),
            "yaw_rate":   (PID_YAW_RATE_KP,   PID_YAW_RATE_KI,   PID_YAW_RATE_KD)
        },
        "run_name":         run_name,
        "epochs":           args.epochs,
        "lr":               args.lr,
    }, path_dict)

    print(f"\n[Saved] {path_full}")
    print(f"[Saved] {path_dict}")
    print(f"\nDone! Run  validate_nn_cf_pid_pybullet.py --weights {os.path.basename(path_dict)}  to validate.")
