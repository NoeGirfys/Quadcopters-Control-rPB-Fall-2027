#!/usr/bin/env python3
"""
Train a concurrent NN high-level controller for Crazyflie 2.x.

Implements Analytic Policy Gradient (APG) from Wiedemann et al. (ICRA 2023).
The NN replaces the position + velocity PIDs and is queried ONCE per chunk,
outputting all T future actions concurrently.

Pipeline (per 500 Hz step):
  NN action (100 Hz) -> Attitude PID (500 Hz) -> Rate PID (500 Hz)
  -> Power Distribution -> PWM -> Wrench -> Nonlinear Dynamics

- Attitude + Rate PIDs: exact CF2.1+ firmware gains (cf_firmware_pid_sim.py)
- Dynamics: pybullet-drones DYN mode (train_nn_cf2x.py), semi-implicit Euler
- Curriculum learning: state reset + detach when error > tau_div

Usage:
    cd training_regulation_simu_and_real
    python train_nn_cf_pid.py
    python train_nn_cf_pid.py --epochs 1000 --lr 5e-4 --t_sim 4.0
    python train_nn_cf_pid.py --t_chunk 0.4 --hidden 128 --tag exp1
"""

import argparse
import itertools
import math

import numpy as np
import torch
import torch.nn as nn

import sys
import os
# On ajoute le dossier parent (Semester_Project) au sys.path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

# =====================================================================
# 1. Physical Constants  (CF2X — from gym-pybullet-drones URDF)
# =====================================================================

M   = 0.027        # mass [kg]
G   = 9.8          # gravity [m/s^2]
I_X = 1.4e-5       # Ixx [kg*m^2]
I_Y = 1.4e-5       # Iyy
I_Z = 2.17e-5      # Izz
L   = 0.0397       # arm length [m]
KF  = 3.16e-10     # thrust coeff [N/RPM^2]
KM  = 7.94e-12     # torque coeff [N*m/RPM^2]
T2W = 2.25

GRAVITY   = M * G
HOVER_RPM = np.sqrt(GRAVITY / (4 * KF))
MAX_RPM   = np.sqrt((T2W * GRAVITY) / (4 * KF))
from crazyflie_firmware.constants import UINT16_MAX

# CF2.1+ per-motor thrust with battery compensation
CF2_THRUST_MAX_PER_MOTOR = 0.12  # N

# ---- PID Gains (from cf_firmware_pid_sim.py / platform_defaults_cf2.h) ----
from crazyflie_firmware.constants import (
    PID_ROLL_KP, PID_ROLL_KI, PID_ROLL_KD, PID_ROLL_INTEGRATION_LIMIT,
    PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD, PID_PITCH_INTEGRATION_LIMIT,
    PID_YAW_KP, PID_YAW_KI, PID_YAW_KD, PID_YAW_INTEGRATION_LIMIT,

    PID_ROLL_RATE_KP, PID_ROLL_RATE_KI, PID_ROLL_RATE_KD, PID_ROLL_RATE_INTEGRATION_LIMIT,
    PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD, PID_PITCH_RATE_INTEGRATION_LIMIT,
    PID_YAW_RATE_KP, PID_YAW_RATE_KI, PID_YAW_RATE_KD, PID_YAW_RATE_INTEGRATION_LIMIT,
)

# Attitude PID
ATT_KP   = torch.tensor([PID_ROLL_KP, PID_PITCH_KP, PID_YAW_KP])    # roll, pitch, yaw
ATT_KI   = torch.tensor([PID_ROLL_KI, PID_PITCH_KI, PID_YAW_KI])
ATT_KD   = torch.tensor([PID_ROLL_KD, PID_PITCH_KD, PID_YAW_KD])
ATT_ILIM = torch.tensor([PID_ROLL_INTEGRATION_LIMIT, PID_PITCH_INTEGRATION_LIMIT, PID_YAW_INTEGRATION_LIMIT])

# Rate PID
RATE_KP   = torch.tensor([PID_ROLL_RATE_KP, PID_PITCH_RATE_KP, PID_YAW_RATE_KP])
RATE_KI   = torch.tensor([PID_ROLL_RATE_KI, PID_PITCH_RATE_KI, PID_YAW_RATE_KI])
RATE_KD   = torch.tensor([PID_ROLL_RATE_KD, PID_PITCH_RATE_KD, PID_YAW_RATE_KD])
RATE_ILIM = torch.tensor([PID_ROLL_RATE_INTEGRATION_LIMIT, PID_PITCH_RATE_INTEGRATION_LIMIT, PID_YAW_RATE_INTEGRATION_LIMIT])

# Velocity PID output limits (NN output bounds)
from crazyflie_firmware.constants import PID_VEL_ROLL_MAX, PID_VEL_PITCH_MAX
YAW_RATE_MAX      = 200.0  # deg/s

# ---- Timing ----
from crazyflie_firmware.constants import POSITION_RATE, ATTITUDE_RATE, ATTITUDE_UPDATE_DT
NN_FREQ       = POSITION_RATE    # Hz (replaces position controller)
PID_STEPS_PER_NN = ATTITUDE_RATE // NN_FREQ  # 5

# ---- Constantes du filtre moteur ----
MOTOR_TAU = 0.02  # 20ms de délai mécanique/électrique
MOTOR_ALPHA = ATTITUDE_UPDATE_DT / (MOTOR_TAU + ATTITUDE_UPDATE_DT)

# ---- State scaling (same as train_nn_cf2x.py) ----
X_SCALE = np.array([
    1.0, 1.0,                            # x, vx
    1.0, 1.0,                            # y, vy
    1.0, 1.0,                            # z, vz
    np.deg2rad(30), np.deg2rad(200),     # phi, p
    np.deg2rad(30), np.deg2rad(200),     # theta, q
    np.deg2rad(45), np.deg2rad(120),     # psi, r
], dtype=np.float32)

POS_SCALE = 1.0  # for relative targets [m]

# ---- Cost weights ----
Q_DIAG = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,                  # pos & vel (1/max^2, max=1)
    1/np.deg2rad(30)**2, 1/np.deg2rad(200)**2,       # phi, p
    1/np.deg2rad(30)**2, 1/np.deg2rad(200)**2,       # theta, q
    1/np.deg2rad(45)**2, 1/np.deg2rad(120)**2,       # psi, r
], dtype=np.float32)

# Hover thrust in uint16 units
# At hover: each motor PWM = thrust_u16, physical thrust = (pwm/65535)*0.12 N
# 4 motors: 4*(thrust_u16/65535)*0.12 = mg → thrust_u16 = mg*65535/(4*0.12)
HOVER_THRUST_U16 = GRAVITY * UINT16_MAX / (4 * CF2_THRUST_MAX_PER_MOTOR)
# ≈ 36118


# =====================================================================
# 2. Nonlinear Dynamics  (exact copy from train_nn_cf2x.py)
# =====================================================================

def rotation_matrix_zyx(phi, theta, psi):
    """Intrinsic ZYX rotation matrix. (B,) -> (B,3,3)."""
    cphi = torch.cos(phi);   sphi = torch.sin(phi)
    cth  = torch.cos(theta); sth  = torch.sin(theta)
    cpsi = torch.cos(psi);   spsi = torch.sin(psi)
    R = torch.stack([
        torch.stack([cth*cpsi, sphi*sth*cpsi - cphi*spsi, cphi*sth*cpsi + sphi*spsi], -1),
        torch.stack([cth*spsi, sphi*sth*spsi + cphi*cpsi, cphi*sth*spsi - sphi*cpsi], -1),
        torch.stack([-sth,     sphi*cth,                   cphi*cth],                  -1),
    ], dim=-2)
    return R


def dynamics_substep(state, rpm, dt):
    """One semi-implicit Euler step. Replicates BaseAviary._dynamics().

    state : (B, 12)  [x, vx, y, vy, z, vz, phi, p, theta, q, psi, r]
    rpm   : (B, 4)   [M1, M2, M3, M4]
    """
    x, vx     = state[:, 0], state[:, 1]
    y, vy     = state[:, 2], state[:, 3]
    z, vz     = state[:, 4], state[:, 5]
    phi, p    = state[:, 6], state[:, 7]
    theta, q  = state[:, 8], state[:, 9]
    psi, r    = state[:, 10], state[:, 11]

    # --- 1. Calcul des forces et couples à partir des RPMs ---
    # Correspond à: forces = np.array(rpm**2) * self.KF
    forces = (rpm ** 2) * KF
    z_torques = (rpm ** 2) * KM

    F = forces.sum(dim=-1)

    # Mixage CF2X (Identique à pybullet-drones DroneModel.CF2X)
    _a = L / math.sqrt(2)
    tau_x = -(forces[:, 0] + forces[:, 1] - forces[:, 2] - forces[:, 3]) * _a
    tau_y = (-forces[:, 0] + forces[:, 1] + forces[:, 2] - forces[:, 3]) * _a
    tau_z = -z_torques[:, 0] + z_torques[:, 1] - z_torques[:, 2] + z_torques[:, 3]

    # --- 2. Poussée dans le repère monde ---
    R = rotation_matrix_zyx(phi, theta, psi)
    thrust_world = F.unsqueeze(-1) * R[:, :, 2]
    
    ax = thrust_world[:, 0] / M
    ay = thrust_world[:, 1] / M
    az = thrust_world[:, 2] / M - G

    # --- 3. Couplage gyroscopique ---
    gyro_x = (I_Z - I_Y) * q * r
    gyro_y = (I_X - I_Z) * p * r
    gyro_z = (I_Y - I_X) * p * q
    
    p_dot = (tau_x - gyro_x) / I_X
    q_dot = (tau_y - gyro_y) / I_Y
    r_dot = (tau_z - gyro_z) / I_Z

    # --- 4. Euler Semi-implicite ---
    vx_n = vx + dt * ax;  vy_n = vy + dt * ay;  vz_n = vz + dt * az
    p_n  = p  + dt * p_dot;  q_n = q + dt * q_dot;  r_n = r + dt * r_dot
    
    x_n     = x     + dt * vx_n;  y_n     = y     + dt * vy_n;  z_n     = z     + dt * vz_n
    phi_n   = phi   + dt * p_n;   theta_n = theta + dt * q_n;   psi_n   = psi   + dt * r_n

    return torch.stack([x_n, vx_n, y_n, vy_n, z_n, vz_n,
                        phi_n, p_n, theta_n, q_n, psi_n, r_n], dim=-1)


# =====================================================================
# 3. Differentiable PID Functions
# =====================================================================

def pid_update_vec(desired, measured, prev_measured, integral,
                   kp, ki, kd, dt, i_limit, yaw_mask=None):
    """Vectorised PID update over (B, 3) tensors — all axes at once.

    yaw_mask: (3,) bool tensor, True for the yaw axis (angle wrapping).
    Returns (output, new_integral, new_prev_measured) each (B, 3).
    """
    error = desired - measured
    delta = -(measured - prev_measured)

    # Yaw angle wrapping for axis 2 only
    if yaw_mask is not None:
        ym = yaw_mask.unsqueeze(0)  # (1, 3)
        error = torch.where(ym, error - 360.0 * torch.round(error / 360.0), error)
        delta = torch.where(ym, delta - 360.0 * torch.round(delta / 360.0), delta)

    out_p = kp * error
    out_d = kd * (delta / dt)

    new_integral = integral + error * dt
    new_integral = torch.clamp(new_integral, -i_limit, i_limit)
    out_i = ki * new_integral

    return out_p + out_d + out_i, new_integral, measured


# Pre-compute allocation coefficients (constants)
_L_SQRT2 = L / math.sqrt(2)
_KM_KF   = KM / KF
_THRUST_SCALE = CF2_THRUST_MAX_PER_MOTOR / UINT16_MAX


# =====================================================================
# 4. Full Simulation Step (NN action -> PIDs -> motors -> dynamics)
# =====================================================================

# Cache for PID gains on the right device
_pid_cache = {}

def _get_pid_gains(dev):
    """Cache PID gain tensors on the target device."""
    if dev not in _pid_cache:
        yaw_mask = torch.tensor([False, False, True], device=dev)
        _pid_cache[dev] = {
            'att_kp': ATT_KP.to(dev), 'att_ki': ATT_KI.to(dev),
            'att_kd': ATT_KD.to(dev), 'att_il': ATT_ILIM.to(dev),
            'rate_kp': RATE_KP.to(dev), 'rate_ki': RATE_KI.to(dev),
            'rate_kd': RATE_KD.to(dev), 'rate_il': RATE_ILIM.to(dev),
            'yaw_mask': yaw_mask,
        }
    return _pid_cache[dev]

_MOTORS_PWM_BITS = 8
_MOTORS_PWM_SHIFT = 16 - _MOTORS_PWM_BITS  # 8


def pwm_to_rpm_torch(motor_pwm: torch.Tensor) -> torch.Tensor:
    """Convertit les PWM (0-65535) en RPM avec troncature 8-bit hardware.

    Réplique la chaîne réelle CF2.1+ :
      1. Troncature timer 8-bit : arrondi vers le bas au multiple de 256 le plus proche
         [FW] motors.c:115  motorsConv16ToBits
      2. Mapping linéaire uint16 → poussée [N]
         [FW] motors.c:165  motorsCompensateBatteryVoltage
      3. thrust = KF × rpm²  →  rpm = sqrt(thrust / KF)

    La troncature n'est pas différentiable; on utilise un straight-through
    estimator : forward = valeur tronquée, backward = gradient identité.
    Cela conserve la fidélité physique tout en permettant la rétropropagation.
    """
    # Straight-through truncation: floor to nearest multiple of 256
    # Equivalent to: int(pwm) >> 8 << 8
    scale = float(1 << _MOTORS_PWM_SHIFT)  # 256.0
    pwm_trunc = motor_pwm - (motor_pwm % scale)  # floor to multiple of 256
    # Straight-through: use truncated value in forward, pass gradient through
    pwm_trunc = motor_pwm + (pwm_trunc - motor_pwm).detach()

    thrust = (pwm_trunc / UINT16_MAX) * CF2_THRUST_MAX_PER_MOTOR
    # Empêche un gradient infini si thrust = 0
    thrust = torch.clamp(thrust, min=1e-3)
    return torch.sqrt(thrust / KF)

def one_pid_step(thrust_u16, roll_des_deg, pitch_des_deg, yaw_rate_deg,
                 state_12, pid_state):
    """One 500 Hz step: attitude PID -> rate PID -> power dist -> dynamics.

    Fully vectorised over the 3 PID axes (roll, pitch, yaw).
    pid_state = (att_integ (B,3), att_prev (B,3),
                 rate_integ (B,3), rate_prev (B,3),
                 yaw_setpoint (B,))
    """
    att_integ, att_prev, rate_integ, rate_prev, yaw_sp, motor_rpm = pid_state
    g = _get_pid_gains(state_12.device)

    # --- Extract state in firmware convention (pitch & pitch-rate negated) ---
    RAD2DEG = 180.0 / math.pi
    actual_att = torch.stack([
        state_12[:, 6] * RAD2DEG,       # roll
        -state_12[:, 8] * RAD2DEG,      # pitch (negated for firmware)
        state_12[:, 10] * RAD2DEG,      # yaw
    ], dim=-1)  # (B, 3)

    actual_gyro = torch.stack([
        state_12[:, 7] * RAD2DEG,       # gyro roll
        -state_12[:, 9] * RAD2DEG,      # gyro pitch (negated)
        state_12[:, 11] * RAD2DEG,      # gyro yaw
    ], dim=-1)  # (B, 3)

    # --- Yaw setpoint accumulation ---
    new_yaw_sp = yaw_sp + yaw_rate_deg * ATTITUDE_UPDATE_DT
    new_yaw_sp = new_yaw_sp - 360.0 * torch.round(new_yaw_sp / 360.0)

    # --- Attitude PID (vectorised over 3 axes) ---
    desired_att = torch.stack([roll_des_deg, pitch_des_deg, new_yaw_sp], dim=-1)

    rate_desired, new_att_integ, new_att_prev = pid_update_vec(
        desired_att, actual_att, att_prev, att_integ,
        g['att_kp'], g['att_ki'], g['att_kd'], ATTITUDE_UPDATE_DT, g['att_il'],
        yaw_mask=g['yaw_mask'])

    # --- Rate PID (vectorised over 3 axes) ---
    motor_cmds, new_rate_integ, new_rate_prev = pid_update_vec(
        rate_desired, actual_gyro, rate_prev, rate_integ,
        g['rate_kp'], g['rate_ki'], g['rate_kd'], ATTITUDE_UPDATE_DT, g['rate_il'])
    motor_cmds = torch.clamp(motor_cmds, -32767, 32767)

    roll_cmd  = motor_cmds[:, 0]
    pitch_cmd = motor_cmds[:, 1]
    yaw_cmd   = -motor_cmds[:, 2]   # firmware negates yaw output

    # --- Power distribution (CF2X, firmware ordering) ---
    r_half = roll_cmd * 0.5
    p_half = pitch_cmd * 0.5
    motor_pwms = torch.stack([
        thrust_u16 - r_half + p_half + yaw_cmd,   # M1 front-right
        thrust_u16 - r_half - p_half - yaw_cmd,   # M2 rear-right
        thrust_u16 + r_half - p_half + yaw_cmd,   # M3 rear-left
        thrust_u16 + r_half + p_half - yaw_cmd,   # M4 front-left
    ], dim=-1)  # (B, 4)

    # Cap: preserve attitude authority
    highest = motor_pwms.max(dim=-1, keepdim=True).values
    reduction = torch.clamp(highest - UINT16_MAX, min=0)
    motor_pwms = torch.clamp(motor_pwms - reduction, min=0, max=UINT16_MAX)

    # --- PWMs -> wrench -> dynamics ---
    # --- PWMs -> RPMs -> dynamics ---
    rpm_cmd = pwm_to_rpm_torch(motor_pwms)
    new_motor_rpm = MOTOR_ALPHA * rpm_cmd + (1.0 - MOTOR_ALPHA) * motor_rpm
    
    new_state = dynamics_substep(state_12, new_motor_rpm, ATTITUDE_UPDATE_DT)
    
    new_pid_state = (new_att_integ, new_att_prev, new_rate_integ,
                     new_rate_prev, new_yaw_sp, new_motor_rpm)
    
    return new_state, new_pid_state


def one_nn_step(state, nn_action, pid_state):
    """Execute one NN step (100 Hz) = 5 PID steps (500 Hz).

    nn_action: (B, 4) [thrust_u16, roll_deg, pitch_deg, yaw_rate_deg_s]
    """
    thrust    = nn_action[:, 0]
    roll_deg  = nn_action[:, 1]
    pitch_deg = nn_action[:, 2]
    yaw_rate  = nn_action[:, 3]

    for _ in range(PID_STEPS_PER_NN):
        state, pid_state = one_pid_step(
            thrust, roll_deg, pitch_deg, yaw_rate, state, pid_state)
    return state, pid_state


# =====================================================================
# 5. PID State Initialization
# =====================================================================

def init_pid_state(state_12):
    """Create initial PID state from dynamics state."""
    B = state_12.shape[0]
    dev = state_12.device

    roll_deg  = torch.rad2deg(state_12[:, 6])
    pitch_deg = -torch.rad2deg(state_12[:, 8])
    yaw_deg   = torch.rad2deg(state_12[:, 10])
    gyro_r = torch.rad2deg(state_12[:, 7])
    gyro_p = -torch.rad2deg(state_12[:, 9])
    gyro_y = torch.rad2deg(state_12[:, 11])

    att_integ  = torch.zeros(B, 3, device=dev)
    att_prev   = torch.stack([roll_deg, pitch_deg, yaw_deg], dim=-1)
    rate_integ = torch.zeros(B, 3, device=dev)
    rate_prev  = torch.stack([gyro_r, gyro_p, gyro_y], dim=-1)
    yaw_sp     = yaw_deg.clone()
    
    # NOUVEAU: Initialiser la vitesse des moteurs au stationnaire
    motor_rpm  = torch.full((B, 4), HOVER_RPM, device=dev, dtype=torch.float32)

    return (att_integ, att_prev, rate_integ, rate_prev, yaw_sp, motor_rpm)


def reset_pid_diverged(pid_state, diverged_mask):
    """Zero PID state for diverged drones (detached)."""
    # NOUVEAU: Récupérer motor_rpm
    att_integ, att_prev, rate_integ, rate_prev, yaw_sp, motor_rpm = pid_state
    
    m = diverged_mask.unsqueeze(1)   # (B, 1)
    m1 = diverged_mask               # (B,)
    
    zero3 = torch.zeros_like(att_integ)
    zero1 = torch.zeros_like(yaw_sp)
    # NOUVEAU: Moteurs au stationnaire pour les drones reset
    hover4 = torch.full_like(motor_rpm, HOVER_RPM) 
    
    return (
        torch.where(m, zero3, att_integ),
        torch.where(m, zero3, att_prev),
        torch.where(m, zero3, rate_integ),
        torch.where(m, zero3, rate_prev),
        torch.where(m1, zero1, yaw_sp),
        torch.where(m, hover4, motor_rpm), # NOUVEAU
    )


# =====================================================================
# 6. Concurrent Neural Network
# =====================================================================

class ConcurrentPolicyMLP(nn.Module):
    """Concurrent NN controller (Wiedemann et al. architecture).

    Queried once per chunk. Outputs all T future actions simultaneously.

    Input:  12D state + T*3D relative targets = (12 + T*3)
    Output: T * (thrust_u16, roll_deg, pitch_deg, yaw_rate_deg/s) = T*4

    Hover initialization: last-layer bias so initial output is
    [hover_thrust, 0, 0, 0] regardless of input.
    """
    def __init__(self, T, hidden=128):
        super().__init__()
        self.T = T
        input_dim = 12 + T * 3
        output_dim = T * 4

        self.register_buffer("x_scale",
            torch.tensor(X_SCALE, dtype=torch.float32))
        self.register_buffer("pos_scale",
            torch.tensor(POS_SCALE, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )
        self._init_hover()

    def _init_hover(self):
        """Initialize last layer: small weights + hover bias."""
        last = self.net[-1]
        # Small random weights so hidden layers get gradients from epoch 1
        nn.init.normal_(last.weight, std=0.01)
        nn.init.zeros_(last.bias)

        # thrust bias: sigmoid(b) * 65535 = hover_thrust
        hover_ratio = HOVER_THRUST_U16 / UINT16_MAX
        thrust_bias = float(np.log(hover_ratio / (1.0 - hover_ratio)))
        for t in range(self.T):
            last.bias.data[t * 4 + 0] = thrust_bias
            # roll, pitch, yaw_rate biases stay 0 -> tanh(0)*max = 0

    def forward(self, state, targets_rel):
        """
        state       : (B, 12)
        targets_rel : (B, T, 3)
        Returns     : (B, T, 4) [thrust_u16, roll_deg, pitch_deg, yaw_rate]
        """
        B = state.shape[0]
        state_n = state / self.x_scale
        targets_n = targets_rel / self.pos_scale
        targets_flat = targets_n.reshape(B, -1)

        x = torch.cat([state_n, targets_flat], dim=-1)
        raw = self.net(x).reshape(B, self.T, 4)

        thrust  = torch.sigmoid(raw[..., 0]) * UINT16_MAX
        roll    = torch.tanh(raw[..., 1]) * PID_VEL_ROLL_MAX
        pitch   = torch.tanh(raw[..., 2]) * PID_VEL_PITCH_MAX
        yaw_r   = torch.tanh(raw[..., 3]) * YAW_RATE_MAX

        return torch.stack([thrust, roll, pitch, yaw_r], dim=-1)


# =====================================================================
# 7. Rollout with Curriculum
# =====================================================================

def compute_relative_targets(state, T, target_pos=None):
    """Compute T relative 3D targets for the regulation-to-origin task.

    target_pos: (3,) or None (defaults to origin).
    Returns: (B, T, 3)
    """
    B = state.shape[0]
    dev = state.device
    current_pos = state[:, [0, 2, 4]]  # (B, 3) [x, y, z]

    if target_pos is None:
        target = torch.zeros(1, 3, device=dev)
    else:
        target = target_pos.unsqueeze(0).to(dev)

    # All T targets are the same for regulation
    rel = (target - current_pos).unsqueeze(1).expand(B, T, 3)
    return rel


def rollout(policy, x0, n_chunks, tau_div=None, log_extra=False):
    """Full rollout: n_chunks * T NN steps, each = 5 PID steps.

    Args:
        log_extra: if True, also return A (NN actions) and R (motor RPMs).
                   Keep False during training to avoid memory overhead.

    Returns:
        X : (B, n_chunks*T, 12) state trajectory at 100 Hz.
        A : (B, n_chunks*T, 4)  NN actions [thrust_u16, roll°, pitch°, yaw_rate°/s]
            — only when log_extra=True
        R : (B, n_chunks*T, 4)  motor RPMs [M1..M4]
            — only when log_extra=True
    """
    T = policy.T
    B = x0.shape[0]
    dev = x0.device
    total_nn_steps = n_chunks * T

    X = torch.zeros(B, total_nn_steps, 12, device=dev)
    if log_extra:
        A = torch.zeros(B, total_nn_steps, 4, device=dev)
        R = torch.zeros(B, total_nn_steps, 4, device=dev)

    state = x0
    pid_state = init_pid_state(state)
    step_idx = 0

    for chunk in range(n_chunks):
        # Compute relative targets from current position
        targets_rel = compute_relative_targets(state, T)

        # Query NN once for the entire chunk
        actions = policy(state, targets_rel)   # (B, T, 4)

        for t in range(T):
            nn_action = actions[:, t, :]
            state, pid_state = one_nn_step(state, nn_action, pid_state)
            X[:, step_idx, :] = state
            if log_extra:
                A[:, step_idx, :] = nn_action
                R[:, step_idx, :] = pid_state[5]   # motor_rpm is index 5
            step_idx += 1

            # Curriculum reset
            if tau_div is not None:
                pos_err = state[:, [0, 2, 4]].norm(dim=1)
                diverged = pos_err > tau_div
                if diverged.any():
                    print(f"  [Curriculum] Resetting {diverged.sum().item()}/{B} drones at step {step_idx} (pos_err > {tau_div:.2f})")
                    mask = diverged.unsqueeze(1).expand_as(state)
                    state = torch.where(mask, torch.zeros_like(state), state)
                    pid_state = reset_pid_diverged(pid_state, diverged)

    if log_extra:
        return X, A, R
    return X


# =====================================================================
# 8. Cost Function
# =====================================================================

def trajectory_cost(X, terminal_weight=10.0, pos_weight=10.0):
    """Quadratic cost: position-dominant + small state penalty.

    pos_weight scales the position terms relative to velocity/angle terms.
    This encourages the NN to actually move toward target, not just hover.
    """
    Q = torch.as_tensor(Q_DIAG, dtype=X.dtype, device=X.device)

    # Scale position terms more heavily
    Q_scaled = Q.clone()
    Q_scaled[[0, 2, 4]] *= pos_weight   # x, y, z positions

    cost_running = (X**2 * Q_scaled).sum(dim=2).mean(dim=0).sum()

    if terminal_weight > 0:
        pos_T = X[:, -1, [0, 2, 4]]
        cost_running += terminal_weight * (pos_T**2).sum(dim=1).mean()

    return cost_running


# =====================================================================
# 9. Training
# =====================================================================

def make_x0_batch(xyz_list, device="cpu"):
    """Create (B, 12) initial states from list of (x, y, z). All rest = 0."""
    X0 = torch.zeros(len(xyz_list), 12, dtype=torch.float32, device=device)
    for b, (x, y, z) in enumerate(xyz_list):
        X0[b, 0] = x;  X0[b, 2] = y;  X0[b, 4] = z
    return X0


def generate_cube_points(half_side=0.3):
    """27 initial positions: vertices + edges + faces + center."""
    vals = [-half_side, 0.0, half_side]
    return list(itertools.product(vals, repeat=3))

import signal
class GracefulKiller:
    """Intercepte Ctrl+C pour arrêter l'entraînement proprement à la fin d'un epoch."""
    def __init__(self):
        self.kill_now = False
        self.interrupt_count = 0
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.interrupt_count += 1
        if self.interrupt_count == 1:
            print("\n[Interruption] Ctrl+C détecté ! L'entraînement s'arrêtera à la fin de cet epoch pour sauvegarder proprement.")
            print("Appuyez à nouveau sur Ctrl+C pour forcer un arrêt immédiat (sans sauvegarde).")
            self.kill_now = True
        else:
            print("\n[Arrêt Forcé] Double Ctrl+C détecté. Arrêt immédiat !")
            sys.exit(1)

def train(epochs=500, lr=1e-2, hidden=128, t_chunk=0.01, t_sim=2.0,
          half_side=0.3, terminal_weight=10.0,
          tau_start=0.5, tau_end=2.0, device="cpu"):
    """Train the concurrent NN controller with curriculum learning."""

    killer = GracefulKiller()

    T = int(t_chunk * NN_FREQ)          # NN steps per chunk
    n_chunks = int(t_sim / t_chunk)     # number of chunks
    total_steps = T * n_chunks

    print(f"[Config] T_chunk={T} steps ({t_chunk}s), "
          f"n_chunks={n_chunks}, total={total_steps} NN steps ({t_sim}s)")
    print(f"[Config] PID substeps per NN step: {PID_STEPS_PER_NN}")
    print(f"[Config] Total dynamics steps: {total_steps * PID_STEPS_PER_NN}")

    policy = ConcurrentPolicyMLP(T=T, hidden=hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    cube_pts = generate_cube_points(half_side)
    x0 = make_x0_batch(cube_pts, device=device)
    print(f"[Train] {len(cube_pts)} initial points, ±{half_side}m cube")

    # ==========================================================
    # PRÉPARATION POUR LES PLOTS
    # ==========================================================
    # Création du dossier
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NN_training_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Génération d'un set fixe de points de test pour toute la durée de l'entraînement
    rng = np.random.default_rng(42)
    # 10 points c'est un bon compromis pour que le plot reste lisible
    test_pts = [(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3)) for _ in range(10)]
    eval_labels = [f"({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})" for p in test_pts]
    # ==========================================================

    # Verify hover initialization
    with torch.no_grad():
        test_state = torch.zeros(1, 12, device=device)
        test_targets = torch.zeros(1, T, 3, device=device)
        test_out = policy(test_state, test_targets)
        print(f"[Init] NN at x=0: thrust={test_out[0,0,0]:.0f} "
              f"(hover≈{HOVER_THRUST_U16:.0f}), "
              f"roll={test_out[0,0,1]:.2f}°, pitch={test_out[0,0,2]:.2f}°, "
              f"yaw_r={test_out[0,0,3]:.2f}°/s")

    for ep in range(epochs):
        # Curriculum: tau_div linearly increases over epochs
        progress = ep / max(epochs - 1, 1)
        tau_div = tau_start + (tau_end - tau_start) * progress

        X = rollout(policy, x0, n_chunks, tau_div=tau_div)
        loss = trajectory_cost(X, terminal_weight)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        # PARE-FEU ANTI-NAN
        has_nan = any(torch.isnan(p.grad).any() for p in policy.parameters() if p.grad is not None)
        if not has_nan:
            opt.step()
            scheduler.step()
        else:
            print(f"[Warning] NaN detected in gradients at epoch {ep+1}. Skipping optimizer step to prevent corruption.")

        with torch.no_grad():
            pos_end = X[:, -1, [0, 2, 4]].norm(dim=1)
            pos_mid = X[:, total_steps // 2, [0, 2, 4]].norm(dim=1)
            print(f"  [ep {ep+1:4d}/{epochs}]  loss={loss.item():.4e}  "
                    f"|pos_T|={pos_end.mean():.4f} (max {pos_end.max():.4f})  "
                    f"|pos_mid|={pos_mid.mean():.4f}  tau={tau_div:.3f}")

        # ==========================================================
        # ÉVALUATION ET SAUVEGARDE DU PLOT
        # ==========================================================
        
        with torch.no_grad():
            X_eval, A_eval, R_eval = evaluate(policy, test_pts, n_chunks, device=device, verbose=False)

            # Nom du fichier : epoch_0001.png, epoch_0002.png...
            plot_file = os.path.join(plot_dir, f"epoch_{ep+1:04d}.png")
            save_plots(X_eval, A_eval, R_eval, eval_labels, plot_file, verbose=False)
        
        if killer.kill_now:
            print(f"\n[Arrêt Propre] Fin prématurée demandée à l'epoch {ep+1}.")
            break  # On sort de la boucle for

    return policy


# =====================================================================
# 10. Evaluation
# =====================================================================

@torch.no_grad()
def evaluate(policy, test_pts, n_chunks, device="cpu", verbose=True):
    """Evaluate policy on test_pts.

    Returns (X, A, R):
        X : (B, total, 12)  state trajectory
        A : (B, total, 4)   NN actions [thrust_u16, roll°, pitch°, yaw_rate°/s]
        R : (B, total, 4)   motor RPMs [M1..M4]
    """
    T = policy.T
    x0 = make_x0_batch(test_pts, device=device)
    X, A, R = rollout(policy, x0, n_chunks, tau_div=None, log_extra=True)

    if verbose:
        pos_end = X[:, -1, [0, 2, 4]].norm(dim=1)
        vel_end = X[:, -1, [1, 3, 5]].norm(dim=1)
        print(f"\n[Eval] {len(test_pts)} test points, "
              f"{n_chunks * T} NN steps ({n_chunks * T / NN_FREQ:.1f}s):")
        print(f"  Terminal pos error  mean={pos_end.mean():.5f} m  max={pos_end.max():.5f} m")
        print(f"  Terminal velocity   mean={vel_end.mean():.5f} m/s  max={vel_end.max():.5f} m/s")

    return X, A, R


def save_plots(X, A, R, labels, filename="training_result.png", verbose=True):
    """Plot full state + NN outputs + motor RPMs.

    Layout: 6 rows × 3 cols
      Row 0 : x, y, z  [m]
      Row 1 : vx, vy, vz  [m/s]
      Row 2 : roll (phi), pitch (theta), yaw (psi)  [deg]
      Row 3 : roll rate (p), pitch rate (q), yaw rate (r)  [deg/s]
      Row 4 : NN thrust [u16], NN roll_des [deg], NN pitch_des [deg]
      Row 5 : NN yaw_rate [deg/s], RPM M1+M2, RPM M3+M4
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    RAD2DEG = 180.0 / np.pi
    X_np = X.cpu().numpy()                   # (B, T, 12)
    A_np = A.cpu().numpy()                   # (B, T, 4)
    R_np = R.cpu().numpy()                   # (B, T, 4)
    B = X_np.shape[0]
    T_total = X_np.shape[1]
    t_axis = np.arange(1, T_total + 1) / NN_FREQ
    n_traj = min(B, 10)

    fig, axes = plt.subplots(6, 3, figsize=(15, 20), sharex=True)
    fig.suptitle("Concurrent NN Controller with CF Firmware PID", fontsize=13)

    def _plot_row(row, data_list, ylabel_list, title_list, hline=None):
        """Plot one row: data_list[j] is (B, T) array for subplot (row, j), or None to skip."""
        for j, (data, ylabel, title) in enumerate(zip(data_list, ylabel_list, title_list)):
            ax = axes[row, j]
            if data is None:
                ax.set_visible(False)
                continue
            for b in range(n_traj):
                ax.plot(t_axis, data[b], linewidth=0.7, alpha=0.65,
                        label=labels[b] if (row == 0 and j == 0) else None)
            if hline is not None:
                ax.axhline(hline, color="k", linewidth=0.6, linestyle="--", alpha=0.5)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=8)
            ax.grid(True, alpha=0.3)

    # Row 0 — positions [m]
    _plot_row(0,
              [X_np[:, :, 0], X_np[:, :, 2], X_np[:, :, 4]],
              ["x (m)", "y (m)", "z (m)"],
              ["Position x", "Position y", "Position z"],
              hline=0.0)
    axes[0, 0].legend(fontsize=5, ncol=2)

    # Row 1 — velocities [m/s]
    _plot_row(1,
              [X_np[:, :, 1], X_np[:, :, 3], X_np[:, :, 5]],
              ["vx (m/s)", "vy (m/s)", "vz (m/s)"],
              ["Velocity vx", "Velocity vy", "Velocity vz"],
              hline=0.0)

    # Row 2 — angles [deg]
    _plot_row(2,
              [X_np[:, :, 6] * RAD2DEG, X_np[:, :, 8] * RAD2DEG, X_np[:, :, 10] * RAD2DEG],
              ["phi (deg)", "theta (deg)", "psi (deg)"],
              ["Roll angle", "Pitch angle", "Yaw angle"],
              hline=0.0)

    # Row 3 — angular rates [deg/s]
    _plot_row(3,
              [X_np[:, :, 7] * RAD2DEG, X_np[:, :, 9] * RAD2DEG, X_np[:, :, 11] * RAD2DEG],
              ["p (deg/s)", "q (deg/s)", "r (deg/s)"],
              ["Roll rate p", "Pitch rate q", "Yaw rate r"],
              hline=0.0)

    # Row 4 — NN outputs: thrust, roll_des, pitch_des
    _plot_row(4,
              [A_np[:, :, 0], A_np[:, :, 1], A_np[:, :, 2]],
              ["thrust (u16)", "roll_des (deg)", "pitch_des (deg)"],
              ["NN Thrust", "NN Roll setpoint", "NN Pitch setpoint"])
    axes[4, 0].axhline(HOVER_THRUST_U16, color="gray", linewidth=0.8,
                       linestyle=":", label=f"hover≈{HOVER_THRUST_U16:.0f}")
    axes[4, 0].legend(fontsize=6)

    # Row 5 — NN yaw_rate + RPMs
    _plot_row(5,
              [A_np[:, :, 3], None, None],
              ["yaw_rate (deg/s)", "", ""],
              ["NN Yaw rate setpoint", "", ""])

    # RPM M1+M2 in (5,1), M3+M4 in (5,2)
    motor_colors = ["C0", "C1", "C2", "C3"]
    for m_idx, (col, motor_name) in enumerate([(1, "M1"), (1, "M2"), (2, "M3"), (2, "M4")]):
        ax = axes[5, col]
        for b in range(n_traj):
            ax.plot(t_axis, R_np[b, :, m_idx], linewidth=0.7, alpha=0.55,
                    color=motor_colors[m_idx], label=motor_name if b == 0 else None)
    axes[5, 1].set_title("RPM M1 & M2", fontsize=8)
    axes[5, 1].set_ylabel("RPM", fontsize=8)
    axes[5, 1].legend(fontsize=6)
    axes[5, 1].grid(True, alpha=0.3)
    axes[5, 2].set_title("RPM M3 & M4", fontsize=8)
    axes[5, 2].set_ylabel("RPM", fontsize=8)
    axes[5, 2].legend(fontsize=6)
    axes[5, 2].grid(True, alpha=0.3)
    axes[5, 5 - 5].axhline(HOVER_RPM, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
    axes[5, 1].axhline(HOVER_RPM, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)
    axes[5, 2].axhline(HOVER_RPM, color="gray", linewidth=0.6, linestyle=":", alpha=0.7)

    # x-label only on bottom row
    for j in range(3):
        axes[5, j].set_xlabel("Time (s)", fontsize=8)

    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    if verbose:
        print(f"[Saved] {filename}")


# =====================================================================
# 11. Main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train concurrent NN controller with CF firmware PID")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--t_chunk", type=float, default=0.01,
                        help="Chunk duration in seconds (T = t_chunk * 100)")
    parser.add_argument("--t_sim", type=float, default=2.0,
                        help="Total simulation duration in seconds")
    parser.add_argument("--half_side", type=float, default=0.5,
                        help="Half-side of initial position cube [m]")
    parser.add_argument("--tau_start", type=float, default=0.8,
                        help="Curriculum: initial divergence threshold [m]")
    parser.add_argument("--tau_end", type=float, default=2.0,
                        help="Curriculum: final divergence threshold [m]")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")
    print(f"[Hover thrust] {HOVER_THRUST_U16:.0f} / {UINT16_MAX:.0f} "
          f"({HOVER_THRUST_U16/UINT16_MAX*100:.1f}%)")

    policy = train(
        epochs=args.epochs, lr=args.lr, hidden=args.hidden,
        t_chunk=args.t_chunk, t_sim=args.t_sim,
        half_side=args.half_side, terminal_weight=10.0,
        tau_start=args.tau_start, tau_end=args.tau_end,
        device=device)

    # Evaluate on random test points
    T = int(args.t_chunk * NN_FREQ)
    n_chunks = int(args.t_sim / args.t_chunk)
    rng = np.random.default_rng(42)
    test_pts = [(rng.uniform(-0.3, 0.3),
                 rng.uniform(-0.3, 0.3),
                 rng.uniform(-0.3, 0.3)) for _ in range(20)]
    X_eval, A_eval, R_eval = evaluate(policy, test_pts, n_chunks, device=device, verbose=True)

    # Plot
    labels = [f"({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})" for p in test_pts]
    tag = f"_{args.tag}" if args.tag else ""
    plot_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f"train_cf_pid_result{tag}.png")
    save_plots(X_eval, A_eval, R_eval, labels, plot_file, verbose=True)

    # Save checkpoint
    out_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_name = (f"trained_cf_pid_T{T}_ch{n_chunks}_h{args.hidden}"
                 f"_ep{args.epochs}{tag}.pt")
    ckpt_path = os.path.join(out_dir, ckpt_name)
    torch.save({
        "state_dict": policy.cpu().state_dict(),
        "T": T,
        "n_chunks": n_chunks,
        "hidden": args.hidden,
        "x_scale": X_SCALE,
        "pos_scale": POS_SCALE,
        "nn_freq": NN_FREQ,
        "att_rate": ATTITUDE_RATE,
        "t_chunk": args.t_chunk,
        "t_sim": args.t_sim,
        "epochs": args.epochs,
        "lr": args.lr,
        "KF": KF, "KM": KM, "L": L, "M": M, "G": G,
        "MAX_RPM": MAX_RPM,
    }, ckpt_path)
    print(f"[Saved] {ckpt_path}")
    print("\nDone!")
