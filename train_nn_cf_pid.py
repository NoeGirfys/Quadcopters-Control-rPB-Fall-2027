#!/usr/bin/env python3
"""
Train a neural network controller for the Crazyflie 2.x using the real
firmware PID cascade as a differentiable inner loop.

The NN outputs a setpoint (position, velocity, attitude, or attitude rate)
which is then fed through a fully differentiable PyTorch reimplementation of
the CF2.1+ firmware PID cascade.  End-to-end gradients flow: NN weights →
setpoint → PID states → RPMs → dynamics → trajectory cost.

Setpoint modes (--setpoint_mode):
  pos_sp   – NN outputs (x, y, z) setpoint [m]  → all 4 PID levels
  vel_sp   – NN outputs (vx, vy, vz) [m/s]      → bypass pos PID
  att_sp   – NN outputs (roll, pitch [deg], yaw_rate [deg/s], thrust [u16])
                                                  → bypass pos+vel PIDs
  rate_sp  – NN outputs (p, q, r [deg/s], thrust [u16])
                                                  → bypass pos+vel+att PIDs

Physics modes:
  nonlinear (default) – full rotation-matrix dynamics, semi-implicit Euler
  linear (--linear)   – ZOH-discretized affine model

Usage:
    python train_nn_cf_pid.py
    python train_nn_cf_pid.py --setpoint_mode att_sp
    python train_nn_cf_pid.py --setpoint_mode rate_sp --linear
    python train_nn_cf_pid.py --flowdeck --flowdeck_delay_ms 110
    python train_nn_cf_pid.py --epochs 1000 --lr 5e-4 --tag exp1
"""

import argparse
import itertools
import os

import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
from scipy.signal import cont2discrete


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
PID_ROLL_RATE_KP  = 250.0;  PID_ROLL_RATE_KI  = 500.0;  PID_ROLL_RATE_KD  = 2.5
PID_ROLL_RATE_INTEGRATION_LIMIT  = 33.3
PID_PITCH_RATE_KP = 250.0;  PID_PITCH_RATE_KI = 500.0;  PID_PITCH_RATE_KD = 2.5
PID_PITCH_RATE_INTEGRATION_LIMIT = 33.3
PID_YAW_RATE_KP   = 120.0;  PID_YAW_RATE_KI   = 16.7;   PID_YAW_RATE_KD   = 0.0
PID_YAW_RATE_INTEGRATION_LIMIT   = 166.7

# --- Attitude PIDs ---
PID_ROLL_KP  = 6.0;  PID_ROLL_KI  = 3.0;  PID_ROLL_KD  = 0.0
PID_ROLL_INTEGRATION_LIMIT  = 20.0
PID_PITCH_KP = 6.0;  PID_PITCH_KI = 3.0;  PID_PITCH_KD = 0.0
PID_PITCH_INTEGRATION_LIMIT = 20.0
PID_YAW_KP   = 6.0;  PID_YAW_KI   = 1.0;  PID_YAW_KD   = 0.35
PID_YAW_INTEGRATION_LIMIT   = 360.0

# --- Velocity PIDs ---
PID_VEL_X_KP = 5.0;  PID_VEL_X_KI = 1.0;  PID_VEL_X_KD = 0.0
PID_VEL_Y_KP = 5.0;  PID_VEL_Y_KI = 1.0;  PID_VEL_Y_KD = 0.0
PID_VEL_Z_KP = 25.0; PID_VEL_Z_KI = 15.0; PID_VEL_Z_KD = 0.0

# --- Position PIDs ---
PID_POS_X_KP = 1.0;  PID_POS_X_KI = 0.0;  PID_POS_X_KD = 0.0
PID_POS_Y_KP = 1.0;  PID_POS_Y_KI = 0.0;  PID_POS_Y_KD = 0.0
PID_POS_Z_KP = 2.0;  PID_POS_Z_KI = 0.5;  PID_POS_Z_KD = 0.0

# --- Limits ---
PID_VEL_ROLL_MAX     = 20.0    # deg
PID_VEL_PITCH_MAX    = 20.0    # deg
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
POSITION_RATE      = 100       # Hz
ATTITUDE_UPDATE_DT = 1.0 / ATTITUDE_RATE   # 0.002 s
POSITION_UPDATE_DT = 1.0 / POSITION_RATE   # 0.01  s
POS_EVERY          = ATTITUDE_RATE // POSITION_RATE  # = 5

# Motor dynamics filter
MOTOR_TAU = 0.02   # s  (brushed motor time constant)
MOTOR_ALPHA = ATTITUDE_UPDATE_DT / (MOTOR_TAU + ATTITUDE_UPDATE_DT)


# =====================================================================
# 3. State scaling and cost matrices
# =====================================================================

X_MAX    = 1.0;   X_DMAX    = 1.0
Y_MAX    = 1.0;   Y_DMAX    = 1.0
Z_MAX    = 1.0;   Z_DMAX    = 1.0
PHI_MAX  = np.deg2rad(30);  PHI_DMAX  = np.deg2rad(200)
THETA_MAX= np.deg2rad(30);  THETA_DMAX= np.deg2rad(200)
PSI_MAX  = np.deg2rad(45);  PSI_DMAX  = np.deg2rad(120)

X_SCALE = np.array([X_MAX, X_DMAX, Y_MAX, Y_DMAX, Z_MAX, Z_DMAX,
                    PHI_MAX, PHI_DMAX, THETA_MAX, THETA_DMAX,
                    PSI_MAX, PSI_DMAX], dtype=np.float32)

STATE_CLIP_MIN = np.array([
    -5.0 * X_MAX, -5.0 * X_DMAX, 
    -5.0 * Y_MAX, -5.0 * Y_DMAX, 
    -2.0 * Z_MAX, -5.0 * Z_DMAX, # On autorise le drone à tomber un peu (z < 0)
    -np.pi, -10.0 * PHI_DMAX, 
    -np.pi, -10.0 * THETA_DMAX, 
    -np.pi, -10.0 * PSI_DMAX
], dtype=np.float32)

STATE_CLIP_MAX = np.array([
     5.0 * X_MAX,  5.0 * X_DMAX, 
     5.0 * Y_MAX,  5.0 * Y_DMAX, 
     5.0 * Z_MAX,  5.0 * Z_DMAX, 
     np.pi,  10.0 * PHI_DMAX, 
     np.pi,  10.0 * THETA_DMAX, 
     np.pi,  10.0 * PSI_DMAX
], dtype=np.float32)

Q_DIAG = np.array([
    1/X_MAX**2,     1/X_DMAX**2,
    1/Y_MAX**2,     1/Y_DMAX**2,
    1/Z_MAX**2,     1/Z_DMAX**2,
    1/PHI_MAX**2,   1/PHI_DMAX**2,
    1/THETA_MAX**2, 1/THETA_DMAX**2,
    1/PSI_MAX**2,   1/PSI_DMAX**2,
], dtype=np.float32)

def get_action_cost_weights(setpoint_mode):
    """Renvoie les poids de la pénalité quadratique pour la sortie du NN."""
    if setpoint_mode == 'pos_sp':
        # Pénalise les grands setpoints de position (en mètres)
        return np.array([1/X_MAX**2, 1/Y_MAX**2, 1/Z_MAX**2], dtype=np.float32)
    elif setpoint_mode == 'vel_sp':
        # Pénalise les grandes vitesses demandées
        return np.array([1/X_DMAX**2, 1/Y_DMAX**2, 1/Z_DMAX**2], dtype=np.float32)
    elif setpoint_mode == 'att_sp':
        # roll(deg), pitch(deg), yaw_rate(deg/s), thrust(u16)
        return np.array([
            1 / PolicyMLPPID.ATT_RP_SCALE**2,
            1 / PolicyMLPPID.ATT_RP_SCALE**2,
            1 / PolicyMLPPID.ATT_YR_SCALE**2,
            1 / PolicyMLPPID.THRUST_HOVER**2  # Pénalise l'écart par rapport à 0, à ajuster si besoin
        ], dtype=np.float32)
    else: # rate_sp
        return np.array([
            1 / PolicyMLPPID.RATE_SCALE**2,
            1 / PolicyMLPPID.RATE_SCALE**2,
            1 / PolicyMLPPID.RATE_SCALE**2,
            1 / PolicyMLPPID.THRUST_HOVER**2
        ], dtype=np.float32)


# =====================================================================
# 4. Simulation timing
# =====================================================================

DT_INNER   = ATTITUDE_UPDATE_DT        # 0.002 s per inner step
T_SIM      = 4.0                       # simulation duration [s]
INNER_STEPS = int(T_SIM * ATTITUDE_RATE)  # = 2000

# Pre-compute allocation matrix (CF2X X-configuration)
# motor layout from validate_nn_pybullet.py / cf_firmware_pid_sim.py
_a = KF * L / np.sqrt(2)
ALLOC_MATRIX = np.array([
    [ KF,   KF,   KF,   KF  ],
    [ -_a,  -_a,   _a,   _a  ],
    [ -_a,   _a,   _a,  -_a  ],
    [ -KM,   KM,  -KM,   KM  ],
], dtype=np.float32)
ALLOC_MATRIX_T = torch.tensor(ALLOC_MATRIX.T, dtype=torch.float32)  # (4,4)

print(f"[CF2X] m={M}, Ixx={I_X}, KF={KF}, KM={KM}")
print(f"[CF2X] HOVER_RPM={HOVER_RPM:.1f}, MAX_RPM={MAX_RPM:.1f}")
print(f"[Timing] dt={DT_INNER*1000:.1f}ms, inner_steps={INNER_STEPS}, "
      f"pos_every={POS_EVERY}")


# =====================================================================
# 5. PID State Layout
# =====================================================================
#
# pid_state: (B, 28)
#
#  Indices  PID
#  [0:2]    pidX          (integ, prevMeas)
#  [2:4]    pidY
#  [4:6]    pidZ
#  [6:8]    pidVX
#  [8:10]   pidVY
#  [10:12]  pidVZ
#  [12:14]  pidRoll
#  [14:16]  pidPitch
#  [16:18]  pidYaw
#  [18:20]  pidRollRate
#  [20:22]  pidPitchRate
#  [22:24]  pidYawRate
#  [24]     attitude_desired_yaw  (deg)
#  [25]     last_att_roll         (deg)
#  [26]     last_att_pitch        (deg)
#  [27]     actuator_thrust       (uint16)

IDX_PX   = (0,  2)   # pidX
IDX_PY   = (2,  4)
IDX_PZ   = (4,  6)
IDX_PVX  = (6,  8)
IDX_PVY  = (8,  10)
IDX_PVZ  = (10, 12)
IDX_PR   = (12, 14)  # pidRoll
IDX_PP   = (14, 16)  # pidPitch
IDX_PYW  = (16, 18)  # pidYaw
IDX_PRR  = (18, 20)  # pidRollRate
IDX_PPR  = (20, 22)  # pidPitchRate
IDX_PYR  = (22, 24)  # pidYawRate
IDX_YAW_DES = 24
IDX_LAST_ROLL = 25
IDX_LAST_PITCH = 26
IDX_THRUST = 27
PID_STATE_DIM = 28

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


# =====================================================================
# 8. Differentiable Position Controller  (pos → vel → att+thrust)
# =====================================================================

def position_controller_update(
    pid_state: torch.Tensor,    # (B, 28)
    state_pos: torch.Tensor,    # (B, 3)  [x, y, z] meters
    state_vel: torch.Tensor,    # (B, 3)  [vx, vy, vz] m/s
    setpoint_pos: torch.Tensor, # (B, 3)  [x, y, z] meters (desired)
    state_yaw_rad: torch.Tensor,# (B,)    yaw in radians
) -> tuple:
    """Run position → velocity → attitude+thrust cascade (100 Hz).

    Returns (thrust_u16, roll_deg, pitch_deg, pid_state_new).
    All intermediate values are in degrees or m/s as per the firmware.
    """
    B = state_pos.shape[0]
    ps = pid_state.clone()

    cosyaw = torch.cos(state_yaw_rad)   # (B,)
    sinyaw = torch.sin(state_yaw_rad)   # (B,)

    sp_x = setpoint_pos[:, 0];  sp_y = setpoint_pos[:, 1];  sp_z = setpoint_pos[:, 2]
    st_x = state_pos[:, 0];     st_y = state_pos[:, 1];     st_z = state_pos[:, 2]
    st_vx = state_vel[:, 0];    st_vy = state_vel[:, 1];    st_vz = state_vel[:, 2]

    # Rotate setpoint to body-yaw frame
    setp_body_x =  sp_x * cosyaw + sp_y * sinyaw
    setp_body_y = -sp_x * sinyaw + sp_y * cosyaw

    # Rotate state position to body-yaw frame
    state_body_x =  st_x * cosyaw + st_y * sinyaw
    state_body_y = -st_x * sinyaw + st_y * cosyaw

    # --- Position PIDs → desired velocity ---
    pos_vel_x_lim = PID_POS_VEL_X_MAX * VEL_MAX_OVERHEAD
    pos_vel_y_lim = PID_POS_VEL_Y_MAX * VEL_MAX_OVERHEAD
    pos_vel_z_lim = max(PID_POS_VEL_Z_MAX, 0.5) * VEL_MAX_OVERHEAD

    vel_sp_x, integ_px, _ = pid_update_batch(
        ps[:, IDX_PX[0]], ps[:, IDX_PX[0]+1],
        state_body_x, setp_body_x,
        PID_POS_X_KP, PID_POS_X_KI, PID_POS_X_KD,
        POSITION_UPDATE_DT, DEFAULT_ILIM, pos_vel_x_lim)
    ps = _set_pid(ps, IDX_PX, integ_px, state_body_x)

    vel_sp_y, integ_py, _ = pid_update_batch(
        ps[:, IDX_PY[0]], ps[:, IDX_PY[0]+1],
        state_body_y, setp_body_y,
        PID_POS_Y_KP, PID_POS_Y_KI, PID_POS_Y_KD,
        POSITION_UPDATE_DT, DEFAULT_ILIM, pos_vel_y_lim)
    ps = _set_pid(ps, IDX_PY, integ_py, state_body_y)

    vel_sp_z, integ_pz, _ = pid_update_batch(
        ps[:, IDX_PZ[0]], ps[:, IDX_PZ[0]+1],
        st_z, sp_z,
        PID_POS_Z_KP, PID_POS_Z_KI, PID_POS_Z_KD,
        POSITION_UPDATE_DT, DEFAULT_ILIM, pos_vel_z_lim)
    ps = _set_pid(ps, IDX_PZ, integ_pz, st_z)

    # Rotate state velocity to body-yaw frame
    state_body_vx =  st_vx * cosyaw + st_vy * sinyaw
    state_body_vy = -st_vx * sinyaw + st_vy * cosyaw

    # --- Velocity PIDs → roll/pitch/thrust ---
    rp_limit_r = PID_VEL_ROLL_MAX  * RP_LIMIT_OVERHEAD
    rp_limit_p = PID_VEL_PITCH_MAX * RP_LIMIT_OVERHEAD
    vz_limit    = UINT16_MAX / 2.0 / THRUST_SCALE

    pitch_raw, integ_pvx, _ = pid_update_batch(
        ps[:, IDX_PVX[0]], ps[:, IDX_PVX[0]+1],
        state_body_vx, vel_sp_x,
        PID_VEL_X_KP, PID_VEL_X_KI, PID_VEL_X_KD,
        POSITION_UPDATE_DT, DEFAULT_ILIM, rp_limit_p)
    ps = _set_pid(ps, IDX_PVX, integ_pvx, state_body_vx)
    pitch_deg = torch.clamp(-pitch_raw, -PID_VEL_PITCH_MAX, PID_VEL_PITCH_MAX)

    roll_raw, integ_pvy, _ = pid_update_batch(
        ps[:, IDX_PVY[0]], ps[:, IDX_PVY[0]+1],
        state_body_vy, vel_sp_y,
        PID_VEL_Y_KP, PID_VEL_Y_KI, PID_VEL_Y_KD,
        POSITION_UPDATE_DT, DEFAULT_ILIM, rp_limit_r)
    ps = _set_pid(ps, IDX_PVY, integ_pvy, state_body_vy)
    roll_deg = torch.clamp(-roll_raw, -PID_VEL_ROLL_MAX, PID_VEL_ROLL_MAX)

    thrust_raw, integ_pvz, _ = pid_update_batch(
        ps[:, IDX_PVZ[0]], ps[:, IDX_PVZ[0]+1],
        st_vz, vel_sp_z,
        PID_VEL_Z_KP, PID_VEL_Z_KI, PID_VEL_Z_KD,
        POSITION_UPDATE_DT, DEFAULT_ILIM, vz_limit)
    ps = _set_pid(ps, IDX_PVZ, integ_pvz, st_vz)

    thrust_u16 = thrust_raw * THRUST_SCALE + PID_VEL_THRUST_BASE
    thrust_u16 = torch.clamp(thrust_u16, PID_VEL_THRUST_MIN, UINT16_MAX)

    return thrust_u16, roll_deg, pitch_deg, ps


def velocity_controller_update(
    pid_state: torch.Tensor,    # (B, 28)
    state_vel: torch.Tensor,    # (B, 3)  [vx, vy, vz] m/s
    setpoint_vel: torch.Tensor, # (B, 3)  [vx, vy, vz] m/s (desired, body-yaw frame)
    state_yaw_rad: torch.Tensor,# (B,)
) -> tuple:
    """Run only the velocity → attitude+thrust cascade (bypasses position PID).

    Returns (thrust_u16, roll_deg, pitch_deg, pid_state_new).
    """
    ps = pid_state.clone()

    cosyaw = torch.cos(state_yaw_rad)
    sinyaw = torch.sin(state_yaw_rad)

    st_vx = state_vel[:, 0];  st_vy = state_vel[:, 1];  st_vz = state_vel[:, 2]

    # Rotate state velocity to body-yaw frame
    state_body_vx =  st_vx * cosyaw + st_vy * sinyaw
    state_body_vy = -st_vx * sinyaw + st_vy * cosyaw

    vel_sp_x = setpoint_vel[:, 0]
    vel_sp_y = setpoint_vel[:, 1]
    vel_sp_z = setpoint_vel[:, 2]

    rp_limit_r = PID_VEL_ROLL_MAX  * RP_LIMIT_OVERHEAD
    rp_limit_p = PID_VEL_PITCH_MAX * RP_LIMIT_OVERHEAD
    vz_limit    = UINT16_MAX / 2.0 / THRUST_SCALE

    pitch_raw, integ_pvx, _ = pid_update_batch(
        ps[:, IDX_PVX[0]], ps[:, IDX_PVX[0]+1],
        state_body_vx, vel_sp_x,
        PID_VEL_X_KP, PID_VEL_X_KI, PID_VEL_X_KD,
        POSITION_UPDATE_DT, DEFAULT_ILIM, rp_limit_p)
    ps = _set_pid(ps, IDX_PVX, integ_pvx, state_body_vx)
    pitch_deg = torch.clamp(-pitch_raw, -PID_VEL_PITCH_MAX, PID_VEL_PITCH_MAX)

    roll_raw, integ_pvy, _ = pid_update_batch(
        ps[:, IDX_PVY[0]], ps[:, IDX_PVY[0]+1],
        state_body_vy, vel_sp_y,
        PID_VEL_Y_KP, PID_VEL_Y_KI, PID_VEL_Y_KD,
        POSITION_UPDATE_DT, DEFAULT_ILIM, rp_limit_r)
    ps = _set_pid(ps, IDX_PVY, integ_pvy, state_body_vy)
    roll_deg = torch.clamp(-roll_raw, -PID_VEL_ROLL_MAX, PID_VEL_ROLL_MAX)

    thrust_raw, integ_pvz, _ = pid_update_batch(
        ps[:, IDX_PVZ[0]], ps[:, IDX_PVZ[0]+1],
        st_vz, vel_sp_z,
        PID_VEL_Z_KP, PID_VEL_Z_KI, PID_VEL_Z_KD,
        POSITION_UPDATE_DT, DEFAULT_ILIM, vz_limit)
    ps = _set_pid(ps, IDX_PVZ, integ_pvz, st_vz)

    thrust_u16 = thrust_raw * THRUST_SCALE + PID_VEL_THRUST_BASE
    thrust_u16 = torch.clamp(thrust_u16, PID_VEL_THRUST_MIN, UINT16_MAX)

    return thrust_u16, roll_deg, pitch_deg, ps


# =====================================================================
# 9. Differentiable Attitude Controller  (att → rate → motor cmds)
# =====================================================================

def attitude_controller_update(
    pid_state: torch.Tensor,         # (B, 28)
    state_rpy_deg: torch.Tensor,     # (B, 3)  [roll, pitch, yaw] deg
    state_gyro_deg: torch.Tensor,    # (B, 3)  [p, q, r] deg/s  (body frame)
    desired_roll_deg: torch.Tensor,  # (B,)
    desired_pitch_deg: torch.Tensor, # (B,)
) -> tuple:
    """Run attitude → rate PID cascade (500 Hz).

    attitude_desired_yaw is maintained inside pid_state[IDX_YAW_DES].
    Returns (roll_cmd, pitch_cmd, yaw_cmd, pid_state_new).
    roll_cmd / pitch_cmd / yaw_cmd are floats in [-32767, 32767].
    """
    ps = pid_state.clone()

    roll_a  = state_rpy_deg[:, 0]
    pitch_a = state_rpy_deg[:, 1]
    yaw_a   = state_rpy_deg[:, 2]

    gyro_x = state_gyro_deg[:, 0]
    gyro_y = state_gyro_deg[:, 1]
    gyro_z = state_gyro_deg[:, 2]

    yaw_des = ps[:, IDX_YAW_DES]   # running yaw setpoint [deg]

    # --- Attitude PIDs → desired rates ---
    roll_rate_des, integ_pr, _ = pid_update_batch(
        ps[:, IDX_PR[0]], ps[:, IDX_PR[0]+1],
        roll_a, desired_roll_deg,
        PID_ROLL_KP, PID_ROLL_KI, PID_ROLL_KD,
        ATTITUDE_UPDATE_DT, PID_ROLL_INTEGRATION_LIMIT)
    ps = _set_pid(ps, IDX_PR, integ_pr, roll_a)

    pitch_rate_des, integ_pp, _ = pid_update_batch(
        ps[:, IDX_PP[0]], ps[:, IDX_PP[0]+1],
        pitch_a, desired_pitch_deg,
        PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD,
        ATTITUDE_UPDATE_DT, PID_PITCH_INTEGRATION_LIMIT)
    ps = _set_pid(ps, IDX_PP, integ_pp, pitch_a)

    yaw_rate_des, integ_pyw, _ = pid_update_batch(
        ps[:, IDX_PYW[0]], ps[:, IDX_PYW[0]+1],
        yaw_a, yaw_des,
        PID_YAW_KP, PID_YAW_KI, PID_YAW_KD,
        ATTITUDE_UPDATE_DT, PID_YAW_INTEGRATION_LIMIT,
        is_yaw=True)
    ps = _set_pid(ps, IDX_PYW, integ_pyw, yaw_a)

    # --- Rate PIDs → motor commands ---
    roll_cmd, integ_prr, _ = pid_update_batch(
        ps[:, IDX_PRR[0]], ps[:, IDX_PRR[0]+1],
        gyro_x, roll_rate_des,
        PID_ROLL_RATE_KP, PID_ROLL_RATE_KI, PID_ROLL_RATE_KD,
        ATTITUDE_UPDATE_DT, PID_ROLL_RATE_INTEGRATION_LIMIT)
    ps = _set_pid(ps, IDX_PRR, integ_prr, gyro_x)
    roll_cmd = torch.clamp(roll_cmd, -32767.0, 32767.0)

    pitch_cmd, integ_ppr, _ = pid_update_batch(
        ps[:, IDX_PPR[0]], ps[:, IDX_PPR[0]+1],
        gyro_y, pitch_rate_des,
        PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD,
        ATTITUDE_UPDATE_DT, PID_PITCH_RATE_INTEGRATION_LIMIT)
    ps = _set_pid(ps, IDX_PPR, integ_ppr, gyro_y)
    pitch_cmd = torch.clamp(pitch_cmd, -32767.0, 32767.0)

    yaw_cmd, integ_pyr, _ = pid_update_batch(
        ps[:, IDX_PYR[0]], ps[:, IDX_PYR[0]+1],
        gyro_z, yaw_rate_des,
        PID_YAW_RATE_KP, PID_YAW_RATE_KI, PID_YAW_RATE_KD,
        ATTITUDE_UPDATE_DT, PID_YAW_RATE_INTEGRATION_LIMIT)
    ps = _set_pid(ps, IDX_PYR, integ_pyr, gyro_z)
    yaw_cmd = torch.clamp(yaw_cmd, -32767.0, 32767.0)

    # Negate yaw output (matches firmware controller_pid.c:131)
    yaw_cmd = -yaw_cmd

    return roll_cmd, pitch_cmd, yaw_cmd, ps


def rate_controller_update(
    pid_state: torch.Tensor,          # (B, 28)
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
# 9. Power Distribution  (thrust + roll/pitch/yaw → 4 motor PWMs)
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
# 10. PWM → RPM  (differentiable)
# =====================================================================

def pwm_to_rpm_torch(motor_pwm: torch.Tensor) -> torch.Tensor:
    """Convert uint16 PWM → RPM via thrust = (pwm/65535)*0.12 N.

    Skips 8-bit truncation (non-differentiable) for training.
    """
    thrust = (motor_pwm / UINT16_MAX) * CF2_THRUST_MAX_PER_MOTOR
    thrust = torch.clamp(thrust, min=1e-5)
    return torch.sqrt(thrust / KF)


# =====================================================================
# 11. Motor Dynamics Filter  (1st-order LP, differentiable)
# =====================================================================

def motor_filter_step(
    rpm_prev: torch.Tensor,  # (B, 4)
    rpm_cmd: torch.Tensor,   # (B, 4)
) -> torch.Tensor:           # (B, 4)
    return MOTOR_ALPHA * rpm_cmd + (1.0 - MOTOR_ALPHA) * rpm_prev


# =====================================================================
# 12. RPM → Wrench  (allocation matrix)
# =====================================================================

def rpm_to_wrench(rpm: torch.Tensor, device) -> torch.Tensor:
    """Convert motor RPMs to wrench [F, tau_x, tau_y, tau_z].

    rpm: (B, 4)  → wrench: (B, 4)
    """
    omega_sq = rpm ** 2
    A = ALLOC_MATRIX_T.to(device)          # (4, 4)
    return omega_sq @ A                    # (B, 4)


# =====================================================================
# 13. Nonlinear Dynamics  (reused from train_nn_cf2x.py)
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


def dynamics_substep(state: torch.Tensor, wrench: torch.Tensor,
                     dt: float) -> torch.Tensor:
    """One semi-implicit Euler sub-step.  state: (B, 12), wrench: (B, 4)."""
    x     = state[:, 0];  vx = state[:, 1]
    y     = state[:, 2];  vy = state[:, 3]
    z     = state[:, 4];  vz = state[:, 5]
    phi   = state[:, 6];  p  = state[:, 7]
    theta = state[:, 8];  q  = state[:, 9]
    psi   = state[:, 10]; r  = state[:, 11]

    F     = wrench[:, 0]
    tau_x = wrench[:, 1]
    tau_y = wrench[:, 2]
    tau_z = wrench[:, 3]

    R = rotation_matrix_zyx(phi, theta, psi)
    thrust_world = F.unsqueeze(-1) * R[:, :, 2]

    ax = thrust_world[:, 0] / M
    ay = thrust_world[:, 1] / M
    az = thrust_world[:, 2] / M - G

    gyro_x = (I_Z - I_Y) * q * r
    gyro_y = (I_X - I_Z) * p * r
    gyro_z = (I_Y - I_X) * p * q

    p_dot = (tau_x - gyro_x) / I_X
    q_dot = (tau_y - gyro_y) / I_Y
    r_dot = (tau_z - gyro_z) / I_Z

    vx_new = vx + dt * ax;   vy_new = vy + dt * ay;   vz_new = vz + dt * az
    p_new  = p  + dt * p_dot; q_new = q + dt * q_dot; r_new  = r + dt * r_dot

    x_new     = x     + dt * vx_new
    y_new     = y     + dt * vy_new
    z_new     = z     + dt * vz_new
    phi_new   = phi   + dt * p_new
    theta_new = theta + dt * q_new
    psi_new   = psi   + dt * r_new

    return torch.stack([
        x_new, vx_new, y_new, vy_new, z_new, vz_new,
        phi_new, p_new, theta_new, q_new, psi_new, r_new
    ], dim=-1)


# =====================================================================
# 14. Linear (ZOH) Dynamics at 500 Hz
# =====================================================================

def _build_linear_model_500hz():
    """ZOH discretization of the hover linear model at dt=ATTITUDE_UPDATE_DT."""
    A = np.zeros((12, 12))
    A[0,1]=1; A[2,3]=1; A[4,5]=1; A[6,7]=1; A[8,9]=1; A[10,11]=1
    A[1,8] = G;  A[3,6] = -G
    B = np.zeros((12, 4))
    B[5,0]=1/M; B[7,1]=1/I_X; B[9,2]=1/I_Y; B[11,3]=1/I_Z

    c = np.zeros((12,1)); c[5,0] = -G
    n, m = 12, 4
    A_aug = np.zeros((n+1,n+1)); A_aug[:n,:n]=A; A_aug[:n,n]=c.squeeze()
    B_aug = np.zeros((n+1,m)); B_aug[:n,:]=B
    C_aug = np.zeros((n,n+1)); C_aug[:,:n]=np.eye(n)
    D_aug = np.zeros((n,m))
    Ad_aug, Bd_aug, _, _, _ = cont2discrete(
        (A_aug, B_aug, C_aug, D_aug), ATTITUDE_UPDATE_DT, method='zoh')
    Ad = Ad_aug[:n,:n]
    Bd = Bd_aug[:n,:]
    d  = Ad_aug[:n, n]
    return Ad, Bd, d


_Ad_500, _Bd_500, _d_500 = _build_linear_model_500hz()
print(f"[Linear 500Hz] hover offset d[5]={_d_500[5]:.6f} (should be ~-{G*ATTITUDE_UPDATE_DT:.6f})")


def dynamics_step_linear(state: torch.Tensor, wrench: torch.Tensor,
                         device) -> torch.Tensor:
    """One ZOH linear step at 500 Hz.  state: (B, 12), wrench: (B, 4)."""
    Ad = torch.tensor(_Ad_500, dtype=torch.float32, device=device)
    Bd = torch.tensor(_Bd_500, dtype=torch.float32, device=device)
    d  = torch.tensor(_d_500,  dtype=torch.float32, device=device).unsqueeze(0)
    return (state @ Ad.T) + (wrench @ Bd.T) + d


# =====================================================================
# 15. Neural Network  (setpoint-mode-specific head)
# =====================================================================

class PolicyMLPPID(nn.Module):
    """MLP controller outputting setpoints for the PID cascade.

    setpoint_mode determines output semantics and activation:
      'pos_sp'  → (x, y, z) [m], tanh * pos_scale
      'vel_sp'  → (vx, vy, vz) [m/s], tanh * vel_scale
      'att_sp'  → (roll_deg, pitch_deg, yaw_rate_deg_s, thrust_u16)
      'rate_sp' → (roll_rate_deg_s, pitch_rate_deg_s, yaw_rate_deg_s, thrust_u16)
    """

    POS_SCALE  = 2.0    # m
    VEL_SCALE  = 1.0    # m/s
    ATT_RP_SCALE    = 20.0   # deg  (PID_VEL_ROLL/PITCH_MAX)
    ATT_YR_SCALE    = 200.0  # deg/s
    RATE_SCALE      = 720.0  # deg/s
    THRUST_HOVER    = PID_VEL_THRUST_BASE  # = 36000

    def __init__(self, x_scale: np.ndarray, setpoint_mode: str, hidden: int = 64):
        super().__init__()
        assert setpoint_mode in ('pos_sp', 'vel_sp', 'att_sp', 'rate_sp')
        self.setpoint_mode = setpoint_mode

        output_dim = 3 if setpoint_mode in ('pos_sp', 'vel_sp') else 4

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

        if self.setpoint_mode in ('att_sp', 'rate_sp'):
            # Initialize thrust to hover: sigmoid(b) * 65535 = THRUST_HOVER
            ratio = self.THRUST_HOVER / UINT16_MAX
            last.bias.data[3] = float(np.log(ratio / (1.0 - ratio)))
        # For pos_sp and vel_sp, zero bias → output (0,0,0) at init = hover

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = x / self.x_scale
        raw = self.net(x_n)

        if self.setpoint_mode == 'pos_sp':
            return torch.tanh(raw) * self.POS_SCALE

        if self.setpoint_mode == 'vel_sp':
            return torch.tanh(raw) * self.VEL_SCALE

        if self.setpoint_mode == 'att_sp':
            roll  = torch.tanh(raw[..., [0]]) * self.ATT_RP_SCALE
            pitch = torch.tanh(raw[..., [1]]) * self.ATT_RP_SCALE
            yr    = torch.tanh(raw[..., [2]]) * self.ATT_YR_SCALE
            thr   = torch.sigmoid(raw[..., [3]]) * UINT16_MAX
            return torch.cat([roll, pitch, yr, thr], dim=-1)

        # rate_sp
        p_rate = torch.tanh(raw[..., [0]]) * self.RATE_SCALE
        q_rate = torch.tanh(raw[..., [1]]) * self.RATE_SCALE
        r_rate = torch.tanh(raw[..., [2]]) * self.RATE_SCALE
        thr    = torch.sigmoid(raw[..., [3]]) * UINT16_MAX
        return torch.cat([p_rate, q_rate, r_rate, thr], dim=-1)


# =====================================================================
# 16. FlowDeck Delay Buffer  (differentiable)
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
# 17. Yaw accumulator helper
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
# 18. Main Rollout Function
# =====================================================================

def rollout_pid(
    policy: PolicyMLPPID,
    x0: torch.Tensor,               # (B, 12)
    setpoint_mode: str,
    inner_steps: int = INNER_STEPS,
    use_flowdeck: bool = False,
    flowdeck_delay_ms: float = 110.0,
    use_motor_filter: bool = True,
    linearized: bool = False,
) -> tuple:
    """Roll out the policy through the differentiable PID cascade.

    Returns (X, cost_sum) where:
      X: (B, inner_steps//POS_EVERY, 12) state trajectory sampled at 100 Hz
      cost_sum: scalar (sum of quadratic costs)
    """
    B = x0.shape[0]
    dev = x0.device

    # --- Initialize PID state ---
    pid_state = torch.zeros(B, PID_STATE_DIM, device=dev, dtype=torch.float32)

    # prevMeas for pos PIDs = initial position (so D term is zero on step 1)
    pid_state = _set_pid(pid_state, IDX_PX,
                         pid_state[:, IDX_PX[0]], x0[:, 0])
    pid_state = _set_pid(pid_state, IDX_PY,
                         pid_state[:, IDX_PY[0]], x0[:, 2])
    pid_state = _set_pid(pid_state, IDX_PZ,
                         pid_state[:, IDX_PZ[0]], x0[:, 4])

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
    R_diag = get_action_cost_weights(setpoint_mode)
    R = torch.tensor(R_diag, dtype=torch.float32, device=dev)

    state = x0.clone()
    cost_sum = torch.zeros(1, device=dev)

    # Storage for sampled trajectory (at POSITION_RATE = 100 Hz)
    n_samples = inner_steps // POS_EVERY
    X = torch.zeros(B, n_samples, 12, device=dev)
    sample_idx = 0

    # Cached setpoints (updated every POS_EVERY steps)
    # Initialized to values that produce hover
    thrust_cached   = torch.full((B,), PID_VEL_THRUST_BASE, device=dev)
    roll_d_cached   = torch.zeros(B, device=dev)
    pitch_d_cached  = torch.zeros(B, device=dev)

    # For att_sp / rate_sp: yaw_rate and rates cached
    yaw_rate_cached  = torch.zeros(B, device=dev)
    rate_r_cached    = torch.zeros(B, device=dev)
    rate_p_cached    = torch.zeros(B, device=dev)

    nn_out_prev = None
    smoothness_weight = 0.5  # Poids de la pénalité de variation

    for k in range(inner_steps):

        # Extract state components
        pos   = state[:, [0, 2, 4]]              # (B, 3)
        vel   = state[:, [1, 3, 5]]              # (B, 3)
        phi   = state[:, 6];   p_rate = state[:, 7]
        theta = state[:, 8];   q_rate = state[:, 9]
        psi   = state[:, 10];  r_rate = state[:, 11]

        rpy_deg  = torch.stack([
            torch.rad2deg(phi), torch.rad2deg(theta), torch.rad2deg(psi)
        ], dim=-1)
        gyro_deg = torch.stack([
            torch.rad2deg(p_rate), torch.rad2deg(q_rate), torch.rad2deg(r_rate)
        ], dim=-1)

        # ---- FlowDeck push (every inner step = 500 Hz) ----
        if use_flowdeck:
            flowdeck.push(pos, vel)

        # =========================================================
        # INFERENCE DU NN
        # =========================================================
        # Si on est en mode 'pos_sp' ou 'vel_sp', la consigne (et la boucle pos) tourne à 100 Hz
        if setpoint_mode in ('pos_sp', 'vel_sp'):
            if k % POS_EVERY == 0:
                if use_flowdeck:
                    pos_pid, vel_pid = flowdeck.read_delayed()
                else:
                    pos_pid, vel_pid = pos, vel

                nn_out = policy(state)
                
                if setpoint_mode == 'pos_sp':
                    thrust_cached, roll_d_cached, pitch_d_cached, pid_state = \
                        position_controller_update(pid_state, pos_pid, vel_pid, nn_out, psi)
                else:
                    thrust_cached, roll_d_cached, pitch_d_cached, pid_state = \
                        velocity_controller_update(pid_state, vel_pid, nn_out, psi)
                        
                pid_state = _set_scalar(pid_state, IDX_THRUST, thrust_cached)
                pid_state = _set_scalar(pid_state, IDX_LAST_ROLL, roll_d_cached)
                pid_state = _set_scalar(pid_state, IDX_LAST_PITCH, pitch_d_cached)

        # Si on est en mode 'att_sp' ou 'rate_sp', le NN tourne à 500 Hz !
        else:
            nn_out = policy(state) # Appelé à chaque itération k
            
            if setpoint_mode == 'att_sp':
                roll_d_cached  = nn_out[:, 0]
                pitch_d_cached = nn_out[:, 1]
                yaw_rate_cached = nn_out[:, 2]
                thrust_cached  = nn_out[:, 3]
                
                pid_state = _set_scalar(pid_state, IDX_THRUST, thrust_cached)
                pid_state = _set_scalar(pid_state, IDX_LAST_ROLL, roll_d_cached)
                pid_state = _set_scalar(pid_state, IDX_LAST_PITCH, pitch_d_cached)
            else: # rate_sp
                rate_r_cached   = nn_out[:, 0]
                rate_p_cached   = nn_out[:, 1]
                yaw_rate_cached = nn_out[:, 2]
                thrust_cached   = nn_out[:, 3]
                pid_state = _set_scalar(pid_state, IDX_THRUST, thrust_cached)

        # ---- Yaw accumulation (every inner step, from yaw_rate) ----
        if setpoint_mode in ('att_sp', 'rate_sp'):
            pid_state = _accumulate_yaw(pid_state, yaw_rate_cached)

        # ---- Attitude + Rate PIDs (every inner step = 500 Hz) ----
        thrust = pid_state[:, IDX_THRUST]
        r_desired = pid_state[:, IDX_LAST_ROLL]
        p_desired = pid_state[:, IDX_LAST_PITCH]

        if setpoint_mode in ('pos_sp', 'vel_sp', 'att_sp'):
            roll_cmd, pitch_cmd, yaw_cmd, pid_state = \
                attitude_controller_update(
                    pid_state, rpy_deg, gyro_deg, r_desired, p_desired)
        else:  # rate_sp
            roll_cmd, pitch_cmd, yaw_cmd, pid_state = \
                rate_controller_update(
                    pid_state, gyro_deg,
                    rate_r_cached, rate_p_cached, yaw_rate_cached)

        # ---- Power distribution ----
        motor_pwm = power_distribute(thrust, roll_cmd, pitch_cmd, yaw_cmd)

        # ---- PWM → RPM ----
        rpm_cmd = pwm_to_rpm_torch(motor_pwm)

        # ---- Motor dynamics filter ----
        if use_motor_filter:
            motor_rpm = motor_filter_step(motor_rpm, rpm_cmd)
        else:
            motor_rpm = rpm_cmd

        # ---- RPM → wrench ----
        wrench = rpm_to_wrench(motor_rpm, dev)

        # ---- Physics step ----
        if linearized:
            state = dynamics_step_linear(state, wrench, dev)
        else:
            state = dynamics_substep(state, wrench, DT_INNER)

        # ---- State clipping (to stabilize training and avoid NaN) ----
        clip_min_t = torch.tensor(STATE_CLIP_MIN, device=dev)
        clip_max_t = torch.tensor(STATE_CLIP_MAX, device=dev)
        state = torch.max(torch.min(state, clip_max_t), clip_min_t)

        # /!\ NOUVEAU : Le remède anti-explosion de gradient
        # On force les gradients à rester entre -10 et +10 lorsqu'ils remontent le temps
        if state.requires_grad:
            state.register_hook(lambda grad: torch.clamp(grad, min=-10.0, max=10.0))
            
        if pid_state.requires_grad:
            pid_state.register_hook(lambda grad: torch.clamp(grad, min=-10.0, max=10.0))

        # ---- Sample trajectory & accumulate cost (every POS_EVERY steps) ----
        if (k + 1) % POS_EVERY == 0:
            X[:, sample_idx, :] = state
            cost_state = (state ** 2 * Q).sum(dim=1).mean()
            cost_input = (nn_out ** 2 * R).sum(dim=1).mean()
            # 3. NOUVEAU : Coût de lissage (Smoothness)
            cost_smooth = torch.zeros_like(cost_input)
            if nn_out_prev is not None:
                # Pénalise la différence au carré entre la consigne actuelle et la précédente
                cost_smooth = smoothness_weight * ((nn_out - nn_out_prev) ** 2 * R).sum(dim=1).mean()
            nn_out_prev = nn_out
            cost_step = cost_state + cost_input + cost_smooth
            cost_sum = cost_sum + cost_step
            sample_idx += 1

    return X, cost_sum


# =====================================================================
# 19. Training Utilities
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
# 20. Training Loop
# =====================================================================

def train(
    epochs: int = 500,
    lr: float = 1e-3,
    hidden: int = 64,
    setpoint_mode: str = "pos_sp",
    linearized: bool = False,
    use_flowdeck: bool = False,
    flowdeck_delay_ms: float = 110.0,
    use_motor_filter: bool = True,
    half_side: float = 0.3,
    t_sim: float = T_SIM,
    device: str = "cpu",
):
    inner_steps = int(t_sim * ATTITUDE_RATE)

    policy = PolicyMLPPID(X_SCALE, setpoint_mode, hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    cube_pts = generate_cube_points(half_side)
    x0_batch = make_x0_batch(cube_pts, device=device)

    # Verify hover init
    with torch.no_grad():
        nn_test = policy(torch.zeros(1, 12, device=device))
        print(f"\n[Init] NN output at x=0: {nn_test[0].cpu().numpy()}")

    dyn_label = "LINEAR" if linearized else "NONLINEAR"
    fd_label  = f" + FlowDeck({flowdeck_delay_ms}ms)" if use_flowdeck else ""
    print(f"[Train] mode={setpoint_mode}, dyn={dyn_label}{fd_label}")
    print(f"[Train] {len(cube_pts)} pts, epochs={epochs}, lr={lr}, "
          f"t_sim={t_sim}s ({inner_steps} inner steps)")

    for ep in range(epochs):
        X, cost = rollout_pid(
            policy, x0_batch, setpoint_mode,
            inner_steps=inner_steps,
            use_flowdeck=use_flowdeck,
            flowdeck_delay_ms=flowdeck_delay_ms,
            use_motor_filter=use_motor_filter,
            linearized=linearized,
        )

        opt.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        scheduler.step()

        #if (ep + 1) % 50 == 0 or ep == 0:
        with torch.no_grad():
            pos_end = X[:, -1, [0, 2, 4]].norm(dim=1)
            print(f"  [ep {ep+1:4d}/{epochs}]  cost={cost.item():.4e}  "
                    f"|x_T|_mean={pos_end.mean().item():.4f}  "
                    f"|x_T|_max={pos_end.max().item():.4f}")

    return policy


# =====================================================================
# 21. Main
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setpoint_mode", type=str, default="pos_sp",
                        choices=["pos_sp", "vel_sp", "att_sp", "rate_sp"])
    parser.add_argument("--linear", action="store_true",
                        help="Use linearized ZOH dynamics")
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
        setpoint_mode=args.setpoint_mode,
        linearized=args.linear,
        use_flowdeck=args.flowdeck,
        flowdeck_delay_ms=args.flowdeck_delay_ms,
        use_motor_filter=not args.no_motor_filter,
        half_side=args.half_side,
        t_sim=args.t_sim,
        device=device,
    )

    # Save
    out_dir  = os.path.dirname(os.path.abspath(__file__))
    run_name = build_run_name(args.setpoint_mode, args.linear,
                              args.epochs, args.lr, args.hidden,
                              args.flowdeck, args.tag)
    path_full = os.path.join(out_dir, f"trained_policy_{run_name}.pt")
    path_dict = os.path.join(out_dir, f"trained_weights_{run_name}.pt")

    torch.save(policy.cpu(), path_full)
    torch.save({
        "state_dict":       policy.state_dict(),
        "hidden":           args.hidden,
        "x_scale":          X_SCALE,
        "setpoint_mode":    args.setpoint_mode,
        "ctrl_freq":        ATTITUDE_RATE,
        "pos_freq":         POSITION_RATE,
        "Ts":               ATTITUDE_UPDATE_DT,
        "KF": KF, "KM": KM, "L": L, "M": M, "G": G,
        "MAX_RPM":          MAX_RPM,
        "linearized":       args.linear,
        "use_flowdeck":     args.flowdeck,
        "flowdeck_delay_ms": args.flowdeck_delay_ms,
        "use_motor_filter": not args.no_motor_filter,
        "pid_gains": {
            "roll_rate":  (PID_ROLL_RATE_KP,  PID_ROLL_RATE_KI,  PID_ROLL_RATE_KD),
            "pitch_rate": (PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD),
            "yaw_rate":   (PID_YAW_RATE_KP,   PID_YAW_RATE_KI,   PID_YAW_RATE_KD),
            "roll":       (PID_ROLL_KP,  PID_ROLL_KI,  PID_ROLL_KD),
            "pitch":      (PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD),
            "yaw":        (PID_YAW_KP,   PID_YAW_KI,   PID_YAW_KD),
            "vel_x":      (PID_VEL_X_KP, PID_VEL_X_KI, PID_VEL_X_KD),
            "vel_y":      (PID_VEL_Y_KP, PID_VEL_Y_KI, PID_VEL_Y_KD),
            "vel_z":      (PID_VEL_Z_KP, PID_VEL_Z_KI, PID_VEL_Z_KD),
            "pos_x":      (PID_POS_X_KP, PID_POS_X_KI, PID_POS_X_KD),
            "pos_y":      (PID_POS_Y_KP, PID_POS_Y_KI, PID_POS_Y_KD),
            "pos_z":      (PID_POS_Z_KP, PID_POS_Z_KI, PID_POS_Z_KD),
        },
        "run_name":         run_name,
        "train_dynamics":   "linear" if args.linear else "nonlinear",
        "epochs":           args.epochs,
        "lr":               args.lr,
    }, path_dict)

    print(f"\n[Saved] {path_full}")
    print(f"[Saved] {path_dict}")
    print(f"\nDone! Run  validate_nn_cf_pid_pybullet.py --weights {os.path.basename(path_dict)}  to validate.")
