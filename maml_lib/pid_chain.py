"""Differentiable firmware-like attitude+rate PID, mass-task aware.

Refactored from training_regulation_simu_and_real/train_nn_cf_pid.py so
that the PID chain accepts (a) any ``dynamics_step`` and (b) any
``MassParams``. The PID gains themselves are *not* mass-aware (the
firmware doesn't change them); only the dynamics step downstream is.
"""
import math

import torch

from . import config as C
from .mass_params import MassParams


_pid_cache: dict = {}


def _get_pid_gains(dev):
    if dev not in _pid_cache:
        yaw_mask = torch.tensor([False, False, True], device=dev)
        _pid_cache[dev] = {
            'att_kp':  C.ATT_KP.to(dev),  'att_ki':  C.ATT_KI.to(dev),
            'att_kd':  C.ATT_KD.to(dev),  'att_il':  C.ATT_ILIM.to(dev),
            'rate_kp': C.RATE_KP.to(dev), 'rate_ki': C.RATE_KI.to(dev),
            'rate_kd': C.RATE_KD.to(dev), 'rate_il': C.RATE_ILIM.to(dev),
            'yaw_mask': yaw_mask,
        }
    return _pid_cache[dev]


_MOTORS_PWM_BITS  = 8
_MOTORS_PWM_SHIFT = 16 - _MOTORS_PWM_BITS  # 8


def pwm_to_rpm(motor_pwm: torch.Tensor) -> torch.Tensor:
    """Replicate the CF2.1+ PWM 8-bit hardware truncation, then map to RPM.

    Straight-through estimator on the truncation: forward uses the
    truncated value, backward propagates through as identity.
    """
    scale = float(1 << _MOTORS_PWM_SHIFT)
    pwm_trunc = motor_pwm - (motor_pwm % scale)
    pwm_trunc = motor_pwm + (pwm_trunc - motor_pwm).detach()
    thrust = (pwm_trunc / C.UINT16_MAX) * C.CF2_THRUST_MAX_PER_MOTOR
    thrust = torch.clamp(thrust, min=1e-3)
    return torch.sqrt(thrust / C.KF)


def pid_update_vec(desired, measured, prev_measured, integral,
                   kp, ki, kd, dt, i_limit, yaw_mask=None):
    """Vectorised PID update over (B, 3) tensors (3 axes at once)."""
    error = desired - measured
    delta = -(measured - prev_measured)
    if yaw_mask is not None:
        ym = yaw_mask.unsqueeze(0)
        error = torch.where(ym, error - 360.0 * torch.round(error / 360.0), error)
        delta = torch.where(ym, delta - 360.0 * torch.round(delta / 360.0), delta)
    out_p = kp * error
    out_d = kd * (delta / dt)
    new_integral = integral + error * dt
    new_integral = torch.clamp(new_integral, -i_limit, i_limit)
    out_i = ki * new_integral
    return out_p + out_d + out_i, new_integral, measured


def init_pid_state(state_12: torch.Tensor, hover_rpm: torch.Tensor):
    """Initialize the firmware PID state to match the current attitude.

    ``hover_rpm`` (1,) is the task-specific hover RPM, used to bootstrap
    the motor low-pass filter.
    """
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

    h = hover_rpm.to(dev).expand(B).unsqueeze(-1).expand(B, 4).contiguous()
    motor_rpm = h.clone()
    return (att_integ, att_prev, rate_integ, rate_prev, yaw_sp, motor_rpm)


def reset_pid_diverged(pid_state, diverged_mask, hover_rpm):
    """Zero PID state for diverged drones (used during curriculum)."""
    att_integ, att_prev, rate_integ, rate_prev, yaw_sp, motor_rpm = pid_state
    m  = diverged_mask.unsqueeze(1)
    m1 = diverged_mask
    zero3 = torch.zeros_like(att_integ)
    zero1 = torch.zeros_like(yaw_sp)
    h = hover_rpm.to(motor_rpm.device).unsqueeze(-1).expand_as(motor_rpm)
    return (
        torch.where(m, zero3, att_integ),
        torch.where(m, zero3, att_prev),
        torch.where(m, zero3, rate_integ),
        torch.where(m, zero3, rate_prev),
        torch.where(m1, zero1, yaw_sp),
        torch.where(m, h, motor_rpm),
    )


def one_pid_step(thrust_u16, roll_des_deg, pitch_des_deg, yaw_rate_deg,
                 state_12, pid_state, mass: MassParams, dynamics_step):
    """One ATTITUDE_RATE step. Closes the loop with `dynamics_step`."""
    att_integ, att_prev, rate_integ, rate_prev, yaw_sp, motor_rpm = pid_state
    g = _get_pid_gains(state_12.device)

    RAD2DEG = 180.0 / math.pi
    actual_att = torch.stack([
        state_12[:, 6] * RAD2DEG,
        -state_12[:, 8] * RAD2DEG,
        state_12[:, 10] * RAD2DEG,
    ], dim=-1)
    actual_gyro = torch.stack([
        state_12[:, 7] * RAD2DEG,
        -state_12[:, 9] * RAD2DEG,
        state_12[:, 11] * RAD2DEG,
    ], dim=-1)

    new_yaw_sp = yaw_sp + yaw_rate_deg * C.ATTITUDE_UPDATE_DT
    new_yaw_sp = new_yaw_sp - 360.0 * torch.round(new_yaw_sp / 360.0)

    desired_att = torch.stack([roll_des_deg, pitch_des_deg, new_yaw_sp], dim=-1)
    rate_desired, new_att_integ, new_att_prev = pid_update_vec(
        desired_att, actual_att, att_prev, att_integ,
        g['att_kp'], g['att_ki'], g['att_kd'], C.ATTITUDE_UPDATE_DT, g['att_il'],
        yaw_mask=g['yaw_mask'])

    motor_cmds, new_rate_integ, new_rate_prev = pid_update_vec(
        rate_desired, actual_gyro, rate_prev, rate_integ,
        g['rate_kp'], g['rate_ki'], g['rate_kd'], C.ATTITUDE_UPDATE_DT, g['rate_il'])
    motor_cmds = torch.clamp(motor_cmds, -32767, 32767)
    roll_cmd  = motor_cmds[:, 0]
    pitch_cmd = motor_cmds[:, 1]
    yaw_cmd   = -motor_cmds[:, 2]

    r_half = roll_cmd * 0.5
    p_half = pitch_cmd * 0.5
    motor_pwms = torch.stack([
        thrust_u16 - r_half + p_half + yaw_cmd,
        thrust_u16 - r_half - p_half - yaw_cmd,
        thrust_u16 + r_half - p_half + yaw_cmd,
        thrust_u16 + r_half + p_half - yaw_cmd,
    ], dim=-1)
    highest = motor_pwms.max(dim=-1, keepdim=True).values
    reduction = torch.clamp(highest - C.UINT16_MAX, min=0)
    motor_pwms = torch.clamp(motor_pwms - reduction, min=0, max=C.UINT16_MAX)

    rpm_cmd = pwm_to_rpm(motor_pwms)
    new_motor_rpm = C.MOTOR_ALPHA * rpm_cmd + (1.0 - C.MOTOR_ALPHA) * motor_rpm

    new_state = dynamics_step(state_12, new_motor_rpm, C.ATTITUDE_UPDATE_DT, mass)

    new_pid_state = (new_att_integ, new_att_prev,
                     new_rate_integ, new_rate_prev,
                     new_yaw_sp, new_motor_rpm)
    return new_state, new_pid_state


def one_nn_step(state, nn_action, pid_state, mass: MassParams, dynamics_step):
    """One NN step (NN_FREQ Hz) = PID_STEPS_PER_NN PID steps."""
    thrust    = nn_action[:, 0]
    roll_deg  = nn_action[:, 1]
    pitch_deg = nn_action[:, 2]
    yaw_rate  = nn_action[:, 3]
    for _ in range(C.PID_STEPS_PER_NN):
        state, pid_state = one_pid_step(
            thrust, roll_deg, pitch_deg, yaw_rate,
            state, pid_state, mass, dynamics_step)
    return state, pid_state
