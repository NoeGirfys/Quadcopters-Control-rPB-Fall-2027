#!/usr/bin/env python3
"""
Crazyflie Firmware-Faithful PID Controller — DEBUG / DIAGNOSTIC version
========================================================================

Same pipeline as cf_firmware_pid_sim.py but with:
  • CSV logging of every PID stage at every control step
  • A simpler diagnostic trajectory (takeoff → hover → +X → +Y → hover → land)
  • Automatic matplotlib plots after the run

Usage:
  python cf_firmware_pid_debug.py                      # circle trajectory (original)
  python cf_firmware_pid_debug.py --trajectory simple   # simple step trajectory
  python cf_firmware_pid_debug.py --no-gui              # headless (faster)
"""

import os, sys, time, math, argparse, csv
import numpy as np

# ---------------------------------------------------------------------------
try:
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.utils import sync
    HAS_PYBULLET_DRONES = True
except ImportError:
    HAS_PYBULLET_DRONES = False

# ===================================================================
#  CONSTANTS  (identical to main script)
# ===================================================================
RATE_MAIN_LOOP = 1000
ATTITUDE_RATE  = 500
POSITION_RATE  = 100
ATTITUDE_UPDATE_DT = 1.0 / ATTITUDE_RATE
POSITION_UPDATE_DT = 1.0 / POSITION_RATE

ARM_LENGTH = 0.046
CF_MASS    = 0.027
UINT16_MAX = 65535

# ---- Rate PID gains ----
PID_ROLL_RATE_KP  = 250.0;  PID_ROLL_RATE_KI  = 500.0;  PID_ROLL_RATE_KD  = 2.5
PID_ROLL_RATE_KFF = 0.0;    PID_ROLL_RATE_INTEGRATION_LIMIT  = 33.3
PID_PITCH_RATE_KP = 250.0;  PID_PITCH_RATE_KI = 500.0;  PID_PITCH_RATE_KD = 2.5
PID_PITCH_RATE_KFF= 0.0;    PID_PITCH_RATE_INTEGRATION_LIMIT = 33.3
PID_YAW_RATE_KP   = 120.0;  PID_YAW_RATE_KI   = 16.7;   PID_YAW_RATE_KD   = 0.0
PID_YAW_RATE_KFF  = 0.0;    PID_YAW_RATE_INTEGRATION_LIMIT   = 166.7

# ---- Attitude PID gains ----
PID_ROLL_KP  = 6.0;  PID_ROLL_KI  = 3.0;  PID_ROLL_KD  = 0.0
PID_ROLL_KFF = 0.0;  PID_ROLL_INTEGRATION_LIMIT  = 20.0
PID_PITCH_KP = 6.0;  PID_PITCH_KI = 3.0;  PID_PITCH_KD = 0.0
PID_PITCH_KFF= 0.0;  PID_PITCH_INTEGRATION_LIMIT = 20.0
PID_YAW_KP   = 6.0;  PID_YAW_KI   = 1.0;  PID_YAW_KD   = 0.35
PID_YAW_KFF  = 0.0;  PID_YAW_INTEGRATION_LIMIT   = 360.0

# ---- Velocity PID gains ----
PID_VEL_X_KP = 25.0;  PID_VEL_X_KI = 1.0;  PID_VEL_X_KD = 0.0;  PID_VEL_X_KFF = 0.0
PID_VEL_Y_KP = 25.0;  PID_VEL_Y_KI = 1.0;  PID_VEL_Y_KD = 0.0;  PID_VEL_Y_KFF = 0.0
PID_VEL_Z_KP = 25.0;  PID_VEL_Z_KI = 15.0; PID_VEL_Z_KD = 0.0;  PID_VEL_Z_KFF = 0.0

# ---- Position PID gains ----
PID_POS_X_KP = 2.0;  PID_POS_X_KI = 0.0;  PID_POS_X_KD = 0.0;  PID_POS_X_KFF = 0.0
PID_POS_Y_KP = 2.0;  PID_POS_Y_KI = 0.0;  PID_POS_Y_KD = 0.0;  PID_POS_Y_KFF = 0.0
PID_POS_Z_KP = 2.0;  PID_POS_Z_KI = 0.5;  PID_POS_Z_KD = 0.0;  PID_POS_Z_KFF = 0.0

# ---- Limits ----
PID_VEL_ROLL_MAX  = 20.0
PID_VEL_PITCH_MAX = 20.0
PID_VEL_THRUST_BASE = 36000.0
PID_VEL_THRUST_MIN  = 20000.0
THRUST_SCALE = 1000.0
PID_POS_VEL_X_MAX = 1.0
PID_POS_VEL_Y_MAX = 1.0
PID_POS_VEL_Z_MAX = 1.0
VEL_MAX_OVERHEAD = 1.10
RP_LIMIT_OVERHEAD = 1.10

# ---- Filters ----
ATTITUDE_LPF_ENABLE   = False;  ATTITUDE_LPF_CUTOFF   = 15.0
ATTITUDE_RATE_LPF_ENABLE = False
ATTITUDE_ROLL_RATE_LPF_CUTOFF  = 30.0
ATTITUDE_PITCH_RATE_LPF_CUTOFF = 30.0
ATTITUDE_YAW_RATE_LPF_CUTOFF   = 30.0
POS_XY_FILT_ENABLE  = True;   POS_XY_FILT_CUTOFF  = 20.0
POS_Z_FILT_ENABLE   = True;   POS_Z_FILT_CUTOFF   = 20.0
VEL_XY_FILT_ENABLE  = True;   VEL_XY_FILT_CUTOFF  = 20.0
VEL_Z_FILT_ENABLE   = True;   VEL_Z_FILT_CUTOFF   = 20.0
DEFAULT_PID_INTEGRATION_LIMIT = 5000.0
DEFAULT_PID_OUTPUT_LIMIT      = 0.0


# ===================================================================
#  Lpf2pData (identical)
# ===================================================================
class Lpf2pData:
    def __init__(self, sample_freq, cutoff_freq):
        self.delay_element_1 = 0.0
        self.delay_element_2 = 0.0
        self.set_cutoff(sample_freq, cutoff_freq)

    def set_cutoff(self, sample_freq, cutoff_freq):
        fr  = sample_freq / cutoff_freq
        ohm = math.tan(math.pi / fr)
        c   = 1.0 + 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm
        self.b0 = ohm * ohm / c
        self.b1 = 2.0 * self.b0
        self.b2 = self.b0
        self.a1 = 2.0 * (ohm * ohm - 1.0) / c
        self.a2 = (1.0 - 2.0 * math.cos(math.pi / 4.0) * ohm + ohm * ohm) / c
        self.delay_element_1 = 0.0
        self.delay_element_2 = 0.0

    def apply(self, sample):
        delay_element_0 = (sample
                           - self.delay_element_1 * self.a1
                           - self.delay_element_2 * self.a2)
        if not math.isfinite(delay_element_0):
            delay_element_0 = sample
        output = (delay_element_0 * self.b0
                  + self.delay_element_1 * self.b1
                  + self.delay_element_2 * self.b2)
        self.delay_element_2 = self.delay_element_1
        self.delay_element_1 = delay_element_0
        return output


# ===================================================================
#  PidObject (identical, but exposes internal state for logging)
# ===================================================================
class PidObject:
    def __init__(self, kp, ki, kd, kff, dt, sampling_rate, cutoff_freq,
                 enable_d_filter, desired=0.0):
        self.kp  = kp;  self.ki  = ki;  self.kd  = kd;  self.kff = kff
        self.dt  = dt
        self.desired      = desired
        self.error        = 0.0
        self.prevMeasured = 0.0
        self.integ        = 0.0
        self.deriv        = 0.0
        self.outP = 0.0; self.outI = 0.0; self.outD = 0.0; self.outFF = 0.0
        self.iLimit      = DEFAULT_PID_INTEGRATION_LIMIT
        self.outputLimit  = DEFAULT_PID_OUTPUT_LIMIT
        self.enableDFilter = enable_d_filter
        self.dFilter = Lpf2pData(sampling_rate, cutoff_freq) if enable_d_filter else None

    def update(self, measured, is_yaw_angle=False):
        self.error = self.desired - measured
        if is_yaw_angle:
            if self.error > 180.0:   self.error -= 360.0
            elif self.error < -180.0: self.error += 360.0
        self.outP = self.kp * self.error
        output = self.outP

        delta = -(measured - self.prevMeasured)
        if is_yaw_angle:
            if delta > 180.0:   delta -= 360.0
            elif delta < -180.0: delta += 360.0
        if self.enableDFilter:
            self.deriv = self.dFilter.apply(delta / self.dt)
        else:
            self.deriv = delta / self.dt
        if math.isnan(self.deriv): self.deriv = 0.0
        self.outD = self.kd * self.deriv
        output += self.outD

        self.integ += self.error * self.dt
        if self.iLimit != 0:
            self.integ = max(-self.iLimit, min(self.iLimit, self.integ))
        self.outI = self.ki * self.integ
        output += self.outI

        self.outFF = self.kff * self.desired
        output += self.outFF

        if self.outputLimit != 0:
            output = max(-self.outputLimit, min(self.outputLimit, output))
        self.prevMeasured = measured
        return output

    def reset(self, actual):
        self.error = 0.0; self.prevMeasured = actual
        self.integ = 0.0; self.deriv = 0.0

    def set_desired(self, desired):
        self.desired = desired

    def set_integral_limit(self, limit):
        self.iLimit = limit


# ===================================================================
#  Position Controller (identical)
# ===================================================================
class CrazyfliePositionController:
    def __init__(self):
        dt = POSITION_UPDATE_DT
        self.pidX  = PidObject(PID_POS_X_KP, PID_POS_X_KI, PID_POS_X_KD, PID_POS_X_KFF, dt, POSITION_RATE, POS_XY_FILT_CUTOFF, POS_XY_FILT_ENABLE)
        self.pidY  = PidObject(PID_POS_Y_KP, PID_POS_Y_KI, PID_POS_Y_KD, PID_POS_Y_KFF, dt, POSITION_RATE, POS_XY_FILT_CUTOFF, POS_XY_FILT_ENABLE)
        self.pidZ  = PidObject(PID_POS_Z_KP, PID_POS_Z_KI, PID_POS_Z_KD, PID_POS_Z_KFF, dt, POSITION_RATE, POS_Z_FILT_CUTOFF,  POS_Z_FILT_ENABLE)
        self.pidVX = PidObject(PID_VEL_X_KP, PID_VEL_X_KI, PID_VEL_X_KD, PID_VEL_X_KFF, dt, POSITION_RATE, VEL_XY_FILT_CUTOFF, VEL_XY_FILT_ENABLE)
        self.pidVY = PidObject(PID_VEL_Y_KP, PID_VEL_Y_KI, PID_VEL_Y_KD, PID_VEL_Y_KFF, dt, POSITION_RATE, VEL_XY_FILT_CUTOFF, VEL_XY_FILT_ENABLE)
        self.pidVZ = PidObject(PID_VEL_Z_KP, PID_VEL_Z_KI, PID_VEL_Z_KD, PID_VEL_Z_KFF, dt, POSITION_RATE, VEL_Z_FILT_CUTOFF,  VEL_Z_FILT_ENABLE)
        self.thrustBase = PID_VEL_THRUST_BASE
        self.thrustMin  = PID_VEL_THRUST_MIN

    def update(self, setpoint_pos, state_pos, state_vel, state_yaw_deg):
        sp_x, sp_y, sp_z = setpoint_pos
        st_x, st_y, st_z = state_pos
        st_vx, st_vy, st_vz = state_vel

        self.pidX.outputLimit = PID_POS_VEL_X_MAX * VEL_MAX_OVERHEAD
        self.pidY.outputLimit = PID_POS_VEL_Y_MAX * VEL_MAX_OVERHEAD
        self.pidZ.outputLimit = max(PID_POS_VEL_Z_MAX, 0.5) * VEL_MAX_OVERHEAD

        cosyaw = math.cos(math.radians(state_yaw_deg))
        sinyaw = math.sin(math.radians(state_yaw_deg))

        setp_body_x  =  sp_x * cosyaw + sp_y * sinyaw
        setp_body_y  = -sp_x * sinyaw + sp_y * cosyaw
        state_body_x =  st_x * cosyaw + st_y * sinyaw
        state_body_y = -st_x * sinyaw + st_y * cosyaw

        self.pidX.set_desired(setp_body_x)
        vel_sp_x = self.pidX.update(state_body_x)
        self.pidY.set_desired(setp_body_y)
        vel_sp_y = self.pidY.update(state_body_y)
        self.pidZ.set_desired(sp_z)
        vel_sp_z = self.pidZ.update(st_z)

        self.pidVX.outputLimit = PID_VEL_PITCH_MAX * RP_LIMIT_OVERHEAD
        self.pidVY.outputLimit = PID_VEL_ROLL_MAX  * RP_LIMIT_OVERHEAD
        self.pidVZ.outputLimit = (UINT16_MAX / 2.0 / THRUST_SCALE)

        state_body_vx =  st_vx * cosyaw + st_vy * sinyaw
        state_body_vy = -st_vx * sinyaw + st_vy * cosyaw

        self.pidVX.set_desired(vel_sp_x)
        pitch_deg = -self.pidVX.update(state_body_vx)
        self.pidVY.set_desired(vel_sp_y)
        roll_deg  = -self.pidVY.update(state_body_vy)

        roll_deg  = max(-PID_VEL_ROLL_MAX,  min(PID_VEL_ROLL_MAX,  roll_deg))
        pitch_deg = max(-PID_VEL_PITCH_MAX, min(PID_VEL_PITCH_MAX, pitch_deg))

        self.pidVZ.set_desired(vel_sp_z)
        thrust_raw = self.pidVZ.update(st_vz)
        thrust = thrust_raw * THRUST_SCALE + self.thrustBase
        if thrust < self.thrustMin: thrust = self.thrustMin
        thrust = max(0.0, min(UINT16_MAX, thrust))

        # Store intermediate values for logging
        self._last_vel_sp = (vel_sp_x, vel_sp_y, vel_sp_z)
        self._last_body_vel = (state_body_vx, state_body_vy, st_vz)
        self._last_thrust_raw = thrust_raw

        return thrust, roll_deg, pitch_deg

    def reset_all(self, x, y, z):
        self.pidX.reset(x); self.pidY.reset(y); self.pidZ.reset(z)
        self.pidVX.reset(0); self.pidVY.reset(0); self.pidVZ.reset(0)


# ===================================================================
#  Attitude Controller (identical)
# ===================================================================
class CrazyflieAttitudeController:
    def __init__(self):
        dt = ATTITUDE_UPDATE_DT
        self.pidRollRate  = PidObject(PID_ROLL_RATE_KP,  PID_ROLL_RATE_KI,  PID_ROLL_RATE_KD,  PID_ROLL_RATE_KFF,  dt, ATTITUDE_RATE, ATTITUDE_ROLL_RATE_LPF_CUTOFF,  ATTITUDE_RATE_LPF_ENABLE)
        self.pidPitchRate = PidObject(PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD, PID_PITCH_RATE_KFF, dt, ATTITUDE_RATE, ATTITUDE_PITCH_RATE_LPF_CUTOFF, ATTITUDE_RATE_LPF_ENABLE)
        self.pidYawRate   = PidObject(PID_YAW_RATE_KP,   PID_YAW_RATE_KI,   PID_YAW_RATE_KD,   PID_YAW_RATE_KFF,   dt, ATTITUDE_RATE, ATTITUDE_YAW_RATE_LPF_CUTOFF,   ATTITUDE_RATE_LPF_ENABLE)
        self.pidRollRate.set_integral_limit(PID_ROLL_RATE_INTEGRATION_LIMIT)
        self.pidPitchRate.set_integral_limit(PID_PITCH_RATE_INTEGRATION_LIMIT)
        self.pidYawRate.set_integral_limit(PID_YAW_RATE_INTEGRATION_LIMIT)

        self.pidRoll  = PidObject(PID_ROLL_KP,  PID_ROLL_KI,  PID_ROLL_KD,  PID_ROLL_KFF,  dt, ATTITUDE_RATE, ATTITUDE_LPF_CUTOFF, ATTITUDE_LPF_ENABLE)
        self.pidPitch = PidObject(PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD, PID_PITCH_KFF, dt, ATTITUDE_RATE, ATTITUDE_LPF_CUTOFF, ATTITUDE_LPF_ENABLE)
        self.pidYaw   = PidObject(PID_YAW_KP,   PID_YAW_KI,   PID_YAW_KD,   PID_YAW_KFF,   dt, ATTITUDE_RATE, ATTITUDE_LPF_CUTOFF, ATTITUDE_LPF_ENABLE)
        self.pidRoll.set_integral_limit(PID_ROLL_INTEGRATION_LIMIT)
        self.pidPitch.set_integral_limit(PID_PITCH_INTEGRATION_LIMIT)
        self.pidYaw.set_integral_limit(PID_YAW_INTEGRATION_LIMIT)

    def correct_attitude(self, roll_a, pitch_a, yaw_a, roll_d, pitch_d, yaw_d):
        self.pidRoll.set_desired(roll_d);   rr = self.pidRoll.update(roll_a)
        self.pidPitch.set_desired(pitch_d); pr = self.pidPitch.update(pitch_a)
        self.pidYaw.set_desired(yaw_d);     yr = self.pidYaw.update(yaw_a, is_yaw_angle=True)
        return rr, pr, yr

    def correct_rate(self, gx, gy, gz, rrd, prd, yrd):
        self.pidRollRate.set_desired(rrd)
        ro = self._sat16(self.pidRollRate.update(gx))
        self.pidPitchRate.set_desired(prd)
        po = self._sat16(self.pidPitchRate.update(gy))
        self.pidYawRate.set_desired(yrd)
        yo = self._sat16(self.pidYawRate.update(gz))
        return ro, po, yo

    def reset_all(self, r, p, y):
        self.pidRoll.reset(r); self.pidPitch.reset(p); self.pidYaw.reset(y)
        self.pidRollRate.reset(0); self.pidPitchRate.reset(0); self.pidYawRate.reset(0)

    @staticmethod
    def _sat16(v):
        return int(max(-32767, min(32767, v)))


# ===================================================================
#  Power Distribution (identical)
# ===================================================================
class CrazyfliePowerDistribution:
    def __init__(self, idle_thrust=0):
        self.idle_thrust = idle_thrust

    def distribute(self, thrust, roll, pitch, yaw):
        r = roll / 2.0;  p = pitch / 2.0
        m1 = thrust - r + p + yaw
        m2 = thrust - r - p - yaw
        m3 = thrust + r - p + yaw
        m4 = thrust + r + p - yaw
        return self.cap([m1, m2, m3, m4])

    def cap(self, motors):
        highest = max(motors)
        reduction = max(0, highest - UINT16_MAX)
        result = []
        for t in motors:
            val = max(self.idle_thrust, t - reduction)
            result.append(max(0.0, min(UINT16_MAX, val)))
        return result


# ===================================================================
#  Top-level controller (identical logic, but exposes internals)
# ===================================================================
def cap_angle(a):
    while a > 180: a -= 360
    while a < -180: a += 360
    return a


class CrazyflieFirmwarePID:
    def __init__(self):
        self.pos_ctrl = CrazyfliePositionController()
        self.att_ctrl = CrazyflieAttitudeController()
        self.power    = CrazyfliePowerDistribution()
        self.attitude_desired_yaw = 0.0
        self.actuator_thrust = 0.0
        self._tick = 0
        self._last_att_roll  = 0.0
        self._last_att_pitch = 0.0

        # --- Diagnostic storage (filled on every update) ---
        self.diag = {}

    def update(self, setpoint_pos, setpoint_yaw_rate, state_pos, state_vel,
               state_rpy_deg, state_gyro_deg):
        roll_a, pitch_a, yaw_a = state_rpy_deg
        gyro_x, gyro_y, gyro_z = state_gyro_deg

        self.attitude_desired_yaw = cap_angle(
            self.attitude_desired_yaw + setpoint_yaw_rate * ATTITUDE_UPDATE_DT)

        # --- Position controller (100 Hz) ---
        pos_ran = (self._tick % (ATTITUDE_RATE // POSITION_RATE) == 0)
        if pos_ran:
            self.actuator_thrust, att_roll, att_pitch = self.pos_ctrl.update(
                setpoint_pos, state_pos, state_vel, yaw_a)
            self._last_att_roll  = att_roll
            self._last_att_pitch = att_pitch

        # --- Attitude controller (500 Hz) ---
        roll_rate_des, pitch_rate_des, yaw_rate_des = \
            self.att_ctrl.correct_attitude(
                roll_a, pitch_a, yaw_a,
                self._last_att_roll, self._last_att_pitch,
                self.attitude_desired_yaw)

        # --- Rate controller ---
        roll_cmd, pitch_cmd, yaw_cmd = \
            self.att_ctrl.correct_rate(
                gyro_x, gyro_y, gyro_z,
                roll_rate_des, pitch_rate_des, yaw_rate_des)
        yaw_cmd = -yaw_cmd

        thrust = self.actuator_thrust

        if thrust == 0:
            self.att_ctrl.reset_all(roll_a, pitch_a, yaw_a)
            self.pos_ctrl.reset_all(state_pos[0], state_pos[1], state_pos[2])
            self.attitude_desired_yaw = yaw_a
            self._tick += 1
            self.diag = {}
            return [0.0, 0.0, 0.0, 0.0]

        motor_pwm = self.power.distribute(thrust, roll_cmd, pitch_cmd, yaw_cmd)

        # --- Store diagnostics ---
        pc = self.pos_ctrl
        ac = self.att_ctrl
        self.diag = {
            'pos_ran': pos_ran,
            # Position PID outputs (= velocity setpoints, body-yaw-aligned)
            'vel_sp_x': getattr(pc, '_last_vel_sp', (0,0,0))[0],
            'vel_sp_y': getattr(pc, '_last_vel_sp', (0,0,0))[1],
            'vel_sp_z': getattr(pc, '_last_vel_sp', (0,0,0))[2],
            # Velocity PID internals
            'pidVX_err': pc.pidVX.error, 'pidVX_P': pc.pidVX.outP,
            'pidVX_I': pc.pidVX.outI,    'pidVX_integ': pc.pidVX.integ,
            'pidVY_err': pc.pidVY.error, 'pidVY_P': pc.pidVY.outP,
            'pidVY_I': pc.pidVY.outI,    'pidVY_integ': pc.pidVY.integ,
            'pidVZ_err': pc.pidVZ.error, 'pidVZ_P': pc.pidVZ.outP,
            'pidVZ_I': pc.pidVZ.outI,    'pidVZ_integ': pc.pidVZ.integ,
            'thrust_raw': getattr(pc, '_last_thrust_raw', 0.0),
            # Attitude desired from vel PID
            'att_roll_d': self._last_att_roll,
            'att_pitch_d': self._last_att_pitch,
            'att_yaw_d': self.attitude_desired_yaw,
            # Attitude PID outputs (= rate setpoints)
            'rate_roll_d': roll_rate_des,
            'rate_pitch_d': pitch_rate_des,
            'rate_yaw_d': yaw_rate_des,
            # Attitude PID internals
            'pidRoll_err': ac.pidRoll.error, 'pidRoll_integ': ac.pidRoll.integ,
            'pidPitch_err': ac.pidPitch.error, 'pidPitch_integ': ac.pidPitch.integ,
            # Rate PID outputs (= motor commands)
            'roll_cmd': roll_cmd,
            'pitch_cmd': pitch_cmd,
            'yaw_cmd': yaw_cmd,
            # Rate PID internals
            'pidRR_err': ac.pidRollRate.error,  'pidRR_P': ac.pidRollRate.outP,
            'pidRR_I': ac.pidRollRate.outI,     'pidRR_integ': ac.pidRollRate.integ,
            'pidPR_err': ac.pidPitchRate.error, 'pidPR_P': ac.pidPitchRate.outP,
            'pidPR_I': ac.pidPitchRate.outI,    'pidPR_integ': ac.pidPitchRate.integ,
            # Thrust & motors
            'thrust': thrust,
            'm1': motor_pwm[0], 'm2': motor_pwm[1],
            'm3': motor_pwm[2], 'm4': motor_pwm[3],
        }

        self._tick += 1
        return motor_pwm

    def reset(self, state_rpy_deg, state_pos):
        self.att_ctrl.reset_all(*state_rpy_deg)
        self.pos_ctrl.reset_all(*state_pos)
        self.attitude_desired_yaw = state_rpy_deg[2]
        self._tick = 0
        self._last_att_roll = 0.0
        self._last_att_pitch = 0.0


# ===================================================================
#  Motor Dynamics Filter (for simulation only)
# ===================================================================

class MotorDynamicsFilter:
    """First-order low-pass filter simulating brushed motor response.

    On the real CF 2.1+, brushed coreless motors have τ ≈ 15–25 ms.
    In PyBullet PYB mode forces are instant → rate PID limit-cycles.
    This filter restores a realistic motor bandwidth.

    Model: rpm(t+dt) = α · rpm_cmd + (1-α) · rpm(t)
           α = dt / (τ + dt)
    """

    def __init__(self, n_motors=4, tau=0.02, dt=0.002):
        self.alpha = dt / (tau + dt)
        self.rpm = np.zeros(n_motors)

    def apply(self, rpm_cmd):
        self.rpm = self.alpha * rpm_cmd + (1.0 - self.alpha) * self.rpm
        return self.rpm.copy()

    def reset(self):
        self.rpm[:] = 0.0


# ===================================================================
#  PWM → RPM
# ===================================================================
def pwm_to_rpm(pwm_values, max_rpm):
    rpms = []
    for pwm in pwm_values:
        ratio = max(0.0, min(1.0, pwm / UINT16_MAX))
        rpms.append(math.sqrt(ratio) * max_rpm)
    return np.array(rpms)


# ===================================================================
#  Trajectories
# ===================================================================
def generate_circle_trajectory(ctrl_freq, duration_sec, hover_height=0.5, radius=0.5):
    """Original circle trajectory."""
    n = int(ctrl_freq * duration_sec)
    wp = np.zeros((n, 4))
    t_take = 3.0;  t_circ = duration_sec - 6.0;  t_land = 3.0
    n_take = int(ctrl_freq * t_take)
    n_circ = int(ctrl_freq * t_circ)
    n_land = n - n_take - n_circ

    for i in range(n_take):
        t = i / n_take
        wp[i] = [0, 0, hover_height * (3*t**2 - 2*t**3), 0]
    for i in range(n_circ):
        t = i / n_circ
        a = t * 2 * math.pi
        wp[n_take+i] = [radius*math.cos(a)-radius, radius*math.sin(a), hover_height, 0]
    for i in range(n_land):
        t = i / n_land
        wp[n_take+n_circ+i] = [0, 0, hover_height*(1-(3*t**2-2*t**3)), 0]
    return wp


def generate_simple_trajectory(ctrl_freq, duration_sec=20, hover_height=0.5, step_dist=0.3):
    """
    Simple diagnostic trajectory:
      0-3s    takeoff to hover_height
      3-6s    hover at (0, 0, h)
      6-9s    move to (+step_dist, 0, h)   — step in X
      9-12s   hold at (+step_dist, 0, h)
      12-15s  move to (+step_dist, +step_dist, h)  — step in Y
      15-18s  hold at (+step_dist, +step_dist, h)
      18-21s  land at (+step_dist, +step_dist, 0)
    """
    n = int(ctrl_freq * duration_sec)
    wp = np.zeros((n, 4))

    def smooth(t):
        t = max(0, min(1, t))
        return 3*t**2 - 2*t**3

    for i in range(n):
        t = i / ctrl_freq  # time in seconds
        if t < 3.0:
            # Takeoff
            wp[i] = [0, 0, hover_height * smooth(t/3.0), 0]
        elif t < 6.0:
            # Hover
            wp[i] = [0, 0, hover_height, 0]
        elif t < 9.0:
            # Step in X
            frac = smooth((t - 6.0) / 3.0)
            wp[i] = [step_dist * frac, 0, hover_height, 0]
        elif t < 12.0:
            # Hold
            wp[i] = [step_dist, 0, hover_height, 0]
        elif t < 15.0:
            # Step in Y
            frac = smooth((t - 12.0) / 3.0)
            wp[i] = [step_dist, step_dist * frac, hover_height, 0]
        elif t < 18.0:
            # Hold
            wp[i] = [step_dist, step_dist, hover_height, 0]
        else:
            # Land
            frac = smooth((t - 18.0) / max(duration_sec - 18.0, 0.1))
            wp[i] = [step_dist, step_dist, hover_height * (1 - frac), 0]
    return wp


# ===================================================================
#  State extraction (identical)
# ===================================================================
def obs_to_firmware_state(obs):
    pos   = obs[0:3]
    rpy   = np.degrees(obs[7:10])
    vel   = obs[10:13]
    ang_v = obs[13:16]

    # ── Pitch convention fix ──────────────────────────────────────
    # pybullet (Z-up, right-hand rule about Y): positive pitch = nose DOWN
    # Crazyflie firmware (aerospace): positive pitch = nose UP
    # Roll and yaw conventions match. Only pitch (and gyro_y) must be negated.
    rpy[1] = -rpy[1]

    r, p, y_ = obs[7:10]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y_), math.sin(y_)
    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,          cp*sr,           cp*cr       ]
    ])
    gyro_body = R.T @ ang_v
    gyro_deg  = np.degrees(gyro_body)

    # Negate pitch rate for the same convention reason
    gyro_deg[1] = -gyro_deg[1]

    return pos, vel, rpy, gyro_deg


# ===================================================================
#  MAIN — Debug simulation
# ===================================================================
def run_debug(trajectory='circle', duration_sec=15, gui=True,
              hover_height=0.5, radius=0.5, csv_path='debug_log.csv'):

    if not HAS_PYBULLET_DRONES:
        print("ERROR: gym-pybullet-drones not found.")
        sys.exit(1)

    PYB_FREQ  = 1000
    CTRL_FREQ = 500

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.02]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.PYB,
        pyb_freq=PYB_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False
    )

    MAX_RPM = env.MAX_RPM
    HOVER_RPM = env.HOVER_RPM

    # Compute theoretical hover PWM for pybullet
    # force_hover = m*g/4 = KF * rpm_hover² → rpm_hover = HOVER_RPM
    # pwm_hover = (HOVER_RPM / MAX_RPM)² * 65535
    hover_pwm_theory = (HOVER_RPM / MAX_RPM)**2 * UINT16_MAX
    print(f"[DEBUG] MAX_RPM={MAX_RPM:.1f}  HOVER_RPM={HOVER_RPM:.1f}")
    print(f"[DEBUG] Theoretical hover PWM for pybullet: {hover_pwm_theory:.0f}")
    print(f"[DEBUG] Firmware THRUST_BASE:               {PID_VEL_THRUST_BASE:.0f}")
    print(f"[DEBUG] Mismatch ratio: {PID_VEL_THRUST_BASE/hover_pwm_theory:.3f}x")
    print(f"[DEBUG] KF={env.KF:.4e}  KM={env.KM:.4e}  M={env.M:.4f}  L={env.L:.4f}")

    ctrl = CrazyflieFirmwarePID()

    # Motor dynamics filter — simulates real motor response lag (τ ≈ 20 ms)
    motor_filter = MotorDynamicsFilter(n_motors=4, tau=0.02, dt=1.0/CTRL_FREQ)

    if trajectory == 'circle':
        waypoints = generate_circle_trajectory(CTRL_FREQ, duration_sec, hover_height, radius)
    else:
        waypoints = generate_simple_trajectory(CTRL_FREQ, duration_sec, hover_height)

    n_steps = len(waypoints)
    print(f"[DEBUG] Trajectory: {trajectory}, {n_steps} steps, {n_steps/CTRL_FREQ:.1f}s")

    # --- Reset controller with initial state ---
    obs_init, _, _, _, _ = env.step(np.zeros((1,4)))
    pos0, vel0, rpy0, gyro0 = obs_to_firmware_state(obs_init[0])
    ctrl.reset(rpy0, pos0)
    print(f"[DEBUG] Initial state: pos={pos0}, rpy={rpy0}")

    # --- Open CSV ---
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    header = [
        'step', 'time',
        # Setpoint
        'sp_x', 'sp_y', 'sp_z', 'sp_yaw_rate',
        # State
        'pos_x', 'pos_y', 'pos_z',
        'vel_x', 'vel_y', 'vel_z',
        'roll', 'pitch', 'yaw',
        'gyro_x', 'gyro_y', 'gyro_z',
        # Position PID → velocity setpoints
        'vel_sp_x', 'vel_sp_y', 'vel_sp_z',
        # Velocity PID internals
        'pidVX_err', 'pidVX_P', 'pidVX_I', 'pidVX_integ',
        'pidVY_err', 'pidVY_P', 'pidVY_I', 'pidVY_integ',
        'pidVZ_err', 'pidVZ_P', 'pidVZ_I', 'pidVZ_integ',
        'thrust_raw',
        # Attitude desired
        'att_roll_d', 'att_pitch_d', 'att_yaw_d',
        # Rate desired
        'rate_roll_d', 'rate_pitch_d', 'rate_yaw_d',
        # Attitude PID internals
        'pidRoll_err', 'pidRoll_integ',
        'pidPitch_err', 'pidPitch_integ',
        # Rate PID outputs
        'roll_cmd', 'pitch_cmd', 'yaw_cmd',
        # Rate PID internals
        'pidRR_err', 'pidRR_P', 'pidRR_I', 'pidRR_integ',
        'pidPR_err', 'pidPR_P', 'pidPR_I', 'pidPR_integ',
        # Thrust & motors
        'thrust', 'm1', 'm2', 'm3', 'm4',
        'rpm0', 'rpm1', 'rpm2', 'rpm3',
    ]
    writer.writerow(header)

    action = np.zeros((1, 4))
    START = time.time()
    crashed = False

    for i in range(n_steps):
        obs, _, _, _, _ = env.step(action)
        pos, vel, rpy_deg, gyro_deg = obs_to_firmware_state(obs[0])

        sp = waypoints[i]
        setpoint_pos      = sp[0:3]
        setpoint_yaw_rate = sp[3]

        motor_pwm = ctrl.update(setpoint_pos, setpoint_yaw_rate,
                                pos, vel, rpy_deg, gyro_deg)

        rpms = pwm_to_rpm(motor_pwm, MAX_RPM)
        rpms = motor_filter.apply(rpms)
        action[0, :] = rpms

        # --- Crash detection ---
        if pos[2] < -0.1 or abs(rpy_deg[0]) > 75 or abs(rpy_deg[1]) > 75:
            print(f"\n[CRASH] t={i/CTRL_FREQ:.3f}s  pos={pos}  rpy={rpy_deg}")
            print(f"  Last diag: thrust={ctrl.diag.get('thrust',0):.0f}  "
                  f"att_d=({ctrl.diag.get('att_roll_d',0):.2f}, {ctrl.diag.get('att_pitch_d',0):.2f})  "
                  f"rate_d=({ctrl.diag.get('rate_roll_d',0):.1f}, {ctrl.diag.get('rate_pitch_d',0):.1f})  "
                  f"roll_cmd={ctrl.diag.get('roll_cmd',0)}  pitch_cmd={ctrl.diag.get('pitch_cmd',0)}")
            crashed = True

        # --- Write CSV row ---
        d = ctrl.diag
        if d:
            row = [
                i, i / CTRL_FREQ,
                sp[0], sp[1], sp[2], sp[3],
                pos[0], pos[1], pos[2],
                vel[0], vel[1], vel[2],
                rpy_deg[0], rpy_deg[1], rpy_deg[2],
                gyro_deg[0], gyro_deg[1], gyro_deg[2],
                d.get('vel_sp_x',0), d.get('vel_sp_y',0), d.get('vel_sp_z',0),
                d.get('pidVX_err',0), d.get('pidVX_P',0), d.get('pidVX_I',0), d.get('pidVX_integ',0),
                d.get('pidVY_err',0), d.get('pidVY_P',0), d.get('pidVY_I',0), d.get('pidVY_integ',0),
                d.get('pidVZ_err',0), d.get('pidVZ_P',0), d.get('pidVZ_I',0), d.get('pidVZ_integ',0),
                d.get('thrust_raw',0),
                d.get('att_roll_d',0), d.get('att_pitch_d',0), d.get('att_yaw_d',0),
                d.get('rate_roll_d',0), d.get('rate_pitch_d',0), d.get('rate_yaw_d',0),
                d.get('pidRoll_err',0), d.get('pidRoll_integ',0),
                d.get('pidPitch_err',0), d.get('pidPitch_integ',0),
                d.get('roll_cmd',0), d.get('pitch_cmd',0), d.get('yaw_cmd',0),
                d.get('pidRR_err',0), d.get('pidRR_P',0), d.get('pidRR_I',0), d.get('pidRR_integ',0),
                d.get('pidPR_err',0), d.get('pidPR_P',0), d.get('pidPR_I',0), d.get('pidPR_integ',0),
                d.get('thrust',0), d.get('m1',0), d.get('m2',0), d.get('m3',0), d.get('m4',0),
                rpms[0], rpms[1], rpms[2], rpms[3],
            ]
            writer.writerow(row)

        if i % (CTRL_FREQ * 1) == 0:
            t = i / CTRL_FREQ
            print(f"  t={t:5.1f}s  pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}]  "
                  f"sp=[{sp[0]:+.3f},{sp[1]:+.3f},{sp[2]:.3f}]  "
                  f"rpy=[{rpy_deg[0]:+.1f},{rpy_deg[1]:+.1f},{rpy_deg[2]:+.1f}]  "
                  f"thrust={ctrl.diag.get('thrust',0):.0f}")

        if crashed and i / CTRL_FREQ > 1.0:
            # Log a bit more after crash then stop
            if pos[2] < -0.5:
                print("[DEBUG] Stopping — drone fell below -0.5m")
                break

        env.render()
        if gui:
            sync(i, START, 1.0 / CTRL_FREQ)

    csv_file.close()
    env.close()
    print(f"\n[DEBUG] Log saved to {csv_path}")
    print(f"[DEBUG] {i+1} steps logged.")
    return csv_path


# ===================================================================
#  Plotting
# ===================================================================
def plot_log(csv_path):
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError:
        print("[PLOT] matplotlib or pandas not found — skipping plots.")
        print("       Install with: pip install matplotlib pandas")
        return

    df = pd.read_csv(csv_path)
    t = df['time']

    fig, axes = plt.subplots(6, 2, figsize=(16, 20), sharex=True)
    fig.suptitle(f'Firmware PID Debug — {csv_path}', fontsize=14)

    # --- Row 0: Position tracking ---
    ax = axes[0, 0]
    ax.plot(t, df['sp_x'], '--', label='sp_x')
    ax.plot(t, df['sp_y'], '--', label='sp_y')
    ax.plot(t, df['sp_z'], '--', label='sp_z')
    ax.plot(t, df['pos_x'], label='x')
    ax.plot(t, df['pos_y'], label='y')
    ax.plot(t, df['pos_z'], label='z')
    ax.set_ylabel('Position [m]')
    ax.legend(fontsize=7, ncol=3)
    ax.set_title('Position tracking')
    ax.grid(True, alpha=0.3)

    # --- Row 0 right: Velocity ---
    ax = axes[0, 1]
    ax.plot(t, df['vel_sp_x'], '--', alpha=0.5, label='vel_sp_x')
    ax.plot(t, df['vel_sp_y'], '--', alpha=0.5, label='vel_sp_y')
    ax.plot(t, df['vel_sp_z'], '--', alpha=0.5, label='vel_sp_z')
    ax.plot(t, df['vel_x'], label='vx')
    ax.plot(t, df['vel_y'], label='vy')
    ax.plot(t, df['vel_z'], label='vz')
    ax.set_ylabel('Velocity [m/s]')
    ax.legend(fontsize=7, ncol=3)
    ax.set_title('Velocity tracking')
    ax.grid(True, alpha=0.3)

    # --- Row 1: Attitude tracking ---
    ax = axes[1, 0]
    ax.plot(t, df['att_roll_d'], '--', label='roll_d')
    ax.plot(t, df['att_pitch_d'], '--', label='pitch_d')
    ax.plot(t, df['roll'], label='roll')
    ax.plot(t, df['pitch'], label='pitch')
    ax.set_ylabel('Angle [deg]')
    ax.legend(fontsize=7)
    ax.set_title('Attitude (roll/pitch) tracking')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, df['att_yaw_d'], '--', label='yaw_d')
    ax.plot(t, df['yaw'], label='yaw')
    ax.set_ylabel('Angle [deg]')
    ax.legend(fontsize=7)
    ax.set_title('Yaw tracking')
    ax.grid(True, alpha=0.3)

    # --- Row 2: Rate tracking ---
    ax = axes[2, 0]
    ax.plot(t, df['rate_roll_d'], '--', label='roll_rate_d')
    ax.plot(t, df['rate_pitch_d'], '--', label='pitch_rate_d')
    ax.plot(t, df['gyro_x'], label='gyro_x')
    ax.plot(t, df['gyro_y'], label='gyro_y')
    ax.set_ylabel('Rate [deg/s]')
    ax.legend(fontsize=7)
    ax.set_title('Angular rate tracking')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(t, df['rate_yaw_d'], '--', label='yaw_rate_d')
    ax.plot(t, df['gyro_z'], label='gyro_z')
    ax.set_ylabel('Rate [deg/s]')
    ax.legend(fontsize=7)
    ax.set_title('Yaw rate tracking')
    ax.grid(True, alpha=0.3)

    # --- Row 3: Thrust & Motor commands ---
    ax = axes[3, 0]
    ax.plot(t, df['thrust'], label='thrust (uint16)', color='k')
    ax.axhline(PID_VEL_THRUST_BASE, ls=':', color='gray', label=f'THRUST_BASE={PID_VEL_THRUST_BASE:.0f}')
    hover_pwm = (df['thrust'].iloc[0:1].values[0] if len(df) > 0 else 36000)  # just for reference
    ax.set_ylabel('Thrust [uint16]')
    ax.legend(fontsize=7)
    ax.set_title('Thrust command')
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.plot(t, df['m1'], label='M1 (FR)', alpha=0.7)
    ax.plot(t, df['m2'], label='M2 (BR)', alpha=0.7)
    ax.plot(t, df['m3'], label='M3 (BL)', alpha=0.7)
    ax.plot(t, df['m4'], label='M4 (FL)', alpha=0.7)
    ax.set_ylabel('Motor PWM [uint16]')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Individual motor PWMs')
    ax.grid(True, alpha=0.3)

    # --- Row 4: Rate PID internals ---
    ax = axes[4, 0]
    ax.plot(t, df['roll_cmd'], label='roll_cmd', alpha=0.7)
    ax.plot(t, df['pitch_cmd'], label='pitch_cmd', alpha=0.7)
    ax.plot(t, df['yaw_cmd'], label='yaw_cmd', alpha=0.7)
    ax.set_ylabel('int16')
    ax.legend(fontsize=7)
    ax.set_title('Rate PID outputs (motor cmds)')
    ax.grid(True, alpha=0.3)

    ax = axes[4, 1]
    ax.plot(t, df['pidRR_P'], label='rollRate P', alpha=0.7)
    ax.plot(t, df['pidRR_I'], label='rollRate I', alpha=0.7)
    ax.plot(t, df['pidPR_P'], label='pitchRate P', alpha=0.7)
    ax.plot(t, df['pidPR_I'], label='pitchRate I', alpha=0.7)
    ax.set_ylabel('PID terms')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Rate PID P & I terms')
    ax.grid(True, alpha=0.3)

    # --- Row 5: VZ PID & RPMs ---
    ax = axes[5, 0]
    ax.plot(t, df['pidVZ_err'], label='VZ error', alpha=0.7)
    ax.plot(t, df['pidVZ_I'], label='VZ I-out', alpha=0.7)
    ax.plot(t, df['thrust_raw'], label='thrust_raw', alpha=0.7)
    ax.set_ylabel('VZ PID')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7)
    ax.set_title('Velocity Z PID internals')
    ax.grid(True, alpha=0.3)

    ax = axes[5, 1]
    ax.plot(t, df['rpm0'], label='RPM0', alpha=0.7)
    ax.plot(t, df['rpm1'], label='RPM1', alpha=0.7)
    ax.plot(t, df['rpm2'], label='RPM2', alpha=0.7)
    ax.plot(t, df['rpm3'], label='RPM3', alpha=0.7)
    ax.set_ylabel('RPM')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Motor RPMs')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = csv_path.replace('.csv', '.png')
    plt.savefig(plot_path, dpi=150)
    print(f"[PLOT] Saved to {plot_path}")
    plt.show()


# ===================================================================
#  CLI
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="CF firmware PID — debug version")
    parser.add_argument('--trajectory', default='circle', choices=['circle', 'simple'],
                        help='Trajectory type (default: circle)')
    parser.add_argument('--duration', default=15, type=float,
                        help='Duration [s] (default: 15 for circle, 21 for simple)')
    parser.add_argument('--height', default=0.5, type=float)
    parser.add_argument('--radius', default=0.5, type=float)
    parser.add_argument('--gui', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--no-gui', dest='gui', action='store_false')
    parser.add_argument('--csv', default='debug_log.csv', help='Output CSV path')
    parser.add_argument('--plot', default=True, type=lambda x: x.lower() == 'true')
    parser.add_argument('--no-plot', dest='plot', action='store_false')
    args = parser.parse_args()

    if args.trajectory == 'simple' and args.duration == 15:
        args.duration = 21  # default for simple trajectory

    csv_path = run_debug(
        trajectory=args.trajectory,
        duration_sec=args.duration,
        gui=args.gui,
        hover_height=args.height,
        radius=args.radius,
        csv_path=args.csv
    )

    if args.plot:
        plot_log(csv_path)


if __name__ == "__main__":
    main()