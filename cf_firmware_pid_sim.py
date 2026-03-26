#!/usr/bin/env python3
"""
Crazyflie Firmware-Faithful PID Controller — Simulation & Real Drone
=====================================================================

This script replicates the EXACT PID control pipeline from the Crazyflie
firmware (crazyflie-firmware-master), and uses it to fly a drone
(takeoff → circle → land) inside the gym-pybullet-drones simulator.

An optional --mode argument selects how much of the pipeline runs on the PC
vs. on the real Crazyflie via Crazyradio:

  sim        – Full pipeline in Python → RPMs → PyBullet           (default)
  attitude   – pos + vel PIDs on PC    → send_setpoint (att+thrust to drone)
  rate       – pos + vel + att PIDs    → send_setpoint in rate mode
  position   – trajectory only         → send_position_setpoint (all PIDs on drone)

Firmware source references are given as comments of the form:
  # [FW] path/to/file.c:LINE

All PID gains and constants come from:
  platform_defaults_cf2.h   (Crazyflie 2.1+ default gains)
  platform_defaults.h       (filter / rate defaults)

Author : Claude / Anthropic  (based on Bitcraze firmware, GPLv3)
Date   : 2026-03
"""

import os
import sys
import time
import math
import argparse
import numpy as np

# ---------------------------------------------------------------------------
# gym-pybullet-drones imports (only needed for sim mode)
# ---------------------------------------------------------------------------
try:
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.utils import sync
    HAS_PYBULLET_DRONES = True
except ImportError:
    HAS_PYBULLET_DRONES = False

# ---------------------------------------------------------------------------
# cflib imports (only needed for real drone modes)
# ---------------------------------------------------------------------------
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    HAS_CFLIB = True
except ImportError:
    HAS_CFLIB = False


# ===================================================================
#  CONSTANTS — from the firmware
# ===================================================================

# --- Rates ----------------------------------------------------------
# [FW] src/modules/interface/stabilizer_types.h:364-372
RATE_MAIN_LOOP = 1000   # Hz
ATTITUDE_RATE  = 500    # Hz
POSITION_RATE  = 100    # Hz
ATTITUDE_UPDATE_DT = 1.0 / ATTITUDE_RATE   # 0.002 s
POSITION_UPDATE_DT = 1.0 / POSITION_RATE   # 0.01  s

# --- Physical constants  -------------------------------------------
# [FW] src/platform/interface/platform_defaults_cf2.h:47,56
ARM_LENGTH = 0.046       # m
CF_MASS    = 0.027       # kg  (CF2X URDF uses 0.027)
UINT16_MAX = 65535

# ===================================================================
#  PID GAINS — platform_defaults_cf2.h:96-176
# ===================================================================

# ---- Rate PID gains  (attitude_pid_controller.c) ------------------
# [FW] platform_defaults_cf2.h:96-112
PID_ROLL_RATE_KP  = 250.0;  PID_ROLL_RATE_KI  = 500.0;  PID_ROLL_RATE_KD  = 2.5
PID_ROLL_RATE_KFF = 0.0;    PID_ROLL_RATE_INTEGRATION_LIMIT  = 33.3

PID_PITCH_RATE_KP = 250.0;  PID_PITCH_RATE_KI = 500.0;  PID_PITCH_RATE_KD = 2.5
PID_PITCH_RATE_KFF= 0.0;    PID_PITCH_RATE_INTEGRATION_LIMIT = 33.3

PID_YAW_RATE_KP   = 120.0;  PID_YAW_RATE_KI   = 16.7;   PID_YAW_RATE_KD   = 0.0
PID_YAW_RATE_KFF  = 0.0;    PID_YAW_RATE_INTEGRATION_LIMIT   = 166.7

# ---- Attitude PID gains  -----------------------------------------
# [FW] platform_defaults_cf2.h:114-130
PID_ROLL_KP  = 6.0;  PID_ROLL_KI  = 3.0;  PID_ROLL_KD  = 0.0
PID_ROLL_KFF = 0.0;  PID_ROLL_INTEGRATION_LIMIT  = 20.0

PID_PITCH_KP = 6.0;  PID_PITCH_KI = 3.0;  PID_PITCH_KD = 0.0
PID_PITCH_KFF= 0.0;  PID_PITCH_INTEGRATION_LIMIT = 20.0

PID_YAW_KP   = 6.0;  PID_YAW_KI   = 1.0;  PID_YAW_KD   = 0.35
PID_YAW_KFF  = 0.0;  PID_YAW_INTEGRATION_LIMIT   = 360.0

# ---- Velocity PID gains  -----------------------------------------
# [FW] platform_defaults_cf2.h:132-145
PID_VEL_X_KP = 5.0;  PID_VEL_X_KI = 1.0;  PID_VEL_X_KD = 0.0;  PID_VEL_X_KFF = 0.0
PID_VEL_Y_KP = 5.0;  PID_VEL_Y_KI = 1.0;  PID_VEL_Y_KD = 0.0;  PID_VEL_Y_KFF = 0.0
PID_VEL_Z_KP = 25.0;  PID_VEL_Z_KI = 15.0; PID_VEL_Z_KD = 0.0;  PID_VEL_Z_KFF = 0.0

# ---- Position PID gains  -----------------------------------------
# [FW] platform_defaults_cf2.h:158-176
PID_POS_X_KP = 1.5;  PID_POS_X_KI = 0.0;  PID_POS_X_KD = 0.0;  PID_POS_X_KFF = 0.0
PID_POS_Y_KP = 1.5;  PID_POS_Y_KI = 0.0;  PID_POS_Y_KD = 0.0;  PID_POS_Y_KFF = 0.0
PID_POS_Z_KP = 2.0;  PID_POS_Z_KI = 0.5;  PID_POS_Z_KD = 0.0;  PID_POS_Z_KFF = 0.0

# ---- Velocity / position limits  ---------------------------------
# [FW] platform_defaults_cf2.h:152-175
PID_VEL_ROLL_MAX  = 20.0   # deg
PID_VEL_PITCH_MAX = 20.0   # deg
PID_VEL_THRUST_BASE = 36000.0
PID_VEL_THRUST_MIN  = 20000.0
THRUST_SCALE = 1000.0

PID_POS_VEL_X_MAX = 1.0    # m/s
PID_POS_VEL_Y_MAX = 1.0
PID_POS_VEL_Z_MAX = 1.0

VEL_MAX_OVERHEAD = 1.10
RP_LIMIT_OVERHEAD = 1.10

# ---- Filter defaults  ---------------------------------------------
# [FW] platform_defaults.h:97-147
ATTITUDE_LPF_ENABLE   = False
ATTITUDE_LPF_CUTOFF   = 15.0    # Hz
ATTITUDE_RATE_LPF_ENABLE = False
ATTITUDE_ROLL_RATE_LPF_CUTOFF  = 30.0   # Hz
ATTITUDE_PITCH_RATE_LPF_CUTOFF = 30.0
ATTITUDE_YAW_RATE_LPF_CUTOFF   = 30.0

POS_XY_FILT_ENABLE  = True;   POS_XY_FILT_CUTOFF  = 20.0
POS_Z_FILT_ENABLE   = True;   POS_Z_FILT_CUTOFF   = 20.0
VEL_XY_FILT_ENABLE  = True;   VEL_XY_FILT_CUTOFF  = 20.0
VEL_Z_FILT_ENABLE   = True;   VEL_Z_FILT_CUTOFF   = 20.0

# ---- PID generic defaults  ----------------------------------------
# [FW] src/utils/interface/pid.h:33-34
DEFAULT_PID_INTEGRATION_LIMIT = 5000.0
DEFAULT_PID_OUTPUT_LIMIT      = 0.0      # 0 = no limit


# ===================================================================
#  Lpf2pData — 2-pole low-pass filter (Butterworth)
#  Mirrors: src/utils/src/filter.c  lpf2pInit / lpf2pApply
# ===================================================================

class Lpf2pData:
    """Second-order (2-pole) Butterworth low-pass filter.

    Translated 1:1 from the firmware's lpf2pInit / lpf2pApply.
    # [FW] src/utils/src/filter.c:184-225
    """

    def __init__(self, sample_freq: float, cutoff_freq: float):
        self.delay_element_1 = 0.0
        self.delay_element_2 = 0.0
        self.set_cutoff(sample_freq, cutoff_freq)

    def set_cutoff(self, sample_freq: float, cutoff_freq: float):
        # [FW] filter.c  lpf2pSetCutoffFreq
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

    def apply(self, sample: float) -> float:
        # [FW] filter.c  lpf2pApply
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
#  PidObject — generic PID controller
#  Mirrors: src/utils/src/pid.c
# ===================================================================

class PidObject:
    """Single-axis PID controller.

    Translated 1:1 from the firmware's pid.c.
    Key design choices preserved:
      - Derivative acts on measurement (not error) to avoid derivative kick
      - Integral clamping via iLimit
      - Output clamping via outputLimit
      - Optional 2nd-order low-pass on the D term
      - Feed-forward term on desired
      - Yaw angle wrapping (±180°)

    # [FW] src/utils/src/pid.c
    """

    def __init__(self, kp: float, ki: float, kd: float, kff: float,
                 dt: float, sampling_rate: float, cutoff_freq: float,
                 enable_d_filter: bool, desired: float = 0.0):
        self.kp  = kp
        self.ki  = ki
        self.kd  = kd
        self.kff = kff
        self.dt  = dt

        self.desired      = desired
        self.error        = 0.0
        self.prevMeasured = 0.0
        self.integ        = 0.0
        self.deriv        = 0.0

        self.outP  = 0.0
        self.outI  = 0.0
        self.outD  = 0.0
        self.outFF = 0.0

        # [FW] pid.h:33-34
        self.iLimit      = DEFAULT_PID_INTEGRATION_LIMIT
        self.outputLimit  = DEFAULT_PID_OUTPUT_LIMIT

        self.enableDFilter = enable_d_filter
        self.dFilter = None
        if self.enableDFilter:
            self.dFilter = Lpf2pData(sampling_rate, cutoff_freq)

    # [FW] pid.c:58 — pidUpdate
    def update(self, measured: float, is_yaw_angle: bool = False) -> float:
        output = 0.0

        self.error = self.desired - measured

        # Wrap yaw error to ±180°
        # [FW] pid.c:64-70
        if is_yaw_angle:
            if self.error > 180.0:
                self.error -= 360.0
            elif self.error < -180.0:
                self.error += 360.0

        # --- Proportional ---
        self.outP = self.kp * self.error
        output += self.outP

        # --- Derivative (on measurement, not error) ---
        # [FW] pid.c:82
        delta = -(measured - self.prevMeasured)
        if is_yaw_angle:
            if delta > 180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0

        # [FW] pid.c:96-101  (CONFIG_CONTROLLER_PID_FILTER_ALL not set by default)
        if self.enableDFilter:
            self.deriv = self.dFilter.apply(delta / self.dt)
        else:
            self.deriv = delta / self.dt

        if math.isnan(self.deriv):
            self.deriv = 0.0
        self.outD = self.kd * self.deriv
        output += self.outD

        # --- Integral ---
        # [FW] pid.c:108-116
        self.integ += self.error * self.dt
        if self.iLimit != 0:
            self.integ = max(-self.iLimit, min(self.iLimit, self.integ))
        self.outI = self.ki * self.integ
        output += self.outI

        # --- Feed-forward ---
        # [FW] pid.c:119
        self.outFF = self.kff * self.desired
        output += self.outFF

        # --- Output limit ---
        # [FW] pid.c:137-139
        if self.outputLimit != 0:
            output = max(-self.outputLimit, min(self.outputLimit, output))

        self.prevMeasured = measured
        return output

    # [FW] pid.c:151
    def reset(self, actual: float):
        self.error        = 0.0
        self.prevMeasured = actual
        self.integ        = 0.0
        self.deriv        = 0.0

    def set_desired(self, desired: float):
        self.desired = desired

    def set_integral_limit(self, limit: float):
        self.iLimit = limit


# ===================================================================
#  Position Controller (pos → vel → attitude + thrust)
#  Mirrors: src/modules/src/controller/position_controller_pid.c
# ===================================================================

class CrazyfliePositionController:
    """Position + Velocity PID cascade.

    Pipeline:
      position error → pidX/Y/Z → desired velocity (body-yaw-aligned)
      velocity error → pidVX/VY/VZ → desired attitude (roll, pitch) + thrust

    Coordinate frame: body-yaw-aligned for X/Y, global for Z.
    # [FW] src/modules/src/controller/position_controller_pid.c
    """

    def __init__(self):
        dt = POSITION_UPDATE_DT   # 0.01 s  (100 Hz)

        # ---- Position PIDs  [FW] position_controller_pid.c:87-163 --
        self.pidX = PidObject(PID_POS_X_KP, PID_POS_X_KI, PID_POS_X_KD,
                              PID_POS_X_KFF, dt, POSITION_RATE,
                              POS_XY_FILT_CUTOFF, POS_XY_FILT_ENABLE)
        self.pidY = PidObject(PID_POS_Y_KP, PID_POS_Y_KI, PID_POS_Y_KD,
                              PID_POS_Y_KFF, dt, POSITION_RATE,
                              POS_XY_FILT_CUTOFF, POS_XY_FILT_ENABLE)
        self.pidZ = PidObject(PID_POS_Z_KP, PID_POS_Z_KI, PID_POS_Z_KD,
                              PID_POS_Z_KFF, dt, POSITION_RATE,
                              POS_Z_FILT_CUTOFF, POS_Z_FILT_ENABLE)

        # ---- Velocity PIDs  [FW] position_controller_pid.c:88-127 --
        self.pidVX = PidObject(PID_VEL_X_KP, PID_VEL_X_KI, PID_VEL_X_KD,
                               PID_VEL_X_KFF, dt, POSITION_RATE,
                               VEL_XY_FILT_CUTOFF, VEL_XY_FILT_ENABLE)
        self.pidVY = PidObject(PID_VEL_Y_KP, PID_VEL_Y_KI, PID_VEL_Y_KD,
                               PID_VEL_Y_KFF, dt, POSITION_RATE,
                               VEL_XY_FILT_CUTOFF, VEL_XY_FILT_ENABLE)
        self.pidVZ = PidObject(PID_VEL_Z_KP, PID_VEL_Z_KI, PID_VEL_Z_KD,
                               PID_VEL_Z_KFF, dt, POSITION_RATE,
                               VEL_Z_FILT_CUTOFF, VEL_Z_FILT_ENABLE)

        self.thrustBase = PID_VEL_THRUST_BASE
        self.thrustMin  = PID_VEL_THRUST_MIN

    def update(self, setpoint_pos, state_pos, state_vel, state_yaw_deg):
        """Run the full position → velocity → attitude+thrust cascade.

        All angles in DEGREES, positions in meters, velocities in m/s.

        Parameters
        ----------
        setpoint_pos : (x, y, z)       — desired position [m], global frame
        state_pos    : (x, y, z)       — current position [m], global frame
        state_vel    : (vx, vy, vz)    — current velocity [m/s], global frame
        state_yaw_deg: float           — current yaw [deg]

        Returns
        -------
        thrust_u16  : float  — thrust command [0..65535]
        roll_deg    : float  — desired roll  [deg]
        pitch_deg   : float  — desired pitch [deg]
        """
        sp_x, sp_y, sp_z = setpoint_pos
        st_x, st_y, st_z = state_pos
        st_vx, st_vy, st_vz = state_vel

        # --- Set output limits ---
        # [FW] position_controller_pid.c:196-200
        self.pidX.outputLimit = PID_POS_VEL_X_MAX * VEL_MAX_OVERHEAD
        self.pidY.outputLimit = PID_POS_VEL_Y_MAX * VEL_MAX_OVERHEAD
        self.pidZ.outputLimit = max(PID_POS_VEL_Z_MAX, 0.5) * VEL_MAX_OVERHEAD

        # --- Rotate setpoint and state to body-yaw-aligned frame ---
        # [FW] position_controller_pid.c:202-209
        cosyaw = math.cos(math.radians(state_yaw_deg))
        sinyaw = math.sin(math.radians(state_yaw_deg))

        setp_body_x =  sp_x * cosyaw + sp_y * sinyaw
        setp_body_y = -sp_x * sinyaw + sp_y * cosyaw

        state_body_x =  st_x * cosyaw + st_y * sinyaw
        state_body_y = -st_x * sinyaw + st_y * cosyaw

        # --- Position → desired velocity (body-yaw-aligned) ---
        # [FW] position_controller_pid.c:219-231 (modeAbs for all axes)
        self.pidX.set_desired(setp_body_x)
        vel_sp_x = self.pidX.update(state_body_x)

        self.pidY.set_desired(setp_body_y)
        vel_sp_y = self.pidY.update(state_body_y)

        self.pidZ.set_desired(sp_z)
        vel_sp_z = self.pidZ.update(st_z)

        # --- Velocity → attitude + thrust ---
        # [FW] position_controller_pid.c:236-267
        self.pidVX.outputLimit = PID_VEL_PITCH_MAX * RP_LIMIT_OVERHEAD
        self.pidVY.outputLimit = PID_VEL_ROLL_MAX  * RP_LIMIT_OVERHEAD
        self.pidVZ.outputLimit = (UINT16_MAX / 2.0 / THRUST_SCALE)

        # Rotate state velocity to body-yaw-aligned
        # [FW] position_controller_pid.c:247-248
        state_body_vx =  st_vx * cosyaw + st_vy * sinyaw
        state_body_vy = -st_vx * sinyaw + st_vy * cosyaw

        # Roll and Pitch from velocity PID
        # [FW] position_controller_pid.c:251-255
        self.pidVX.set_desired(vel_sp_x)
        pitch_deg = -self.pidVX.update(state_body_vx)

        self.pidVY.set_desired(vel_sp_y)
        roll_deg  = -self.pidVY.update(state_body_vy)

        roll_deg  = max(-PID_VEL_ROLL_MAX,  min(PID_VEL_ROLL_MAX,  roll_deg))
        pitch_deg = max(-PID_VEL_PITCH_MAX, min(PID_VEL_PITCH_MAX, pitch_deg))

        # Thrust from Z velocity PID
        # [FW] position_controller_pid.c:258-266
        self.pidVZ.set_desired(vel_sp_z)
        thrust_raw = self.pidVZ.update(st_vz)
        thrust = thrust_raw * THRUST_SCALE + self.thrustBase
        if thrust < self.thrustMin:
            thrust = self.thrustMin
        thrust = max(0.0, min(UINT16_MAX, thrust))

        return thrust, roll_deg, pitch_deg

    def reset_all(self, x, y, z):
        # [FW] position_controller_pid.c:269-277
        self.pidX.reset(x);   self.pidY.reset(y);   self.pidZ.reset(z)
        self.pidVX.reset(0);  self.pidVY.reset(0);  self.pidVZ.reset(0)


# ===================================================================
#  Attitude Controller  (attitude → rate → motor torques)
#  Mirrors: src/modules/src/controller/attitude_pid_controller.c
# ===================================================================

class CrazyflieAttitudeController:
    """Attitude (angle) + Rate (angular velocity) PID cascade.

    Pipeline:
      attitude error → pidRoll/Pitch/Yaw → desired angular rate [deg/s]
      rate error     → pidRollRate/PitchRate/YawRate → motor commands (int16)

    All units are DEGREES and DEGREES/S.
    # [FW] src/modules/src/controller/attitude_pid_controller.c
    """

    def __init__(self):
        dt = ATTITUDE_UPDATE_DT   # 0.002 s  (500 Hz)

        # ---- Rate PIDs  [FW] attitude_pid_controller.c:57-76 ------
        self.pidRollRate  = PidObject(
            PID_ROLL_RATE_KP, PID_ROLL_RATE_KI, PID_ROLL_RATE_KD,
            PID_ROLL_RATE_KFF, dt, ATTITUDE_RATE,
            ATTITUDE_ROLL_RATE_LPF_CUTOFF, ATTITUDE_RATE_LPF_ENABLE)
        self.pidPitchRate = PidObject(
            PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD,
            PID_PITCH_RATE_KFF, dt, ATTITUDE_RATE,
            ATTITUDE_PITCH_RATE_LPF_CUTOFF, ATTITUDE_RATE_LPF_ENABLE)
        self.pidYawRate   = PidObject(
            PID_YAW_RATE_KP, PID_YAW_RATE_KI, PID_YAW_RATE_KD,
            PID_YAW_RATE_KFF, dt, ATTITUDE_RATE,
            ATTITUDE_YAW_RATE_LPF_CUTOFF, ATTITUDE_RATE_LPF_ENABLE)

        # [FW] attitude_pid_controller.c:118-120
        self.pidRollRate.set_integral_limit(PID_ROLL_RATE_INTEGRATION_LIMIT)
        self.pidPitchRate.set_integral_limit(PID_PITCH_RATE_INTEGRATION_LIMIT)
        self.pidYawRate.set_integral_limit(PID_YAW_RATE_INTEGRATION_LIMIT)

        # ---- Attitude PIDs  [FW] attitude_pid_controller.c:78-97 --
        self.pidRoll  = PidObject(
            PID_ROLL_KP, PID_ROLL_KI, PID_ROLL_KD, PID_ROLL_KFF,
            dt, ATTITUDE_RATE, ATTITUDE_LPF_CUTOFF, ATTITUDE_LPF_ENABLE)
        self.pidPitch = PidObject(
            PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD, PID_PITCH_KFF,
            dt, ATTITUDE_RATE, ATTITUDE_LPF_CUTOFF, ATTITUDE_LPF_ENABLE)
        self.pidYaw   = PidObject(
            PID_YAW_KP, PID_YAW_KI, PID_YAW_KD, PID_YAW_KFF,
            dt, ATTITUDE_RATE, ATTITUDE_LPF_CUTOFF, ATTITUDE_LPF_ENABLE)

        # [FW] attitude_pid_controller.c:129-131
        self.pidRoll.set_integral_limit(PID_ROLL_INTEGRATION_LIMIT)
        self.pidPitch.set_integral_limit(PID_PITCH_INTEGRATION_LIMIT)
        self.pidYaw.set_integral_limit(PID_YAW_INTEGRATION_LIMIT)

    def correct_attitude(self, roll_actual, pitch_actual, yaw_actual,
                         roll_desired, pitch_desired, yaw_desired):
        """Attitude PID → desired angular rates [deg/s].

        # [FW] attitude_pid_controller.c:156-171
        """
        self.pidRoll.set_desired(roll_desired)
        roll_rate_desired  = self.pidRoll.update(roll_actual)

        self.pidPitch.set_desired(pitch_desired)
        pitch_rate_desired = self.pidPitch.update(pitch_actual)

        self.pidYaw.set_desired(yaw_desired)
        yaw_rate_desired   = self.pidYaw.update(yaw_actual, is_yaw_angle=True)

        return roll_rate_desired, pitch_rate_desired, yaw_rate_desired

    def correct_rate(self, gyro_roll, gyro_pitch, gyro_yaw,
                     rate_desired_roll, rate_desired_pitch, rate_desired_yaw):
        """Rate PID → motor commands (int16).

        gyro values are body-frame angular rates in DEG/S.
        # [FW] attitude_pid_controller.c:141-153
        """
        self.pidRollRate.set_desired(rate_desired_roll)
        roll_out  = self._saturate_i16(self.pidRollRate.update(gyro_roll))

        self.pidPitchRate.set_desired(rate_desired_pitch)
        pitch_out = self._saturate_i16(self.pidPitchRate.update(gyro_pitch))

        self.pidYawRate.set_desired(rate_desired_yaw)
        yaw_out   = self._saturate_i16(self.pidYawRate.update(gyro_yaw))

        return roll_out, pitch_out, yaw_out

    def reset_all(self, roll, pitch, yaw):
        # [FW] attitude_pid_controller.c:183-191
        self.pidRoll.reset(roll)
        self.pidPitch.reset(pitch)
        self.pidYaw.reset(yaw)
        self.pidRollRate.reset(0)
        self.pidPitchRate.reset(0)
        self.pidYawRate.reset(0)

    @staticmethod
    def _saturate_i16(val):
        # [FW] attitude_pid_controller.c:46-55
        if val > 32767:
            return 32767
        elif val < -32767:
            return -32767
        return int(val)


# ===================================================================
#  Power Distribution  (legacy mode — thrust + roll/pitch/yaw → 4 motor PWMs)
#  Mirrors: src/modules/src/power_distribution_quadrotor.c
# ===================================================================

class CrazyfliePowerDistribution:
    """Motor mixing for the standard quadrotor (legacy control mode).

    Motor layout (CF2 X-configuration, viewed from above):
    ┌─────────────────────────────────┐
    │         Front                   │
    │    M4 (CCW)   M1 (CW)          │
    │         \\   /                  │
    │          \\ /                   │
    │           X                     │
    │          / \\                   │
    │         /   \\                  │
    │    M3 (CW)    M2 (CCW)         │
    │         Back                    │
    └─────────────────────────────────┘

    Firmware index M1..M4 maps to pybullet prop0..prop3.

    # [FW] src/modules/src/power_distribution_quadrotor.c:84-93
    """

    def __init__(self, idle_thrust=0):
        self.idle_thrust = idle_thrust

    def distribute(self, thrust, roll, pitch, yaw):
        """Convert thrust (uint16) + roll/pitch/yaw (int16) to 4 motor PWMs.

        Returns motor_pwm[0..3] as floats in [0, 65535].
        # [FW] power_distribution_quadrotor.c:84-93
        """
        r = roll  / 2.0
        p = pitch / 2.0

        # [FW] power_distribution_quadrotor.c:89-92
        m1 = thrust - r + p + yaw    # front-right  (CW)
        m2 = thrust - r - p - yaw    # rear-right   (CCW)
        m3 = thrust + r - p + yaw    # rear-left    (CW)
        m4 = thrust + r + p - yaw    # front-left   (CCW)

        return self.cap([m1, m2, m3, m4])

    def cap(self, motor_thrust_uncapped):
        """Cap motor values — prioritize attitude over thrust.

        If any motor exceeds 65535, all are shifted down equally.
        # [FW] power_distribution_quadrotor.c:181-211
        """
        highest = max(motor_thrust_uncapped)
        reduction = 0
        if highest > UINT16_MAX:
            reduction = highest - UINT16_MAX

        result = []
        for t in motor_thrust_uncapped:
            val = t - reduction
            if val < self.idle_thrust:
                val = self.idle_thrust
            result.append(max(0.0, min(UINT16_MAX, val)))
        return result


# ===================================================================
#  Top-Level Controller  (controllerPid)
#  Mirrors: src/modules/src/controller/controller_pid.c
# ===================================================================

def cap_angle(angle):
    """Wrap angle to ±180°.
    # [FW] controller_pid.c:42-54
    """
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


class CrazyflieFirmwarePID:
    """Complete firmware PID cascade: position → velocity → attitude → rate → motors.

    This class ties together all four PID stages exactly as in the firmware's
    controllerPid() function, plus power distribution to get motor PWMs.

    # [FW] src/modules/src/controller/controller_pid.c:56-163
    """

    def __init__(self):
        self.pos_ctrl = CrazyfliePositionController()
        self.att_ctrl = CrazyflieAttitudeController()
        self.power    = CrazyfliePowerDistribution()

        self.attitude_desired_yaw = 0.0   # persistent yaw setpoint
        self.actuator_thrust = 0.0

        # Internal state for multi-rate scheduling
        self._tick = 0

    def update(self, setpoint_pos, setpoint_yaw_rate, state_pos, state_vel,
               state_rpy_deg, state_gyro_deg):
        """Run one full control step.

        Parameters
        ----------
        setpoint_pos     : (x, y, z)  — desired position [m], global
        setpoint_yaw_rate: float      — desired yaw rate [deg/s]
        state_pos        : (x, y, z)  — current position [m], global
        state_vel        : (vx,vy,vz) — current velocity [m/s], global
        state_rpy_deg    : (r, p, y)  — current attitude [deg]
        state_gyro_deg   : (gx,gy,gz) — body-frame angular rates [deg/s]

        Returns
        -------
        motor_pwm : list of 4 floats in [0, 65535]
        """
        roll_a, pitch_a, yaw_a = state_rpy_deg
        gyro_x, gyro_y, gyro_z = state_gyro_deg

        # === Yaw setpoint accumulation (rate mode for yaw) ===
        # [FW] controller_pid.c:65-66
        self.attitude_desired_yaw = cap_angle(
            self.attitude_desired_yaw + setpoint_yaw_rate * ATTITUDE_UPDATE_DT)
        self.attitude_desired_yaw = cap_angle(self.attitude_desired_yaw)

        # === Position controller (runs at POSITION_RATE = 100 Hz) ===
        # [FW] controller_pid.c:93-94
        #
        # In the firmware, the main loop runs at 1000 Hz and the position
        # controller fires every 10th tick (RATE_DO_EXECUTE).
        # Here we schedule it every 5th attitude step since attitude is 500 Hz.
        if self._tick % (ATTITUDE_RATE // POSITION_RATE) == 0:
            self.actuator_thrust, att_roll, att_pitch = self.pos_ctrl.update(
                setpoint_pos, state_pos, state_vel, yaw_a)
        else:
            # Reuse last position controller output
            # (att_roll / att_pitch are set only when pos ctrl runs;
            #  between runs the firmware also reuses the last values)
            att_roll  = getattr(self, '_last_att_roll', 0.0)
            att_pitch = getattr(self, '_last_att_pitch', 0.0)

        self._last_att_roll  = att_roll  if self._tick % (ATTITUDE_RATE // POSITION_RATE) == 0 else getattr(self, '_last_att_roll', 0.0)
        self._last_att_pitch = att_pitch if self._tick % (ATTITUDE_RATE // POSITION_RATE) == 0 else getattr(self, '_last_att_pitch', 0.0)

        # === Attitude controller (runs at ATTITUDE_RATE = 500 Hz) ===
        # [FW] controller_pid.c:107-109
        roll_rate_des, pitch_rate_des, yaw_rate_des = \
            self.att_ctrl.correct_attitude(
                roll_a, pitch_a, yaw_a,
                self._last_att_roll, self._last_att_pitch,
                self.attitude_desired_yaw)

        # === Rate controller ===
        # [FW] controller_pid.c:124-125
        # NOTE: firmware negates gyro.y because of sensor mounting orientation.
        # In simulation, we negate it in obs_to_firmware_state() instead,
        # to account for the pitch convention difference (pybullet Z-up
        # vs. firmware aerospace convention).
        roll_cmd, pitch_cmd, yaw_cmd = \
            self.att_ctrl.correct_rate(
                gyro_x, gyro_y, gyro_z,
                roll_rate_des, pitch_rate_des, yaw_rate_des)

        # [FW] controller_pid.c:131  —  yaw output is negated
        yaw_cmd = -yaw_cmd

        thrust = self.actuator_thrust

        # === Zero-thrust safety ===
        # [FW] controller_pid.c:145-162
        if thrust == 0:
            self.att_ctrl.reset_all(roll_a, pitch_a, yaw_a)
            self.pos_ctrl.reset_all(state_pos[0], state_pos[1], state_pos[2])
            self.attitude_desired_yaw = yaw_a
            self._tick += 1
            return [0.0, 0.0, 0.0, 0.0]

        # === Power distribution ===
        # [FW] power_distribution_quadrotor.c:84-93
        motor_pwm = self.power.distribute(thrust, roll_cmd, pitch_cmd, yaw_cmd)

        self._tick += 1
        return motor_pwm

    def reset(self, state_rpy_deg, state_pos):
        self.att_ctrl.reset_all(*state_rpy_deg)
        self.pos_ctrl.reset_all(*state_pos)
        self.attitude_desired_yaw = state_rpy_deg[2]
        self._tick = 0
        self._last_att_roll  = 0.0
        self._last_att_pitch = 0.0


# ===================================================================
#  Motor Dynamics Filter (for simulation only)
# ===================================================================

class MotorDynamicsFilter:
    """First-order low-pass filter simulating brushed motor response.

    On the real Crazyflie 2.1+, the brushed coreless motors have a
    mechanical+electrical time constant of roughly 15–25 ms.  In PyBullet's
    PYB physics mode, forces are applied instantaneously (zero motor lag),
    which makes the firmware's high-gain rate PID oscillate in a limit
    cycle.  This filter restores a realistic motor bandwidth.

    Model:  rpm(t+dt) = α · rpm_cmd + (1-α) · rpm(t)
            α = dt / (τ + dt)

    With τ = 0.02 s and dt = 0.002 s (500 Hz control):  α ≈ 0.091
    """

    def __init__(self, n_motors=4, tau=0.02, dt=0.002):
        self.alpha = dt / (tau + dt)
        self.rpm = np.zeros(n_motors)

    def apply(self, rpm_cmd):
        """Filter commanded RPMs through first-order motor dynamics."""
        self.rpm = self.alpha * rpm_cmd + (1.0 - self.alpha) * self.rpm
        return self.rpm.copy()

    def reset(self):
        self.rpm[:] = 0.0


# ===================================================================
#  Flow Deck v2 State Estimator Simulator
# ===================================================================

class FlowDeckSimulator:
    """Simulates the transport delay of the Crazyflie Flow deck v2.

    The primary effect responsible for PID oscillations is the latency
    between the physical state and what the position PID receives:
      - PMW3901 sampling at 100 Hz       → 10 ms per measurement
      - I2C transfer + EKF update        → ~5 ms
      - Total round-trip to PID input    → ~15 ms

    This delay reduces the phase margin of the closed-loop system.  The
    default CF2.1+ gains were tuned for Lighthouse/Loco (<5 ms latency);
    with the Flow deck the same gains are often marginally stable → oscillations.

    An optional Ornstein-Uhlenbeck velocity noise model (vel_noise > 0) is
    available for realism testing, but is disabled by default: deterministic
    delay-only runs are more useful for systematic PID tuning.

    Parameters
    ----------
    ctrl_freq : int   — simulation control frequency [Hz]           (default 500)
    delay_ms  : float — total sensor-to-PID latency [ms]           (default 25)
    vel_noise : float — OU velocity noise std at ref_height [m/s]
                        Set to 0 (default) to disable noise.
    ref_height: float — reference height for noise scaling [m]     (default 0.5)
    noise_tau : float — OU correlation time [s]                    (default 0.5)
    """

    FLOW_HZ = 100   # PMW3901 update rate as configured in the CF firmware

    def __init__(self, ctrl_freq=500, delay_ms=110,
                 vel_noise=0.0, ref_height=0.5, noise_tau=0.5):
        self.ctrl_freq  = ctrl_freq
        self.delay_ms   = delay_ms
        self.vel_noise  = vel_noise
        self.ref_height = ref_height
        self.noise_tau  = noise_tau

        # Delay buffer: holds the last `delay_steps` (pos, vel) pairs
        delay_steps = max(1, round(delay_ms * 1e-3 * ctrl_freq))
        self._delay_steps = delay_steps
        self._buf_pos = [np.zeros(3)] * delay_steps
        self._buf_vel = [np.zeros(3)] * delay_steps
        self._head = 0   # circular buffer write index

        # OU noise state (only used when vel_noise > 0)
        self._dt_flow  = 1.0 / self.FLOW_HZ
        self._ou_alpha = math.exp(-self._dt_flow / noise_tau) if noise_tau > 0 else 0.0
        self._ou_x     = 0.0
        self._ou_y     = 0.0
        self._step     = 0
        self._steps_per_flow = ctrl_freq // self.FLOW_HZ

    def reset(self, true_pos):
        self._buf_pos = [true_pos.copy() for _ in range(self._delay_steps)]
        self._buf_vel = [np.zeros(3)     for _ in range(self._delay_steps)]
        self._head  = 0
        self._ou_x  = 0.0
        self._ou_y  = 0.0
        self._step  = 0

    def update(self, true_pos, true_vel):
        """Advance one control step.

        Returns (pos_delayed, vel_delayed) — the state seen by the position
        PID after the Flow deck's transport delay.  If vel_noise > 0, OU
        noise is added at 100 Hz before the delay buffer.
        """
        self._step += 1

        pos_in = true_pos.copy()
        vel_in = true_vel.copy()

        # Optional OU velocity noise (100 Hz ticks only)
        if self.vel_noise > 0 and self._step % self._steps_per_flow == 0:
            h = max(true_pos[2], 0.02)
            sigma_v     = self.vel_noise * (h / self.ref_height)
            sigma_drive = sigma_v * math.sqrt(1.0 - self._ou_alpha ** 2)
            self._ou_x  = self._ou_alpha * self._ou_x + sigma_drive * np.random.randn()
            self._ou_y  = self._ou_alpha * self._ou_y + sigma_drive * np.random.randn()
            vel_in[0]  += self._ou_x
            vel_in[1]  += self._ou_y
            pos_in[0]  += self._ou_x * self._dt_flow
            pos_in[1]  += self._ou_y * self._dt_flow

        # Write current (possibly noisy) state into circular delay buffer
        self._buf_pos[self._head] = pos_in
        self._buf_vel[self._head] = vel_in
        self._head = (self._head + 1) % self._delay_steps

        # Read oldest entry (= state from delay_ms ago)
        tail = self._head   # after increment, head points to oldest slot
        return self._buf_pos[tail].copy(), self._buf_vel[tail].copy()


# ===================================================================
#  PWM → RPM conversion (for pybullet-drones)
# ===================================================================

# CF2.1+ per-motor thrust limits (with battery compensation ON by default)
# [FW] src/platform/interface/platform_defaults_cf2.h:68-75
# The firmware maps uint16 [0..65535] linearly to [0..THRUST_MAX] in Newtons.
# Battery compensation adjusts the PWM duty cycle so that the same uint16
# value produces the same physical thrust regardless of battery voltage.
CF2_THRUST_MAX_PER_MOTOR = 0.12     # N  (CONFIG_CRAZYFLIE_21_PLUS)
CF2_THRUST_MIN_PER_MOTOR = 0.01282  # N

# [FW] motors.h:48 — the hardware timer only has 8-bit resolution
MOTORS_PWM_BITS = 8


def pwm_to_rpm(pwm_values, kf, truncate_8bit=True):
    """Convert firmware uint16 PWM [0..65535] to RPM for pybullet.

    Faithfully replicates the real CF2.1+ signal chain:
      1. (optional) Truncate to 8-bit timer resolution, like the real hardware
      2. Map uint16 linearly to [0, THRUST_MAX] Newtons (battery compensation)
      3. Convert thrust to RPM via  thrust = KF × rpm²

    Parameters
    ----------
    pwm_values : list/array of 4 uint16 PWMs
    kf         : float — propeller thrust coefficient from the URDF [N/rpm²]
    truncate_8bit : bool — if True, simulate the 8-bit timer truncation
                    (default True for maximum fidelity; set False if you want
                    the "ideal" mapping without quantization noise)

    Returns
    -------
    np.ndarray of 4 RPM values
    """
    rpms = []
    for pwm in pwm_values:
        pwm = max(0.0, min(UINT16_MAX, pwm))

        # [FW] motors.c:115  motorsConv16ToBits — 8-bit truncation
        if truncate_8bit:
            pwm = float(int(pwm) >> (16 - MOTORS_PWM_BITS)
                        << (16 - MOTORS_PWM_BITS))

        # [FW] motors.c:165  motorsCompensateBatteryVoltage
        # With battery compensation ON (default), the firmware ensures
        # that uint16 maps linearly to thrust:
        #   thrust_per_motor = (pwm / 65535) * THRUST_MAX
        thrust = (pwm / UINT16_MAX) * CF2_THRUST_MAX_PER_MOTOR

        # Convert thrust (Newtons) → RPM for pybullet
        #   thrust = KF × rpm²  →  rpm = sqrt(thrust / KF)
        if thrust <= 0:
            rpms.append(0.0)
        else:
            rpms.append(math.sqrt(thrust / kf))

    return np.array(rpms)


# ===================================================================
#  Trajectory generation  (takeoff → circle → land)
# ===================================================================

def generate_trajectory(ctrl_freq, duration_sec, hover_height=0.5, radius=0.5):
    """Generate a smooth takeoff → circle → landing trajectory.

    Returns
    -------
    waypoints : np.ndarray of shape (N, 4) — [x, y, z, yaw_rate_deg_s]
    """
    n_steps = int(ctrl_freq * duration_sec)
    waypoints = np.zeros((n_steps, 4))

    takeoff_time   = 2    # seconds
    circle_time    = duration_sec - 2 * takeoff_time   # seconds of circling

    takeoff_steps  = int(ctrl_freq * takeoff_time)
    circle_steps   = int(ctrl_freq * circle_time)
    landing_steps  = n_steps - takeoff_steps - circle_steps

    # --- Phase 1: Takeoff (vertical climb to hover_height) ---
    for i in range(takeoff_steps):
        t = i / takeoff_steps
        z = hover_height * t
        waypoints[i] = [0.0, 0.0, z, 0.0]

    # --- Phase 2: Circle at hover_height ---
    for i in range(circle_steps):
        t = i / circle_steps
        angle = t * 2 * math.pi   # one full circle
        x = radius * math.cos(angle) - radius  # start at (0,0)
        y = radius * math.sin(angle)
        waypoints[takeoff_steps + i] = [x, y, hover_height, 0.0]

    # --- Phase 3: Landing (back to center, descend) ---
    # First, the circle ends near (0, 0).  Descend smoothly.
    for i in range(landing_steps):
        t = i / landing_steps
        z = hover_height * (1.0 - t)
        waypoints[takeoff_steps + circle_steps + i] = [0.0, 0.0, z, 0.0]

    return waypoints


# ===================================================================
#  Helper: extract firmware-compatible state from pybullet obs
# ===================================================================

def obs_to_firmware_state(obs):
    """Convert pybullet-drones observation (20,) to firmware-style state.

    obs layout (from BaseAviary._getDroneStateVector):
      [0:3]   position  (x, y, z)        — global frame [m]
      [3:7]   quaternion (qx, qy, qz, qw) — pybullet convention
      [7:10]  rpy       (roll, pitch, yaw) — [rad]
      [10:13] velocity  (vx, vy, vz)      — global frame [m/s]
      [13:16] angular velocity (wx, wy, wz) — global frame [rad/s]
      [16:20] last motor RPMs
    """
    pos    = obs[0:3]                        # [m] global
    rpy    = np.degrees(obs[7:10])           # [deg]
    vel    = obs[10:13]                      # [m/s] global
    ang_v  = obs[13:16]                      # [rad/s] global

    # ── Pitch convention fix ──────────────────────────────────────
    # pybullet (Z-up frame, right-hand rule about Y):
    #   positive pitch = nose DOWN
    # Crazyflie firmware (aerospace convention):
    #   positive pitch = nose UP
    #
    # Roll and yaw conventions match between the two frames.
    # Only pitch (and the corresponding body-frame pitch rate, gyro_y)
    # must be negated to convert from pybullet to firmware convention.
    rpy[1] = -rpy[1]

    # Convert angular velocity from world frame to body frame
    # [FW] In reality the gyro measures body-frame rates directly.
    # In pybullet, ang_v is world-frame → we rotate: ω_body = Rᵀ @ ω_world
    r, p, y_ = obs[7:10]   # radians
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y_), math.sin(y_)

    # Rotation matrix (ZYX Euler: yaw-pitch-roll)
    R = np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [  -sp,          cp*sr,           cp*cr       ]
    ])
    gyro_body = R.T @ ang_v          # [rad/s] body frame
    gyro_deg  = np.degrees(gyro_body) # [deg/s] body frame

    # Negate pitch rate for the same convention reason as pitch angle
    gyro_deg[1] = -gyro_deg[1]

    return pos, vel, rpy, gyro_deg


# ===================================================================
#  MAIN — Simulation mode
# ===================================================================

def run_sim(duration_sec=15, gui=True, hover_height=0.5, radius=0.5, plot=True,
            simulate_flow_deck=False):
    """Run the full firmware PID pipeline in PyBullet simulation.

    Parameters
    ----------
    simulate_flow_deck : bool
        If True, replace the perfect PyBullet state with a simulated Flow
        deck v2 estimate (noise + delay) before feeding it to the position
        PID.  This reproduces the oscillations seen on the real drone.
    """
    if not HAS_PYBULLET_DRONES:
        print("ERROR: gym-pybullet-drones not found. Install it first.")
        sys.exit(1)

    # We run pybullet physics at 1000 Hz and our controller at 500 Hz
    # (matching the firmware's ATTITUDE_RATE).
    # The position controller runs internally at 100 Hz (every 5th step).
    PYB_FREQ  = 1000
    CTRL_FREQ = 500

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.02]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.PYB_GND_DRAG_DW,
        pyb_freq=PYB_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False
    )

    # Extract drone constants from the environment
    MAX_RPM = env.MAX_RPM
    KF = env.KF    # thrust coefficient: thrust_per_prop = KF * rpm²

    # Initialize controller
    ctrl = CrazyflieFirmwarePID()

    # Motor dynamics filter — simulates real motor response lag
    # τ ≈ 20 ms is typical for CF2.1+ brushed coreless motors
    motor_filter = MotorDynamicsFilter(n_motors=4, tau=0.02, dt=1.0/CTRL_FREQ)

    # Pre-arm: initialize motor filter and first action at hover RPM
    # so the drone doesn't freefall on the first env.step()
    HOVER_RPM = env.HOVER_RPM
    motor_filter.rpm = np.full(4, HOVER_RPM)

    # Flow deck simulator (optional) — models sensor noise + delay
    if simulate_flow_deck:
        flow_deck = FlowDeckSimulator(ctrl_freq=CTRL_FREQ)
        flow_deck.reset(np.array([0.0, 0.0, 0.02]))
        noise_str = (f", vel_noise={flow_deck.vel_noise} m/s (OU τ={flow_deck.noise_tau}s)"
                     if flow_deck.vel_noise > 0 else ", noise OFF")
        print(f"[SIM] Flow deck simulation ON — delay={flow_deck.delay_ms} ms{noise_str}")
    else:
        flow_deck = None

    # Generate trajectory
    waypoints = generate_trajectory(CTRL_FREQ, duration_sec,
                                    hover_height=hover_height, radius=radius)
    n_steps = len(waypoints)

    print(f"[SIM] PyBullet freq: {PYB_FREQ} Hz, Control freq: {CTRL_FREQ} Hz")
    print(f"[SIM] MAX_RPM: {MAX_RPM:.1f}, HOVER_RPM: {HOVER_RPM:.1f}, KF: {KF:.4e}")
    print(f"[SIM] Firmware THRUST_MAX/motor: {CF2_THRUST_MAX_PER_MOTOR*1000:.1f} mN, "
          f"RPM at THRUST_MAX: {math.sqrt(CF2_THRUST_MAX_PER_MOTOR/KF):.1f}")
    print(f"[SIM] Trajectory: {n_steps} steps, {duration_sec}s")
    print(f"[SIM] Phases: takeoff 3s → circle {duration_sec-6}s → land 3s")

    action = np.full((1, 4), HOVER_RPM)
    START = time.time()

    # --- Data logging arrays ---
    log_t   = np.zeros(n_steps)
    log_sp  = np.zeros((n_steps, 3))
    log_pos = np.zeros((n_steps, 3))
    log_vel = np.zeros((n_steps, 3))
    log_rpy = np.zeros((n_steps, 3))
    log_rpms = np.zeros((n_steps, 4))
    log_thrust = np.zeros(n_steps)

    for i in range(n_steps):
        # --- Step the simulation ---
        obs, _, _, _, _ = env.step(action)

        # --- Extract state ---
        pos, vel, rpy_deg, gyro_deg = obs_to_firmware_state(obs[0])

        # --- Optionally replace pos/vel with Flow deck estimate ---
        # Attitude (rpy) and gyro come from the IMU which is much more
        # accurate; only position and velocity are affected by the flow deck.
        if flow_deck is not None:
            pos_ctrl, vel_ctrl = flow_deck.update(pos, vel)
        else:
            pos_ctrl, vel_ctrl = pos, vel

        # --- Get setpoint ---
        sp = waypoints[i]
        setpoint_pos      = sp[0:3]
        setpoint_yaw_rate = sp[3]

        # --- Run firmware controller ---
        motor_pwm = ctrl.update(
            setpoint_pos, setpoint_yaw_rate,
            pos_ctrl, vel_ctrl, rpy_deg, gyro_deg
        )

        # --- Convert PWM to RPM for pybullet ---
        rpms = pwm_to_rpm(motor_pwm, KF)

        # --- Apply motor dynamics filter (simulates real motor lag) ---
        rpms = motor_filter.apply(rpms)

        action[0, :] = rpms

        # --- Log ---
        # Log pos_ctrl/vel_ctrl (what the PID sees) so that the comparison
        # plot is on the same footing as the real drone's stateEstimate.
        log_t[i]      = i / CTRL_FREQ
        log_sp[i]     = setpoint_pos
        log_pos[i]    = pos_ctrl
        log_vel[i]    = vel_ctrl
        log_rpy[i]    = rpy_deg
        log_rpms[i]   = rpms
        log_thrust[i] = ctrl.actuator_thrust

        # --- Print status periodically ---
        if i % (CTRL_FREQ * 1) == 0:
            t = i / CTRL_FREQ
            print(f"  t={t:5.1f}s  pos=[{pos_ctrl[0]:+.3f}, {pos_ctrl[1]:+.3f}, {pos_ctrl[2]:.3f}]  "
                  f"sp=[{sp[0]:+.3f}, {sp[1]:+.3f}, {sp[2]:.3f}]  "
                  f"thrust={ctrl.actuator_thrust:.0f}  "
                  f"rpms=[{rpms[0]:.0f},{rpms[1]:.0f},{rpms[2]:.0f},{rpms[3]:.0f}]")

        # --- Render & sync ---
        #env.render()
        if gui:
            sync(i, START, 1.0 / CTRL_FREQ)

    env.close()
    print("[SIM] Done.")

    sim_data = dict(t=log_t, sp=log_sp, pos=log_pos, vel=log_vel,
                    rpy=log_rpy, rpms=log_rpms, thrust=log_thrust)
    if plot:
        _plot_sim(log_t, log_sp, log_pos, log_vel, log_rpy, log_rpms, log_thrust)
    return sim_data


def _plot_sim(t, sp, pos, vel, rpy, rpms, thrust):
    """Plot simulation results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not found — skipping plots.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle('Crazyflie firmware PID — simulation results', fontsize=14)

    # -- Position tracking --
    ax = axes[0, 0]
    ax.plot(t, sp[:, 0], '--', label='sp_x', alpha=0.7)
    ax.plot(t, sp[:, 1], '--', label='sp_y', alpha=0.7)
    ax.plot(t, sp[:, 2], '--', label='sp_z', alpha=0.7)
    ax.plot(t, pos[:, 0], label='x')
    ax.plot(t, pos[:, 1], label='y')
    ax.plot(t, pos[:, 2], label='z')
    ax.set_ylabel('Position [m]')
    ax.legend(fontsize=7, ncol=3)
    ax.set_title('Position tracking')
    ax.grid(True, alpha=0.3)

    # -- Velocity --
    ax = axes[0, 1]
    ax.plot(t, vel[:, 0], label='vx')
    ax.plot(t, vel[:, 1], label='vy')
    ax.plot(t, vel[:, 2], label='vz')
    ax.set_ylabel('Velocity [m/s]')
    ax.legend(fontsize=7)
    ax.set_title('Velocity')
    ax.grid(True, alpha=0.3)

    # -- Attitude --
    ax = axes[1, 0]
    ax.plot(t, rpy[:, 0], label='roll')
    ax.plot(t, rpy[:, 1], label='pitch')
    ax.set_ylabel('Angle [deg]')
    ax.legend(fontsize=7)
    ax.set_title('Roll / Pitch')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, rpy[:, 2], label='yaw', color='green')
    ax.set_ylabel('Angle [deg]')
    ax.legend(fontsize=7)
    ax.set_title('Yaw')
    ax.grid(True, alpha=0.3)

    # -- Thrust --
    ax = axes[2, 0]
    ax.plot(t, thrust, label='thrust', color='black')
    ax.axhline(PID_VEL_THRUST_BASE, ls=':', color='gray',
               label=f'THRUST_BASE={PID_VEL_THRUST_BASE:.0f}')
    ax.set_ylabel('Thrust [uint16]')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7)
    ax.set_title('Thrust command')
    ax.grid(True, alpha=0.3)

    # -- RPMs --
    ax = axes[2, 1]
    ax.plot(t, rpms[:, 0], label='M1 (FR)', alpha=0.7)
    ax.plot(t, rpms[:, 1], label='M2 (BR)', alpha=0.7)
    ax.plot(t, rpms[:, 2], label='M3 (BL)', alpha=0.7)
    ax.plot(t, rpms[:, 3], label='M4 (FL)', alpha=0.7)
    ax.set_ylabel('RPM')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Motor RPMs')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sim_results.png', dpi=150)
    print("[PLOT] Saved to sim_results.png")
    plt.show()


def _plot_comparison(real, sim):
    """3×3 grid comparing simulation vs real drone: position (with setpoints),
    velocity, and attitude (roll/pitch/yaw).

    Parameters
    ----------
    real : dict with keys t, pos, vel, rpy  (Nx3 arrays, time in seconds)
                          sp_t, sp           (setpoint times and positions)
    sim  : dict returned by run_sim(plot=False) — keys t, pos, vel, rpy, sp
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not found — skipping comparison plot.")
        return

    fig, axes = plt.subplots(3, 3, figsize=(16, 10))
    fig.suptitle('Simulation vs Real Drone — State Comparison', fontsize=14)

    pos_labels = ['x [m]',  'y [m]',  'z [m]']
    vel_labels = ['vx [m/s]', 'vy [m/s]', 'vz [m/s]']
    att_labels = ['roll [deg]', 'pitch [deg]', 'yaw [deg]']

    for col in range(3):
        # ── Row 0 : position ──────────────────────────────────────────
        ax = axes[0, col]
        ax.plot(sim['t'],      sim['pos'][:, col],  color='steelblue', label='sim',      lw=1.5)
        ax.plot(real['t'],     real['pos'][:, col], color='tomato',    label='real',     lw=1.5, alpha=0.85)
        if len(real['sp_t']) > 0:
            ax.step(real['sp_t'], real['sp'][:, col],  color='gray',      label='setpoint', lw=1.0,
                    where='post', linestyle='--', alpha=0.7)
        ax.set_ylabel(pos_labels[col])
        ax.set_title(f'Position {["x","y","z"][col]}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ── Row 1 : velocity ──────────────────────────────────────────
        ax = axes[1, col]
        ax.plot(sim['t'],  sim['vel'][:, col],  color='steelblue', label='sim',  lw=1.5)
        ax.plot(real['t'], real['vel'][:, col], color='tomato',    label='real', lw=1.5, alpha=0.85)
        ax.set_ylabel(vel_labels[col])
        ax.set_title(f'Velocity {["vx","vy","vz"][col]}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ── Row 2 : attitude ──────────────────────────────────────────
        ax = axes[2, col]
        ax.plot(sim['t'],  sim['rpy'][:, col],  color='steelblue', label='sim',  lw=1.5)
        ax.plot(real['t'], real['rpy'][:, col], color='tomato',    label='real', lw=1.5, alpha=0.85)
        ax.set_ylabel(att_labels[col])
        ax.set_xlabel('Time [s]')
        ax.set_title(['Roll', 'Pitch', 'Yaw'][col])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150)
    print("[PLOT] Saved to comparison_results.png")
    plt.show()


# ===================================================================
#  Push Python PID constants → real Crazyflie firmware via cflib
# ===================================================================

def push_pid_gains_to_drone(cf):
    """Write the Python PID constants to the drone's onboard firmware.

    All gains defined at the top of this file are sent via cf.param.set_value()
    so that the real drone uses the exact same tuning as the simulation.
    Changes are volatile: they reset to firmware defaults on power-cycle.

    Firmware parameter groups (from platform_defaults_cf2.h):
      posCtlPid   — position PID  (x/y/z)
      velCtlPid   — velocity PID  (vx/vy/vz)
      pid_attitude — attitude PID (roll/pitch/yaw)
      pid_rate    — angular rate PID (roll/pitch/yaw)
    """
    gains = {
        # ── Position PID ────────────────────────────────────────────
        'posCtlPid.xKp': PID_POS_X_KP,  'posCtlPid.xKi': PID_POS_X_KI,  'posCtlPid.xKd': PID_POS_X_KD,
        'posCtlPid.yKp': PID_POS_Y_KP,  'posCtlPid.yKi': PID_POS_Y_KI,  'posCtlPid.yKd': PID_POS_Y_KD,
        'posCtlPid.zKp': PID_POS_Z_KP,  'posCtlPid.zKi': PID_POS_Z_KI,  'posCtlPid.zKd': PID_POS_Z_KD,
        # ── Velocity PID ────────────────────────────────────────────
        'velCtlPid.vxKp': PID_VEL_X_KP,  'velCtlPid.vxKi': PID_VEL_X_KI,  'velCtlPid.vxKd': PID_VEL_X_KD,
        'velCtlPid.vyKp': PID_VEL_Y_KP,  'velCtlPid.vyKi': PID_VEL_Y_KI,  'velCtlPid.vyKd': PID_VEL_Y_KD,
        'velCtlPid.vzKp': PID_VEL_Z_KP,  'velCtlPid.vzKi': PID_VEL_Z_KI,  'velCtlPid.vzKd': PID_VEL_Z_KD,
        # ── Attitude PID ────────────────────────────────────────────
        'pid_attitude.roll_kp':  PID_ROLL_KP,   'pid_attitude.roll_ki':  PID_ROLL_KI,   'pid_attitude.roll_kd':  PID_ROLL_KD,
        'pid_attitude.pitch_kp': PID_PITCH_KP,  'pid_attitude.pitch_ki': PID_PITCH_KI,  'pid_attitude.pitch_kd': PID_PITCH_KD,
        'pid_attitude.yaw_kp':   PID_YAW_KP,    'pid_attitude.yaw_ki':   PID_YAW_KI,    'pid_attitude.yaw_kd':   PID_YAW_KD,
        # ── Rate PID ────────────────────────────────────────────────
        'pid_rate.roll_kp':  PID_ROLL_RATE_KP,   'pid_rate.roll_ki':  PID_ROLL_RATE_KI,   'pid_rate.roll_kd':  PID_ROLL_RATE_KD,
        'pid_rate.pitch_kp': PID_PITCH_RATE_KP,  'pid_rate.pitch_ki': PID_PITCH_RATE_KI,  'pid_rate.pitch_kd': PID_PITCH_RATE_KD,
        'pid_rate.yaw_kp':   PID_YAW_RATE_KP,    'pid_rate.yaw_ki':   PID_YAW_RATE_KI,    'pid_rate.yaw_kd':   PID_YAW_RATE_KD,
    }

    print("[GAINS] Pushing PID gains to drone...")
    for param, value in gains.items():
        cf.param.set_value(param, str(value))
    print(f"[GAINS] Done — {len(gains)} parameters written.")


# ===================================================================
#  MAIN — Real drone modes
# ===================================================================

def run_real(mode, uri="radio://0/80/2M/E7E7E7E7E7",
             duration_sec=15, hover_height=0.5, radius=0.5,
             push_gains=False):
    """Run with a real Crazyflie drone.

    Parameters
    ----------
    mode : str
        'attitude' — PC runs pos+vel PIDs, sends attitude+thrust to drone
        'rate'     — PC runs pos+vel+att PIDs, sends rate+thrust to drone
        'position' — Sends position setpoint, drone runs all PIDs
    uri : str
        Crazyflie radio URI
    push_gains : bool
        If True, overwrite the drone's onboard PID gains with the Python
        constants defined at the top of this file before flying.
    """
    if not HAS_CFLIB:
        print("ERROR: cflib not found. pip install cflib")
        sys.exit(1)

    cflib.crtp.init_drivers()

    # Control rate for the PC-side loop
    if mode == "position":
        CTRL_FREQ = 10     # position setpoints don't need high rate
    else:
        CTRL_FREQ = 100    # attitude / rate commands: 100 Hz is typical

    waypoints = generate_trajectory(CTRL_FREQ, duration_sec,
                                    hover_height=hover_height, radius=radius)

    # We only need the position controller for 'attitude' and 'rate' modes
    pos_ctrl = CrazyfliePositionController() if mode != "position" else None
    att_ctrl = CrazyflieAttitudeController() if mode == "rate"    else None

    print(f"[REAL] Mode: {mode}")
    print(f"[REAL] Connecting to {uri} ...")

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        print("[REAL] Connected!")

        if push_gains:
            push_pid_gains_to_drone(cf)

        # For rate mode, disable the onboard stabilizer's attitude PID
        # so send_setpoint sends rate commands directly.
        # [FW] The firmware checks flightmode.stabModeRoll/Pitch/Yaw params.
        if mode == "rate":
            cf.param.set_value('flightmode.stabModeRoll',  '0')
            cf.param.set_value('flightmode.stabModeRoll',  '0')
            cf.param.set_value('flightmode.stabModePitch', '0')
            cf.param.set_value('flightmode.stabModeYaw',   '0')
            print("[REAL] Rate mode: disabled onboard attitude stabilization")

        # Set up logging to read back state (position + velocity + attitude)
        # This requires a positioning system (Lighthouse / Loco / MoCap)
        from cflib.crazyflie.log import LogConfig
        log_state = LogConfig(name='State', period_in_ms=10)  # 100 Hz
        log_state.add_variable('stateEstimate.x',  'float')
        log_state.add_variable('stateEstimate.y',  'float')
        log_state.add_variable('stateEstimate.z',  'float')
        log_state.add_variable('stateEstimate.vx', 'float')
        log_state.add_variable('stateEstimate.vy', 'float')
        log_state.add_variable('stateEstimate.vz', 'float')

        log_att = LogConfig(name='Attitude', period_in_ms=10)
        log_att.add_variable('stabilizer.roll',  'float')
        log_att.add_variable('stabilizer.pitch', 'float')
        log_att.add_variable('stabilizer.yaw',   'float')
        # Gyro for rate mode
        if mode == "rate":
            log_att.add_variable('gyro.x', 'float')
            log_att.add_variable('gyro.y', 'float')
            log_att.add_variable('gyro.z', 'float')

        # Shared state dict updated by log callbacks
        drone_state = {
            'x': 0, 'y': 0, 'z': 0, 'vx': 0, 'vy': 0, 'vz': 0,
            'roll': 0, 'pitch': 0, 'yaw': 0,
            'gyro_x': 0, 'gyro_y': 0, 'gyro_z': 0
        }

        # Timestamped data logs — filled during flight for comparison plot
        real_log    = []   # one entry per _state_cb callback (~100 Hz)
        real_sp_log = []   # one entry per setpoint command sent
        flight_start = None  # set just before the main flight loop

        def _state_cb(timestamp, data, logconf):
            drone_state['x']  = data['stateEstimate.x']
            drone_state['y']  = data['stateEstimate.y']
            drone_state['z']  = data['stateEstimate.z']
            drone_state['vx'] = data['stateEstimate.vx']
            drone_state['vy'] = data['stateEstimate.vy']
            drone_state['vz'] = data['stateEstimate.vz']
            # Record full state snapshot (attitude values come from last _att_cb)
            if flight_start is not None:
                real_log.append({
                    't':     time.time() - flight_start,
                    'x':     drone_state['x'],   'y':   drone_state['y'],   'z':   drone_state['z'],
                    'vx':    drone_state['vx'],  'vy':  drone_state['vy'],  'vz':  drone_state['vz'],
                    'roll':  drone_state['roll'], 'pitch': drone_state['pitch'], 'yaw': drone_state['yaw'],
                })

        def _att_cb(timestamp, data, logconf):
            drone_state['roll']  = data['stabilizer.roll']
            drone_state['pitch'] = data['stabilizer.pitch']
            drone_state['yaw']   = data['stabilizer.yaw']
            if mode == "rate":
                drone_state['gyro_x'] = data['gyro.x']
                drone_state['gyro_y'] = data['gyro.y']
                drone_state['gyro_z'] = data['gyro.z']

        log_state.data_received_cb.add_callback(_state_cb)
        log_att.data_received_cb.add_callback(_att_cb)
        cf.log.add_config(log_state)
        cf.log.add_config(log_att)
        log_state.start()
        log_att.start()

        # ── Reset Kalman estimator ────────────────────────────────────
        # Without this, the drone's position estimate keeps the value from
        # the previous flight (or wherever the estimator drifted to).
        # All trajectory setpoints are in the estimator frame, so if the
        # estimator thinks the drone is at (1, 0, 0) the first setpoint
        # (0, 0, z) will make it fly sideways — the drift you observe.
        # After reset the estimator restarts from (0, 0, 0), which becomes
        # the origin for the whole trajectory.
        print("[REAL] Resetting Kalman estimator...")
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)   # let filter converge and receive first log packets
        print(f"[REAL] Estimator ready — pos=({drone_state['x']:.3f}, "
              f"{drone_state['y']:.3f}, {drone_state['z']:.3f})")

        # ── Unlock commander watchdog ─────────────────────────────────
        # The firmware requires receiving setpoints before it accepts real
        # commands (and after each re-arm following a landing).
        for _ in range(50):
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.02)
        print("[REAL] Commander unlocked. Flying trajectory...")

        START = time.time()
        flight_start = START   # enable data logging in _state_cb
        try:
            for i in range(len(waypoints)):
                sp = waypoints[i]
                sp_pos = sp[0:3]
                sp_yaw_rate = sp[3]

                # Log setpoint for every control step (all modes)
                real_sp_log.append({'t': time.time() - flight_start, 'pos': sp_pos.copy()})

                if mode == "position":
                    # ─── Mode: position ───────────────────────────
                    # All PIDs run on the drone.
                    # send_position_setpoint(x, y, z, yaw)
                    cf.commander.send_position_setpoint(
                        sp_pos[0], sp_pos[1], sp_pos[2], 0.0)

                elif mode == "attitude":
                    # ─── Mode: attitude ───────────────────────────
                    # PC runs pos+vel PIDs → sends attitude + thrust
                    # Drone runs attitude + rate PIDs.
                    #
                    # send_setpoint(roll, pitch, yawrate, thrust)
                    #   roll, pitch: degrees
                    #   yawrate: deg/s
                    #   thrust: uint16 [0..65535]
                    state_pos = [drone_state['x'], drone_state['y'], drone_state['z']]
                    state_vel = [drone_state['vx'], drone_state['vy'], drone_state['vz']]
                    yaw_deg   = drone_state['yaw']

                    thrust, roll_d, pitch_d = pos_ctrl.update(
                        sp_pos, state_pos, state_vel, yaw_deg)

                    cf.commander.send_setpoint(
                        roll_d, pitch_d, sp_yaw_rate, int(thrust))

                elif mode == "rate":
                    # ─── Mode: rate ───────────────────────────────
                    # PC runs pos+vel+attitude PIDs → sends rate + thrust
                    # Drone runs only rate PID.
                    #
                    # With stabMode disabled, send_setpoint interprets
                    # roll/pitch as rate commands (deg/s).
                    state_pos = [drone_state['x'], drone_state['y'], drone_state['z']]
                    state_vel = [drone_state['vx'], drone_state['vy'], drone_state['vz']]
                    rpy_deg   = [drone_state['roll'], drone_state['pitch'], drone_state['yaw']]

                    thrust, roll_d, pitch_d = pos_ctrl.update(
                        sp_pos, state_pos, state_vel, rpy_deg[2])

                    roll_rate_d, pitch_rate_d, yaw_rate_d = att_ctrl.correct_attitude(
                        rpy_deg[0], rpy_deg[1], rpy_deg[2],
                        roll_d, pitch_d, 0.0)  # yaw desired = 0

                    cf.commander.send_setpoint(
                        roll_rate_d, pitch_rate_d, yaw_rate_d, int(thrust))

                # --- Timing ---
                if i % (CTRL_FREQ * 1) == 0:
                    t = i / CTRL_FREQ
                    print(f"  t={t:5.1f}s  pos=[{drone_state['x']:+.3f}, "
                          f"{drone_state['y']:+.3f}, {drone_state['z']:.3f}]  "
                          f"sp=[{sp_pos[0]:+.3f}, {sp_pos[1]:+.3f}, {sp_pos[2]:.3f}]")

                sync(i, START, 1.0 / CTRL_FREQ)

        except KeyboardInterrupt:
            print("\n[REAL] Interrupted! Soft landing...")
        finally:
            # ─── Smooth landing ───────────────────────────────
            # Descend from current position to ~5 cm, then cut motors.
            # Uses position mode for simplicity and safety (the onboard
            # PIDs handle attitude even if we were in rate mode before).
            #
            # First, restore normal stabilization so send_position_setpoint works.
            if mode == "rate":
                cf.param.set_value('flightmode.stabModeRoll',  '1')
                cf.param.set_value('flightmode.stabModePitch', '1')
                cf.param.set_value('flightmode.stabModeYaw',   '1')
                time.sleep(0.05)

            land_x = drone_state['x']
            land_y = drone_state['y']
            land_z = drone_state['z']
            land_duration = max(1.0, land_z / 0.3)  # descend at ~0.3 m/s
            land_freq = 20  # Hz — position setpoints don't need high rate
            land_steps = int(land_freq * land_duration)
            cutoff_z = 0.05  # m — cut motors below this height

            print(f"[REAL] Landing from z={land_z:.2f}m over {land_duration:.1f}s...")

            land_start = time.time()
            for j in range(land_steps):
                frac = (j + 1) / land_steps
                # Smooth cubic descent
                z = land_z * (1.0 - (3 * frac**2 - 2 * frac**3))
                if z < cutoff_z:
                    break
                real_sp_log.append({'t': time.time() - flight_start,
                                    'pos': np.array([land_x, land_y, z])})
                cf.commander.send_position_setpoint(land_x, land_y, z, 0.0)
                time.sleep(1.0 / land_freq)

            # Notify end of setpoints — this tells the firmware the PC-side
            # commander is stopping cleanly, rather than vanishing (watchdog
            # timeout).  The firmware then idles the motors gracefully without
            # entering a hard-locked state that would require a power cycle.
            cf.commander.send_notify_setpoint_stop()
            time.sleep(0.1)

            log_state.stop()
            log_att.stop()

            # Restore normal flight mode (if not already done above)
            if mode == "rate":
                # Already restored above before landing
                pass

            print("[REAL] Landed and cleaned up.")

        # ─── Post-flight: run headless sim + comparison plot ──────────────
        if real_log:
            real_t   = np.array([e['t']   for e in real_log])
            real_pos = np.array([[e['x'],  e['y'],  e['z']]       for e in real_log])
            real_vel = np.array([[e['vx'], e['vy'], e['vz']]      for e in real_log])
            real_rpy = np.array([[e['roll'], e['pitch'], e['yaw']] for e in real_log])
            sp_t_arr = np.array([e['t']   for e in real_sp_log])
            sp_arr   = np.array([e['pos'] for e in real_sp_log])
            real_data = dict(t=real_t, pos=real_pos, vel=real_vel, rpy=real_rpy,
                             sp_t=sp_t_arr, sp=sp_arr)

            print("[REAL] Running headless simulation (with Flow deck model) for comparison...")
            if HAS_PYBULLET_DRONES:
                sim_data = run_sim(duration_sec=duration_sec, gui=False,
                                   hover_height=hover_height, radius=radius, plot=False,
                                   simulate_flow_deck=False)
                _plot_comparison(real_data, sim_data)
            else:
                print("[REAL] pybullet-drones not found — skipping comparison plot.")


# ===================================================================
#  CLI entry point
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Crazyflie firmware-faithful PID — sim & real drone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  sim        Full firmware PID in Python → RPMs → PyBullet  (default)
  attitude   Pos+vel PIDs on PC → send_setpoint(att, thrust) to drone
  rate       Pos+vel+att PIDs on PC → send_setpoint(rate, thrust)
  position   Only trajectory → send_position_setpoint (all PIDs on drone)
        """)
    parser.add_argument('--mode', default='sim',
                        choices=['sim', 'attitude', 'rate', 'position'],
                        help='Control mode (default: sim)')
    parser.add_argument('--duration', default=15, type=float,
                        help='Flight duration in seconds (default: 15)')
    parser.add_argument('--height', default=1.0, type=float,
                        help='Hover height in meters (default: 1.0)')
    parser.add_argument('--radius', default=0.5, type=float,
                        help='Circle radius in meters (default: 0.5)')
    parser.add_argument('--gui', default=True, type=lambda x: x.lower() == 'true',
                        help='PyBullet GUI (default: True)')
    parser.add_argument('--uri', default='radio://0/80/2M/E7E7E7E7E7',
                        help='Crazyflie radio URI (for real modes)')
    parser.add_argument('--flow-deck-sim', action='store_true',
                        help='Simulate Flow deck v2 noise+delay in sim mode')
    parser.add_argument('--push-gains', action='store_true',
                        help='Overwrite drone PID gains with Python constants before flying')
    args = parser.parse_args()

    if args.mode == 'sim':
        run_sim(duration_sec=args.duration, gui=args.gui,
                hover_height=args.height, radius=args.radius,
                simulate_flow_deck=args.flow_deck_sim)
    else:
        run_real(mode=args.mode, uri=args.uri,
                 duration_sec=args.duration, hover_height=args.height,
                 radius=args.radius, push_gains=args.push_gains)


if __name__ == "__main__":
    main()