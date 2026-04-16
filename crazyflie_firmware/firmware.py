"""
Crazyflie Firmware PID Classes
================================

Python translation of the official Crazyflie firmware PID pipeline:
  - Lpf2pData                  (src/utils/src/filter.c)
  - PidObject                  (src/utils/src/pid.c)
  - CrazyfliePositionController (src/modules/src/controller/position_controller_pid.c)
  - CrazyflieAttitudeController (src/modules/src/controller/attitude_pid_controller.c)
  - CrazyfliePowerDistribution  (src/modules/src/power_distribution_quadrotor.c)
  - CrazyflieFirmwarePID        (src/modules/src/controller/controller_pid.c)

Firmware source references are given as comments of the form:
  # [FW] path/to/file.c:LINE
"""

import math

from crazyflie_firmware.constants import (
    ATTITUDE_RATE, POSITION_RATE,
    ATTITUDE_UPDATE_DT, POSITION_UPDATE_DT,
    UINT16_MAX,
    PID_ROLL_RATE_KP, PID_ROLL_RATE_KI, PID_ROLL_RATE_KD,
    PID_ROLL_RATE_KFF, PID_ROLL_RATE_INTEGRATION_LIMIT,
    PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD,
    PID_PITCH_RATE_KFF, PID_PITCH_RATE_INTEGRATION_LIMIT,
    PID_YAW_RATE_KP, PID_YAW_RATE_KI, PID_YAW_RATE_KD,
    PID_YAW_RATE_KFF, PID_YAW_RATE_INTEGRATION_LIMIT,
    PID_ROLL_KP, PID_ROLL_KI, PID_ROLL_KD,
    PID_ROLL_KFF, PID_ROLL_INTEGRATION_LIMIT,
    PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD,
    PID_PITCH_KFF, PID_PITCH_INTEGRATION_LIMIT,
    PID_YAW_KP, PID_YAW_KI, PID_YAW_KD,
    PID_YAW_KFF, PID_YAW_INTEGRATION_LIMIT,
    PID_VEL_X_KP, PID_VEL_X_KI, PID_VEL_X_KD, PID_VEL_X_KFF,
    PID_VEL_Y_KP, PID_VEL_Y_KI, PID_VEL_Y_KD, PID_VEL_Y_KFF,
    PID_VEL_Z_KP, PID_VEL_Z_KI, PID_VEL_Z_KD, PID_VEL_Z_KFF,
    PID_POS_X_KP, PID_POS_X_KI, PID_POS_X_KD, PID_POS_X_KFF,
    PID_POS_Y_KP, PID_POS_Y_KI, PID_POS_Y_KD, PID_POS_Y_KFF,
    PID_POS_Z_KP, PID_POS_Z_KI, PID_POS_Z_KD, PID_POS_Z_KFF,
    PID_VEL_ROLL_MAX, PID_VEL_PITCH_MAX,
    PID_VEL_THRUST_BASE, PID_VEL_THRUST_MIN, THRUST_SCALE,
    PID_POS_VEL_X_MAX, PID_POS_VEL_Y_MAX, PID_POS_VEL_Z_MAX,
    VEL_MAX_OVERHEAD, RP_LIMIT_OVERHEAD,
    ATTITUDE_LPF_ENABLE, ATTITUDE_LPF_CUTOFF,
    ATTITUDE_RATE_LPF_ENABLE,
    ATTITUDE_ROLL_RATE_LPF_CUTOFF, ATTITUDE_PITCH_RATE_LPF_CUTOFF,
    ATTITUDE_YAW_RATE_LPF_CUTOFF,
    POS_XY_FILT_ENABLE, POS_XY_FILT_CUTOFF,
    POS_Z_FILT_ENABLE, POS_Z_FILT_CUTOFF,
    VEL_XY_FILT_ENABLE, VEL_XY_FILT_CUTOFF,
    VEL_Z_FILT_ENABLE, VEL_Z_FILT_CUTOFF,
    DEFAULT_PID_INTEGRATION_LIMIT, DEFAULT_PID_OUTPUT_LIMIT,
)


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
