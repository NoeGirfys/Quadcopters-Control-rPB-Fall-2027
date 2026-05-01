"""Physical constants and timing for the CF2X drone (mass-baseline values).

Identical to ``training_regulation_simu_and_real/train_nn_cf_pid.py``.
The MAML training uses these as the *baseline* drone parameters; tasks
modify them by attaching an offset mass.
"""
import math
import os
import sys

import numpy as np
import torch

# Make crazyflie_firmware importable.
_PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT_DIR not in sys.path:
    sys.path.append(_PARENT_DIR)

from crazyflie_firmware.constants import (  # noqa: E402
    UINT16_MAX,
    PID_ROLL_KP, PID_ROLL_KI, PID_ROLL_KD, PID_ROLL_INTEGRATION_LIMIT,
    PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD, PID_PITCH_INTEGRATION_LIMIT,
    PID_YAW_KP, PID_YAW_KI, PID_YAW_KD, PID_YAW_INTEGRATION_LIMIT,
    PID_ROLL_RATE_KP, PID_ROLL_RATE_KI, PID_ROLL_RATE_KD,
    PID_ROLL_RATE_INTEGRATION_LIMIT,
    PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD,
    PID_PITCH_RATE_INTEGRATION_LIMIT,
    PID_YAW_RATE_KP, PID_YAW_RATE_KI, PID_YAW_RATE_KD,
    PID_YAW_RATE_INTEGRATION_LIMIT,
    PID_VEL_ROLL_MAX, PID_VEL_PITCH_MAX,
    POSITION_RATE, ATTITUDE_RATE, ATTITUDE_UPDATE_DT,
)

# ---- Physical (CF2X URDF, baseline drone) ----
M_BASE = 0.027        # baseline mass [kg]
G      = 9.8          # gravity [m/s^2]
I_X    = 1.4e-5       # Ixx [kg*m^2]
I_Y    = 1.4e-5
I_Z    = 2.17e-5
L      = 0.0397       # arm length [m]
KF     = 3.16e-10     # thrust coeff [N/RPM^2]
KM     = 7.94e-12     # torque coeff [N*m/RPM^2]
T2W    = 2.25

CF2_THRUST_MAX_PER_MOTOR = 0.12   # N

I_DRONE = np.diag([I_X, I_Y, I_Z]).astype(np.float32)

GRAVITY_BASE   = M_BASE * G
HOVER_RPM_BASE = math.sqrt(GRAVITY_BASE / (4 * KF))
MAX_RPM        = math.sqrt((T2W * GRAVITY_BASE) / (4 * KF))

# Motor positions (CF2X X-shape), body frame, geometric center origin.
# Verified against the mixing in train_nn_cf_pid.py:
#   tau_x = -(F1+F2-F3-F4)*a   tau_y = (-F1+F2+F3-F4)*a
# with p × F = (py*Fz, -px*Fz, 0).
_a = L / math.sqrt(2)
MOTOR_POS = np.array([
    [+_a, -_a, 0.0],   # M1 front-right
    [-_a, -_a, 0.0],   # M2 rear-right
    [-_a, +_a, 0.0],   # M3 rear-left
    [+_a, +_a, 0.0],   # M4 front-left
], dtype=np.float32)

# Yaw torque sign per motor (CF2X mixing in train_nn_cf_pid):
# tau_z = -M_yaw_1 + M_yaw_2 - M_yaw_3 + M_yaw_4
YAW_SIGNS = np.array([-1.0, +1.0, -1.0, +1.0], dtype=np.float32)

# ---- PID gains ----
ATT_KP   = torch.tensor([PID_ROLL_KP, PID_PITCH_KP, PID_YAW_KP])
ATT_KI   = torch.tensor([PID_ROLL_KI, PID_PITCH_KI, PID_YAW_KI])
ATT_KD   = torch.tensor([PID_ROLL_KD, PID_PITCH_KD, PID_YAW_KD])
ATT_ILIM = torch.tensor([PID_ROLL_INTEGRATION_LIMIT, PID_PITCH_INTEGRATION_LIMIT,
                         PID_YAW_INTEGRATION_LIMIT])

RATE_KP   = torch.tensor([PID_ROLL_RATE_KP, PID_PITCH_RATE_KP, PID_YAW_RATE_KP])
RATE_KI   = torch.tensor([PID_ROLL_RATE_KI, PID_PITCH_RATE_KI, PID_YAW_RATE_KI])
RATE_KD   = torch.tensor([PID_ROLL_RATE_KD, PID_PITCH_RATE_KD, PID_YAW_RATE_KD])
RATE_ILIM = torch.tensor([PID_ROLL_RATE_INTEGRATION_LIMIT,
                          PID_PITCH_RATE_INTEGRATION_LIMIT,
                          PID_YAW_RATE_INTEGRATION_LIMIT])

YAW_RATE_MAX = 200.0
NN_FREQ          = POSITION_RATE
PID_STEPS_PER_NN = ATTITUDE_RATE // NN_FREQ
MOTOR_TAU        = 0.02
MOTOR_ALPHA      = ATTITUDE_UPDATE_DT / (MOTOR_TAU + ATTITUDE_UPDATE_DT)

X_SCALE = np.array([
    1.0, 1.0,
    1.0, 1.0,
    1.0, 1.0,
    np.deg2rad(30), np.deg2rad(200),
    np.deg2rad(30), np.deg2rad(200),
    np.deg2rad(45), np.deg2rad(120),
], dtype=np.float32)
POS_SCALE = 1.0

OBS_NOISE_STD = torch.tensor([
    0.01, 0.05,
    0.01, 0.05,
    0.01, 0.05,
    math.radians(2),  math.radians(10),
    math.radians(2),  math.radians(10),
    math.radians(2),  math.radians(10),
], dtype=torch.float32)

Q_DIAG = np.array([
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1 / np.deg2rad(30) ** 2, 1 / np.deg2rad(200) ** 2,
    1 / np.deg2rad(30) ** 2, 1 / np.deg2rad(200) ** 2,
    1 / np.deg2rad(45) ** 2, 1 / np.deg2rad(120) ** 2,
], dtype=np.float32)

HOVER_THRUST_U16_BASE = (
    GRAVITY_BASE * UINT16_MAX / (4 * CF2_THRUST_MAX_PER_MOTOR)
)
