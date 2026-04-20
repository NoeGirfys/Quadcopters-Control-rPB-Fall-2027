"""
Crazyflie Firmware Constants
============================

All PID gains and constants from the official Crazyflie firmware:
  platform_defaults_cf2.h   (Crazyflie 2.1+ default gains)
  platform_defaults.h       (filter / rate defaults)
  src/utils/interface/pid.h (PID generic defaults)
"""

# ===================================================================
#  RATES
# ===================================================================
# [FW] src/modules/interface/stabilizer_types.h:364-372
RATE_MAIN_LOOP = 1000   # Hz
ATTITUDE_RATE  = 500    # Hz
POSITION_RATE  = 100    # Hz
ATTITUDE_UPDATE_DT = 1.0 / ATTITUDE_RATE   # 0.002 s
POSITION_UPDATE_DT = 1.0 / POSITION_RATE   # 0.01  s

# ===================================================================
#  PHYSICAL CONSTANTS
# ===================================================================
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
PID_ROLL_KFF = 5.0;  PID_ROLL_INTEGRATION_LIMIT  = 20.0

PID_PITCH_KP = 6.0;  PID_PITCH_KI = 3.0;  PID_PITCH_KD = 0.0
PID_PITCH_KFF= 5.0;  PID_PITCH_INTEGRATION_LIMIT = 20.0

PID_YAW_KP   = 6.0;  PID_YAW_KI   = 1.0;  PID_YAW_KD   = 0.35
PID_YAW_KFF  = 0.0;  PID_YAW_INTEGRATION_LIMIT   = 360.0

# ---- Velocity PID gains  -----------------------------------------
# [FW] platform_defaults_cf2.h:132-145
PID_VEL_X_KP = 25.0;  PID_VEL_X_KI = 1.0;  PID_VEL_X_KD = 0.0;  PID_VEL_X_KFF = 10.0
PID_VEL_Y_KP = 25.0;  PID_VEL_Y_KI = 1.0;  PID_VEL_Y_KD = 0.0;  PID_VEL_Y_KFF = 10.0
PID_VEL_Z_KP = 25.0;  PID_VEL_Z_KI = 15.0; PID_VEL_Z_KD = 0.0;  PID_VEL_Z_KFF = 0.0

# ---- Position PID gains  -----------------------------------------
# [FW] platform_defaults_cf2.h:158-176
PID_POS_X_KP = 2.0;  PID_POS_X_KI = 0.5;  PID_POS_X_KD = 0.0;  PID_POS_X_KFF = 0.0
PID_POS_Y_KP = 2.0;  PID_POS_Y_KI = 0.5;  PID_POS_Y_KD = 0.0;  PID_POS_Y_KFF = 0.0
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
