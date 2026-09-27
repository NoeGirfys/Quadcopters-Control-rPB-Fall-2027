import numpy as np


# Quadcopter model
M = 1.0                             # mass in kg
G = 9.81                            # gravity in m/s^2
I_X = 0.11                          # moment of inertia around x-axis in kg*m^2
I_Y = 0.11                          # moment of inertia around y-axis in kg*m^2
I_Z = 0.04                          # moment of inertia around z-axis in kg*m^2

F_MAX = 2 * M * G                   # max thrust in N
TAU_X_MAX = 0.08                    # max torque around x-axis in N*m
TAU_Y_MAX = 0.08                    # max torque around y-axis in N*m
TAU_Z_MAX = 0.05                    # max torque around z-axis in N*m
U_MAX = np.array([F_MAX, TAU_X_MAX, TAU_Y_MAX, TAU_Z_MAX], dtype=np.float32)
NB_INPUTS = len(U_MAX)

X_MAX = 5.0                         # max position in x in m
X_DMAX = 5.0                        # max velocity in x in m/s
Y_MAX = 5.0                         # max position in y in m
Y_DMAX = 5.0                        # max velocity in y in m/s
Z_MAX = 5.0                         # max position in z in m
Z_DMAX = 5.0                        # max velocity in z in m/s
PHI_MAX = np.deg2rad(45)            # max roll angle in rad 
PHI_DMAX = np.deg2rad(120)          # max roll velocity in rad/s (120 deg/s = full rotation in 3s)
THETA_MAX = np.deg2rad(45)          # max pitch angle in rad
THETA_DMAX = np.deg2rad(120)        # max pitch velocity in rad/s
PSI_MAX = np.deg2rad(45)            # max yaw angle in rad
PSI_DMAX = np.deg2rad(90)           # max yaw velocity in rad/s
X_SCALE = np.array([X_MAX, X_DMAX, Y_MAX, Y_DMAX, Z_MAX, Z_DMAX, PHI_MAX, PHI_DMAX, THETA_MAX, THETA_DMAX, PSI_MAX, PSI_DMAX], dtype=np.float32)
NB_STATES = len(X_SCALE)

Q_DIAG = np.array([8/X_MAX**2,   1/X_DMAX**2, 
                   4/Y_MAX**2,   1/Y_DMAX**2, 
                   4/Z_MAX**2,   1/Z_DMAX**2, 
                   0,            1/PHI_DMAX**2,
                   0,            1/THETA_DMAX**2,
                   4/PSI_MAX**2, 1/PSI_DMAX**2], dtype=np.float32)
Q = np.diag(Q_DIAG) # task specific cost for disturbance wind on x and y axis
Q_TERM = 0 # terminal cost weight for position states
R_DIAG = np.array([0, 0 ,0, 0], dtype=np.float32) # no cost on control inputs
R = np.diag(R_DIAG)


# Discretization and simulation
T_D = 0.1
T_SIM = 20.0
T_STEPS = int(T_SIM / T_D)

