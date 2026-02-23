"""
LQR controller for Crazyflie 2.x in gym-pybullet-drones.

Drop-in replacement for the pid.py example: same structure, same env,
but replaces DSLPIDControl with a discrete-time LQR designed from the
linearized hover model.

All physical parameters (m, L, J, kf, km, max_rpm, t2w, max_speed)
are read directly from the environment after init — no hardcoded values.

Usage:
    python lqr_from_pid.py
    python lqr_from_pid.py --num_drones 1 --gui True --T 15

Based on: gym_pybullet_drones/examples/pid.py
"""

import os
import time
import argparse
import math
import numpy as np
import scipy.linalg as la
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


# ══════════════════════════════════════════════════════════════════════
#  1.  LQR DESIGN  — reads params from the environment
# ══════════════════════════════════════════════════════════════════════

def build_linear_model(m, g, Ixx, Iyy, Izz):
    """
    12-state linearized hover model.
    State:  x = [X, Ẋ, Y, Ẏ, Z, Ż, φ, φ̇, θ, θ̇, ψ, ψ̇]
    Input:  u = [ΔF, τ_x, τ_y, τ_z]
    """
    A = np.zeros((12, 12))
    A[0,1] = 1;  A[2,3] = 1;  A[4,5] = 1
    A[6,7] = 1;  A[8,9] = 1;  A[10,11] = 1
    A[1,8] =  g     # Ẍ ← θ
    A[3,6] = -g     # Ÿ ← φ

    B = np.zeros((12, 4))
    B[5,0]  = 1.0/m
    B[7,1]  = 1.0/Ixx
    B[9,2]  = 1.0/Iyy
    B[11,3] = 1.0/Izz
    return A, B


def build_lqr_gain(env, ctrl_freq):
    """
    Build the LQR gain K from the environment's URDF parameters.
    
    Reads from env: M, L, J, KF, KM, MAX_RPM, THRUST2WEIGHT_RATIO,
                    MAX_SPEED_KMH, GRAVITY
    
    Returns: K (4×12), and derived constants needed for wrench→RPM.
    """
    # ── Physical parameters directly from the environment ──
    m   = env.M                     # mass [kg]
    g   = env.G                     # gravity [m/s²]
    Ixx = env.J[0, 0]              # moments of inertia [kg·m²]
    Iyy = env.J[1, 1]
    Izz = env.J[2, 2]
    L   = env.L                     # arm length [m]
    kf  = env.KF                    # thrust coefficient
    km  = env.KM                    # torque coefficient
    t2w = env.THRUST2WEIGHT_RATIO   # thrust-to-weight ratio
    max_rpm = env.MAX_RPM
    max_speed_kmh = env.MAX_SPEED_KMH

    print(f"\n{'='*60}")
    print(f"  Drone parameters read from environment (URDF)")
    print(f"{'='*60}")
    print(f"  Mass:        {m:.6f} kg")
    print(f"  Arm length:  {L:.6f} m")
    print(f"  Ixx:         {Ixx:.2e} kg·m²")
    print(f"  Iyy:         {Iyy:.2e} kg·m²")
    print(f"  Izz:         {Izz:.2e} kg·m²")
    print(f"  kf:          {kf:.2e}")
    print(f"  km:          {km:.2e}")
    print(f"  T/W ratio:   {t2w:.2f}")
    print(f"  Max RPM:     {max_rpm:.2f}")
    print(f"  Max speed:   {max_speed_kmh:.1f} km/h")

    # ── Derived limits (for Bryson's rule) ──
    # Max thrust per motor
    F_one_motor_max = kf * max_rpm**2
    F_total_max = 4 * F_one_motor_max           # all 4 at max
    F_hover = m * g
    delta_F_max = F_total_max - F_hover          # max upward thrust deviation

    # Max torques from the allocation matrix
    d = L / np.sqrt(2)  # effective moment arm for X-config
    tau_xy_max = 2 * d * F_one_motor_max         # roll/pitch: 2 motors at max diff
    tau_z_max  = 2 * km * max_rpm**2             # yaw: 2 motors CW vs 2 CCW

    # Max velocities
    max_speed_ms = max_speed_kmh / 3.6

    # Max angles: from t2w we can estimate max tilt while maintaining altitude
    # At max tilt angle α: F_total·cos(α) = mg → cos(α) = 1/t2w → α = acos(1/t2w)
    max_tilt = np.arccos(1.0 / t2w) if t2w > 1.0 else np.deg2rad(30)
    max_tilt = min(max_tilt, np.deg2rad(60))     # cap at 60° for linearization validity

    # Max angular rate: rough estimate from max torque / inertia × time_const
    max_ang_rate = tau_xy_max / min(Ixx, Iyy) * 0.1   # ~100ms to reach
    max_ang_rate = min(max_ang_rate, np.deg2rad(500))  # cap

    max_yaw = np.deg2rad(180)                     # yaw can go anywhere
    max_yaw_rate = tau_z_max / Izz * 0.1
    max_yaw_rate = min(max_yaw_rate, np.deg2rad(200))

    # Arena size as position limit
    pos_max = 2.0  # meters — reasonable for indoor Crazyflie

    print(f"\n  Derived limits for Bryson's rule:")
    print(f"  ΔF_max:      {delta_F_max*1e3:.1f} mN  (hover={F_hover*1e3:.1f} mN)")
    print(f"  τ_xy_max:    {tau_xy_max*1e6:.1f} µNm")
    print(f"  τ_z_max:     {tau_z_max*1e6:.1f} µNm")
    print(f"  Max tilt:    {np.rad2deg(max_tilt):.1f}°")
    print(f"  Max ω:       {np.rad2deg(max_ang_rate):.0f} °/s")
    print(f"  Max speed:   {max_speed_ms:.1f} m/s")

    # ── Bryson's Q and R ──
    q_diag = np.array([
        1/pos_max**2,         1/max_speed_ms**2,       # X, Ẋ
        1/pos_max**2,         1/max_speed_ms**2,       # Y, Ẏ
        1/pos_max**2,         1/max_speed_ms**2,       # Z, Ż
        1/max_tilt**2,        1/max_ang_rate**2,       # φ, φ̇
        1/max_tilt**2,        1/max_ang_rate**2,       # θ, θ̇
        1/max_yaw**2,         1/max_yaw_rate**2,       # ψ, ψ̇
    ])
    Q = np.diag(q_diag)

    r_diag = np.array([
        1/delta_F_max**2,
        1/tau_xy_max**2,
        1/tau_xy_max**2,
        1/tau_z_max**2,
    ])
    R = np.diag(r_diag)

    # ── Discretize + solve DARE ──
    Ts = 1.0 / ctrl_freq
    Ac, Bc = build_linear_model(m, g, Ixx, Iyy, Izz)
    sys_d = cont2discrete((Ac, Bc, np.eye(12), np.zeros((12, 4))), Ts, method='zoh')
    Ad, Bd = sys_d[0], sys_d[1]

    P = la.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)

    print(f"\n  LQR gain K max per row:")
    for i, name in enumerate(['ΔF', 'τx', 'τy', 'τz']):
        print(f"    {name}: {np.abs(K[i]).max():.6f}")

    # Pack drone constants needed for wrench→RPM
    drone_params = {
        'm': m, 'g': g, 'L': L, 'kf': kf, 'km': km,
        'max_rpm': max_rpm, 'F_hover': F_hover,
        'delta_F_max': delta_F_max,
        'tau_xy_max': tau_xy_max, 'tau_z_max': tau_z_max,
    }
    return K, drone_params


# ══════════════════════════════════════════════════════════════════════
#  2.  WRENCH → RPM ALLOCATION
# ══════════════════════════════════════════════════════════════════════

def build_allocation_inv(L, kf, km):
    """Inverse of [F, τx, τy, τz] = A @ [ω0², ω1², ω2², ω3²]"""
    d = L / np.sqrt(2)
    A = np.array([
        [ kf,      kf,      kf,      kf     ],
        [+d*kf,   -d*kf,   -d*kf,   +d*kf  ],
        [+d*kf,   +d*kf,   -d*kf,   -d*kf  ],
        [-km,     +km,     -km,     +km    ],
    ])
    return np.linalg.inv(A)


def wrench_to_rpm(F_total, tau_x, tau_y, tau_z, A_inv, max_rpm):
    """Convert total thrust + torques → 4 motor RPMs."""
    wrench = np.array([F_total, tau_x, tau_y, tau_z])
    rpm_sq = A_inv @ wrench
    rpm_sq = np.clip(rpm_sq, 0, max_rpm**2)
    return np.sqrt(rpm_sq)


# ══════════════════════════════════════════════════════════════════════
#  3.  STATE EXTRACTION + HELPERS
# ══════════════════════════════════════════════════════════════════════

def rotation_matrix_ZYX(phi, theta, psi):
    """R such that v_world = R @ v_body."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [  -st,             ct*sp,             ct*cp ],
    ])


def obs_to_state(obs):
    """
    Convert the 20-dim observation vector from CtrlAviary to our 12-state:
    obs = [x,y,z, q1,q2,q3,q4, r,p,y, vx,vy,vz, wx,wy,wz, p0,p1,p2,p3]
           0,1,2   3, 4, 5, 6  7,8,9  10,11,12  13,14,15  16,17,18,19

    Output: [X, Ẋ, Y, Ẏ, Z, Ż, φ, φ̇, θ, θ̇, ψ, ψ̇]

    NOTE: obs[13:16] angular velocity is in WORLD frame (from PyBullet).
    We rotate to body frame for consistency with the linearized model.
    """
    pos = obs[0:3]
    rpy = obs[7:10]
    vel = obs[10:13]
    ang_w = obs[13:16]  # world frame

    # Rotate angular velocity to body frame
    R = rotation_matrix_ZYX(rpy[0], rpy[1], rpy[2])
    ang_b = R.T @ ang_w

    return np.array([
        pos[0], vel[0],     # X, Ẋ
        pos[1], vel[1],     # Y, Ẏ
        pos[2], vel[2],     # Z, Ż
        rpy[0], ang_b[0],   # φ, φ̇
        rpy[1], ang_b[1],   # θ, θ̇
        rpy[2], ang_b[2],   # ψ, ψ̇
    ])


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_error(state, x_ref):
    err = state - x_ref
    err[6]  = wrap_angle(err[6])
    err[8]  = wrap_angle(err[8])
    err[10] = wrap_angle(err[10])
    return err


def make_target_state(target_pos, target_yaw=0.0):
    """12-state reference: hover at target_pos with given yaw."""
    xref = np.zeros(12)
    xref[0] = target_pos[0]
    xref[2] = target_pos[1]
    xref[4] = target_pos[2]
    xref[10] = target_yaw
    return xref


# ══════════════════════════════════════════════════════════════════════
#  4.  LQR CONTROL (replacement for DSLPIDControl.computeControlFromState)
# ══════════════════════════════════════════════════════════════════════

class LQRControl:
    """
    Drop-in replacement for DSLPIDControl.
    
    Computes RPMs from state observation + target position via LQR.
    Includes:
      - World→body angular velocity rotation
      - Angle wrapping
      - Anti-flip safety cutoff
      - Z-axis integral term for steady-state accuracy
    """
    def __init__(self, K, drone_params, ctrl_freq):
        self.K = K
        self.p = drone_params
        self.A_inv = build_allocation_inv(
            drone_params['L'], drone_params['kf'], drone_params['km']
        )
        self.Ts = 1.0 / ctrl_freq

        # Integral on Z error
        self.z_integral = 0.0
        self.KI_Z = 0.5

        # Safety angle limit
        self.ANGLE_SAFE = np.deg2rad(50)

    def reset(self):
        self.z_integral = 0.0

    def computeControlFromState(self, control_timestep, state, target_pos, target_rpy=np.zeros(3)):
        """
        Same signature as DSLPIDControl.computeControlFromState.
        
        Parameters
        ----------
        control_timestep : float (unused, we use self.Ts)
        state : ndarray (20,) — observation from CtrlAviary
        target_pos : ndarray (3,) — desired [x, y, z]
        target_rpy : ndarray (3,) — desired [roll, pitch, yaw]
        
        Returns
        -------
        rpm : ndarray (4,) — motor RPMs
        pos : ndarray (3,) — current position (for logging)
        yaw : float — current yaw (for logging)
        """
        # 1) Extract 12-state from 20-dim observation
        x = obs_to_state(state)

        # 2) Build reference
        x_ref = make_target_state(target_pos, target_yaw=target_rpy[2])

        # 3) Error
        error = compute_error(x, x_ref)

        # 4) Safety check
        if abs(x[6]) > self.ANGLE_SAFE or abs(x[8]) > self.ANGLE_SAFE:
            # Reduce thrust, zero torques — let it stabilize
            hover_rpm = np.sqrt(self.p['F_hover'] / (4 * self.p['kf']))
            rpm = np.full(4, hover_rpm * 0.6)
            self.z_integral = 0.0
            return rpm, x[np.array([0,2,4])], x[10]

        # 5) LQR
        u = -self.K @ error          # [ΔF, τx, τy, τz]

        # 6) Z integral
        z_err = error[4]
        self.z_integral += z_err * self.Ts
        self.z_integral = np.clip(self.z_integral, -0.5, 0.5)

        delta_F = u[0] - self.KI_Z * self.z_integral
        F_total = self.p['F_hover'] + delta_F
        F_total = np.clip(F_total, 0, 4 * self.p['kf'] * self.p['max_rpm']**2)

        # Torque clamp (use derived limits, with some margin)
        tau_x = np.clip(u[1], -self.p['tau_xy_max'] * 0.8, self.p['tau_xy_max'] * 0.8)
        tau_y = np.clip(u[2], -self.p['tau_xy_max'] * 0.8, self.p['tau_xy_max'] * 0.8)
        tau_z = np.clip(u[3], -self.p['tau_z_max']  * 0.8, self.p['tau_z_max']  * 0.8)

        # 7) Wrench → RPMs
        rpm = wrench_to_rpm(F_total, tau_x, tau_y, tau_z,
                            self.A_inv, self.p['max_rpm'])

        return rpm, x[np.array([0,2,4])], x[10]


# ══════════════════════════════════════════════════════════════════════
#  5.  MAIN — same structure as pid.py
# ══════════════════════════════════════════════════════════════════════

DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 3
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = True
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 12
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


def run(
    drone=DEFAULT_DRONES,
    num_drones=DEFAULT_NUM_DRONES,
    physics=DEFAULT_PHYSICS,
    gui=DEFAULT_GUI,
    record_video=DEFAULT_RECORD_VISION,
    plot=DEFAULT_PLOT,
    user_debug_gui=DEFAULT_USER_DEBUG_GUI,
    obstacles=DEFAULT_OBSTACLES,
    simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
    control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
    duration_sec=DEFAULT_DURATION_SEC,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    colab=DEFAULT_COLAB
):
    #### Initialize the simulation #############################
    H = .1
    H_STEP = .05
    R = .3
    INIT_XYZS = np.array([
        [R*np.cos((i/6)*2*np.pi+np.pi/2),
         R*np.sin((i/6)*2*np.pi+np.pi/2) - R,
         H + i*H_STEP]
        for i in range(num_drones)
    ])
    INIT_RPYS = np.array([
        [0, 0, i * (np.pi/2)/num_drones]
        for i in range(num_drones)
    ])

    #### Initialize a circular trajectory ######################
    PERIOD = 10
    NUM_WP = control_freq_hz * PERIOD
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = (
            R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2) + INIT_XYZS[0, 0],
            R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2) - R + INIT_XYZS[0, 1],
            0
        )
    wp_counters = np.array([
        int((i*NUM_WP/6) % NUM_WP) for i in range(num_drones)
    ])

    #### Create the environment ################################
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui
    )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab
    )

    #### Design LQR from env parameters ########################
    K, drone_params = build_lqr_gain(env, control_freq_hz)
    ctrl = [LQRControl(K, drone_params, control_freq_hz) for _ in range(num_drones)]

    print(f"\n{'='*60}")
    print(f"  Starting simulation: {num_drones} drones, {duration_sec}s")
    print(f"  Physics: {simulation_freq_hz} Hz, Control: {control_freq_hz} Hz")
    print(f"{'='*60}\n")

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    START = time.time()

    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute LQR control for the current waypoint ##########
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                control_timestep=env.CTRL_TIMESTEP,
                state=obs[j],
                target_pos=np.hstack([
                    TARGET_POS[wp_counters[j], 0:2],
                    INIT_XYZS[j, 2]
                ]),
                target_rpy=INIT_RPYS[j, :]
            )

        #### Go to the next waypoint and loop ######################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP-1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(
                drone=j,
                timestamp=i/env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack([
                    TARGET_POS[wp_counters[j], 0:2],
                    INIT_XYZS[j, 2],
                    INIT_RPYS[j, :],
                    np.zeros(6)
                ])
            )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Save the simulation results ###########################
    logger.save()
    logger.save_as_csv("lqr")

    #### Plot the simulation results ###########################
    if plot:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LQR flight script (drop-in replacement for pid.py)')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,            type=DroneModel, help='Drone model', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES,        type=int,        help='Number of drones', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,           type=Physics,    help='Physics updates', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,               type=str2bool,   help='PyBullet GUI', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION,     type=str2bool,   help='Record video', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,              type=str2bool,   help='Plot results', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI,    type=str2bool,   help='Debug GUI', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,         type=str2bool,   help='Add obstacles', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,      help='Sim freq Hz', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ,   type=int,       help='Ctrl freq Hz', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC,       type=int,       help='Duration (s)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER,      type=str,       help='Output folder', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,              type=bool,      help='Colab mode', metavar='')
    ARGS = parser.parse_args()
    run(**vars(ARGS))