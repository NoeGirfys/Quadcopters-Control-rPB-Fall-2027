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

Date   : 2026-03

Usage :
    cd circle_comparison_simu_and_real
    python cf_firmware_pid_sim.py [--mode MODE]
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Firmware constants and classes — from crazyflie_firmware
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(SCRIPT_DIR, ".."))
from crazyflie_firmware.constants import UINT16_MAX  # noqa: E402


# ---------------------------------------------------------------------------
# Firmware classes — from crazyflie_firmware
# ---------------------------------------------------------------------------
from crazyflie_firmware.firmware import (  # noqa: E402
    Lpf2pData,
    PidObject,
    CrazyfliePowerDistribution,
    cap_angle,
    CrazyflieFirmwarePID,
)


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

def generate_trajectory(ctrl_freq, duration_sec, hover_height=0.5, radius=0.5,
                        start_xy=(0.0, 0.0)):
    """Generate a smooth takeoff → circle → landing trajectory.

    The whole trajectory is rigidly translated so that takeoff and landing
    happen at (start_xy[0], start_xy[1]) in the world frame.  This lets the
    real drone start from an arbitrary mocap position without jerking
    toward the origin on takeoff.

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

    ox, oy = start_xy   # trajectory origin in world frame

    # --- Phase 1: Takeoff (vertical climb to hover_height) ---
    for i in range(takeoff_steps):
        t = i / takeoff_steps
        z = hover_height * t
        waypoints[i] = [ox, oy, z, 0.0]

    # --- Phase 2: Circle at hover_height ---
    for i in range(circle_steps):
        t = i / circle_steps
        angle = t * 2 * math.pi   # one full circle
        x = radius * math.cos(angle) - radius  # starts at (0,0) relative
        y = radius * math.sin(angle)
        waypoints[takeoff_steps + i] = [ox + x, oy + y, hover_height, 0.0]

    # --- Phase 3: Landing (back to origin, descend) ---
    for i in range(landing_steps):
        t = i / landing_steps
        z = hover_height * (1.0 - t)
        waypoints[takeoff_steps + circle_steps + i] = [ox, oy, z, 0.0]

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
    from crazyflie_firmware.constants import PID_VEL_THRUST_BASE
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
    plt.savefig(os.path.join(SCRIPT_DIR, 'sim_results.png'), dpi=150)
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
    plt.savefig(os.path.join(SCRIPT_DIR, 'comparison_results.png'), dpi=150)
    print("[PLOT] Saved to comparison_results.png")
    plt.show()


def _plot_jitter(t_wall_log: list, t_nominal_log: list, target_freq: float) -> None:
    """Plot cumulative drift vs nominal timeline and inter-send interval (jitter)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not found — skipping jitter plot.")
        return

    t_wall = np.array(t_wall_log)
    t_nominal = np.array(t_nominal_log)
    drift_ms = (t_wall - t_wall[0] - (t_nominal - t_nominal[0])) * 1000.0
    intervals_ms = np.diff(t_wall) * 1000.0
    nominal_interval_ms = 1000.0 / target_freq

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    fig.suptitle(f'Send Timing — Real Drone (nominal: {nominal_interval_ms:.1f} ms @ {target_freq:.0f} Hz)',
                 fontsize=13)

    ax = axes[0]
    ax.plot(t_nominal, drift_ms, lw=0.8, alpha=0.9, label='drift vs nominal [ms]')
    ax.axhline(0, color='red', ls='--', lw=1.2, label='perfect timing')
    ax.set_ylabel('Drift [ms]')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=8)
    ax.set_title('Cumulative drift vs nominal timeline')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t_nominal[1:], intervals_ms, lw=0.6, alpha=0.7, label='inter-send interval [ms]')
    ax.axhline(nominal_interval_ms, color='red', ls='--', lw=1.2,
               label=f'nominal {nominal_interval_ms:.1f} ms')
    ax.set_ylabel('Interval [ms]')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=8)
    ax.set_title('Inter-send interval (jitter)')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(SCRIPT_DIR, 'jitter_results.png')
    fig.savefig(save_path, dpi=150)
    print(f"[PLOT] Saved to {save_path}")
    print(f"[TIMING] n={len(intervals_ms)}  "
          f"mean={np.mean(intervals_ms):.2f} ms  "
          f"std={np.std(intervals_ms):.2f} ms  "
          f"min={np.min(intervals_ms):.2f} ms  "
          f"max={np.max(intervals_ms):.2f} ms  "
          f"final_drift={drift_ms[-1]:.1f} ms")
    plt.show()


# ===================================================================
#  Flight-quality metrics — circle phase only
# ===================================================================

_TAKEOFF_DURATION = 2  # seconds — must match generate_trajectory


def compute_and_save_metrics(real_log, duration_sec,
                              positioning_system, output_dir):
    """Compute and persist tracking metrics for the circle phase.

    All setpoints come from the drone's own log at 100 Hz — no interpolation.
    posCtl.targetX/Y/Z and posCtl.targetVX/VY/VZ are in the body-yaw-aligned
    (BYA) frame and are rotated back to the world frame using the concurrent
    yaw estimate before computing errors.

    Cascade:
      Position  : posCtl.targetX/Y/Z (BYA→world)  vs  stateEstimate.x/y/z
      Velocity  : posCtl.targetVX/VY/VZ (BYA→world) vs  stateEstimate.vx/vy/vz
      Attitude  : controller.roll/pitch/yaw          vs  stabilizer.roll/pitch/yaw
      Rate      : controller.rollRate/pitchRate/yawRate vs gyro.x/y/z
    """
    import json
    from datetime import datetime

    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    circle_start = float(_TAKEOFF_DURATION)
    circle_end   = float(duration_sec - _TAKEOFF_DURATION)

    if not real_log:
        print("[METRICS] No flight data — skipping.")
        return None

    all_t      = np.array([e['t']                                               for e in real_log])
    all_pos    = np.array([[e['x'],       e['y'],       e['z']]                 for e in real_log])
    all_vel    = np.array([[e['vx'],      e['vy'],      e['vz']]                for e in real_log])
    all_sp_bya = np.array([[e.get('sp_x', 0.0), e.get('sp_y', 0.0),
                             e.get('sp_z', 0.0)]                                for e in real_log])
    all_sv_bya = np.array([[e.get('sp_vx', 0.0), e.get('sp_vy', 0.0),
                             e.get('sp_vz', 0.0)]                               for e in real_log])
    all_rpy    = np.array([[e['roll'],    e['pitch'],   e['yaw']]               for e in real_log])
    all_sp_att = np.array([[e.get('sp_roll', 0.0), e.get('sp_pitch', 0.0),
                             e.get('sp_yaw',  0.0)]                             for e in real_log])
    all_gyro   = np.array([[e.get('gx', 0.0),  e.get('gy', 0.0),
                             e.get('gz', 0.0)]                                  for e in real_log])
    all_sp_gyr = np.array([[e.get('sp_gx', 0.0), e.get('sp_gy', 0.0),
                             e.get('sp_gz', 0.0)]                               for e in real_log])

    # ── Filter to circle phase ────────────────────────────────────────────
    mask   = (all_t >= circle_start) & (all_t <= circle_end)
    t      = all_t[mask]
    pos    = all_pos[mask]
    vel    = all_vel[mask]
    sp_bya = all_sp_bya[mask];  sv_bya = all_sv_bya[mask]
    rpy    = all_rpy[mask];     sp_att = all_sp_att[mask]
    gyro   = all_gyro[mask];    sp_gyr = all_sp_gyr[mask]

    if len(t) < 2:
        print("[METRICS] Circle phase too short — skipping metrics.")
        return None

    # ── Body-yaw-aligned → world frame ────────────────────────────────────
    # posCtl uses a frame rotated by the drone's current yaw around Z.
    # Inverse rotation: x_w = x_bya·cos(ψ) − y_bya·sin(ψ)
    #                   y_w = x_bya·sin(ψ) + y_bya·cos(ψ)
    yaw_rad = np.radians(rpy[:, 2])
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)

    sp_interp = np.column_stack([
        sp_bya[:, 0] * c - sp_bya[:, 1] * s,   # x world
        sp_bya[:, 0] * s + sp_bya[:, 1] * c,   # y world
        sp_bya[:, 2],                            # z unchanged
    ])
    sp_vel = np.column_stack([
        sv_bya[:, 0] * c - sv_bya[:, 1] * s,
        sv_bya[:, 0] * s + sv_bya[:, 1] * c,
        sv_bya[:, 2],
    ])

    pos_err    = pos - sp_interp
    pos_err_3d = np.linalg.norm(pos_err, axis=1)

    # ── Velocity error ────────────────────────────────────────────────────
    vel_err    = vel - sp_vel
    vel_err_3d = np.linalg.norm(vel_err, axis=1)

    # ── Attitude error (wrap yaw to [−180, 180]) ──────────────────────────
    att_err = rpy - sp_att
    att_err[:, 2] = ((att_err[:, 2] + 180) % 360) - 180

    # ── Rate error ─────────────────────────────────────────────────────────
    rate_err    = gyro - sp_gyr
    rate_err_3d = np.linalg.norm(rate_err, axis=1)

    def _rmse(e):  return float(np.sqrt(np.mean(e ** 2)))
    def _mae(e):   return float(np.mean(np.abs(e)))
    def _max(e):   return float(np.max(np.abs(e)))

    metrics = {
        'positioning_system': positioning_system,
        'timestamp':          timestamp,
        'circle_phase_s':     [circle_start, circle_end],
        'n_samples':          int(len(t)),
        'position_tracking': {
            'rmse_3d_m':    float(np.sqrt(np.mean(pos_err_3d ** 2))),
            'rmse_x_m':     _rmse(pos_err[:, 0]),
            'rmse_y_m':     _rmse(pos_err[:, 1]),
            'rmse_z_m':     _rmse(pos_err[:, 2]),
            'mae_3d_m':     float(np.mean(pos_err_3d)),
            'max_err_3d_m': float(np.max(pos_err_3d)),
        },
        'velocity_tracking': {
            'rmse_3d_m_s':  float(np.sqrt(np.mean(vel_err_3d ** 2))),
            'rmse_vx_m_s':  _rmse(vel_err[:, 0]),
            'rmse_vy_m_s':  _rmse(vel_err[:, 1]),
            'rmse_vz_m_s':  _rmse(vel_err[:, 2]),
            'mae_3d_m_s':   float(np.mean(vel_err_3d)),
            'max_err_3d_m_s': float(np.max(vel_err_3d)),
        },
        'attitude_tracking': {
            'rmse_roll_deg':  _rmse(att_err[:, 0]),
            'rmse_pitch_deg': _rmse(att_err[:, 1]),
            'rmse_yaw_deg':   _rmse(att_err[:, 2]),
            'mae_roll_deg':   _mae(att_err[:, 0]),
            'mae_pitch_deg':  _mae(att_err[:, 1]),
            'mae_yaw_deg':    _mae(att_err[:, 2]),
            'max_roll_deg':   _max(att_err[:, 0]),
            'max_pitch_deg':  _max(att_err[:, 1]),
            'max_yaw_deg':    _max(att_err[:, 2]),
        },
        'rate_tracking': {
            'rmse_3d_deg_s':  float(np.sqrt(np.mean(rate_err_3d ** 2))),
            'rmse_gx_deg_s':  _rmse(rate_err[:, 0]),
            'rmse_gy_deg_s':  _rmse(rate_err[:, 1]),
            'rmse_gz_deg_s':  _rmse(rate_err[:, 2]),
            'mae_3d_deg_s':   float(np.mean(rate_err_3d)),
            'max_err_3d_deg_s': float(np.max(rate_err_3d)),
        },
    }

    # ── Print summary ─────────────────────────────────────────────────────
    pt = metrics['position_tracking']
    vt = metrics['velocity_tracking']
    at = metrics['attitude_tracking']
    rt = metrics['rate_tracking']

    print("\n" + "=" * 62)
    print(f" METRICS  [{positioning_system.upper()}]  "
          f"circle {circle_start:.0f}–{circle_end:.0f} s  ({len(t)} samples)")
    print("=" * 62)
    print(f"  [Position]  RMSE 3-D    : {pt['rmse_3d_m']*100:6.2f} cm")
    print(f"              RMSE x/y/z  : {pt['rmse_x_m']*100:.2f} / "
          f"{pt['rmse_y_m']*100:.2f} / {pt['rmse_z_m']*100:.2f} cm")
    print(f"              MAE 3-D     : {pt['mae_3d_m']*100:.2f} cm   "
          f"max: {pt['max_err_3d_m']*100:.2f} cm")
    print(f"  [Velocity]  RMSE 3-D    : {vt['rmse_3d_m_s']*100:6.2f} cm/s")
    print(f"              RMSE vx/vy/vz: {vt['rmse_vx_m_s']*100:.2f} / "
          f"{vt['rmse_vy_m_s']*100:.2f} / {vt['rmse_vz_m_s']*100:.2f} cm/s")
    print(f"  [Attitude]  RMSE r/p/y  : {at['rmse_roll_deg']:.2f} / "
          f"{at['rmse_pitch_deg']:.2f} / {at['rmse_yaw_deg']:.2f} °")
    print(f"              MAE  r/p/y  : {at['mae_roll_deg']:.2f} / "
          f"{at['mae_pitch_deg']:.2f} / {at['mae_yaw_deg']:.2f} °")
    print(f"  [Rate]      RMSE 3-D    : {rt['rmse_3d_deg_s']:6.1f} °/s")
    print(f"              RMSE gx/gy/gz: {rt['rmse_gx_deg_s']:.1f} / "
          f"{rt['rmse_gy_deg_s']:.1f} / {rt['rmse_gz_deg_s']:.1f} °/s")
    print("=" * 62 + "\n")

    # ── Save JSON ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir,
                             f'metrics_{positioning_system}_{timestamp}.json')
    with open(json_path, 'w') as fh:
        json.dump(metrics, fh, indent=2)
    print(f"[METRICS] JSON  → {json_path}")

    # ── Plot ───────────────────────────────────────────────────────────────
    _plot_metrics(
        t, pos, sp_interp, pos_err, pos_err_3d,
        vel, sp_vel, vel_err_3d,
        rpy, sp_att,
        gyro, sp_gyr, rate_err_3d,
        metrics, positioning_system, timestamp, output_dir,
    )

    return metrics


def _plot_metrics(t, pos, sp_interp, pos_err, pos_err_3d,
                  vel, sp_vel, vel_err_3d,
                  rpy, sp_att,
                  gyro, sp_gyr, rate_err_3d,
                  metrics, positioning_system, timestamp, output_dir):
    """5-row × 3-column figure covering the full PID cascade.

    Row 0 — Position x/y/z vs setpoint
    Row 1 — Velocity vx/vy/vz vs setpoint (pos-PID output)
    Row 2 — Attitude roll/pitch vs setpoint (vel-PID output) + yaw actual
    Row 3 — Angular rate gx/gy/gz vs setpoint (att-PID output)
    Row 4 — 3-D position error, 3-D velocity error, top-view X-Y
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[PLOT] matplotlib not found — skipping metrics plot.")
        return

    tr = t - t[0]   # time relative to circle start [s]
    pt = metrics['position_tracking']
    vt = metrics['velocity_tracking']
    at = metrics['attitude_tracking']
    rt = metrics['rate_tracking']

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle(
        f"PID Cascade Quality — {positioning_system.upper()}  |  {timestamp}\n"
        f"Pos RMSE {pt['rmse_3d_m']*100:.2f} cm  |  "
        f"Vel RMSE {vt['rmse_3d_m_s']*100:.2f} cm/s  |  "
        f"Att RMSE r/p {at['rmse_roll_deg']:.2f}/{at['rmse_pitch_deg']:.2f} °  |  "
        f"Rate RMSE {rt['rmse_3d_deg_s']:.1f} °/s",
        fontsize=12,
    )
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.55, wspace=0.35)

    def _tracking_subplot(ax, tr, actual, setpoint, ylabel, title, clr_a='tomato', clr_s='steelblue'):
        ax.plot(tr, actual,   color=clr_a, lw=1.5, label='actual')
        ax.plot(tr, setpoint, color=clr_s, lw=1.0, ls='--', label='setpoint')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 0: position ───────────────────────────────────────────────────
    for col, (lbl, axis) in enumerate(zip(['x [m]', 'y [m]', 'z [m]'], 'xyz')):
        _tracking_subplot(fig.add_subplot(gs[0, col]),
                          tr, pos[:, col], sp_interp[:, col], lbl, f'Pos {axis}')

    # ── Row 1: velocity ───────────────────────────────────────────────────
    for col, (lbl, axis) in enumerate(zip(['vx [m/s]', 'vy [m/s]', 'vz [m/s]'], ['x', 'y', 'z'])):
        _tracking_subplot(fig.add_subplot(gs[1, col]),
                          tr, vel[:, col], sp_vel[:, col], lbl, f'Vel {axis}',
                          clr_a='darkorange', clr_s='royalblue')

    # ── Row 2: attitude (roll, pitch, yaw — all vs setpoint) ─────────────
    att_row = [
        ('roll [°]',  0, at['rmse_roll_deg']),
        ('pitch [°]', 1, at['rmse_pitch_deg']),
        ('yaw [°]',   2, at['rmse_yaw_deg']),
    ]
    for col, (lbl, idx, rmse) in enumerate(att_row):
        ax = fig.add_subplot(gs[2, col])
        ax.plot(tr, rpy[:, idx],    color='seagreen',     lw=1.5, label='actual')
        ax.plot(tr, sp_att[:, idx], color='mediumorchid', lw=1.0, ls='--', label='setpoint')
        ax.set_ylabel(lbl)
        ax.set_title(f'Att {["roll","pitch","yaw"][idx]}  RMSE {rmse:.2f}°')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # ── Row 3: angular rate ───────────────────────────────────────────────
    rate_labels = ['gx [°/s]', 'gy [°/s]', 'gz [°/s]']
    rate_rmses  = [rt['rmse_gx_deg_s'], rt['rmse_gy_deg_s'], rt['rmse_gz_deg_s']]
    for col in range(3):
        ax = fig.add_subplot(gs[3, col])
        ax.plot(tr, gyro[:, col],   color='purple',      lw=1.5, label='actual')
        ax.plot(tr, sp_gyr[:, col], color='darkcyan',    lw=1.0, ls='--', label='setpoint')
        ax.set_ylabel(rate_labels[col])
        ax.set_title(f'Rate {"xyz"[col]}  RMSE {rate_rmses[col]:.1f} °/s')
        ax.legend(fontsize=7)
        ax.set_xlabel('Time [s]')
        ax.grid(True, alpha=0.3)

    # ── Row 4: aggregate error + top-view ────────────────────────────────
    ax = fig.add_subplot(gs[4, 0])
    ax.plot(tr, pos_err_3d, color='firebrick', lw=1.2)
    ax.axhline(pt['rmse_3d_m'], color='steelblue', ls='--', lw=1.0,
               label=f"RMSE {pt['rmse_3d_m']*100:.2f} cm")
    ax.axhline(pt['mae_3d_m'],  color='seagreen',  ls=':',  lw=1.0,
               label=f"MAE  {pt['mae_3d_m']*100:.2f} cm")
    ax.set_ylabel('3-D pos error [m]')
    ax.set_title('Position 3-D error')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[4, 1])
    ax.plot(tr, vel_err_3d, color='darkorange', lw=1.2)
    ax.axhline(vt['rmse_3d_m_s'], color='royalblue', ls='--', lw=1.0,
               label=f"RMSE {vt['rmse_3d_m_s']*100:.2f} cm/s")
    ax.axhline(vt['mae_3d_m_s'],  color='seagreen',  ls=':',  lw=1.0,
               label=f"MAE  {vt['mae_3d_m_s']*100:.2f} cm/s")
    ax.set_ylabel('3-D vel error [m/s]')
    ax.set_title('Velocity 3-D error')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[4, 2])
    ax.plot(sp_interp[:, 0], sp_interp[:, 1],
            color='steelblue', ls='--', lw=1.0, label='setpoint', zorder=2)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=tr, cmap='plasma',
                    s=3, zorder=3, label='actual')
    plt.colorbar(sc, ax=ax, label='t [s]')
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_aspect('equal')
    ax.set_title('Top-view (X-Y)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir,
                            f'metrics_{positioning_system}_{timestamp}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT]    PNG   → {png_path}")
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
    from crazyflie_firmware.constants import (
        PID_POS_X_KP, PID_POS_X_KI, PID_POS_X_KD,
        PID_POS_Y_KP, PID_POS_Y_KI, PID_POS_Y_KD,
        PID_POS_Z_KP, PID_POS_Z_KI, PID_POS_Z_KD,

        PID_VEL_X_KP, PID_VEL_X_KI, PID_VEL_X_KD,
        PID_VEL_Y_KP, PID_VEL_Y_KI, PID_VEL_Y_KD,
        PID_VEL_Z_KP, PID_VEL_Z_KI, PID_VEL_Z_KD,

        PID_ROLL_KP, PID_ROLL_KI, PID_ROLL_KD,
        PID_PITCH_KP, PID_PITCH_KI, PID_PITCH_KD,
        PID_YAW_KP, PID_YAW_KI, PID_YAW_KD,

        PID_ROLL_RATE_KP, PID_ROLL_RATE_KI, PID_ROLL_RATE_KD,
        PID_PITCH_RATE_KP, PID_PITCH_RATE_KI, PID_PITCH_RATE_KD,
        PID_YAW_RATE_KP, PID_YAW_RATE_KI, PID_YAW_RATE_KD,
    )
    
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

def run_real(uri="radio://0/80/2M/E7E7E7E7E7",
             duration_sec=15, hover_height=0.5, radius=0.5,
             push_gains=False, positioning_system='unknown'):
    """Run with a real Crazyflie drone in position-setpoint mode.

    All PIDs (position, velocity, attitude, rate) run on the drone.
    The PC sends only position setpoints at 10 Hz via
    send_position_setpoint(x, y, z, yaw).

    Parameters
    ----------
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

    CTRL_FREQ = 100   # position setpoints don't need high rate
    print(f"[REAL] Connecting to {uri} ...")

    cache_dir = os.path.join(SCRIPT_DIR, 'cache')
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=cache_dir)) as scf:
        cf = scf.cf
        print("[REAL] Connected!")

        # On s'assure de désactiver l'écart-type adaptatif
        cf.param.set_value('motion.adaptive', '0')

        # On augmente l'écart-type fixe du Flowdeck (par défaut à 2.0)
        # La valeur de 10.0 est un bon point de départ pour lisser les oscillations selon les tests de Bitcraze
        cf.param.set_value('motion.flowStdFixed', '10.0')

        if push_gains:
            push_pid_gains_to_drone(cf)

        # Set up logging to read back state (position + velocity + attitude)
        # This requires a positioning system (Lighthouse / Loco / MoCap)
        from cflib.crazyflie.log import LogConfig
        LOG_MS = 50 # 20 Hz
        log_state = LogConfig(name='State', period_in_ms=LOG_MS)  
        log_state.add_variable('stateEstimate.x',  'float')
        log_state.add_variable('stateEstimate.y',  'float')
        log_state.add_variable('stateEstimate.z',  'float')
        log_state.add_variable('stateEstimate.vx', 'float')
        log_state.add_variable('stateEstimate.vy', 'float')
        log_state.add_variable('stateEstimate.vz', 'float')

        # actual attitude + attitude setpoints from vel-PID (6 floats = 24 bytes)
        log_att = LogConfig(name='Attitude', period_in_ms=LOG_MS)
        log_att.add_variable('stabilizer.roll',   'float')
        log_att.add_variable('stabilizer.pitch',  'float')
        log_att.add_variable('stabilizer.yaw',    'float')
        log_att.add_variable('controller.roll',   'float')  # att setpoint from vel-PID [deg]
        log_att.add_variable('controller.pitch',  'float')  # att setpoint from vel-PID [deg]
        log_att.add_variable('controller.yaw',    'float')  # att setpoint from vel-PID [deg]

        # actual gyro + rate setpoints from att-PID (6 floats = 24 bytes)
        log_rate = LogConfig(name='AngRate', period_in_ms=LOG_MS)
        log_rate.add_variable('gyro.x',              'float')
        log_rate.add_variable('gyro.y',              'float')
        log_rate.add_variable('gyro.z',              'float')
        log_rate.add_variable('controller.rollRate',  'float')  # rate setpoint from att-PID [deg/s]
        log_rate.add_variable('controller.pitchRate', 'float')
        log_rate.add_variable('controller.yawRate',   'float')

        # position + velocity setpoints from pos-PID, body-yaw-aligned frame (6 floats = 24 bytes)
        log_vel_sp = LogConfig(name='PidSp', period_in_ms=LOG_MS)
        log_vel_sp.add_variable('posCtl.targetX',  'float')  # pos setpoint BYA [m]
        log_vel_sp.add_variable('posCtl.targetY',  'float')
        log_vel_sp.add_variable('posCtl.targetZ',  'float')
        log_vel_sp.add_variable('posCtl.targetVX', 'float')  # vel setpoint BYA [m/s]
        log_vel_sp.add_variable('posCtl.targetVY', 'float')
        log_vel_sp.add_variable('posCtl.targetVZ', 'float')

        # Shared state dict updated by log callbacks
        drone_state = {
            'x': 0, 'y': 0, 'z': 0, 'vx': 0, 'vy': 0, 'vz': 0,
            'sp_x': 0, 'sp_y': 0, 'sp_z': 0,          # pos setpoints (BYA → world in metrics)
            'sp_vx': 0, 'sp_vy': 0, 'sp_vz': 0,        # vel setpoints (BYA → world in metrics)
            'roll': 0, 'pitch': 0, 'yaw': 0,
            'sp_roll': 0, 'sp_pitch': 0, 'sp_yaw': 0,  # att setpoints [deg]
            'gyro_x': 0, 'gyro_y': 0, 'gyro_z': 0,
            'sp_gx': 0, 'sp_gy': 0, 'sp_gz': 0,        # rate setpoints [deg/s]
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
                    't':  time.time() - flight_start,
                    # actual position & velocity (world frame)
                    'x':  drone_state['x'],     'y':  drone_state['y'],     'z':  drone_state['z'],
                    'vx': drone_state['vx'],    'vy': drone_state['vy'],    'vz': drone_state['vz'],
                    # pos/vel setpoints from pos-PID — body-yaw-aligned; converted in metrics
                    'sp_x':  drone_state['sp_x'],  'sp_y':  drone_state['sp_y'],  'sp_z':  drone_state['sp_z'],
                    'sp_vx': drone_state['sp_vx'], 'sp_vy': drone_state['sp_vy'], 'sp_vz': drone_state['sp_vz'],
                    # actual attitude [deg]
                    'roll': drone_state['roll'], 'pitch': drone_state['pitch'], 'yaw': drone_state['yaw'],
                    # attitude setpoints from vel-PID [deg]
                    'sp_roll': drone_state['sp_roll'], 'sp_pitch': drone_state['sp_pitch'],
                    'sp_yaw':  drone_state['sp_yaw'],
                    # actual angular rates [deg/s]
                    'gx': drone_state['gyro_x'], 'gy': drone_state['gyro_y'], 'gz': drone_state['gyro_z'],
                    # rate setpoints from att-PID [deg/s]
                    'sp_gx': drone_state['sp_gx'], 'sp_gy': drone_state['sp_gy'], 'sp_gz': drone_state['sp_gz'],
                })

        def _att_cb(timestamp, data, logconf):
            drone_state['roll']     = data['stabilizer.roll']
            drone_state['pitch']    = data['stabilizer.pitch']
            drone_state['yaw']      = data['stabilizer.yaw']
            drone_state['sp_roll']  = data['controller.roll']
            drone_state['sp_pitch'] = data['controller.pitch']
            drone_state['sp_yaw']   = data['controller.yaw']

        def _rate_cb(timestamp, data, logconf):
            drone_state['gyro_x'] = data['gyro.x']
            drone_state['gyro_y'] = data['gyro.y']
            drone_state['gyro_z'] = data['gyro.z']
            drone_state['sp_gx']  = data['controller.rollRate']
            drone_state['sp_gy']  = data['controller.pitchRate']
            drone_state['sp_gz']  = data['controller.yawRate']

        def _vel_sp_cb(timestamp, data, logconf):
            drone_state['sp_x']  = data['posCtl.targetX']
            drone_state['sp_y']  = data['posCtl.targetY']
            drone_state['sp_z']  = data['posCtl.targetZ']
            drone_state['sp_vx'] = data['posCtl.targetVX']
            drone_state['sp_vy'] = data['posCtl.targetVY']
            drone_state['sp_vz'] = data['posCtl.targetVZ']

        log_state.data_received_cb.add_callback(_state_cb)
        log_att.data_received_cb.add_callback(_att_cb)
        log_rate.data_received_cb.add_callback(_rate_cb)
        log_vel_sp.data_received_cb.add_callback(_vel_sp_cb)
        cf.log.add_config(log_state)
        cf.log.add_config(log_att)
        cf.log.add_config(log_rate)
        cf.log.add_config(log_vel_sp)
        log_state.start()
        log_att.start()
        log_rate.start()
        log_vel_sp.start()


        # ── Reset Kalman estimator ────────────────────────────────────
        print("[REAL] Resetting Kalman estimator...")
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)   # On attend que le filtre converge ET que les paquets de logs arrivent !
        
        # --- NOUVEAU PLACEMENT ICI ---
        # Maintenant les logs sont à jour avec la vraie position absolue Lighthouse
        start_xy = np.array([drone_state['x'], drone_state['y']]) # <-- Note les crochets []
        print(f"[REAL] Estimator ready — pos=({drone_state['x']:.3f}, "
              f"{drone_state['y']:.3f}, {drone_state['z']:.3f})")

        # ── Generate trajectory centered on the drone's actual start ──
        waypoints = generate_trajectory(CTRL_FREQ, duration_sec,
                                        hover_height=hover_height,
                                        radius=radius, start_xy=start_xy)

        # ── Unlock commander watchdog ─────────────────────────────────
        # The firmware requires receiving setpoints before it accepts real
        # commands (and after each re-arm following a landing).
        for _ in range(50):
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.02)
        print("[REAL] Commander unlocked. Flying trajectory...")

        START = time.time()
        flight_start = START   # enable data logging in _state_cb

        cmd_count = 0

        # Wall-clock timestamp at the start of each iteration (for drift/jitter analysis)
        t_wall_log: list = []
        t_nominal_log: list = []

        def _sync_precise(target: float) -> None:
            """Wait until target (time.perf_counter()). Sleeps most of the remaining
            time, then busy-waits the last 2 ms for accuracy.
            Requires timeBeginPeriod(1) on Windows so that sleep(<15 ms) is accurate."""
            remaining = target - time.perf_counter()
            if remaining > 0.002:
                time.sleep(remaining - 0.002)
            while time.perf_counter() < target:
                pass

        _win_timer_set = False
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
            _win_timer_set = True
        except Exception:
            pass

        ctrl_dt = 1.0 / CTRL_FREQ
        loop_start_perf = time.perf_counter()

        try:
            for i in range(len(waypoints)):
                t_iter_wall = time.time()
                sp = waypoints[i]
                sp_pos = sp[0:3]
                sp_yaw_rate = sp[3]

                real_sp_log.append({'t': time.time() - flight_start, 'pos': sp_pos.copy()})
                cf.commander.send_position_setpoint(sp_pos[0], sp_pos[1], sp_pos[2], 0.0)

                # --- Timing ---
                if i % (CTRL_FREQ * 1) == 0:
                    t = i / CTRL_FREQ
                    print(f"  t={t:5.1f}s  pos=[{drone_state['x']:+.3f}, "
                          f"{drone_state['y']:+.3f}, {drone_state['z']:.3f}]  "
                          f"sp=[{sp_pos[0]:+.3f}, {sp_pos[1]:+.3f}, {sp_pos[2]:.3f}]")

                t_wall_log.append(t_iter_wall)
                t_nominal_log.append(i / CTRL_FREQ)
                cmd_count += 1

                _sync_precise(loop_start_perf + (i + 1) * ctrl_dt)

        except KeyboardInterrupt:
            print("\n[REAL] Interrupted! Soft landing...")
        finally:
            if _win_timer_set:
                ctypes.windll.winmm.timeEndPeriod(1)

            # ─── Smooth landing ───────────────────────────────
            # Descend from current position to ~5 cm, then cut motors.
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
            log_rate.stop()
            log_vel_sp.stop()
            print("[REAL] Landed and cleaned up.")

        # ─── Post-flight: run headless sim + comparison plot ──────────────
        if real_log:
            real_t   = np.array([e['t']   for e in real_log])
            real_pos = np.array([[e['x'],  e['y'],  e['z']]       for e in real_log])
            real_vel = np.array([[e['vx'], e['vy'], e['vz']]      for e in real_log])
            real_rpy = np.array([[e['roll'], e['pitch'], e['yaw']] for e in real_log])
            sp_t_arr = np.array([e['t']   for e in real_sp_log])
            sp_arr   = np.array([e['pos'] for e in real_sp_log])

            real_pos[:, 0] -= start_xy[0]
            real_pos[:, 1] -= start_xy[1]

            sp_arr[:, 0]   -= start_xy[0]
            sp_arr[:, 1]   -= start_xy[1]


            real_data = dict(t=real_t, pos=real_pos, vel=real_vel, rpy=real_rpy,
                             sp_t=sp_t_arr, sp=sp_arr)

            if t_wall_log:
                print("[REAL] Generating timing plot...")
                _plot_jitter(t_wall_log, t_nominal_log, CTRL_FREQ)

            # ── Flight-quality metrics (circle phase only) ─────────────
            metrics_dir = os.path.join(SCRIPT_DIR, 'metrics')
            compute_and_save_metrics(real_log, duration_sec,
                                     positioning_system, metrics_dir)
            # -----------------------------------------------------------

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
  position   Real drone: trajectory only → send_position_setpoint (all PIDs on drone)
        """)
    parser.add_argument('--mode', default='sim',
                        choices=['sim', 'position'],
                        help='sim: full firmware PID in PyBullet | position: real drone, all PIDs on-board (default: sim)')
    parser.add_argument('--duration', default=10, type=float,
                        help='Flight duration in seconds (default: 10)')
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
    parser.add_argument('--positioning', default='unknown',
                        choices=['flowdeck', 'lighthouse', 'loco', 'mocap', 'unknown'],
                        help='Positioning system used — labels output files (default: unknown)')
    args = parser.parse_args()

    if args.mode == 'sim':
        run_sim(duration_sec=args.duration, gui=args.gui,
                hover_height=args.height, radius=args.radius,
                simulate_flow_deck=args.flow_deck_sim)
    else:
        run_real(uri=args.uri,
                 duration_sec=args.duration, hover_height=args.height,
                 radius=args.radius, push_gains=args.push_gains,
                 positioning_system=args.positioning)


if __name__ == "__main__":
    main()