#!/usr/bin/env python3
"""
Crazyflie Firmware-Faithful PID Controller — Simulation & Real Drone
=====================================================================

This script replicates the EXACT PID control pipeline from the Crazyflie
firmware (crazyflie-firmware-master), and uses it to fly a drone
(takeoff → circle → land) inside the gym-pybullet-drones simulator.

An optional --mode argument selects the target:

  sim        – Full pipeline in Python → RPMs → PyBullet           (default)
  position   – OptiTrack mocap → extpose to drone; PC streams
               send_position_setpoint (all PIDs on drone)

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

# ---------------------------------------------------------------------------
# NatNet SDK (OptiTrack streaming for real drone, position mode)
# Bundled in ./NatNetSDK/ — added to sys.path lazily inside the streamer
# so that import failures only affect real-drone runs.
# ---------------------------------------------------------------------------
import threading

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
NATNET_DIR = os.path.join(SCRIPT_DIR, "NatNetSDK")
HAS_NATNET = os.path.isdir(NATNET_DIR) and os.path.isfile(
    os.path.join(NATNET_DIR, "NatNetClient.py"))

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
    CrazyfliePositionController,
    CrazyflieAttitudeController,
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
            simulate_flow_deck=False, setpoint_freq_hz=100):
    """Run the full firmware PID pipeline in PyBullet simulation.

    Parameters
    ----------
    simulate_flow_deck : bool
        If True, replace the perfect PyBullet state with a simulated Flow
        deck v2 estimate (noise + delay) before feeding it to the position
        PID.  This reproduces the oscillations seen on the real drone.
    setpoint_freq_hz : float
        Rate at which a new position setpoint is handed to the firmware PID,
        matching the radio packet rate on the real drone (default 10 Hz).
        The internal PID still runs at CTRL_FREQ (500 Hz); between setpoint
        updates the controller holds the last received setpoint — exactly
        the behavior of the real firmware between two send_position_setpoint
        packets.
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

    # Generate the setpoint trajectory at the radio packet rate, then run
    # the sim at CTRL_FREQ — each setpoint is held for CTRL_FREQ/setpoint_freq_hz
    # PID cycles, mirroring the real drone where the firmware keeps the last
    # send_position_setpoint value until the next radio packet arrives.
    setpoint_waypoints = generate_trajectory(int(setpoint_freq_hz), duration_sec,
                                             hover_height=hover_height, radius=radius)
    n_steps = int(CTRL_FREQ * duration_sec)
    n_sp    = len(setpoint_waypoints)

    print(f"[SIM] PyBullet freq: {PYB_FREQ} Hz, Control freq: {CTRL_FREQ} Hz, "
          f"Setpoint freq: {setpoint_freq_hz} Hz")
    print(f"[SIM] MAX_RPM: {MAX_RPM:.1f}, HOVER_RPM: {HOVER_RPM:.1f}, KF: {KF:.4e}")
    print(f"[SIM] Firmware THRUST_MAX/motor: {CF2_THRUST_MAX_PER_MOTOR*1000:.1f} mN, "
          f"RPM at THRUST_MAX: {math.sqrt(CF2_THRUST_MAX_PER_MOTOR/KF):.1f}")
    print(f"[SIM] Trajectory: {n_steps} PID steps ({n_sp} setpoints), {duration_sec}s")
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

        # --- Get setpoint (held between radio packets, like the real drone) ---
        # Each setpoint is kept for CTRL_FREQ/setpoint_freq_hz PID cycles.
        sp_idx = min(i * n_sp // n_steps, n_sp - 1)
        sp = setpoint_waypoints[sp_idx]
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
#  OptiTrack streaming thread  (NatNet SDK → send_extpose)
# ===================================================================

def _natnet_streamer(cf, server_ip, client_ip, use_multicast, rigid_body_id,
                     stop_event, shared=None, rate_hz=None):
    """Stream OptiTrack rigid-body pose to the Crazyflie via send_extpose,
    using the bundled NatNet SDK directly.

    Motive must be configured with:
      • Streaming Engine ON, Up Axis = Z, Rigid Bodies ON.
      • Multicast or Unicast matching `use_multicast`.
      • A rigid body with streaming ID = `rigid_body_id`.

    The NatNet client spawns its own data thread internally; the per-rigid-body
    callback fires there.  `cf.extpos.send_extpose` is called directly from
    that callback (cflib serializes radio writes via its own queues).

    The shared dict is updated at Motive's native rate so the start-of-flight
    handshake converges quickly.  Radio writes are optionally throttled to
    `rate_hz`; `rate_hz=None` sends every frame.
    """
    if not HAS_NATNET:
        print(f"[MOCAP] NatNet SDK not found at {NATNET_DIR}")
        stop_event.set()
        return

    if NATNET_DIR not in sys.path:
        sys.path.insert(0, NATNET_DIR)
    try:
        from NatNetClient import NatNetClient  # type: ignore
    except Exception as e:
        print(f"[MOCAP] Failed to import NatNetClient: {e}")
        stop_event.set()
        return

    period = (1.0 / rate_hz) if rate_hz else 0.0
    # Watchdog thresholds (seconds)
    RB_LOST_TIMEOUT     = 0.10   # no rigid body for 100 ms → lost
    STREAM_LOST_TIMEOUT = 0.50   # no frame at all for 500 ms → stream down

    state = {
        'last_sent':       0.0,
        'last_rb_time':    None,   # last time the target rigid body was seen
        'last_frame_time': None,   # last time any frame arrived
        'n_rb':            0,      # frames where the target RB was present
        'n_frames':        0,      # total frames received
        'n_sent':          0,      # extpose packets sent to drone
        'n_lost_events':   0,      # tracking-loss transitions
        'tracked':         False,  # current tracking state (latched)
        'warned_send':     False,
    }

    def _on_rigid_body(new_id, position, rotation):
        if new_id != rigid_body_id:
            return
        x, y, z = position
        qx, qy, qz, qw = rotation
        now = time.time()
        state['n_rb'] += 1
        state['last_rb_time'] = now
        if not state['tracked']:
            state['tracked'] = True
            if state['n_lost_events'] > 0:
                print(f"[MOCAP] Rigid body id={rigid_body_id} REGAINED "
                      f"@ pos=({x:+.3f}, {y:+.3f}, {z:+.3f})")
        if shared is not None:
            shared['x'], shared['y'], shared['z'] = x, y, z
            shared['count'] = state['n_rb']

        if period > 0 and (now - state['last_sent']) < period:
            return
        try:
            cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
        except Exception as e:
            if not state['warned_send']:
                print(f"[MOCAP] send_extpose error: {e}")
                state['warned_send'] = True
            return
        state['last_sent'] = now
        state['n_sent'] += 1

    def _on_frame(_data_dict):
        state['n_frames'] += 1
        state['last_frame_time'] = time.time()

    client = NatNetClient()
    client.set_client_address(client_ip)
    client.set_server_address(server_ip)
    client.set_use_multicast(bool(use_multicast))
    client.rigid_body_listener = _on_rigid_body
    client.new_frame_listener = _on_frame
    client.set_print_level(0)

    if not client.run('d'):
        print("[MOCAP] NatNet run() failed — could not open sockets.")
        stop_event.set()
        return

    # Wait briefly for the server handshake to complete.
    t0 = time.time()
    while not client.connected() and (time.time() - t0) < 3.0:
        if stop_event.is_set():
            client.shutdown()
            return
        time.sleep(0.05)

    if not client.connected():
        print(f"[MOCAP] NatNet did not connect to {server_ip} "
              f"(client={client_ip}, multicast={use_multicast}). "
              f"Check Motive Streaming Engine settings.")
        client.shutdown()
        stop_event.set()
        return

    cast_str = "multicast" if use_multicast else "unicast"
    rate_msg = f"throttled to {rate_hz:.0f} Hz" if rate_hz else "every frame"
    print(f"[MOCAP] NatNet connected — server={server_ip} client={client_ip} "
          f"{cast_str}, rigid body id={rigid_body_id} ({rate_msg})")

    try:
        warned_no_frames = False
        warned_no_rb     = False
        while not stop_event.is_set():
            time.sleep(0.05)
            now = time.time()

            # ── Stream watchdog: no frame at all from Motive ──────────
            if state['last_frame_time'] is None:
                if (now - t0) > 3.0 and not warned_no_frames:
                    print("[MOCAP] WARNING: no NatNet frames received yet "
                          "— Motive streaming likely stopped.")
                    warned_no_frames = True
                continue

            stream_age = now - state['last_frame_time']
            if stream_age > STREAM_LOST_TIMEOUT:
                if not warned_no_frames:
                    print(f"[MOCAP] WARNING: NatNet stream stalled "
                          f"({stream_age*1000:.0f} ms since last frame).")
                    warned_no_frames = True
            else:
                if warned_no_frames:
                    print(f"[MOCAP] NatNet stream resumed after "
                          f"{stream_age*1000:.0f} ms gap.")
                    warned_no_frames = False

            # ── Tracking watchdog: frames arriving but no rigid body ──
            if state['last_rb_time'] is None:
                if (now - t0) > 3.0 and not warned_no_rb:
                    print(f"[MOCAP] WARNING: no frames for rigid body "
                          f"id={rigid_body_id} yet — check Motive streaming "
                          f"ID and that the body is visible.")
                    warned_no_rb = True
                continue

            rb_age = now - state['last_rb_time']
            if state['tracked'] and rb_age > RB_LOST_TIMEOUT:
                state['tracked'] = False
                state['n_lost_events'] += 1
                print(f"[MOCAP] *** TRACKING LOST *** rigid body id="
                      f"{rigid_body_id} not seen for {rb_age*1000:.0f} ms "
                      f"(stream still alive @ "
                      f"{1.0/max(stream_age,1e-6):.0f} Hz)")
    finally:
        client.shutdown()
        print(f"[MOCAP] Stopped — frames={state['n_frames']} "
              f"rb_frames={state['n_rb']} sent={state['n_sent']} "
              f"tracking_loss_events={state['n_lost_events']}")


# ===================================================================
#  MAIN — Real drone (position mode, OptiTrack-fed Kalman)
# ===================================================================

def run_real(uri="radio://0/80/2M/E7E7E7E7E7",
             duration_sec=15, hover_height=0.5, radius=0.5,
             push_gains=False,
             natnet_server_ip="192.168.0.24",
             natnet_client_ip="192.168.0.100",
             natnet_multicast=True,
             rigid_body_id=1,
             mocap_rate_hz=None):
    """Run with a real Crazyflie drone in position mode, fed by OptiTrack.

    The PC only streams pose from Motive (via the NatNet SDK) to the drone;
    the drone runs all PIDs internally on its Kalman estimate.

    Parameters
    ----------
    uri : str
        Crazyflie radio URI.
    push_gains : bool
        If True, overwrite the drone's onboard PID gains with the Python
        constants defined at the top of this file before flying.
    natnet_server_ip : str
        IP address of the Motive PC on the OptiTrack network.
    natnet_client_ip : str
        IP address of this PC's network interface used to receive the stream.
    natnet_multicast : bool
        True = multicast (matches PythonSample.py's '0' choice),
        False = unicast.
    rigid_body_id : int
        Streaming ID of the Crazyflie rigid body in Motive (the "User Data"
        ID column in Motive, NOT the rigid body name).
    """
    if not HAS_CFLIB:
        print("ERROR: cflib not found. pip install cflib")
        sys.exit(1)
    if not HAS_NATNET:
        print(f"ERROR: NatNet SDK not found at {NATNET_DIR}")
        sys.exit(1)

    cflib.crtp.init_drivers()

    # Send send_position_setpoint at 100 Hz — matches the Crazyflie
    # ecosystem convention (crazyswarm, Bitcraze examples) and keeps each
    # setpoint step small enough to avoid the high-frequency ringing that
    # appears at 10 Hz.  Bandwidth usage: ~20 B × 100 Hz = 2 kB/s (well
    # within the CRTP link's ~100 kB/s).
    CTRL_FREQ = 100

    # Trajectory is generated later, after we know the mocap start position,
    # so that takeoff happens at the drone's actual location.

    print(f"[REAL] Mode: position (OptiTrack-fed Kalman)")
    print(f"[REAL] Connecting to {uri} ...")

    cache_dir = os.path.join(SCRIPT_DIR, 'cache')
    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache=cache_dir)) as scf:
        cf = scf.cf
        print("[REAL] Connected!")

        # ─── Radio link health monitoring ─────────────────────────────
        # The Crazyradio reports link quality (0-100 %) periodically and
        # fires connection_lost / disconnected on hard radio failures
        # (e.g. "Too many packets lost" from the cflib radio driver).
        #
        # We use these to:
        #   1. log every degradation / loss event
        #   2. trigger a preemptive controlled descent when the link
        #      stays below LINK_QUALITY_ABORT_PCT for too long, before
        #      cflib gives up entirely.
        LINK_QUALITY_WARN_PCT  = 70.0   # transient warning threshold
        LINK_QUALITY_ABORT_PCT = 40.0   # sustained → abort to landing
        LINK_BAD_ABORT_SEC     = 0.5    # how long to stay below abort

        link_status = {
            'quality':      100.0,
            'lost':         False,    # True after connection_lost callback
            'disconnect':   False,    # True after disconnected callback
            'reason':       None,     # error message from connection_lost
            'bad_since':    None,     # time.time() when quality first dropped
            'warn_active':  False,
            'n_low_events': 0,
            'min_quality':  100.0,
        }

        def _link_quality_cb(percent):
            link_status['quality'] = percent
            if percent < link_status['min_quality']:
                link_status['min_quality'] = percent

            now = time.time()
            if percent < LINK_QUALITY_ABORT_PCT:
                if link_status['bad_since'] is None:
                    link_status['bad_since'] = now
            else:
                link_status['bad_since'] = None

            if percent < LINK_QUALITY_WARN_PCT:
                if not link_status['warn_active']:
                    link_status['warn_active'] = True
                    link_status['n_low_events'] += 1
                    print(f"[LINK] WARNING: link quality dropped to "
                          f"{percent:.0f}%")
            else:
                if link_status['warn_active']:
                    print(f"[LINK] Link quality recovered ({percent:.0f}%)")
                    link_status['warn_active'] = False

        def _connection_lost_cb(uri_str, msg):
            link_status['lost']   = True
            link_status['reason'] = msg
            print(f"[LINK] *** CONNECTION LOST *** {uri_str} — {msg}")

        def _disconnected_cb(uri_str):
            link_status['disconnect'] = True
            print(f"[LINK] Disconnected from {uri_str}")

        cf.link_quality_updated.add_callback(_link_quality_cb)
        cf.connection_lost.add_callback(_connection_lost_cb)
        cf.disconnected.add_callback(_disconnected_cb)

        # ─── Estimator & external-position configuration ──────────────
        # Force the Kalman filter (required for mocap-based localization)
        # and set the standard deviation of the incoming extpose packets.
        #   stabilizer.estimator: 1 = complementary, 2 = Kalman
        #   locSrv.extPosStdDev  : position noise std [m]
        #   locSrv.extQuatStdDev : quaternion noise std [rad]
        cf.param.set_value('stabilizer.estimator', '2')
        cf.param.set_value('locSrv.extPosStdDev',  '0.001')
        cf.param.set_value('locSrv.extQuatStdDev', '0.0045')

        if push_gains:
            push_pid_gains_to_drone(cf)

        # Set up logging to read back state (position + velocity + attitude).
        # 10 Hz (period_in_ms=100) keeps radio bandwidth low; the comparison
        # plot still has ~10× duration samples, which is plenty.
        from cflib.crazyflie.log import LogConfig
        LOG_PERIOD_MS = 100   # 10 Hz — tune this if more resolution is needed
        log_state = LogConfig(name='State', period_in_ms=LOG_PERIOD_MS)
        log_state.add_variable('stateEstimate.x',  'float')
        log_state.add_variable('stateEstimate.y',  'float')
        log_state.add_variable('stateEstimate.z',  'float')
        log_state.add_variable('stateEstimate.vx', 'float')
        log_state.add_variable('stateEstimate.vy', 'float')
        log_state.add_variable('stateEstimate.vz', 'float')

        log_att = LogConfig(name='Attitude', period_in_ms=LOG_PERIOD_MS)
        log_att.add_variable('stabilizer.roll',  'float')
        log_att.add_variable('stabilizer.pitch', 'float')
        log_att.add_variable('stabilizer.yaw',   'float')

        # Shared state dict updated by log callbacks
        drone_state = {
            'x': 0, 'y': 0, 'z': 0, 'vx': 0, 'vy': 0, 'vz': 0,
            'roll': 0, 'pitch': 0, 'yaw': 0,
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

        log_state.data_received_cb.add_callback(_state_cb)
        log_att.data_received_cb.add_callback(_att_cb)
        cf.log.add_config(log_state)
        cf.log.add_config(log_att)
        log_state.start()
        log_att.start()

        # ── Start OptiTrack streaming thread ──────────────────────────
        # The Kalman filter needs external pose packets BEFORE the reset
        # so that it converges to the true mocap origin instead of (0,0,0).
        mocap_stop  = threading.Event()
        mocap_shared = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'count': 0}
        mocap_thread = threading.Thread(
            target=_natnet_streamer,
            kwargs=dict(cf=cf,
                        server_ip=natnet_server_ip,
                        client_ip=natnet_client_ip,
                        use_multicast=natnet_multicast,
                        rigid_body_id=rigid_body_id,
                        stop_event=mocap_stop, shared=mocap_shared,
                        rate_hz=mocap_rate_hz),
            daemon=True)
        mocap_thread.start()

        # Wait until the thread has streamed a few frames (or bail out).
        t0 = time.time()
        while mocap_shared['count'] < 30 and not mocap_stop.is_set():
            if time.time() - t0 > 5.0:
                print("[REAL] ERROR: no OptiTrack frames received in 5s. "
                      "Check Motive streaming config, IP, and rigid-body name.")
                mocap_stop.set()
                mocap_thread.join(timeout=1.0)
                return
            time.sleep(0.05)
        start_xy = (mocap_shared['x'], mocap_shared['y'])
        print(f"[REAL] Mocap pose @ start: "
              f"({mocap_shared['x']:.3f}, {mocap_shared['y']:.3f}, {mocap_shared['z']:.3f})")
        print(f"[REAL] Trajectory origin set to start_xy=({start_xy[0]:+.3f}, {start_xy[1]:+.3f})")

        # ── Generate trajectory centered on the drone's actual start ──
        waypoints = generate_trajectory(CTRL_FREQ, duration_sec,
                                        hover_height=hover_height,
                                        radius=radius, start_xy=start_xy)

        # ── Reset Kalman estimator ────────────────────────────────────
        # With extpose packets already streaming, the reset re-initializes
        # the filter around the current mocap pose.
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
                # ─── Link health check ────────────────────────────
                if link_status['lost'] or link_status['disconnect']:
                    print(f"[REAL] Aborting flight loop — link lost "
                          f"(reason: {link_status['reason']}).")
                    break
                if (link_status['bad_since'] is not None
                        and (time.time() - link_status['bad_since'])
                            > LINK_BAD_ABORT_SEC):
                    print(f"[REAL] Aborting flight loop — link quality "
                          f"≤ {LINK_QUALITY_ABORT_PCT:.0f}% for "
                          f">{LINK_BAD_ABORT_SEC*1000:.0f} ms "
                          f"(now {link_status['quality']:.0f}%).")
                    break

                t_iter_wall = time.time()
                sp = waypoints[i]
                sp_pos = sp[0:3]
                sp_yaw_rate = sp[3]

                # Log setpoint for every control step
                real_sp_log.append({'t': time.time() - flight_start, 'pos': sp_pos.copy()})

                # ─── Position mode ────────────────────────────────
                # All PIDs run on the drone; mocap-fed Kalman supplies the state.
                # send_position_setpoint(x, y, z, yaw_deg)
                cf.commander.send_position_setpoint(
                    sp_pos[0], sp_pos[1], sp_pos[2], 0.0)

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
            # Skip the active descent if the radio link is already gone —
            # the firmware will hit its commander watchdog (~500 ms) on its
            # own; trying to send packets only triggers more error logs.
            link_dead = link_status['lost'] or link_status['disconnect']
            if link_dead:
                print(f"[REAL] Skipping smooth landing — link is down "
                      f"(reason: {link_status['reason']}). Firmware watchdog "
                      f"will idle the motors.")
            else:
                land_x = drone_state['x']
                land_y = drone_state['y']
                land_z = drone_state['z']
                land_duration = max(1.0, land_z / 0.3)  # descend at ~0.3 m/s
                land_freq = 20  # Hz — position setpoints don't need high rate
                land_steps = int(land_freq * land_duration)
                cutoff_z = 0.05  # m — cut motors below this height

                print(f"[REAL] Landing from z={land_z:.2f}m over "
                      f"{land_duration:.1f}s...")

                for j in range(land_steps):
                    if link_status['lost'] or link_status['disconnect']:
                        print("[REAL] Link died during landing — aborting "
                              "descent loop.")
                        break
                    frac = (j + 1) / land_steps
                    # Smooth cubic descent
                    z = land_z * (1.0 - (3 * frac**2 - 2 * frac**3))
                    if z < cutoff_z:
                        break
                    real_sp_log.append({'t': time.time() - flight_start,
                                        'pos': np.array([land_x, land_y, z])})
                    try:
                        cf.commander.send_position_setpoint(land_x, land_y, z, 0.0)
                    except Exception as e:
                        print(f"[REAL] send_position_setpoint failed: {e}")
                        break
                    time.sleep(1.0 / land_freq)

                # Notify end of setpoints — tells the firmware the PC-side
                # commander is stopping cleanly, so it idles the motors
                # gracefully instead of hard-locking on watchdog timeout.
                try:
                    cf.commander.send_notify_setpoint_stop()
                except Exception:
                    pass
                time.sleep(0.1)

            try:
                log_state.stop()
                log_att.stop()
            except Exception:
                pass

            # Stop OptiTrack streaming thread cleanly
            mocap_stop.set()
            mocap_thread.join(timeout=2.0)

            # ─── Link health summary ──────────────────────────
            print(f"[LINK] Summary — min_quality={link_status['min_quality']:.0f}%  "
                  f"low_quality_events={link_status['n_low_events']}  "
                  f"connection_lost={link_status['lost']}  "
                  f"disconnected={link_status['disconnect']}")
            if link_status['reason']:
                print(f"[LINK] Failure reason: {link_status['reason']}")

            print("[REAL] Landed and cleaned up.")

        # ─── Post-flight: run headless sim + comparison plot ──────────────
        if real_log:
            real_t   = np.array([e['t']   for e in real_log])
            real_pos = np.array([[e['x'],  e['y'],  e['z']]       for e in real_log])
            real_vel = np.array([[e['vx'], e['vy'], e['vz']]      for e in real_log])
            real_rpy = np.array([[e['roll'], e['pitch'], e['yaw']] for e in real_log])
            sp_t_arr = np.array([e['t']   for e in real_sp_log])
            sp_arr   = np.array([e['pos'] for e in real_sp_log])

            # Remove the trajectory's world-frame offset so the comparison
            # overlays the sim (which always takes off from (0,0,0.02)).
            if real_pos.size > 0:
                real_pos[:, 0] -= start_xy[0]
                real_pos[:, 1] -= start_xy[1]
            if sp_arr.size > 0:
                sp_arr[:, 0]   -= start_xy[0]
                sp_arr[:, 1]   -= start_xy[1]

            real_data = dict(t=real_t, pos=real_pos, vel=real_vel, rpy=real_rpy,
                             sp_t=sp_t_arr, sp=sp_arr)

            if t_wall_log:
                print("[REAL] Generating timing plot...")
                _plot_jitter(t_wall_log, t_nominal_log, CTRL_FREQ)

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
        description="Crazyflie firmware-faithful PID — sim & real drone (OptiTrack)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  sim        Full firmware PID in Python → RPMs → PyBullet           (default)
  position   OptiTrack → extpose → send_position_setpoint (all PIDs on drone)
        """)
    parser.add_argument('--mode', default='sim',
                        choices=['sim', 'position'],
                        help='Control mode (default: sim)')
    parser.add_argument('--duration', default=10, type=float,
                        help='Flight duration in seconds (default: 10)')
    parser.add_argument('--height', default=1.0, type=float,
                        help='Hover height in meters (default: 1.0)')
    parser.add_argument('--radius', default=0.5, type=float,
                        help='Circle radius in meters (default: 0.5)')
    parser.add_argument('--gui', default=True, type=lambda x: x.lower() == 'true',
                        help='PyBullet GUI (default: True)')
    parser.add_argument('--uri', default='radio://0/80/2M/E7E7E7E7E7',
                        help='Crazyflie radio URI (for real drone)')
    parser.add_argument('--flow-deck-sim', action='store_true',
                        help='Simulate Flow deck v2 noise+delay in sim mode')
    parser.add_argument('--push-gains', action='store_true',
                        help='Overwrite drone PID gains with Python constants before flying')
    parser.add_argument('--natnet-server-ip', default='192.168.0.24',
                        help='IP address of the Motive PC / NatNet server '
                             '(default: 192.168.0.24)')
    parser.add_argument('--natnet-client-ip', default='192.168.0.56', # 192.168.0.100 if ethernet, 192.168.0.56 if wifi on TPLink-A2BC
                        help='IP address of this PC on the OptiTrack network '
                             '(default: 192.168.0.56)')
    parser.add_argument('--natnet-unicast', action='store_true',
                        help='Use unicast instead of multicast (default: multicast)')
    parser.add_argument('--rigid-body-id', default=4, type=int,
                        help='Streaming ID of the Crazyflie rigid body in Motive '
                             '(default: 4)')
    parser.add_argument('--mocap-rate', default=None, type=float,
                        help='Throttle extpose packets to this rate [Hz] '
                             '(default: send every NatNet frame)')
    args = parser.parse_args()

    if args.mode == 'sim':
        run_sim(duration_sec=args.duration, gui=args.gui,
                hover_height=args.height, radius=args.radius,
                simulate_flow_deck=args.flow_deck_sim)
    else:
        run_real(uri=args.uri,
                 duration_sec=args.duration, hover_height=args.height,
                 radius=args.radius, push_gains=args.push_gains,
                 natnet_server_ip=args.natnet_server_ip,
                 natnet_client_ip=args.natnet_client_ip,
                 natnet_multicast=(not args.natnet_unicast),
                 rigid_body_id=args.rigid_body_id,
                 mocap_rate_hz=args.mocap_rate)


if __name__ == "__main__":
    main()