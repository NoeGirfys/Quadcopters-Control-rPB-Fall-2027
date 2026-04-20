#!/usr/bin/env python3
"""
NN Controller for Crazyflie — Simulation & Real Drone
======================================================

Uses a trained ConcurrentPolicyMLP (from train_nn_cf_pid.py) to control
the Crazyflie.  The NN replaces the position + velocity PIDs.  Only the
attitude + rate PIDs from the CF firmware remain in the loop.

Modes
-----
  sim       — NN + firmware attitude/rate PIDs → RPMs → PyBullet
  attitude  — NN on PC → send_setpoint(roll, pitch, yaw_rate, thrust) to drone

The NN was trained to regulate to the origin.  A coordinate shift makes
the NN fly to an arbitrary target position (default [0, 0, 1]).

Flight phases:
  1. PID takeoff   — position+velocity PID brings the drone near the target
  2. NN regulation — the trained NN takes over and holds position

Pipeline (NN phase, sim mode, per 500 Hz step):
  NN action (100 Hz) → Attitude PID (500 Hz) → Rate PID (500 Hz)
  → Power Distribution → PWM → RPM → Motor filter → PyBullet

Usage:
  cd training_regulation_simu_and_real
  python fly_nn_cf_pid.py --weights trained_cf_pid_T20_ch50_h64_ep100.pt
  python fly_nn_cf_pid.py --weights ... --mode attitude --uri radio://...
"""

import argparse
import math
import time

import numpy as np
import torch


import sys
import os

# Ajout du dossier parent (Semester_Project) au sys.path
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# gym-pybullet-drones imports
# ---------------------------------------------------------------------------
try:
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.utils import sync
    HAS_PYBULLET_DRONES = True
except ImportError:
    HAS_PYBULLET_DRONES = False

    def sync(i: int, start: float, dt: float) -> None:
        """Fallback sync when pybullet-drones is unavailable."""
        target = start + (i + 1) * dt
        remaining = target - time.time()
        if remaining > 0:
            time.sleep(remaining)

# ---------------------------------------------------------------------------
# cflib imports (only for real drone mode)
# ---------------------------------------------------------------------------
try:
    import cflib.crtp
    from cflib.crazyflie import Crazyflie
    from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
    HAS_CFLIB = True
except ImportError:
    HAS_CFLIB = False

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from crazyflie_firmware.firmware import (
    CrazyfliePositionController,
    CrazyflieAttitudeController,
    CrazyfliePowerDistribution,
    cap_angle,
)
from circle_comparison_simu_and_real.cf_firmware_pid_sim import (
    MotorDynamicsFilter,
    obs_to_firmware_state,
    pwm_to_rpm,
    push_pid_gains_to_drone,
)
from crazyflie_firmware.constants import (
    ATTITUDE_RATE,
    ATTITUDE_UPDATE_DT
)
from train_nn_cf_pid import (
    ConcurrentPolicyMLP,
    NN_FREQ,
    HOVER_THRUST_U16,
)


# ===================================================================
#  Load trained NN
# ===================================================================

def load_nn_policy(ckpt_path: str, device: str = "cpu") -> ConcurrentPolicyMLP:
    """Load a trained ConcurrentPolicyMLP from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    T = ckpt["T"]
    hidden = ckpt["hidden"]
    policy = ConcurrentPolicyMLP(T=T, hidden=hidden).to(device)
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    print(f"[NN] Loaded: T={T}, hidden={hidden}, "
          f"epochs={ckpt.get('epochs', '?')}, "
          f"t_chunk={ckpt.get('t_chunk', '?')}s")
    return policy


# ===================================================================
#  State extraction helpers
# ===================================================================

def obs_to_nn_state(obs: np.ndarray) -> np.ndarray:
    """Convert pybullet-drones observation (20,) to NN state (12,).

    NN state ordering (Goffin convention):
      [x, vx, y, vy, z, vz, phi, p, theta, q, psi, r]

    All in pybullet/physics convention (positive pitch = nose DOWN).
    Angles in radians, angular rates in rad/s (body frame).
    """
    pos = obs[0:3]
    rpy_rad = obs[7:10]    # pybullet convention — no pitch negation
    vel = obs[10:13]
    ang_v = obs[13:16]     # world frame [rad/s]

    # Rotate angular velocity to body frame: ω_body = Rᵀ · ω_world
    r, p, y = rpy_rad
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])
    gyro_body = R.T @ ang_v

    return np.array([
        pos[0], vel[0],  pos[1], vel[1],  pos[2], vel[2],
        rpy_rad[0], gyro_body[0],
        rpy_rad[1], gyro_body[1],
        rpy_rad[2], gyro_body[2],
    ], dtype=np.float32)


def drone_state_to_nn_state(ds: dict) -> np.ndarray:
    """Convert real drone telemetry dict to NN state (12,).

    The Crazyflie reports attitude in firmware convention
    (positive pitch = nose UP).  Pitch and pitch rate are negated
    to match the NN's pybullet convention (positive pitch = nose DOWN).
    """
    roll_rad = math.radians(ds['roll'])
    pitch_rad = -math.radians(ds['pitch'])    # firmware → pybullet
    yaw_rad = math.radians(ds['yaw'])

    p_rad = math.radians(ds.get('gyro_x', 0.0))
    q_rad = -math.radians(ds.get('gyro_y', 0.0))  # firmware → pybullet
    r_rad = math.radians(ds.get('gyro_z', 0.0))

    return np.array([
        ds['x'], ds['vx'],  ds['y'], ds['vy'],  ds['z'], ds['vz'],
        roll_rad, p_rad,  pitch_rad, q_rad,  yaw_rad, r_rad,
    ], dtype=np.float32)


def query_nn_chunk(policy: ConcurrentPolicyMLP, state_12: np.ndarray,
                   target_pos: np.ndarray, device: str = "cpu") -> np.ndarray:
    """Query the NN and return all T actions for the chunk.

    Matches the training rollout exactly: the NN is called once at the start
    of each T-step chunk and all T actions are used sequentially.

    Returns: (T, 4) array of [thrust_u16, roll_deg, pitch_deg, yaw_rate_deg/s]
    """
    T = policy.T

    # Shift positions so target becomes the origin (training convention)
    state_shifted = state_12.copy()
    state_shifted[0] -= target_pos[0]   # x
    state_shifted[2] -= target_pos[1]   # y
    state_shifted[4] -= target_pos[2]   # z

    s = torch.tensor(state_shifted, dtype=torch.float32,
                     device=device).unsqueeze(0)   # (1, 12)

    # Relative targets: all T targets are the same for regulation
    current_pos = s[:, [0, 2, 4]]                      # (1, 3)
    rel = (-current_pos).unsqueeze(1).expand(1, T, 3)  # (1, T, 3)

    with torch.no_grad():
        actions = policy(s, rel)   # (1, T, 4)

    return actions[0].cpu().numpy()   # (T, 4)


# ===================================================================
#  Shared trajectory helpers (identical profiles for sim & real)
# ===================================================================

def takeoff_ramp_z(target_z: float, elapsed: float, max_duration: float,
                   ramp_frac: float = 0.8) -> float:
    """Linear ramp used during Phase 0 (PID takeoff), identical sim & real.

    Reaches target_z at ramp_frac * max_duration, then stays there.
    """
    frac = min(1.0, elapsed / (max_duration * ramp_frac))
    return target_z * frac


def landing_ramp_z(start_z: float, elapsed: float, duration: float) -> float:
    """Smoothstep ramp used during landing, identical sim & real.

    Follows z = start_z * (1 - (3 f^2 - 2 f^3)) with f in [0, 1].
    """
    frac = min(1.0, max(0.0, elapsed / max(duration, 1e-6)))
    return start_z * (1.0 - (3.0 * frac ** 2 - 2.0 * frac ** 3))


# ===================================================================
#  SIM mode
# ===================================================================

def run_sim_nn(ckpt_path: str, takeoff_pos=(0, 0, 0.5), target_pos=(0, 0, 1),
               duration_sec: float = 15, takeoff_max_duration: float = 3.0,
               hover_duration: float = 0.0,
               land_duration: float = None,
               gui: bool = True, plot: bool = True,
               device: str = "cpu"):
    """Run NN controller in PyBullet simulation.

    Phase 0 (PID):   Position+Velocity PID climbs to takeoff_pos
                     using the same linear z-ramp as the real-drone mode
    Phase 1 (HOVER): PID holds takeoff_pos for hover_duration seconds
    Phase 2 (NN):    NN regulates to target_pos
    Phase 3 (LAND):  PID brings the drone down along the same smoothstep
                     profile used by the real-drone landing, so the two
                     trajectories match end-to-end.

    The attitude + rate PIDs run continuously at 500 Hz throughout,
    ensuring a smooth handover between phases.
    """
    if not HAS_PYBULLET_DRONES:
        print("ERROR: gym-pybullet-drones not found.")
        sys.exit(1)

    policy = load_nn_policy(ckpt_path, device)

    PYB_FREQ = 1000
    CTRL_FREQ = ATTITUDE_RATE          # 500 Hz
    nn_divider = CTRL_FREQ // NN_FREQ  # 5 (run NN every 5th step)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.02]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.DYN,
        pyb_freq=PYB_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    KF = env.KF
    HOVER_RPM = env.HOVER_RPM

    # ---- Controllers ------------------------------------------------
    pos_ctrl = CrazyfliePositionController()     # takeoff only
    att_ctrl = CrazyflieAttitudeController()     # continuous
    power = CrazyfliePowerDistribution()
    motor_filter = MotorDynamicsFilter(n_motors=4, tau=0.02, dt=1.0 / CTRL_FREQ)
    motor_filter.rpm = np.full(4, HOVER_RPM)

    nn_target = np.array(target_pos, dtype=np.float64)
    pid_target = np.array(takeoff_pos, dtype=np.float64)

    PHASE_SWITCH_RADIUS = 0.05   # m — switch to HOVER when within 5 cm of pid_target
    PHASE_SWITCH_MAX_SEC = takeoff_max_duration  # hard cap: switch anyway after takeoff_max_duration

    PHASE_PID = 0
    PHASE_HOVER = 1
    PHASE_NN = 2
    PHASE_LAND = 3
    current_phase = PHASE_PID
    hover_start_step: int = 0
    land_start_step: int = 0
    land_start_pos: np.ndarray = pid_target.copy()   # filled when LAND begins

    # Landing duration: same rule as real mode (z / 0.3 m/s, min 1 s) by default.
    # In sim we must pre-size the log arrays, so we estimate using nn_target[2]
    # (the expected end-of-NN altitude). The actual descent profile is computed
    # from the current pos when PHASE_LAND begins, identical to the real mode.
    if land_duration is None:
        land_duration_effective = max(1.0, float(nn_target[2]) / 0.3)
    else:
        land_duration_effective = max(0.0, float(land_duration))

    SETTLE_TIME = 1.0  # On ajoute 1.5s pour voir le drone se poser
    total_duration_sec = duration_sec + land_duration_effective + SETTLE_TIME

    print(f"[SIM] PYB={PYB_FREQ}Hz  CTRL={CTRL_FREQ}Hz  NN={NN_FREQ}Hz")
    print(f"[SIM] PID target={pid_target}  NN target={nn_target}  "
          f"Duration={duration_sec}s  Switch radius={PHASE_SWITCH_RADIUS}m  "
          f"Hover={hover_duration}s  Land={land_duration_effective:.1f}s")

    n_steps = int(CTRL_FREQ * total_duration_sec)

    # Step index at which the NN phase ends and landing starts
    land_start_step_planned = int(CTRL_FREQ * duration_sec)

    # ---- Logging ----------------------------------------------------
    log_t = np.zeros(n_steps)
    log_pos = np.zeros((n_steps, 3))
    log_vel = np.zeros((n_steps, 3))
    log_rpy = np.zeros((n_steps, 3))
    log_rpms = np.zeros((n_steps, 4))
    log_thrust = np.zeros(n_steps)
    log_nn_cmd = np.zeros((n_steps, 4))
    log_phase = np.zeros(n_steps, dtype=int)

    # --- NOUVEAU : Liste pour les temps d'inférence ---
    inference_times = []

    action = np.full((1, 4), HOVER_RPM)
    START = time.time()

    # Current high-level commands (updated at 100 Hz)
    thrust_cmd = float(HOVER_THRUST_U16)
    roll_des = 0.0
    pitch_des = 0.0
    yaw_rate_cmd = 0.0
    yaw_setpoint = 0.0

    for i in range(n_steps):
        obs, _, _, _, _ = env.step(action)

        # Extract firmware-convention state (for attitude/rate PIDs)
        pos, vel, rpy_deg, gyro_deg = obs_to_firmware_state(obs[0])

        # ---- Phase transition check ---------------------------------
        if current_phase == PHASE_PID:
            dist_to_pid_target = np.linalg.norm(pos - pid_target)
            hard_cap_reached = (i >= int(CTRL_FREQ * PHASE_SWITCH_MAX_SEC))
            if dist_to_pid_target <= PHASE_SWITCH_RADIUS or hard_cap_reached:
                reason = "proximity" if dist_to_pid_target <= PHASE_SWITCH_RADIUS else "timeout"
                if hover_duration > 0.0:
                    current_phase = PHASE_HOVER
                    hover_start_step = i
                    print(f"[SIM] HOVER start at t={i / CTRL_FREQ:.1f}s ({reason})  "
                          f"dist={dist_to_pid_target:.3f}m  "
                          f"pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}]  "
                          f"hover={hover_duration}s")
                else:
                    current_phase = PHASE_NN
                    print(f"[SIM] NN takeover at t={i / CTRL_FREQ:.1f}s ({reason})  "
                          f"dist={dist_to_pid_target:.3f}m  "
                          f"pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}]")
        elif current_phase == PHASE_HOVER:
            hover_elapsed = (i - hover_start_step) / CTRL_FREQ
            if hover_elapsed >= hover_duration:
                current_phase = PHASE_NN
                print(f"[SIM] NN takeover at t={i / CTRL_FREQ:.1f}s (hover done)  "
                      f"pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}]")
        elif current_phase == PHASE_NN:
            # Switch to landing once the requested flight duration has elapsed,
            # mirroring the real-mode behaviour (landing happens after the NN phase).
            if land_duration_effective > 0.0 and i >= land_start_step_planned:
                current_phase = PHASE_LAND
                land_start_step = i
                land_start_pos = pos.copy()
                print(f"[SIM] LAND start at t={i / CTRL_FREQ:.1f}s  "
                      f"pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}]  "
                      f"land={land_duration_effective:.1f}s")

        # ---- High-level controller (100 Hz) -------------------------
        if i % nn_divider == 0:
            if current_phase == PHASE_PID:
                # Phase PID: same linear z-ramp as the real-drone mode,
                # so position setpoints match end-to-end.
                elapsed_sec = i / CTRL_FREQ
                z_sp = takeoff_ramp_z(pid_target[2], elapsed_sec,
                                      takeoff_max_duration, ramp_frac=0.8)
                ramped_target = (pid_target[0], pid_target[1], z_sp)
                thrust_cmd, roll_des, pitch_des = pos_ctrl.update(
                    ramped_target, tuple(pos), tuple(vel), rpy_deg[2])
                yaw_rate_cmd = 0.0
            elif current_phase == PHASE_HOVER:
                # Phase HOVER: hold pid_target exactly (same as real mode's
                # send_position_setpoint(pid_target[0], pid_target[1], pid_target[2], 0)).
                thrust_cmd, roll_des, pitch_des = pos_ctrl.update(
                    tuple(pid_target), tuple(pos), tuple(vel), rpy_deg[2])
                yaw_rate_cmd = 0.0
            elif current_phase == PHASE_LAND:
                # Phase LAND: same smoothstep descent as the real-drone mode.
                land_elapsed = (i - land_start_step) / CTRL_FREQ
                z_sp = landing_ramp_z(land_start_pos[2], land_elapsed,
                                      land_duration_effective)
                if z_sp < 0.05:
                    z_sp = 0.05
                land_sp = (land_start_pos[0], land_start_pos[1], z_sp)
                thrust_cmd, roll_des, pitch_des = pos_ctrl.update(
                    land_sp, tuple(pos), tuple(vel), rpy_deg[2])
                yaw_rate_cmd = 0.0
            else:
                # Phase NN: NN controller — query at 100 Hz, use first action
                nn_state = obs_to_nn_state(obs[0])
                t_start = time.perf_counter()
                a = query_nn_chunk(policy, nn_state, nn_target, device)[0]
                t_end = time.perf_counter()
                inference_times.append(t_end - t_start)
                thrust_cmd, roll_des, pitch_des, yaw_rate_cmd = \
                    float(a[0]), float(a[1]), float(a[2]), float(a[3])

        # ---- Yaw setpoint accumulation (500 Hz) ---------------------
        yaw_setpoint = cap_angle(
            yaw_setpoint + yaw_rate_cmd * ATTITUDE_UPDATE_DT)

        # ---- Attitude PID (500 Hz) ----------------------------------
        roll_rate_d, pitch_rate_d, yaw_rate_d = att_ctrl.correct_attitude(
            rpy_deg[0], rpy_deg[1], rpy_deg[2],
            roll_des, pitch_des, yaw_setpoint)

        # ---- Rate PID (500 Hz) --------------------------------------
        roll_cmd, pitch_cmd, yaw_cmd = att_ctrl.correct_rate(
            gyro_deg[0], gyro_deg[1], gyro_deg[2],
            roll_rate_d, pitch_rate_d, yaw_rate_d)
        yaw_cmd = -yaw_cmd   # firmware negates yaw output

        # ---- Zero-thrust safety -------------------------------------
        if thrust_cmd <= 0:
            att_ctrl.reset_all(rpy_deg[0], rpy_deg[1], rpy_deg[2])
            yaw_setpoint = rpy_deg[2]
            action[0, :] = 0
            log_t[i] = i / CTRL_FREQ
            log_pos[i] = pos
            log_vel[i] = vel
            log_rpy[i] = rpy_deg
            continue

        # ---- Power distribution → RPMs ------------------------------
        motor_pwm = power.distribute(thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd)
        rpms = pwm_to_rpm(motor_pwm, KF, truncate_8bit=True)
        rpms = motor_filter.apply(rpms)
        action[0, :] = rpms

        # ---- Log ----------------------------------------------------
        log_t[i] = i / CTRL_FREQ
        log_pos[i] = pos
        log_vel[i] = vel
        log_rpy[i] = rpy_deg
        log_rpms[i] = rpms
        log_thrust[i] = thrust_cmd
        log_nn_cmd[i] = [thrust_cmd, roll_des, pitch_des, yaw_rate_cmd]
        log_phase[i] = current_phase

        #if i % CTRL_FREQ == 0:
        #    t = i / CTRL_FREQ
        #    phase_label = {PHASE_PID: "PID  ", PHASE_HOVER: "HOVER",
        #                   PHASE_NN: "NN   ", PHASE_LAND: "LAND "}[current_phase]
        #    if current_phase == PHASE_NN:
        #        current_target = nn_target
        #    elif current_phase == PHASE_LAND:
        #        current_target = np.array([land_start_pos[0], land_start_pos[1], 0.0])
        #    else:
        #        current_target = pid_target
        #    err = np.linalg.norm(pos - current_target)
        #    print(f"  t={t:5.1f}s [{phase_label}]  "
        #          f"pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:.3f}]  "
        #          f"err={err:.4f}m  thrust={thrust_cmd:.0f}")

        if gui:
            sync(i, START, 1.0 / CTRL_FREQ)

    env.close()
    print("[SIM] Done.")

    # --- NOUVEAU : Affichage des statistiques ---
    if inference_times:
        inf_ms = np.array(inference_times) * 1000.0  # Conversion en ms
        print("\n" + "="*40)
        print("[NN INFERENCE TIME STATS (100 Hz)]")
        print(f"  Min  : {np.min(inf_ms):.3f} ms")
        print(f"  Mean : {np.mean(inf_ms):.3f} ms")
        print(f"  Max  : {np.max(inf_ms):.3f} ms")
        print("="*40 + "\n")

    data = dict(t=log_t, pos=log_pos, vel=log_vel, rpy=log_rpy,
                rpms=log_rpms, thrust=log_thrust, nn_cmd=log_nn_cmd,
                phase=log_phase, pid_target=pid_target, nn_target=nn_target,
                PHASE_PID=PHASE_PID, PHASE_HOVER=PHASE_HOVER,
                PHASE_NN=PHASE_NN, PHASE_LAND=PHASE_LAND)
    if plot:
        _plot_nn_sim(data)
    return data


# ===================================================================
#  SIM plotting
# ===================================================================

def _plot_nn_sim(data: dict):
    """Plot NN simulation results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not found.")
        return

    t = data['t']
    pos = data['pos']
    vel = data['vel']
    rpy = data['rpy']
    rpms = data['rpms']
    thrust = data['thrust']
    nn_cmd = data['nn_cmd']
    pid_target = data['pid_target']
    nn_target = data['nn_target']
    phase = data['phase']
    PHASE_HOVER = data['PHASE_HOVER']
    PHASE_NN = data['PHASE_NN']
    PHASE_LAND = data.get('PHASE_LAND', -1)

    # Actual transition times from phase log
    hover_start_idx = np.argmax(phase == PHASE_HOVER) if np.any(phase == PHASE_HOVER) else None
    nn_start_idx = np.argmax(phase == PHASE_NN) if np.any(phase == PHASE_NN) else len(t) - 1
    nn_switch_sec = t[nn_start_idx]
    land_start_idx = np.argmax(phase == PHASE_LAND) if np.any(phase == PHASE_LAND) else None
    land_switch_sec = t[land_start_idx] if land_start_idx is not None else None

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle('NN Controller + CF Firmware Attitude/Rate PID', fontsize=14)

    # Phase boundary lines on all subplots
    for row in axes:
        for ax in row:
            if hover_start_idx is not None:
                ax.axvline(t[hover_start_idx], color='orange', ls=':', alpha=0.5,
                           label='PID\u2192HOVER')
            ax.axvline(nn_switch_sec, color='red', ls=':', alpha=0.5,
                       label='HOVER\u2192NN' if hover_start_idx is not None else 'PID\u2192NN')
            if land_switch_sec is not None:
                ax.axvline(land_switch_sec, color='purple', ls=':', alpha=0.5,
                           label='NN\u2192LAND')

    # -- Position --
    ax = axes[0, 0]
    ax.plot(t, pos[:, 0], label='x')
    ax.plot(t, pos[:, 1], label='y')
    ax.plot(t, pos[:, 2], label='z')
    for j, c in enumerate(['C0', 'C1', 'C2']):
        ax.axhline(pid_target[j], ls=':', color=c, alpha=0.35,
                   label=f'PID tgt {["x","y","z"][j]}={pid_target[j]:.2f}')
        ax.axhline(nn_target[j], ls='--', color=c, alpha=0.6,
                   label=f'NN tgt {["x","y","z"][j]}={nn_target[j]:.2f}')
    ax.set_ylabel('Position [m]')
    ax.legend(fontsize=6, ncol=3)
    ax.set_title('Position')
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

    # -- NN commands --
    ax = axes[1, 1]
    nn_mask = phase == PHASE_NN
    t_nn = t[nn_mask]
    ax.plot(t_nn, nn_cmd[nn_mask, 1], label='NN roll [deg]', alpha=0.8)
    ax.plot(t_nn, nn_cmd[nn_mask, 2], label='NN pitch [deg]', alpha=0.8)
    ax.plot(t_nn, nn_cmd[nn_mask, 3], label='NN yaw_rate [deg/s]', alpha=0.6)
    ax.set_ylabel('NN attitude cmd')
    ax.legend(fontsize=7)
    ax.set_title('NN Outputs (attitude)')
    ax.grid(True, alpha=0.3)

    # -- Thrust --
    ax = axes[2, 0]
    ax.plot(t, thrust, color='black', label='thrust')
    ax.axhline(HOVER_THRUST_U16, ls=':', color='gray',
               label=f'hover={HOVER_THRUST_U16:.0f}')
    ax.set_ylabel('Thrust [uint16]')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7)
    ax.set_title('Thrust')
    ax.grid(True, alpha=0.3)

    # -- RPMs --
    ax = axes[2, 1]
    for m in range(4):
        ax.plot(t, rpms[:, m], label=f'M{m + 1}', alpha=0.7)
    ax.set_ylabel('RPM')
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7, ncol=2)
    ax.set_title('Motor RPMs')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nn_sim_results.png', dpi=150)
    print("[PLOT] Saved to nn_sim_results.png")


# ===================================================================
#  REAL mode
# ===================================================================

def run_real_nn(ckpt_path: str, takeoff_pos=(0, 0, 0.5), target_pos=(0, 0, 1),
                uri: str = "radio://0/80/2M/E7E7E7E7E7",
                duration_sec: float = 15, takeoff_max_duration: float = 3.0,
                hover_duration: float = 0.0,
                land_duration: float = None,
                push_gains: bool = True,
                device: str = "cpu"):
    """Run NN controller on a real Crazyflie via Crazyradio.

    Phase 0 (PID):   send_position_setpoint climbs to takeoff_pos via a
                     linear z-ramp (shared helper with sim mode)
    Phase 1 (HOVER): holds takeoff_pos for hover_duration seconds
    Phase 2 (NN):    NN outputs attitude + thrust regulating to target_pos
    Phase 3 (LAND):  smoothstep descent to the ground (shared helper with
                     sim mode), using land_duration or ~0.3 m/s default.

    The drone's onboard attitude + rate PIDs handle low-level stabilisation.
    """
    if not HAS_CFLIB:
        print("ERROR: cflib not found.  pip install cflib")
        sys.exit(1)

    policy = load_nn_policy(ckpt_path, device)
    nn_target = np.array(target_pos, dtype=np.float64)
    pid_target = np.array(takeoff_pos, dtype=np.float64)

    cflib.crtp.init_drivers()
    print(f"[REAL] Mode: attitude  PID target: {pid_target}  NN target: {nn_target}  Hover: {hover_duration}s")
    print(f"[REAL] Connecting to {uri}...")

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        print("[REAL] Connected!")

        if push_gains:
            push_pid_gains_to_drone(cf)

        # ---- Flow deck settings -----
        #cf.param.set_value('motion.adaptive', '0')
        #cf.param.set_value('motion.flowStdFixed', '10.0')

        # ---- Logging setup -----
        from cflib.crazyflie.log import LogConfig

        log_state = LogConfig(name='State', period_in_ms=10)
        log_state.add_variable('stateEstimate.x', 'float')
        log_state.add_variable('stateEstimate.y', 'float')
        log_state.add_variable('stateEstimate.z', 'float')
        log_state.add_variable('stateEstimate.vx', 'float')
        log_state.add_variable('stateEstimate.vy', 'float')
        log_state.add_variable('stateEstimate.vz', 'float')

        log_att = LogConfig(name='Attitude', period_in_ms=10)
        log_att.add_variable('stabilizer.roll', 'float')
        log_att.add_variable('stabilizer.pitch', 'float')
        log_att.add_variable('stabilizer.yaw', 'float')
        log_att.add_variable('gyro.x', 'float')
        log_att.add_variable('gyro.y', 'float')
        log_att.add_variable('gyro.z', 'float')

        drone_state = {
            'x': 0.0, 'y': 0.0, 'z': 0.0,
            'vx': 0.0, 'vy': 0.0, 'vz': 0.0,
            'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
        }

        real_log = []
        flight_start = None
        elapsed_phase0 = 0.0
        elapsed_phases01 = 0.0

        def _state_cb(timestamp, data, logconf):
            drone_state['x'] = data['stateEstimate.x']
            drone_state['y'] = data['stateEstimate.y']
            drone_state['z'] = data['stateEstimate.z']
            drone_state['vx'] = data['stateEstimate.vx']
            drone_state['vy'] = data['stateEstimate.vy']
            drone_state['vz'] = data['stateEstimate.vz']
            if flight_start is not None:
                real_log.append({
                    't': time.time() - flight_start,
                    **{k: drone_state[k] for k in drone_state},
                })

        def _att_cb(timestamp, data, logconf):
            drone_state['roll'] = data['stabilizer.roll']
            drone_state['pitch'] = data['stabilizer.pitch']
            drone_state['yaw'] = data['stabilizer.yaw']
            drone_state['gyro_x'] = data['gyro.x']
            drone_state['gyro_y'] = data['gyro.y']
            drone_state['gyro_z'] = data['gyro.z']

        log_state.data_received_cb.add_callback(_state_cb)
        log_att.data_received_cb.add_callback(_att_cb)
        cf.log.add_config(log_state)
        cf.log.add_config(log_att)
        log_state.start()
        log_att.start()

        # ---- Reset Kalman estimator -----
        print("[REAL] Resetting Kalman estimator...")
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(2.0)
        print(f"[REAL] Estimator ready — pos=({drone_state['x']:.3f}, "
              f"{drone_state['y']:.3f}, {drone_state['z']:.3f})")

        # ---- Unlock commander -----
        for _ in range(50):
            cf.commander.send_setpoint(0, 0, 0, 0)
            time.sleep(0.02)
        print("[REAL] Commander unlocked.")

        # ---- Phase 0: PID takeoff via position setpoints -----
        TAKEOFF_FREQ = 20        # Hz — position setpoints don't need high rate
        PHASE_SWITCH_RADIUS = 0.05  # m — switch to HOVER/NN when within 5 cm

        START = time.time()
        flight_start = START

        try:
            print(f"[REAL] Phase 0: PID takeoff to {pid_target} "
                  f"(max {takeoff_max_duration}s, switch radius={PHASE_SWITCH_RADIUS}m)...")
            i = 0
            while True:
                elapsed = time.time() - START
                # Same linear z-ramp as the simulation (shared helper):
                # reaches pid_target[2] at 80% of takeoff_max_duration.
                z = takeoff_ramp_z(pid_target[2], elapsed,
                                   takeoff_max_duration, ramp_frac=0.8)
                cf.commander.send_position_setpoint(
                    pid_target[0], pid_target[1], z, 0.0)

                pos_now = np.array([drone_state['x'], drone_state['y'],
                                    drone_state['z']])
                dist = np.linalg.norm(pos_now - pid_target)
                hard_cap_reached = elapsed >= takeoff_max_duration

                if i % TAKEOFF_FREQ == 0:
                    print(f"  takeoff t={elapsed:.1f}s  "
                          f"pos=[{pos_now[0]:+.3f},{pos_now[1]:+.3f},{pos_now[2]:.3f}]  "
                          f"dist={dist:.3f}m  sp_z={z:.3f}")

                if dist <= PHASE_SWITCH_RADIUS or hard_cap_reached:
                    reason = "proximity" if dist <= PHASE_SWITCH_RADIUS else "timeout"
                    print(f"[REAL] PID done at t={elapsed:.1f}s ({reason})  "
                          f"dist={dist:.3f}m  "
                          f"pos=[{pos_now[0]:+.3f},{pos_now[1]:+.3f},{pos_now[2]:.3f}]")
                    break

                i += 1
                sync(i, START, 1.0 / TAKEOFF_FREQ)

            elapsed_phase0 = time.time() - START

            # ---- Phase 1: HOVER (hold position) -----
            if hover_duration > 0.0:
                print(f"[REAL] Phase 1: HOVER at {pid_target} for {hover_duration}s...")
                hover_steps = int(TAKEOFF_FREQ * hover_duration)
                hover_start = time.time()
                for j in range(hover_steps):
                    cf.commander.send_position_setpoint(
                        pid_target[0], pid_target[1], pid_target[2], 0.0)
                    if j % TAKEOFF_FREQ == 0:
                        pos_now = np.array([drone_state['x'], drone_state['y'],
                                            drone_state['z']])
                        print(f"  hover t={j / TAKEOFF_FREQ:.1f}s  "
                              f"pos=[{pos_now[0]:+.3f},{pos_now[1]:+.3f},{pos_now[2]:.3f}]")
                    sync(j, hover_start, 1.0 / TAKEOFF_FREQ)
                print("[REAL] Hover done.")

            # ---- Phase 2: NN control -----
            elapsed_phases01 = time.time() - START
            nn_duration = max(0.0, duration_sec - elapsed_phases01)
            nn_steps = int(NN_FREQ * nn_duration)
            print(f"[REAL] Phase 2: NN control to {nn_target} for "
                  f"{nn_duration:.0f}s...")

            phase2_start = time.time()
            for i in range(nn_steps):
                nn_state = drone_state_to_nn_state(drone_state)
                thrust, roll_deg, pitch_deg, yaw_rate = \
                    query_nn_chunk(policy, nn_state, nn_target, device)[0]

                # send_setpoint: (roll_deg, pitch_deg, yaw_rate_deg/s,
                #                 thrust_uint16)
                cf.commander.send_setpoint(
                    roll_deg, -pitch_deg, yaw_rate, int(thrust))

                if i % NN_FREQ == 0:
                    t = elapsed_phases01 + i / NN_FREQ
                    err = math.sqrt(
                        (drone_state['x'] - nn_target[0]) ** 2
                        + (drone_state['y'] - nn_target[1]) ** 2
                        + (drone_state['z'] - nn_target[2]) ** 2)
                    print(f"  t={t:5.1f}s [NN]  "
                          f"pos=[{drone_state['x']:+.3f},"
                          f"{drone_state['y']:+.3f},"
                          f"{drone_state['z']:.3f}]  "
                          f"err={err:.4f}m  thrust={thrust:.0f}  "
                          f"r={roll_deg:+.1f} p={pitch_deg:+.1f}")

                sync(i, phase2_start, 1.0 / NN_FREQ)

        except KeyboardInterrupt:
            print("\n[REAL] Interrupted!")
        finally:
            # ---- Smooth landing (same smoothstep profile as sim) -----
            print("[REAL] Landing...")
            land_x = drone_state['x']
            land_y = drone_state['y']
            land_z = drone_state['z']
            # Same default rule as sim: ~0.3 m/s descent, min 1 s.
            if land_duration is None:
                land_duration_effective = max(1.0, land_z / 0.3)
            else:
                land_duration_effective = max(0.0, float(land_duration))
            LAND_FREQ = 20
            land_steps = int(LAND_FREQ * land_duration_effective)
            land_start_t = time.time()

            for j in range(land_steps):
                land_elapsed = time.time() - land_start_t
                z = landing_ramp_z(land_z, land_elapsed, land_duration_effective)
                if z < 0.05:
                    break
                cf.commander.send_position_setpoint(
                    land_x, land_y, z, 0.0)
                time.sleep(1.0 / LAND_FREQ)

            cf.commander.send_notify_setpoint_stop()
            time.sleep(0.1)
            log_state.stop()
            log_att.stop()
            print("[REAL] Landed.")

        # ---- Post-flight plot -----
        if real_log:
            _plot_real_nn(real_log, pid_target, nn_target,
                          elapsed_phase0, elapsed_phases01)


# ===================================================================
#  REAL plotting
# ===================================================================

def _plot_real_nn(real_log: list, pid_target: np.ndarray, nn_target: np.ndarray,
                  elapsed_phase0: float, elapsed_phases01: float):
    """Plot real drone NN control results."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not found.")
        return

    t = np.array([e['t'] for e in real_log])
    pos = np.array([[e['x'], e['y'], e['z']] for e in real_log])
    vel = np.array([[e['vx'], e['vy'], e['vz']] for e in real_log])
    rpy = np.array([[e['roll'], e['pitch'], e['yaw']] for e in real_log])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle('NN Controller — Real Drone', fontsize=14)

    for ax in axes.flat:
        if elapsed_phases01 > elapsed_phase0:
            ax.axvline(elapsed_phase0, color='orange', ls=':', alpha=0.5,
                       label='PID\u2192HOVER')
        ax.axvline(elapsed_phases01, color='red', ls=':', alpha=0.5,
                   label='HOVER\u2192NN' if elapsed_phases01 > elapsed_phase0 else 'PID\u2192NN')

    ax = axes[0, 0]
    ax.plot(t, pos[:, 0], label='x')
    ax.plot(t, pos[:, 1], label='y')
    ax.plot(t, pos[:, 2], label='z')
    for j, c in enumerate(['C0', 'C1', 'C2']):
        ax.axhline(pid_target[j], ls=':', color=c, alpha=0.35,
                   label=f'PID tgt {["x","y","z"][j]}={pid_target[j]:.2f}')
        ax.axhline(nn_target[j], ls='--', color=c, alpha=0.6,
                   label=f'NN tgt {["x","y","z"][j]}={nn_target[j]:.2f}')
    ax.set_ylabel('Position [m]')
    ax.legend(fontsize=6)
    ax.set_title('Position')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, vel[:, 0], label='vx')
    ax.plot(t, vel[:, 1], label='vy')
    ax.plot(t, vel[:, 2], label='vz')
    ax.set_ylabel('Velocity [m/s]')
    ax.legend(fontsize=7)
    ax.set_title('Velocity')
    ax.grid(True, alpha=0.3)

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
    ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7)
    ax.set_title('Yaw')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('nn_real_results.png', dpi=150)
    print("[PLOT] Saved to nn_real_results.png")
    #plt.show()


# ===================================================================
#  CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="NN controller for Crazyflie — sim & real drone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
MODES:
  sim       NN + firmware att/rate PID -> RPMs -> PyBullet  (default)
  attitude  NN on PC -> send_setpoint(roll, pitch, yaw_rate, thrust)
        """)
    parser.add_argument('--weights', required=True,
                        help='Path to trained NN checkpoint (.pt)')
    parser.add_argument('--mode', default='sim',
                        choices=['sim', 'attitude'],
                        help='Control mode (default: sim)')
    parser.add_argument('--duration', default=10, type=float,
                        help='Total flight duration [s] (default: 10)')
    parser.add_argument('--takeoff-max-duration', default=1.0, type=float,
                        help='PID takeoff duration [s] (default: 1.0)')
    parser.add_argument('--PID-target', nargs=3, type=float,
                    default=[0.0, 0.0, 0.7],
                    metavar=('X', 'Y', 'Z'),
                    help='PID takeoff target [x y z] en mètres (default: 0 0 0.7)')
    parser.add_argument('--NN-target', nargs=3, type=float,
                    default=[0.0, 0.0, 1.0],
                    metavar=('X', 'Y', 'Z'),
                    help='NN target [x y z] en mètres (default: 0 0 1)')
    parser.add_argument('--gui', default=True,
                        type=lambda x: x.lower() == 'true',
                        help='PyBullet GUI (default: True)')
    parser.add_argument('--uri', default='radio://0/80/2M/E7E7E7E7E7',
                        help='Crazyflie radio URI (for real mode)')
    parser.add_argument('--hover-duration', default=2.0, type=float,
                        help='Hover duration between PID and NN phases [s] (default: 2.0)')
    parser.add_argument('--land-duration', default=None, type=float,
                        help='Landing duration [s] (default: None). '
                             'Used identically in sim and real modes.')
    parser.add_argument('--no-push-gains', dest='push_gains', action='store_false',
                        help='Skip pushing finetuned PID gains to the drone (real mode only)')
    parser.add_argument('--no-plot', dest='plot', action='store_false',
                        help='Disable post-flight plots')
    args = parser.parse_args()

    takeoff_pos = tuple(args.PID_target)
    target_pos  = tuple(args.NN_target)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.mode == 'sim':
        run_sim_nn(args.weights, takeoff_pos=takeoff_pos, target_pos=target_pos,
                   duration_sec=args.duration, takeoff_max_duration=args.takeoff_max_duration,
                   hover_duration=args.hover_duration,
                   land_duration=args.land_duration,
                   gui=args.gui, plot=args.plot, device=device)
    elif args.mode == 'attitude':
        run_real_nn(args.weights, takeoff_pos=takeoff_pos, target_pos=target_pos,
                    uri=args.uri,
                    duration_sec=args.duration, takeoff_max_duration=args.takeoff_max_duration,
                    hover_duration=args.hover_duration,
                    land_duration=args.land_duration,
                    push_gains=args.push_gains,
                    device=device)


if __name__ == "__main__":
    main()