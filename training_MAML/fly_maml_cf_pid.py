#!/usr/bin/env python3
"""
MAML NN Controller for a (weighted) Crazyflie — Simulation & Real Drone
=======================================================================

Companion of ``training_regulation_simu_and_real/fly_nn_cf_pid.py``, but for
the offset-mass MAML controller (``maml_lib.PolicyMLP``, single-step T=1). The
network replaces the position + velocity PIDs; the firmware attitude + rate
PIDs remain in the loop. Before flying, the policy is adapted few-shot to the
test payload (the lested mass), using the K-shot budget, inner learning rate
and grad-clip stored in the MAML checkpoint.

Flight phases (identical to §5):
  0. PID takeoff   — position+velocity PID climbs to ``takeoff_pos``
  1. HOVER         — PID holds ``takeoff_pos``
  2. NN regulation — the adapted policy flies to ``NN_target`` and holds it
  3. PID landing   — smoothstep descent to the ground

Modes
-----
  sim       — adapted NN + firmware att/rate PIDs → RPMs → PyBullet
              (Physics.DYN_OFFSET, the lested mass set via set_offset_mass)
  attitude  — adapted NN on PC → send_setpoint(roll, pitch, yaw_rate, thrust)
              to the *physically lested* real drone

Controllers (``--controller``) --- the same four compared in simulation:
  pid         — pure firmware: the NN phase is flown by the position controller
                too (no network), as a reference flight
  base        — a jointly-trained baseline (``--baseline-ckpt``), deployed as-is
  base-adapt  — the same baseline, few-shot adapted to the payload
  maml        — the MAML meta-init, few-shot adapted to the payload (default)

All logs are saved to flights/<prefix>_<timestamp>.npz so the figures can be
regenerated offline, and the same plots as §5 are produced.

Usage:
  cd training_MAML
  # simulate a 10 g lested flight with the adapted MAML controller
  python fly_maml_cf_pid.py --maml-ckpt maml_..._offdirmag_ep500.pt --mass 10
  # real lested drone
  python fly_maml_cf_pid.py --maml-ckpt maml_..._offdirmag_ep500.pt --mass 10 \\
      --mode attitude --uri radio://0/80/2M/E7E7E7E7E7
  # baseline (adapted) reference flight, same payload
  python fly_maml_cf_pid.py --maml-ckpt maml_... --baseline-ckpt baseline_... \\
      --controller base --mass 10
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# gym-pybullet-drones (sim mode)
# ---------------------------------------------------------------------------
try:
    from gym_pybullet_drones.utils.enums import DroneModel, Physics
    from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
    from gym_pybullet_drones.utils.utils import sync
    HAS_PYBULLET_DRONES = True
except ImportError:
    HAS_PYBULLET_DRONES = False

    def sync(i, start, dt):
        target = start + (i + 1) * dt
        remaining = target - time.time()
        if remaining > 0:
            time.sleep(remaining)

# ---------------------------------------------------------------------------
# cflib (real mode)
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
try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

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
from crazyflie_firmware.constants import ATTITUDE_RATE, ATTITUDE_UPDATE_DT

from maml_lib import (
    config as C,
    PolicyMLP, compute_mass_params_batched, maml_adapt, DYNAMICS,
)

NN_FREQ = C.NN_FREQ


# ===================================================================
#  State extraction (identical to fly_nn_cf_pid)
# ===================================================================

def obs_to_nn_state(obs: np.ndarray) -> np.ndarray:
    """pybullet obs (20,) -> NN state (12,), Goffin order, radians, body rates."""
    pos = obs[0:3]
    rpy_rad = obs[7:10]
    vel = obs[10:13]
    ang_v = obs[13:16]
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
    """Real drone telemetry -> NN state (12,). Pitch/pitch-rate negated
    (firmware nose-up -> pybullet nose-down)."""
    roll_rad = math.radians(ds['roll'])
    pitch_rad = -math.radians(ds['pitch'])
    yaw_rad = math.radians(ds['yaw'])
    p_rad = math.radians(ds.get('gyro_x', 0.0))
    q_rad = -math.radians(ds.get('gyro_y', 0.0))
    r_rad = math.radians(ds.get('gyro_z', 0.0))
    return np.array([
        ds['x'], ds['vx'],  ds['y'], ds['vy'],  ds['z'], ds['vz'],
        roll_rad, p_rad,  pitch_rad, q_rad,  yaw_rad, r_rad,
    ], dtype=np.float32)


def query_maml_nn(policy, theta, state_12, target_pos, device="cpu") -> np.ndarray:
    """One single-step (T=1) NN query with adapted parameters -> (4,) action
    [thrust_u16, roll_deg, pitch_deg, yaw_rate_deg/s]. Regulation via the
    coordinate shift: target -> origin."""
    s = state_12.copy()
    s[0] -= target_pos[0]; s[2] -= target_pos[1]; s[4] -= target_pos[2]
    st = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)  # (1,12)
    rel = -st[:, [0, 2, 4]]                                                # (1,3)
    with torch.no_grad():
        a = functional_call(policy, theta, (st, rel))                      # (1,4)
    return a[0].cpu().numpy()


# ===================================================================
#  Shared trajectory profiles (identical to fly_nn_cf_pid)
# ===================================================================

def takeoff_ramp_z(target_z, elapsed, max_duration, ramp_frac=0.8):
    frac = min(1.0, elapsed / (max_duration * ramp_frac))
    return target_z * frac


def landing_ramp_z(start_z, elapsed, duration):
    frac = min(1.0, max(0.0, elapsed / max(duration, 1e-6)))
    return start_z * (1.0 - (3.0 * frac ** 2 - 2.0 * frac ** 3))


# ===================================================================
#  Raw-log persistence
# ===================================================================

FLIGHTS_DIR = os.path.join(SCRIPT_DIR, "flights")


def _save_flight_npz(arrays: dict, prefix: str) -> str:
    from datetime import datetime
    os.makedirs(FLIGHTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(FLIGHTS_DIR, f"{prefix}_{ts}.npz")
    np.savez_compressed(path, **arrays)
    print(f"[SAVE] Flight log -> {path}")
    return path


# ===================================================================
#  Load policy + few-shot adapt to the payload
# ===================================================================

def hover_u16_for_mass(m_extra: float) -> float:
    M_total = C.M_BASE + m_extra
    return (M_total * C.G * C.UINT16_MAX) / (4 * C.CF2_THRUST_MAX_PER_MOTOR)


def load_and_adapt(args, device: str):
    """Load the chosen policy and few-shot adapt it to the (centred/offset)
    test payload. Returns (policy, theta, m_extra, r_offset, hover_u16, info).

    For ``--controller pid`` no network is used; (policy, theta) are None.
    """
    maml_ckpt = torch.load(args.maml_ckpt, map_location=device, weights_only=False)
    A = maml_ckpt["args"]
    ts = maml_ckpt["task_set"]
    hidden = int(A.get("hidden", 64))

    m_extra = args.mass * 1e-3
    r_offset = (args.offset_dx, args.offset_dy, args.offset_dz)
    hover_u16 = hover_u16_for_mass(m_extra)
    centred = abs(args.offset_dx) + abs(args.offset_dy) + abs(args.offset_dz) < 1e-9
    info = dict(controller=args.controller, mass_g=args.mass, r_offset=r_offset,
                adapted=False, hidden=hidden)

    if args.controller == "pid":
        print(f"[CTRL] firmware PID reference (no NN)  payload={args.mass:.1f} g "
              f"({'centred' if centred else r_offset})")
        return None, None, m_extra, r_offset, hover_u16, info

    # which checkpoint provides the weights
    if args.controller in ("base", "base-adapt"):
        if not args.baseline_ckpt:
            raise SystemExit("--controller base/base-adapt requires --baseline-ckpt")
        ck = torch.load(args.baseline_ckpt, map_location=device, weights_only=False)
        src = args.baseline_ckpt
    else:  # maml
        ck = maml_ckpt
        src = args.maml_ckpt

    policy = PolicyMLP(hidden=hidden).to(device)
    policy.load_state_dict(ck["state_dict"])
    policy.eval()
    print(f"[CTRL] {args.controller} controller from {os.path.basename(src)}  "
          f"payload={args.mass:.1f} g ({'centred' if centred else r_offset})")

    # parameter dict of the un-adapted policy (used by 'base' and '--no-adapt')
    theta = {n: p.detach().clone() for n, p in policy.named_parameters()}

    # 'maml' and 'base-adapt' are adapted; 'base' is deployed as-is.
    needs_adapt = args.controller in ("maml", "base-adapt")
    if needs_adapt and args.adapt:
        K = int(A.get("k_samples", 5) or 5)
        n_inner = int(A.get("n_inner_steps", 1))
        lr_inner = float(A.get("lr_inner", 0.05))
        clip = float(A.get("inner_grad_clip", 1.0))
        term_w = float(A.get("terminal_weight", 50.0))
        pos_w = float(A.get("pos_weight", 10.0))
        z_w = float(A.get("z_weight", 1.0))
        n_steps = int(float(A.get("t_sim", 3.0)) * C.NN_FREQ)
        tau = (A.get("tau_end") or A.get("tau_div") or None)
        half_side = float(ts.get("half_side", 0.2))
        seed = int(A.get("seed", 0))
        ons_scale = float(A.get("obs_noise_scale", 0.0))
        ons = (C.OBS_NOISE_STD * ons_scale) if ons_scale > 0 else None
        dyn = DYNAMICS[A.get("dynamics", "nonlinear")]

        rng = np.random.default_rng(seed)
        x0 = torch.zeros(K, 12)
        p = torch.from_numpy(rng.uniform(-half_side, half_side, (K, 3)).astype(np.float32))
        x0[:, 0], x0[:, 2], x0[:, 4] = p[:, 0], p[:, 1], p[:, 2]
        mass = compute_mass_params_batched(
            np.full(K, m_extra, dtype=np.float32),
            np.tile(np.asarray(r_offset, dtype=np.float32), (K, 1)))

        print(f"[Adapt] K={K}  n_inner={n_inner}  lr_inner={lr_inner}  "
              f"clip={clip}  noise_scale={ons_scale}  (support sampled at the payload)")
        theta, losses = maml_adapt(
            policy, mass.to(device), x0.to(device), dynamics_step=dyn,
            n_steps=n_steps, n_steps_adapt=n_inner, lr_inner=lr_inner,
            terminal_weight=term_w, pos_weight=pos_w, z_weight=z_w,
            obs_noise_std=ons, tau_div=tau, inner_grad_clip=clip,
            device=device, seed=seed, verbose=True)
        info["adapted"] = True
        info["adapt_losses"] = np.asarray(losses)

    return policy, theta, m_extra, r_offset, hover_u16, info


# ===================================================================
#  Phase constants
# ===================================================================

PHASE_PID, PHASE_HOVER, PHASE_NN, PHASE_LAND = 0, 1, 2, 3


def _outer_command(controller, policy, theta, nn_state, nn_target, pos_ctrl,
                   pos, vel, yaw_deg, device):
    """Outer-loop command for the NN phase: NN query (maml/base) or position
    PID to the NN target (pid). Returns (thrust, roll, pitch, yaw_rate)."""
    if controller == "pid":
        thrust, roll, pitch = pos_ctrl.update(tuple(nn_target), tuple(pos),
                                              tuple(vel), yaw_deg)
        return thrust, roll, pitch, 0.0
    a = query_maml_nn(policy, theta, nn_state, nn_target, device)
    return float(a[0]), float(a[1]), float(a[2]), float(a[3])


# ===================================================================
#  SIM mode (Physics.DYN_OFFSET with the lested mass)
# ===================================================================

def run_sim(args, policy, theta, m_extra, r_offset, hover_u16, info, device):
    if not HAS_PYBULLET_DRONES:
        print("ERROR: gym-pybullet-drones not found."); sys.exit(1)

    takeoff_pos = np.array(args.PID_target, dtype=np.float64)
    nn_target = np.array(args.NN_target, dtype=np.float64)

    PYB_FREQ = 1000
    CTRL_FREQ = ATTITUDE_RATE
    nn_divider = CTRL_FREQ // NN_FREQ

    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.02]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=Physics.DYN_OFFSET, pyb_freq=PYB_FREQ, ctrl_freq=CTRL_FREQ,
        gui=args.gui, record=False, obstacles=False, user_debug_gui=False)
    env.set_offset_mass(m_extra, r_offset)
    KF = env.KF
    M_total = C.M_BASE + m_extra
    HOVER_RPM = math.sqrt(M_total * C.G / (4 * C.KF))

    pos_ctrl = CrazyfliePositionController()
    att_ctrl = CrazyflieAttitudeController()
    power = CrazyfliePowerDistribution()
    motor_filter = MotorDynamicsFilter(n_motors=4, tau=0.02, dt=1.0 / CTRL_FREQ)
    motor_filter.rpm = np.full(4, HOVER_RPM)

    takeoff_max = args.takeoff_max_duration
    hover_duration = args.hover_duration
    if args.land_duration is None:
        land_dur = max(1.0, float(nn_target[2]) / 0.3)
    else:
        land_dur = max(0.0, float(args.land_duration))
    SETTLE = 1.0
    total = args.duration + land_dur + SETTLE
    n_steps = int(CTRL_FREQ * total)
    land_start_planned = int(CTRL_FREQ * args.duration)

    print(f"[SIM] payload={args.mass:.1f}g  PID tgt={takeoff_pos}  NN tgt={nn_target}  "
          f"hover={hover_duration}s  land={land_dur:.1f}s  hover_u16={hover_u16:.0f}")

    log_t = np.zeros(n_steps); log_pos = np.zeros((n_steps, 3))
    log_vel = np.zeros((n_steps, 3)); log_rpy = np.zeros((n_steps, 3))
    log_rpms = np.zeros((n_steps, 4)); log_thrust = np.zeros(n_steps)
    log_nn_cmd = np.zeros((n_steps, 4)); log_phase = np.zeros(n_steps, dtype=int)

    action = np.full((1, 4), HOVER_RPM)
    phase = PHASE_PID
    hover_start = land_start = 0
    land_pos = takeoff_pos.copy()
    thrust_cmd, roll_des, pitch_des, yaw_rate_cmd, yaw_sp = hover_u16, 0.0, 0.0, 0.0, 0.0
    START = time.time()

    for i in range(n_steps):
        obs, _, _, _, _ = env.step(action)
        pos, vel, rpy_deg, gyro_deg = obs_to_firmware_state(obs[0])

        # phase transitions
        if phase == PHASE_PID:
            d = np.linalg.norm(pos - takeoff_pos)
            if d <= 0.05 or i >= int(CTRL_FREQ * takeoff_max):
                if hover_duration > 0:
                    phase, hover_start = PHASE_HOVER, i
                else:
                    phase = PHASE_NN
                print(f"[SIM] PID->{'HOVER' if hover_duration>0 else 'NN'} "
                      f"t={i/CTRL_FREQ:.1f}s pos={np.round(pos,3)}")
        elif phase == PHASE_HOVER:
            if (i - hover_start) / CTRL_FREQ >= hover_duration:
                phase = PHASE_NN
                print(f"[SIM] HOVER->NN t={i/CTRL_FREQ:.1f}s")
        elif phase == PHASE_NN:
            if land_dur > 0 and i >= land_start_planned:
                phase, land_start, land_pos = PHASE_LAND, i, pos.copy()
                print(f"[SIM] NN->LAND t={i/CTRL_FREQ:.1f}s pos={np.round(pos,3)}")

        # outer loop @ 100 Hz
        if i % nn_divider == 0:
            if phase == PHASE_PID:
                z_sp = takeoff_ramp_z(takeoff_pos[2], i / CTRL_FREQ, takeoff_max)
                thrust_cmd, roll_des, pitch_des = pos_ctrl.update(
                    (takeoff_pos[0], takeoff_pos[1], z_sp), tuple(pos), tuple(vel), rpy_deg[2])
                yaw_rate_cmd = 0.0
            elif phase == PHASE_HOVER:
                thrust_cmd, roll_des, pitch_des = pos_ctrl.update(
                    tuple(takeoff_pos), tuple(pos), tuple(vel), rpy_deg[2])
                yaw_rate_cmd = 0.0
            elif phase == PHASE_LAND:
                z_sp = max(0.05, landing_ramp_z(land_pos[2], (i - land_start) / CTRL_FREQ, land_dur))
                thrust_cmd, roll_des, pitch_des = pos_ctrl.update(
                    (land_pos[0], land_pos[1], z_sp), tuple(pos), tuple(vel), rpy_deg[2])
                yaw_rate_cmd = 0.0
            else:  # PHASE_NN
                nn_state = obs_to_nn_state(obs[0])
                thrust_cmd, roll_des, pitch_des, yaw_rate_cmd = _outer_command(
                    args.controller, policy, theta, nn_state, nn_target,
                    pos_ctrl, pos, vel, rpy_deg[2], device)

        yaw_sp = cap_angle(yaw_sp + yaw_rate_cmd * ATTITUDE_UPDATE_DT)
        roll_rate_d, pitch_rate_d, yaw_rate_d = att_ctrl.correct_attitude(
            rpy_deg[0], rpy_deg[1], rpy_deg[2], roll_des, pitch_des, yaw_sp)
        roll_cmd, pitch_cmd, yaw_cmd = att_ctrl.correct_rate(
            gyro_deg[0], gyro_deg[1], gyro_deg[2], roll_rate_d, pitch_rate_d, yaw_rate_d)
        yaw_cmd = -yaw_cmd

        if thrust_cmd <= 0:
            att_ctrl.reset_all(rpy_deg[0], rpy_deg[1], rpy_deg[2]); yaw_sp = rpy_deg[2]
            action[0, :] = 0
            log_t[i], log_pos[i], log_vel[i], log_rpy[i] = i / CTRL_FREQ, pos, vel, rpy_deg
            log_phase[i] = phase
            continue

        motor_pwm = power.distribute(thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd)
        rpms = motor_filter.apply(pwm_to_rpm(motor_pwm, KF, truncate_8bit=True))
        action[0, :] = rpms

        log_t[i] = i / CTRL_FREQ; log_pos[i] = pos; log_vel[i] = vel; log_rpy[i] = rpy_deg
        log_rpms[i] = rpms; log_thrust[i] = thrust_cmd
        log_nn_cmd[i] = [thrust_cmd, roll_des, pitch_des, yaw_rate_cmd]; log_phase[i] = phase
        if args.gui:
            sync(i, START, 1.0 / CTRL_FREQ)

    env.close()
    print("[SIM] Done.")

    data = dict(t=log_t, pos=log_pos, vel=log_vel, rpy=log_rpy, rpms=log_rpms,
                thrust=log_thrust, nn_cmd=log_nn_cmd, phase=log_phase,
                pid_target=takeoff_pos, nn_target=nn_target, hover_u16=hover_u16,
                mass_g=args.mass, controller=args.controller,
                PHASE_PID=PHASE_PID, PHASE_HOVER=PHASE_HOVER,
                PHASE_NN=PHASE_NN, PHASE_LAND=PHASE_LAND)
    _save_flight_npz(data, f"maml_sim_{args.controller}_{args.mass:.0f}g")
    if args.plot:
        _plot_sim(data)
    return data


# ===================================================================
#  REAL mode (attitude setpoints to the physically lested drone)
# ===================================================================

def run_real(args, policy, theta, m_extra, r_offset, hover_u16, info, device):
    if not HAS_CFLIB:
        print("ERROR: cflib not found. pip install cflib"); sys.exit(1)

    takeoff_pos = np.array(args.PID_target, dtype=np.float64)
    nn_target = np.array(args.NN_target, dtype=np.float64)
    is_pid = (args.controller == "pid")

    cflib.crtp.init_drivers()
    print(f"[REAL] payload={args.mass:.1f}g (lested)  controller={args.controller}  "
          f"PID tgt={takeoff_pos}  NN tgt={nn_target}")
    print(f"[REAL] Connecting to {args.uri}...")

    with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        print("[REAL] Connected!")
        if args.push_gains:
            push_pid_gains_to_drone(cf)

        from cflib.crazyflie.log import LogConfig
        log_state = LogConfig(name='State', period_in_ms=10)
        for v in ('x', 'y', 'z', 'vx', 'vy', 'vz'):
            log_state.add_variable(f'stateEstimate.{v}', 'float')
        log_att = LogConfig(name='Attitude', period_in_ms=10)
        for v in ('roll', 'pitch', 'yaw'):
            log_att.add_variable(f'stabilizer.{v}', 'float')
        for v in ('x', 'y', 'z'):
            log_att.add_variable(f'gyro.{v}', 'float')

        ds = {k: 0.0 for k in ('x', 'y', 'z', 'vx', 'vy', 'vz',
                               'roll', 'pitch', 'yaw', 'gyro_x', 'gyro_y', 'gyro_z')}
        real_log, nn_cmd_log = [], []
        flight_start = [None]

        def _state_cb(ts, d, lc):
            ds['x'], ds['y'], ds['z'] = d['stateEstimate.x'], d['stateEstimate.y'], d['stateEstimate.z']
            ds['vx'], ds['vy'], ds['vz'] = d['stateEstimate.vx'], d['stateEstimate.vy'], d['stateEstimate.vz']
            if flight_start[0] is not None:
                real_log.append({'t': time.time() - flight_start[0], **{k: ds[k] for k in ds}})

        def _att_cb(ts, d, lc):
            ds['roll'], ds['pitch'], ds['yaw'] = d['stabilizer.roll'], d['stabilizer.pitch'], d['stabilizer.yaw']
            ds['gyro_x'], ds['gyro_y'], ds['gyro_z'] = d['gyro.x'], d['gyro.y'], d['gyro.z']

        log_state.data_received_cb.add_callback(_state_cb)
        log_att.data_received_cb.add_callback(_att_cb)
        cf.log.add_config(log_state); cf.log.add_config(log_att)
        log_state.start(); log_att.start()

        print("[REAL] Resetting Kalman estimator...")
        cf.param.set_value('kalman.resetEstimation', '1'); time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0'); time.sleep(2.0)
        for _ in range(50):
            cf.commander.send_setpoint(0, 0, 0, 0); time.sleep(0.02)
        print("[REAL] Commander unlocked.")

        TAKEOFF_FREQ = 20
        START = time.time(); flight_start[0] = START
        elapsed_phase0 = elapsed_phases01 = 0.0

        try:
            # Phase 0: PID takeoff
            print(f"[REAL] Phase 0: PID takeoff to {takeoff_pos}...")
            i = 0
            while True:
                elapsed = time.time() - START
                z = takeoff_ramp_z(takeoff_pos[2], elapsed, args.takeoff_max_duration)
                cf.commander.send_position_setpoint(takeoff_pos[0], takeoff_pos[1], z, 0.0)
                pos_now = np.array([ds['x'], ds['y'], ds['z']])
                if np.linalg.norm(pos_now - takeoff_pos) <= 0.05 or elapsed >= args.takeoff_max_duration:
                    break
                i += 1; sync(i, START, 1.0 / TAKEOFF_FREQ)
            elapsed_phase0 = time.time() - START

            # Phase 1: HOVER
            if args.hover_duration > 0:
                print(f"[REAL] Phase 1: HOVER {args.hover_duration}s...")
                hstart = time.time()
                for j in range(int(TAKEOFF_FREQ * args.hover_duration)):
                    cf.commander.send_position_setpoint(*takeoff_pos, 0.0)
                    sync(j, hstart, 1.0 / TAKEOFF_FREQ)
            elapsed_phases01 = time.time() - START

            # Phase 2: NN (or PID) regulation to nn_target
            nn_duration = max(0.0, args.duration - elapsed_phases01)
            print(f"[REAL] Phase 2: {args.controller} to {nn_target} for {nn_duration:.0f}s...")
            try:
                import ctypes; ctypes.windll.winmm.timeBeginPeriod(1); _wt = True
            except Exception:
                _wt = False

            def _sync_precise(target):
                rem = target - time.perf_counter()
                if rem > 0.002:
                    time.sleep(rem - 0.002)
                while time.perf_counter() < target:
                    pass

            if is_pid:
                # pure firmware: stream the NN target as a position setpoint @ 20 Hz
                psteps = int(TAKEOFF_FREQ * nn_duration); pstart = time.time()
                for j in range(psteps):
                    cf.commander.send_position_setpoint(*nn_target, 0.0)
                    sync(j, pstart, 1.0 / TAKEOFF_FREQ)
            else:
                nn_dt = 1.0 / NN_FREQ
                p2 = time.perf_counter()
                for i in range(int(NN_FREQ * nn_duration)):
                    t_wall = time.time(); target_t = p2 + (i + 1) * nn_dt
                    nn_state = drone_state_to_nn_state(ds)
                    a = query_maml_nn(policy, theta, nn_state, nn_target, device)
                    thrust, roll_deg, pitch_deg, yaw_rate = float(a[0]), float(a[1]), float(a[2]), float(a[3])
                    nn_cmd_log.append({'t': elapsed_phases01 + i / NN_FREQ, 't_wall': t_wall,
                                       'thrust': thrust, 'roll': roll_deg,
                                       'pitch': pitch_deg, 'yaw_rate': yaw_rate})
                    # send_setpoint(roll, pitch(neg), yaw_rate(neg), thrust_u16)
                    # Firmware RPYT convention: positive yawrate = clockwise =
                    # decreasing stabilizer.yaw, opposite to the sim/training
                    # convention (yaw_sp += yaw_rate*dt) -> negate.
                    cf.commander.send_setpoint(roll_deg, -pitch_deg, -yaw_rate, int(thrust))
                    if i % NN_FREQ == 0:
                        err = math.dist((ds['x'], ds['y'], ds['z']), nn_target)
                        print(f"  t={elapsed_phases01 + i/NN_FREQ:5.1f}s pos="
                              f"[{ds['x']:+.3f},{ds['y']:+.3f},{ds['z']:.3f}] err={err:.3f}m thr={thrust:.0f}")
                    _sync_precise(target_t)
            if _wt:
                ctypes.windll.winmm.timeEndPeriod(1)

        except KeyboardInterrupt:
            print("\n[REAL] Interrupted!")
        finally:
            # Phase 3: smoothstep landing
            print("[REAL] Landing...")
            lx, ly, lz = ds['x'], ds['y'], ds['z']
            ldur = max(1.0, lz / 0.3) if args.land_duration is None else max(0.0, float(args.land_duration))
            lstart = time.time()
            for j in range(int(20 * ldur)):
                z = landing_ramp_z(lz, time.time() - lstart, ldur)
                if z < 0.05:
                    break
                cf.commander.send_position_setpoint(lx, ly, z, 0.0)
                time.sleep(1.0 / 20)
            cf.commander.send_notify_setpoint_stop(); time.sleep(0.1)
            log_state.stop(); log_att.stop()
            print("[REAL] Landed.")

        if real_log:
            arr = dict(
                t=np.array([e['t'] for e in real_log]),
                pos=np.array([[e['x'], e['y'], e['z']] for e in real_log]),
                vel=np.array([[e['vx'], e['vy'], e['vz']] for e in real_log]),
                rpy=np.array([[e['roll'], e['pitch'], e['yaw']] for e in real_log]),
                gyro=np.array([[e['gyro_x'], e['gyro_y'], e['gyro_z']] for e in real_log]),
                pid_target=takeoff_pos, nn_target=nn_target, hover_u16=hover_u16,
                mass_g=args.mass, controller=args.controller,
                elapsed_phase0=elapsed_phase0, elapsed_phases01=elapsed_phases01)
            if nn_cmd_log:
                arr['nn_cmd_t'] = np.array([e['t'] for e in nn_cmd_log])
                arr['nn_cmd'] = np.array([[e['thrust'], e['roll'], e['pitch'], e['yaw_rate']]
                                          for e in nn_cmd_log])
                arr['nn_cmd_twall'] = np.array([e['t_wall'] for e in nn_cmd_log])
            _save_flight_npz(arr, f"maml_real_{args.controller}_{args.mass:.0f}g")
            if args.plot:
                _plot_real(real_log, takeoff_pos, nn_target, hover_u16,
                           elapsed_phase0, elapsed_phases01, nn_cmd_log)


# ===================================================================
#  Plotting (same layout as §5)
# ===================================================================

def _plot_sim(data):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not found."); return
    t, pos, vel, rpy = data['t'], data['pos'], data['vel'], data['rpy']
    rpms, thrust, nn_cmd = data['rpms'], data['thrust'], data['nn_cmd']
    pid_target, nn_target, phase = data['pid_target'], data['nn_target'], data['phase']
    hover = float(data['hover_u16'])
    hov_i = np.argmax(phase == PHASE_HOVER) if np.any(phase == PHASE_HOVER) else None
    nn_i = np.argmax(phase == PHASE_NN) if np.any(phase == PHASE_NN) else len(t) - 1
    land_i = np.argmax(phase == PHASE_LAND) if np.any(phase == PHASE_LAND) else None

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"MAML controller ({data.get('controller','maml')}) + CF att/rate PID "
                 f"— payload {float(data.get('mass_g',0)):.0f} g", fontsize=14)
    for ax in axes.flat:
        if hov_i is not None:
            ax.axvline(t[hov_i], color='orange', ls=':', alpha=0.5)
        ax.axvline(t[nn_i], color='red', ls=':', alpha=0.5)
        if land_i is not None:
            ax.axvline(t[land_i], color='purple', ls=':', alpha=0.5)

    ax = axes[0, 0]
    for j, lab in enumerate(['x', 'y', 'z']):
        ax.plot(t, pos[:, j], label=lab)
    for j, c in enumerate(['C0', 'C1', 'C2']):
        ax.axhline(nn_target[j], ls='--', color=c, alpha=0.6, label=f'NN tgt {["x","y","z"][j]}={nn_target[j]:.2f}')
    ax.set_ylabel('Position [m]'); ax.legend(fontsize=6, ncol=2); ax.set_title('Position'); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    for j, lab in enumerate(['vx', 'vy', 'vz']):
        ax.plot(t, vel[:, j], label=lab)
    ax.set_ylabel('Velocity [m/s]'); ax.legend(fontsize=7); ax.set_title('Velocity'); ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.plot(t, rpy[:, 0], label='roll'); ax.plot(t, rpy[:, 1], label='pitch')
    ax.set_ylabel('Angle [deg]'); ax.legend(fontsize=7); ax.set_title('Roll / Pitch'); ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    m = phase == PHASE_NN
    ax.plot(t[m], nn_cmd[m, 1], label='NN roll [deg]', alpha=0.8)
    ax.plot(t[m], nn_cmd[m, 2], label='NN pitch [deg]', alpha=0.8)
    ax.plot(t[m], nn_cmd[m, 3], label='NN yaw_rate [deg/s]', alpha=0.6)
    ax.set_ylabel('NN attitude cmd'); ax.legend(fontsize=7); ax.set_title('NN Outputs'); ax.grid(True, alpha=0.3)
    ax = axes[2, 0]
    ax.plot(t, thrust, color='black', label='thrust')
    ax.axhline(hover, ls=':', color='gray', label=f'hover={hover:.0f}')
    ax.set_ylabel('Thrust [uint16]'); ax.set_xlabel('Time [s]'); ax.legend(fontsize=7); ax.set_title('Thrust'); ax.grid(True, alpha=0.3)
    ax = axes[2, 1]
    for k in range(4):
        ax.plot(t, rpms[:, k], label=f'M{k+1}', alpha=0.7)
    ax.set_ylabel('RPM'); ax.set_xlabel('Time [s]'); ax.legend(fontsize=7, ncol=2); ax.set_title('Motor RPMs'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, f"maml_sim_results_{float(data.get('mass_g',0)):.0f}g.png")
    plt.savefig(out, dpi=150); print(f"[PLOT] Saved to {out}")


def _plot_real(real_log, pid_target, nn_target, hover_u16,
               elapsed_phase0, elapsed_phases01, nn_cmd_log=None):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not found."); return
    t = np.array([e['t'] for e in real_log])
    pos = np.array([[e['x'], e['y'], e['z']] for e in real_log])
    vel = np.array([[e['vx'], e['vy'], e['vz']] for e in real_log])
    rpy = np.array([[e['roll'], e['pitch'], e['yaw']] for e in real_log])

    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True)
    fig.suptitle('MAML controller — Real Drone (lested)', fontsize=14)

    def _lines(ax):
        if elapsed_phases01 > elapsed_phase0:
            ax.axvline(elapsed_phase0, color='orange', ls=':', alpha=0.5)
        ax.axvline(elapsed_phases01, color='red', ls=':', alpha=0.5)
    for ax in axes.flat:
        _lines(ax)

    ax = axes[0, 0]
    for j, lab in enumerate(['x', 'y', 'z']):
        ax.plot(t, pos[:, j], label=lab)
    for j, c in enumerate(['C0', 'C1', 'C2']):
        ax.axhline(nn_target[j], ls='--', color=c, alpha=0.6, label=f'NN tgt {["x","y","z"][j]}={nn_target[j]:.2f}')
    ax.set_ylabel('Position [m]'); ax.legend(fontsize=6); ax.set_title('Position'); ax.grid(True, alpha=0.3)
    ax = axes[0, 1]
    for j, lab in enumerate(['vx', 'vy', 'vz']):
        ax.plot(t, vel[:, j], label=lab)
    ax.set_ylabel('Velocity [m/s]'); ax.legend(fontsize=7); ax.set_title('Velocity'); ax.grid(True, alpha=0.3)
    ax = axes[1, 0]
    ax.plot(t, rpy[:, 0], label='roll'); ax.plot(t, rpy[:, 1], label='pitch')
    ax.set_ylabel('Angle [deg]'); ax.legend(fontsize=7); ax.set_title('Roll / Pitch'); ax.grid(True, alpha=0.3)
    ax = axes[1, 1]
    ax.plot(t, rpy[:, 2], label='yaw', color='green')
    ax.set_ylabel('Angle [deg]'); ax.legend(fontsize=7); ax.set_title('Yaw'); ax.grid(True, alpha=0.3)
    ax = axes[2, 0]
    if nn_cmd_log:
        tn = np.array([e['t'] for e in nn_cmd_log])
        ax.plot(tn, [e['roll'] for e in nn_cmd_log], label='NN roll [deg]', alpha=0.8)
        ax.plot(tn, [e['pitch'] for e in nn_cmd_log], label='NN pitch [deg]', alpha=0.8)
        ax.plot(tn, [e['yaw_rate'] for e in nn_cmd_log], label='NN yaw_rate [deg/s]', alpha=0.6)
    ax.set_ylabel('NN attitude cmd'); ax.set_xlabel('Time [s]'); ax.legend(fontsize=7); ax.set_title('NN Outputs'); ax.grid(True, alpha=0.3)
    ax = axes[2, 1]
    if nn_cmd_log:
        tn = np.array([e['t'] for e in nn_cmd_log])
        ax.plot(tn, [e['thrust'] for e in nn_cmd_log], color='black', label='NN thrust')
        ax.axhline(hover_u16, ls=':', color='gray', label=f'hover={hover_u16:.0f}')
    ax.set_ylabel('Thrust [uint16]'); ax.set_xlabel('Time [s]'); ax.legend(fontsize=7); ax.set_title('NN Thrust'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, 'maml_real_results.png')
    plt.savefig(out, dpi=150); print(f"[PLOT] Saved to {out}")


# ===================================================================
#  CLI
# ===================================================================

def main():
    p = argparse.ArgumentParser(
        description="MAML NN controller for a lested Crazyflie — sim & real",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--maml-ckpt', required=True, help='MAML checkpoint (.pt)')
    p.add_argument('--baseline-ckpt', default=None, help='baseline checkpoint (for --controller base)')
    p.add_argument('--controller', default='maml',
                   choices=['maml', 'base', 'base-adapt', 'pid'],
                   help='which of the 4 controllers to fly (default: maml). '
                        "'base' = baseline as-is, 'base-adapt' = baseline "
                        "few-shot adapted, 'maml' = meta-init adapted, "
                        "'pid' = pure firmware reference.")
    p.add_argument('--mass', default=10.0, type=float, help='lested payload [g] (default: 10)')
    p.add_argument('--offset-dx', default=0.0, type=float, help='payload offset dx [m] (default 0 = centred)')
    p.add_argument('--offset-dy', default=0.0, type=float)
    p.add_argument('--offset-dz', default=0.0, type=float)
    p.add_argument('--no-adapt', dest='adapt', action='store_false',
                   help='fly without few-shot adaptation (zero-shot)')
    p.add_argument('--mode', default='sim', choices=['sim', 'attitude'])
    p.add_argument('--duration', default=20.0, type=float, help='total flight duration [s]')
    p.add_argument('--takeoff-max-duration', default=5.0, type=float)
    p.add_argument('--PID-target', nargs=3, type=float, default=[0.0, 0.0, 0.5],
                   metavar=('X', 'Y', 'Z'), help='PID takeoff target [m]')
    p.add_argument('--NN-target', nargs=3, type=float, default=[0.0, 0.0, 1.0],
                   metavar=('X', 'Y', 'Z'), help='NN target [m]')
    p.add_argument('--hover-duration', default=5.0, type=float)
    p.add_argument('--land-duration', default=None, type=float)
    p.add_argument('--gui', default=True, type=lambda x: x.lower() == 'true')
    p.add_argument('--uri', default='radio://0/80/2M/E7E7E7E7E7')
    p.add_argument('--no-push-gains', dest='push_gains', action='store_false')
    p.add_argument('--no-plot', dest='plot', action='store_false')
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy, theta, m_extra, r_offset, hover_u16, info = load_and_adapt(args, device)

    if args.mode == 'sim':
        run_sim(args, policy, theta, m_extra, r_offset, hover_u16, info, device)
    else:
        run_real(args, policy, theta, m_extra, r_offset, hover_u16, info, device)


if __name__ == "__main__":
    main()
