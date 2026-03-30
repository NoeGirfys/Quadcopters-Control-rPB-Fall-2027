#!/usr/bin/env python3
"""
Validate a trained NN+PID controller (from train_nn_cf_pid.py) inside
gym-pybullet-drones.

For each setpoint mode, the NN output is fed into the appropriate stage of
the NumPy firmware PID cascade (from cf_firmware_pid_sim.py):
  pos_sp   → full cascade (pos→vel→att→rate→motors)
  vel_sp   → bypass position PID, run vel→att→rate→motors
  att_sp   → bypass pos+vel, run att→rate→motors
  rate_sp  → bypass pos+vel+att, run rate→motors

Plots 12 state channels over time: position, velocity, attitude, angular rate.
Reports NN forward-pass timing.

Usage:
    python validate_nn_cf_pid_pybullet.py
    python validate_nn_cf_pid_pybullet.py --gui
    python validate_nn_cf_pid_pybullet.py --weights trained_weights_cf_pid_pos_sp_nonlinear_h64_ep500_lr1e-03.pt
    python validate_nn_cf_pid_pybullet.py --flowdeck
    python validate_nn_cf_pid_pybullet.py --physics pyb_gnd_drag_dw
"""

import sys
import os
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn

# ── Add gym-pybullet-drones to path ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYB_DIR    = os.path.join(SCRIPT_DIR, "gym-pybullet-drones-main")
if PYB_DIR not in sys.path:
    sys.path.insert(0, PYB_DIR)

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# ── Import firmware PID (NumPy implementation) ────────────────────────────────
from cf_firmware_pid_sim import (
    CrazyfliePositionController,
    CrazyflieAttitudeController,
    CrazyfliePowerDistribution,
    FlowDeckSimulator,
    MotorDynamicsFilter,
    pwm_to_rpm,
    obs_to_firmware_state,
    # Constants
    ATTITUDE_RATE, POSITION_RATE, ATTITUDE_UPDATE_DT, POSITION_UPDATE_DT,
    PID_VEL_THRUST_BASE, PID_VEL_THRUST_MIN, THRUST_SCALE, UINT16_MAX,
    PID_VEL_ROLL_MAX, PID_VEL_PITCH_MAX,
    PID_VEL_X_KP, PID_VEL_X_KI, PID_VEL_X_KD,
    PID_VEL_Y_KP, PID_VEL_Y_KI, PID_VEL_Y_KD,
    PID_VEL_Z_KP, PID_VEL_Z_KI, PID_VEL_Z_KD,
    VEL_MAX_OVERHEAD, RP_LIMIT_OVERHEAD,
    PID_POS_VEL_X_MAX, PID_POS_VEL_Y_MAX, PID_POS_VEL_Z_MAX,
    CF2_THRUST_MAX_PER_MOTOR,
    cap_angle,
)

POS_EVERY = ATTITUDE_RATE // POSITION_RATE   # = 5

# ─────────────────────────────────────────────────────────────────────────────
# 1. PolicyMLP  (must match train_nn_cf_pid.py exactly)
# ─────────────────────────────────────────────────────────────────────────────
class PolicyMLPPID(nn.Module):
    POS_SCALE       = 2.0
    VEL_SCALE       = 1.0
    ATT_RP_SCALE    = 20.0
    ATT_YR_SCALE    = 200.0
    RATE_SCALE      = 720.0
    THRUST_HOVER    = float(PID_VEL_THRUST_BASE)

    def __init__(self, x_scale, setpoint_mode: str, hidden: int = 64):
        super().__init__()
        assert setpoint_mode in ('pos_sp', 'vel_sp', 'att_sp', 'rate_sp')
        self.setpoint_mode = setpoint_mode
        output_dim = 3 if setpoint_mode in ('pos_sp', 'vel_sp') else 4
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(12, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n = x / self.x_scale
        raw = self.net(x_n)
        if self.setpoint_mode == 'pos_sp':
            return torch.tanh(raw) * self.POS_SCALE
        if self.setpoint_mode == 'vel_sp':
            return torch.tanh(raw) * self.VEL_SCALE
        if self.setpoint_mode == 'att_sp':
            roll  = torch.tanh(raw[..., [0]]) * self.ATT_RP_SCALE
            pitch = torch.tanh(raw[..., [1]]) * self.ATT_RP_SCALE
            yr    = torch.tanh(raw[..., [2]]) * self.ATT_YR_SCALE
            thr   = torch.sigmoid(raw[..., [3]]) * float(UINT16_MAX)
            return torch.cat([roll, pitch, yr, thr], dim=-1)
        # rate_sp
        p_rate = torch.tanh(raw[..., [0]]) * self.RATE_SCALE
        q_rate = torch.tanh(raw[..., [1]]) * self.RATE_SCALE
        r_rate = torch.tanh(raw[..., [2]]) * self.RATE_SCALE
        thr    = torch.sigmoid(raw[..., [3]]) * float(UINT16_MAX)
        return torch.cat([p_rate, q_rate, r_rate, thr], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Velocity-only PID helper  (bypasses position loop)
# ─────────────────────────────────────────────────────────────────────────────

def velocity_pid_update(pos_ctrl: CrazyfliePositionController,
                        state_vel, setpoint_vel_body, state_yaw_deg):
    """Run only the vel→att+thrust part of the position controller.

    Parameters
    ----------
    state_vel        : (vx, vy, vz)   global frame [m/s]
    setpoint_vel_body: (vx, vy, vz)   body-yaw frame [m/s]
    state_yaw_deg    : float          current yaw [deg]

    Returns
    -------
    thrust_u16, roll_deg, pitch_deg
    """
    st_vx, st_vy, st_vz = state_vel
    sp_vx, sp_vy, sp_vz = setpoint_vel_body

    cosyaw = math.cos(math.radians(state_yaw_deg))
    sinyaw = math.sin(math.radians(state_yaw_deg))

    # Rotate state velocity to body-yaw frame
    state_body_vx =  st_vx * cosyaw + st_vy * sinyaw
    state_body_vy = -st_vx * sinyaw + st_vy * cosyaw

    # Set output limits
    pos_ctrl.pidVX.outputLimit = PID_VEL_PITCH_MAX * RP_LIMIT_OVERHEAD
    pos_ctrl.pidVY.outputLimit = PID_VEL_ROLL_MAX  * RP_LIMIT_OVERHEAD
    pos_ctrl.pidVZ.outputLimit = (float(UINT16_MAX) / 2.0 / THRUST_SCALE)

    pos_ctrl.pidVX.set_desired(sp_vx)
    pitch_deg = -pos_ctrl.pidVX.update(state_body_vx)

    pos_ctrl.pidVY.set_desired(sp_vy)
    roll_deg  = -pos_ctrl.pidVY.update(state_body_vy)

    roll_deg  = max(-PID_VEL_ROLL_MAX,  min(PID_VEL_ROLL_MAX,  roll_deg))
    pitch_deg = max(-PID_VEL_PITCH_MAX, min(PID_VEL_PITCH_MAX, pitch_deg))

    pos_ctrl.pidVZ.set_desired(sp_vz)
    thrust_raw = pos_ctrl.pidVZ.update(st_vz)
    thrust = thrust_raw * THRUST_SCALE + pos_ctrl.thrustBase
    if thrust < pos_ctrl.thrustMin:
        thrust = pos_ctrl.thrustMin
    thrust = max(0.0, min(float(UINT16_MAX), thrust))

    return thrust, roll_deg, pitch_deg


# ─────────────────────────────────────────────────────────────────────────────
# 3. State extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def build_goffin_state(pos, vel, rpy_deg, gyro_deg, target_pos=np.zeros(3)):
    """Build 12D error state: [x,ẋ,y,ẏ,z,ż,φ,φ̇,θ,θ̇,ψ,ψ̇] in SI units.

    Angles and angular rates are in RADIANS for NN input (same as training).
    """
    err_pos = pos - target_pos
    state = np.array([
        err_pos[0],              vel[0],
        err_pos[1],              vel[1],
        err_pos[2],              vel[2],
        math.radians(rpy_deg[0]),  math.radians(gyro_deg[0]),
        math.radians(rpy_deg[1]),  math.radians(gyro_deg[1]),
        math.radians(rpy_deg[2]),  math.radians(gyro_deg[2]),
    ], dtype=np.float32)
    return state


def physics_from_string(name: str) -> Physics:
    for p in Physics:
        if p.value == name:
            return p
    raise ValueError(f"Unknown physics '{name}'. Options: {[p.value for p in Physics]}")


def resolve_weights_path(script_dir: str, weights_arg: str) -> str:
    if os.path.isabs(weights_arg):
        return weights_arg
    candidate = os.path.join(script_dir, weights_arg)
    if os.path.isfile(candidate):
        return candidate
    return weights_arg


def build_plot_filename(weights_path, physics_mode, target_pos, duration, flowdeck):
    stem   = os.path.splitext(os.path.basename(weights_path))[0]
    tz     = f"{target_pos[2]:.2f}".replace("-", "m").replace(".", "p")
    dur    = f"{duration:.1f}".replace(".", "p")
    fd_str = "_flowdeck" if flowdeck else ""
    return f"validation_{stem}_phys-{physics_mode}_targetz-{tz}_T-{dur}s{fd_str}.png"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Main simulation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    policy: PolicyMLPPID,
    checkpoint: dict,
    offset_xyz,
    gui: bool = False,
    duration: float = 4.0,
    target_pos: np.ndarray = np.zeros(3),
    physics_mode: Physics = Physics.PYB,
    use_flowdeck: bool = False,
    flowdeck_delay_ms: float = 110.0,
) -> dict:
    """Run the NN+PID controller in pybullet-drones and record the trajectory.

    The simulation runs at ctrl_freq=500 Hz (matching the attitude loop).
    Position PID + NN update every 5th step (100 Hz).

    Parameters
    ----------
    offset_xyz   : (dx, dy, dz)  initial offset from target_pos
    use_flowdeck : bool          enable FlowDeck delay/noise simulation
    """
    KF       = checkpoint["KF"]
    KM       = checkpoint.get("KM", 7.94e-12)
    M_drone  = checkpoint.get("M", 0.027)
    G_val    = checkpoint.get("G", 9.8)
    MAX_RPM  = checkpoint["MAX_RPM"]
    HOVER_RPM_val = float(np.sqrt(M_drone * G_val / (4.0 * KF)))
    setpoint_mode = checkpoint["setpoint_mode"]

    # ── Environment ──────────────────────────────────────────────────────────
    PYB_FREQ  = 1000   # physics steps/s
    CTRL_FREQ = ATTITUDE_RATE  # 500 Hz

    init_xyz = target_pos + np.asarray(offset_xyz, dtype=float)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[init_xyz[0], init_xyz[1], init_xyz[2]]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=physics_mode,
        pyb_freq=PYB_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=gui,
    )

    N_steps = int(duration * CTRL_FREQ)

    # ── PID controllers ───────────────────────────────────────────────────────
    pos_ctrl   = CrazyfliePositionController()
    att_ctrl   = CrazyflieAttitudeController()
    power_dist = CrazyfliePowerDistribution()
    motor_filt = MotorDynamicsFilter(n_motors=4, tau=0.02,
                                     dt=ATTITUDE_UPDATE_DT)
    motor_filt.rpm = np.full(4, HOVER_RPM_val)

    attitude_desired_yaw = 0.0   # deg
    actuator_thrust      = PID_VEL_THRUST_BASE
    last_att_roll        = 0.0
    last_att_pitch       = 0.0
    yaw_rate_sp          = 0.0  # deg/s — used for att_sp / rate_sp
    rate_r_sp            = 0.0  # deg/s — used for rate_sp
    rate_p_sp            = 0.0
    rate_r_sp_yaw        = 0.0

    # ── FlowDeck ──────────────────────────────────────────────────────────────
    flow_deck = None
    if use_flowdeck:
        flow_deck = FlowDeckSimulator(ctrl_freq=CTRL_FREQ,
                                      delay_ms=flowdeck_delay_ms,
                                      vel_noise=0.0)
        flow_deck.reset(init_xyz)

    # ── Logging ───────────────────────────────────────────────────────────────
    pos_log    = np.zeros((N_steps, 3))
    vel_log    = np.zeros((N_steps, 3))
    rpy_log    = np.zeros((N_steps, 3))
    gyro_log   = np.zeros((N_steps, 3))
    wrench_log = np.zeros((N_steps, 4))
    rpm_log    = np.zeros((N_steps, 4))
    time_log   = np.zeros(N_steps)

    # NN timing
    nn_times_ms = []

    print(f"  [{setpoint_mode}] {N_steps} steps from offset={offset_xyz}, "
          f"target={target_pos}, physics={physics_mode.value}")

    action = np.array([[HOVER_RPM_val] * 4])

    for k in range(N_steps):
        # ── Step environment ─────────────────────────────────────────────────
        obs, _, _, _, _ = env.step(action)

        # ── Extract state (firmware convention) ──────────────────────────────
        pos, vel, rpy_deg, gyro_deg = obs_to_firmware_state(obs[0])

        # ── FlowDeck ─────────────────────────────────────────────────────────
        if flow_deck is not None:
            pos_ctrl_input, vel_ctrl_input = flow_deck.update(pos, vel)
        else:
            pos_ctrl_input, vel_ctrl_input = pos, vel

        # ── NN + Position/Velocity PID  (every POS_EVERY steps = 100 Hz) ────
        if k % POS_EVERY == 0:
            # Build 12D Goffin state (error relative to target, rad for angles)
            state_12 = build_goffin_state(pos, vel, rpy_deg, gyro_deg, target_pos)
            state_t  = torch.tensor(state_12, dtype=torch.float32).unsqueeze(0)

            # NN forward pass — timed
            t0 = time.perf_counter()
            with torch.no_grad():
                nn_out = policy(state_t).squeeze(0).numpy()
            nn_times_ms.append((time.perf_counter() - t0) * 1000.0)

            if setpoint_mode == 'pos_sp':
                sp_pos = nn_out   # [x, y, z] m  (global frame, error from origin)
                # The NN learned to regulate to origin, so the setpoint is an
                # absolute position in "error space". Convert to world frame:
                sp_world = sp_pos + target_pos
                actuator_thrust, last_att_roll, last_att_pitch = \
                    pos_ctrl.update(
                        sp_world, pos_ctrl_input, vel_ctrl_input, rpy_deg[2])

            elif setpoint_mode == 'vel_sp':
                sp_vel = nn_out   # [vx, vy, vz] m/s (body-yaw frame)
                actuator_thrust, last_att_roll, last_att_pitch = \
                    velocity_pid_update(
                        pos_ctrl, vel_ctrl_input, sp_vel, rpy_deg[2])

            elif setpoint_mode == 'att_sp':
                last_att_roll  = float(nn_out[0])   # deg
                last_att_pitch = float(nn_out[1])   # deg
                yaw_rate_sp    = float(nn_out[2])   # deg/s
                actuator_thrust = float(nn_out[3])  # uint16
                # Accumulate yaw setpoint
                attitude_desired_yaw = cap_angle(
                    attitude_desired_yaw + yaw_rate_sp * ATTITUDE_UPDATE_DT)

            else:  # rate_sp
                rate_r_sp      = float(nn_out[0])   # deg/s
                rate_p_sp      = float(nn_out[1])
                rate_r_sp_yaw  = float(nn_out[2])
                actuator_thrust = float(nn_out[3])  # uint16
                yaw_rate_sp    = rate_r_sp_yaw
                attitude_desired_yaw = cap_angle(
                    attitude_desired_yaw + yaw_rate_sp * ATTITUDE_UPDATE_DT)

        # ── Yaw accumulation for att_sp/rate_sp (every inner step) ──────────
        if setpoint_mode in ('att_sp', 'rate_sp') and k % POS_EVERY != 0:
            attitude_desired_yaw = cap_angle(
                attitude_desired_yaw + yaw_rate_sp * ATTITUDE_UPDATE_DT)

        # ── Attitude + Rate PIDs (every inner step = 500 Hz) ─────────────────
        if setpoint_mode in ('pos_sp', 'vel_sp', 'att_sp'):
            roll_rate_des, pitch_rate_des, yaw_rate_des = \
                att_ctrl.correct_attitude(
                    rpy_deg[0], rpy_deg[1], rpy_deg[2],
                    last_att_roll, last_att_pitch, attitude_desired_yaw)
        else:  # rate_sp
            roll_rate_des  = rate_r_sp
            pitch_rate_des = rate_p_sp
            yaw_rate_des   = rate_r_sp_yaw

        roll_cmd, pitch_cmd, yaw_cmd = att_ctrl.correct_rate(
            gyro_deg[0], gyro_deg[1], gyro_deg[2],
            roll_rate_des, pitch_rate_des, yaw_rate_des)
        yaw_cmd = -yaw_cmd

        # ── Power distribution ────────────────────────────────────────────────
        motor_pwm = power_dist.distribute(
            actuator_thrust, roll_cmd, pitch_cmd, yaw_cmd)

        # ── PWM → RPM (with 8-bit truncation — matches real hardware) ────────
        rpms = pwm_to_rpm(motor_pwm, KF)

        # ── Motor dynamics filter ─────────────────────────────────────────────
        rpms = motor_filt.apply(rpms)

        action = np.array([rpms])

        # ── Logging ───────────────────────────────────────────────────────────
        # Compute approximate wrench from RPMs for logging
        omega_sq = rpms ** 2
        _a = KF * float(checkpoint.get("L", 0.0397)) / np.sqrt(2)
        alloc = np.array([
            [ KF,   KF,   KF,   KF],
            [ -_a,  -_a,   _a,   _a],
            [ -_a,   _a,   _a,  -_a],
            [ -KM,   KM,  -KM,   KM],
        ])
        wrench_approx = alloc @ omega_sq

        pos_log[k]    = pos
        vel_log[k]    = vel
        rpy_log[k]    = rpy_deg
        gyro_log[k]   = gyro_deg
        wrench_log[k] = wrench_approx
        rpm_log[k]    = rpms
        time_log[k]   = (k + 1) / CTRL_FREQ

    env.close()

    # ── NN timing report ──────────────────────────────────────────────────────
    if nn_times_ms:
        print(f"\n  NN forward pass timing ({len(nn_times_ms)} calls):")
        print(f"    avg: {np.mean(nn_times_ms):.3f} ms")
        print(f"    min: {np.min(nn_times_ms):.3f} ms")
        print(f"    max: {np.max(nn_times_ms):.3f} ms")

    return {
        "pos": pos_log, "vel": vel_log,
        "rpy": rpy_log, "gyro": gyro_log,
        "wrench": wrench_log, "rpm": rpm_log,
        "time": time_log,
        "nn_times_ms": np.array(nn_times_ms),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Results display
# ─────────────────────────────────────────────────────────────────────────────

def print_results(trajs, labels, target_pos):
    print(f"\n  Target position: {target_pos}")
    hdr = f"{'Offset (dx,dy,dz)':>25s}  │  {'|pos_err|':>10s}  {'|vel_T|':>10s}  {'max|rpy|°':>10s}"
    print(f"\n{hdr}")
    print("─" * 72)
    for label, traj in zip(labels, trajs):
        pos_err = np.linalg.norm(traj["pos"][-1] - target_pos)
        vel_T   = np.linalg.norm(traj["vel"][-1])
        rpy_max = np.abs(traj["rpy"]).max()
        print(f"  {label:>23s}  │  {pos_err:10.5f}  "
              f"{vel_T:10.5f}  {rpy_max:10.3f}°")


def save_plots(trajs, labels, target_pos, setpoint_mode,
               filename="validation_results.png"):
    """Save 4×3 time-series plot: position, velocity, attitude, angular rate."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    row_names = ["Position (m)", "Velocity (m/s)", "Attitude (deg)", "Angular rate (deg/s)"]
    col_names = ["X", "Y", "Z"]
    n_rows, n_cols = 4, 3

    fig, axes = plt.subplots(len(trajs), n_rows * n_cols,
                              figsize=(5 * n_rows, 3 * len(trajs)),
                              squeeze=False)

    # Restructure: rows=offsets, but we want rows=channels, cols=axes
    fig2, axes2 = plt.subplots(n_rows, n_cols,
                                figsize=(14, 10), sharex=True)

    colors = plt.cm.tab10.colors

    for i_traj, (traj, label) in enumerate(zip(trajs, labels)):
        t = traj["time"]
        c = colors[i_traj % len(colors)]

        # Row 0: position
        for j in range(3):
            axes2[0, j].plot(t, traj["pos"][:, j], color=c, linewidth=1.2,
                             label=label)
            axes2[0, j].axhline(target_pos[j], color="k", lw=0.8, ls="--")

        # Row 1: velocity
        for j in range(3):
            axes2[1, j].plot(t, traj["vel"][:, j], color=c, linewidth=1.2)
            axes2[1, j].axhline(0, color="k", lw=0.5, ls="--")

        # Row 2: attitude (deg)
        for j in range(3):
            axes2[2, j].plot(t, traj["rpy"][:, j], color=c, linewidth=1.2)
            axes2[2, j].axhline(0, color="k", lw=0.5, ls="--")

        # Row 3: angular rate (deg/s)
        for j in range(3):
            axes2[3, j].plot(t, traj["gyro"][:, j], color=c, linewidth=1.2)
            axes2[3, j].axhline(0, color="k", lw=0.5, ls="--")

    # Labels and titles
    for row_idx, row_name in enumerate(row_names):
        for col_idx, col_name in enumerate(col_names):
            ax = axes2[row_idx, col_idx]
            ax.set_ylabel(f"{row_name} {col_name}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if row_idx == 0:
                ax.set_title(col_name, fontsize=10)

    for col_idx in range(n_cols):
        axes2[-1, col_idx].set_xlabel("Time (s)")

    axes2[0, 0].legend(fontsize=7, loc="upper right")

    fig2.suptitle(
        f"NN+PID Controller Validation — mode={setpoint_mode} "
        f"(target z={target_pos[2]:.2f}m)", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(filename, dpi=150)
    plt.close(fig2)
    plt.close(fig)
    print(f"\n[Saved] {filename}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEFAULT_WEIGHTS = "trained_weights_cf_pid_pos_sp_nonlinear_h64_ep500_lr1e-03.pt"

    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",     action="store_true", default=False)
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS)
    parser.add_argument("--physics", type=str, default="pyb",
                        choices=[p.value for p in Physics])
    parser.add_argument("--duration",  type=float, default=4.0)
    parser.add_argument("--target_z", type=float, default=0.5)
    parser.add_argument("--flowdeck", action="store_true", default=False,
                        help="Enable FlowDeck delay/noise simulation")
    parser.add_argument("--flowdeck_delay_ms", type=float, default=110.0)
    args = parser.parse_args()

    physics_mode = physics_from_string(args.physics)

    # ── Load weights ──────────────────────────────────────────────────────────
    ckpt_path = resolve_weights_path(SCRIPT_DIR, args.weights)
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Weights file not found: {ckpt_path}")
        print("        Run  train_nn_cf_pid.py  first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = PolicyMLPPID(
        x_scale=ckpt["x_scale"],
        setpoint_mode=ckpt["setpoint_mode"],
        hidden=ckpt["hidden"],
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()

    setpoint_mode = ckpt["setpoint_mode"]
    print(f"[Loaded] {ckpt_path}")
    print(f"  setpoint_mode={setpoint_mode}")
    print(f"  trained_dynamics={ckpt.get('train_dynamics', 'unknown')}")
    print(f"  ctrl_freq={ckpt['ctrl_freq']} Hz")
    print(f"  use_flowdeck (train)={ckpt.get('use_flowdeck', False)}")
    print(f"  validation_physics={physics_mode.value}")
    print(f"  validation_flowdeck={args.flowdeck}")

    target_pos = np.array([0.0, 0.0, args.target_z])

    # Determine effective flowdeck setting
    use_flowdeck = args.flowdeck or ckpt.get("use_flowdeck", False)

    # ── Test offsets ──────────────────────────────────────────────────────────
    test_offsets = [
        (0.0,   0.0,   0.0),     # at target
        (0.3,   0.0,   0.0),     # +x offset
        (0.0,   0.3,   0.0),     # +y offset
        (0.0,   0.0,   0.3),     # +z offset
        (-0.2, -0.1,   0.1),     # diagonal
        (0.2,   0.1,  -0.1),     # opposite diagonal
    ]

    trajs  = []
    labels = []
    for offset in test_offsets:
        traj = run_validation(
            policy, ckpt, np.array(offset),
            gui=args.gui,
            duration=args.duration,
            target_pos=target_pos,
            physics_mode=physics_mode,
            use_flowdeck=use_flowdeck,
            flowdeck_delay_ms=args.flowdeck_delay_ms,
        )
        trajs.append(traj)
        labels.append(f"({offset[0]:+.1f},{offset[1]:+.1f},{offset[2]:+.1f})")

    # ── Print summary ────────────────────────────────────────────────────────
    print_results(trajs, labels, target_pos)

    # ── Aggregate NN timing ──────────────────────────────────────────────────
    all_times = np.concatenate([t["nn_times_ms"] for t in trajs])
    print(f"\n[NN timing — all runs combined]  "
          f"avg={np.mean(all_times):.3f}ms  "
          f"min={np.min(all_times):.3f}ms  "
          f"max={np.max(all_times):.3f}ms  "
          f"(N={len(all_times)} calls)")

    # ── Save plots ───────────────────────────────────────────────────────────
    plot_name = build_plot_filename(ckpt_path, physics_mode.value,
                                   target_pos, args.duration, use_flowdeck)
    save_plots(trajs, labels, target_pos, setpoint_mode,
               os.path.join(SCRIPT_DIR, plot_name))

    print("\nDone!")
