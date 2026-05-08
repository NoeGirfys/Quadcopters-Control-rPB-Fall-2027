"""Test a MAML-trained controller in gym-pybullet-drones with offset mass.

Pipeline (matches the training rollout):
  1. Load the checkpoint and rebuild PolicyMLP (with the hover-thrust bias
     that matches M_total = M_base + m_extra).
  2. Build the target task: the held-out motor with the same (m_extra, sigma)
     as training. By default the mass is attached at the target motor's
     centre; ``--target-dx/dy/dz`` lets you place it elsewhere.
  3. Few-shot adaptation: ``maml_adapt`` runs ``--n-inner-steps`` gradient
     steps (defaults to the checkpoint's value) on ``x0_train`` from the
     checkpoint, using ``lr_inner`` and the same ``inner_grad_clip`` as in
     training. Output is a parameter dict ``theta_adapt``.
  4. PyBullet simulation: ``CtrlAviary`` with ``Physics.DYN_OFFSET``, the
     offset mass installed via ``env.set_offset_mass``. The NN runs at
     NN_FREQ Hz and outputs (thrust_u16, roll_deg, pitch_deg, yaw_rate_deg/s);
     the firmware attitude+rate PIDs run downstream at ATTITUDE_RATE Hz.

Phase B for now: the drone is teleported to ``init_pos`` with zero
attitude/velocity and the NN takes over immediately — no firmware PID
takeoff.

Usage:
    cd training_MAML
    python test_maml_pybullet.py --weights maml_linearized_h64_ep500_o2_tm0.pt
    python test_maml_pybullet.py --weights ... --target-dx 0.02 --target-dy 0.0
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

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync

from maml_lib import (
    config as C,
    PolicyMLP, compute_mass_params, maml_adapt,
    DYNAMICS,
)

from crazyflie_firmware.firmware import (
    CrazyflieAttitudeController,
    CrazyfliePowerDistribution,
    cap_angle,
)
from circle_comparison_simu_and_real.cf_firmware_pid_sim import (
    MotorDynamicsFilter,
    obs_to_firmware_state,
    pwm_to_rpm,
)


# ===================================================================
#  State extraction (Goffin convention, identical to fly_nn_cf_pid)
# ===================================================================

def obs_to_nn_state(obs: np.ndarray) -> np.ndarray:
    """Convert pybullet-drones obs (20,) to NN state (12,) in Goffin order."""
    pos = obs[0:3]
    rpy_rad = obs[7:10]
    vel = obs[10:13]
    ang_v = obs[13:16]

    r, pi, y = rpy_rad
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(pi), math.sin(pi)
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


# ===================================================================
#  Load + rebuild policy with adapted weights
# ===================================================================

def load_and_adapt(args, device: str):
    """Load checkpoint, rebuild policy, run maml_adapt, return adapted state."""
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    saved_args = ckpt["args"]
    motor_centers = ckpt["motor_centers"]   # list of (mx, my)
    target_motor  = int(ckpt["target_motor"])
    sigma         = float(ckpt["sigma"])
    m_extra_train = float(ckpt["m_extra"])
    x0_train      = ckpt["x0_train"].to(device)

    # --- Target mass position --------------------------------------
    motor_xy = motor_centers[target_motor]
    if args.target_dx is not None:
        dx = args.target_dx
    else:
        dx = motor_xy[0]
    if args.target_dy is not None:
        dy = args.target_dy
    else:
        dy = motor_xy[1]
    dz = args.target_dz
    m_extra = args.mass if args.mass is not None else m_extra_train

    print(f"[Test] target motor #{target_motor}  centre=({motor_xy[0]*100:+.2f},"
          f"{motor_xy[1]*100:+.2f})cm  σ_train={sigma*100:.2f}cm")
    print(f"[Test] mass position (dx, dy, dz) = "
          f"({dx*100:+.2f}, {dy*100:+.2f}, {dz*100:+.2f})cm  m={m_extra*1e3:.1f}g")

    target_mass = compute_mass_params(m_extra, (dx, dy, dz)).to(device)

    # --- Rebuild policy with the right hover bias ------------------
    M_total = C.M_BASE + m_extra
    hover_thrust_u16 = (M_total * C.G * C.UINT16_MAX) / (4 * C.CF2_THRUST_MAX_PER_MOTOR)
    policy = PolicyMLP(hidden=int(saved_args["hidden"]),
                       hover_thrust_u16=hover_thrust_u16).to(device)
    policy.load_state_dict(ckpt["state_dict"])

    # --- Few-shot adaptation ---------------------------------------
    n_inner = (args.n_inner_steps
               if args.n_inner_steps is not None
               else int(saved_args["n_inner_steps"]))
    lr_inner = (args.lr_inner
                if args.lr_inner is not None
                else float(saved_args["lr_inner"]))
    inner_clip = float(saved_args.get("inner_grad_clip", 1.0))
    pos_w  = float(saved_args.get("pos_weight", 10.0))
    z_w    = float(saved_args.get("z_weight", 1.0))
    term_w = float(saved_args.get("terminal_weight", 50.0))
    tau_div = saved_args.get("tau_div", None)
    tau_div = tau_div if (tau_div is not None and tau_div > 0) else None
    obs_noise_scale = float(saved_args.get("obs_noise_scale", 0.0))
    obs_noise = ((C.OBS_NOISE_STD * obs_noise_scale).to(device)
                 if obs_noise_scale > 0 else None)
    n_steps_rollout = int(float(saved_args["t_sim"]) * C.NN_FREQ)
    dynamics_step = DYNAMICS[saved_args["dynamics"]]

    print(f"[Adapt] n_inner_steps={n_inner}  lr_inner={lr_inner}  "
          f"clip={inner_clip}  rollout={n_steps_rollout} steps")

    theta_adapt, losses = maml_adapt(
        policy, target_mass, x0_train,
        dynamics_step=dynamics_step,
        n_steps=n_steps_rollout,
        n_steps_adapt=n_inner,
        lr_inner=lr_inner,
        terminal_weight=term_w,
        pos_weight=pos_w,
        z_weight=z_w,
        obs_noise_std=obs_noise,
        tau_div=tau_div,
        inner_grad_clip=inner_clip,
        device=device,
        seed=int(saved_args.get("seed", 0)),
        verbose=True,
    )

    return policy, theta_adapt, m_extra, (dx, dy, dz), losses


# ===================================================================
#  Inference wrapper
# ===================================================================

def query_maml_nn(policy: PolicyMLP, theta: dict, state_12: np.ndarray,
                  target_pos: np.ndarray, device: str) -> np.ndarray:
    """One NN query with adapted parameters; returns (4,) action array."""
    state_shift = state_12.copy()
    state_shift[0] -= target_pos[0]
    state_shift[2] -= target_pos[1]
    state_shift[4] -= target_pos[2]

    s = torch.tensor(state_shift, dtype=torch.float32,
                     device=device).unsqueeze(0)             # (1, 12)
    rel = -s[:, [0, 2, 4]]                                   # (1, 3)
    with torch.no_grad():
        a = functional_call(policy, theta, (s, rel))         # (1, 4)
    return a[0].cpu().numpy()


# ===================================================================
#  Sim loop (Phase B: immediate NN takeover)
# ===================================================================

def run_sim(args, policy, theta_adapt, m_extra, r_offset, device: str):
    nn_target = np.array(args.target_pos, dtype=np.float64)
    init_pos  = np.array(args.init_pos,   dtype=np.float64)
    init_rpy  = np.array(args.init_rpy,   dtype=np.float64)

    PYB_FREQ  = 1000
    CTRL_FREQ = C.ATTITUDE_RATE          # 500 Hz
    nn_divider = CTRL_FREQ // C.NN_FREQ  # 5

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=init_pos.reshape(1, 3),
        initial_rpys=init_rpy.reshape(1, 3),
        physics=Physics.DYN_OFFSET,
        pyb_freq=PYB_FREQ,
        ctrl_freq=CTRL_FREQ,
        gui=args.gui,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    env.set_offset_mass(m_extra, r_offset)

    KF = env.KF
    M_total = C.M_BASE + m_extra
    HOVER_RPM_TASK = math.sqrt(M_total * C.G / (4 * C.KF))

    att_ctrl = CrazyflieAttitudeController()
    power = CrazyfliePowerDistribution()
    motor_filter = MotorDynamicsFilter(n_motors=4, tau=0.02, dt=1.0 / CTRL_FREQ)
    motor_filter.rpm = np.full(4, HOVER_RPM_TASK)

    n_steps = int(CTRL_FREQ * args.duration)
    print(f"[SIM] PYB={PYB_FREQ}Hz  CTRL={CTRL_FREQ}Hz  NN={C.NN_FREQ}Hz  "
          f"duration={args.duration}s  n_steps={n_steps}")
    print(f"[SIM] init_pos={init_pos}  init_rpy={init_rpy}  "
          f"NN target={nn_target}")

    log_t   = np.zeros(n_steps)
    log_pos = np.zeros((n_steps, 3))
    log_vel = np.zeros((n_steps, 3))
    log_rpy = np.zeros((n_steps, 3))
    log_rpms = np.zeros((n_steps, 4))
    log_thrust = np.zeros(n_steps)
    log_nn_cmd = np.zeros((n_steps, 4))

    action = np.full((1, 4), HOVER_RPM_TASK)
    START = time.time()

    # Start at hover thrust matching the offset-mass total weight.
    M_total_hover_u16 = (M_total * C.G * C.UINT16_MAX) / (4 * C.CF2_THRUST_MAX_PER_MOTOR)
    thrust_cmd = float(M_total_hover_u16)
    roll_des = 0.0
    pitch_des = 0.0
    yaw_rate_cmd = 0.0
    yaw_setpoint = float(init_rpy[2]) * 180.0 / math.pi

    for i in range(n_steps):
        obs, _, _, _, _ = env.step(action)
        pos, vel, rpy_deg, gyro_deg = obs_to_firmware_state(obs[0])

        # ---- NN query at NN_FREQ ----
        if i % nn_divider == 0:
            nn_state = obs_to_nn_state(obs[0])
            a = query_maml_nn(policy, theta_adapt, nn_state, nn_target, device)
            thrust_cmd = float(a[0])
            roll_des   = float(a[1])
            pitch_des  = float(a[2])
            yaw_rate_cmd = float(a[3])

        # ---- Yaw setpoint accumulation ----
        yaw_setpoint = cap_angle(
            yaw_setpoint + yaw_rate_cmd * C.ATTITUDE_UPDATE_DT)

        # ---- Attitude PID (500 Hz) ----
        roll_rate_d, pitch_rate_d, yaw_rate_d = att_ctrl.correct_attitude(
            rpy_deg[0], rpy_deg[1], rpy_deg[2],
            roll_des, pitch_des, yaw_setpoint)

        # ---- Rate PID (500 Hz) ----
        roll_cmd, pitch_cmd, yaw_cmd = att_ctrl.correct_rate(
            gyro_deg[0], gyro_deg[1], gyro_deg[2],
            roll_rate_d, pitch_rate_d, yaw_rate_d)
        yaw_cmd = -yaw_cmd

        if thrust_cmd <= 0:
            att_ctrl.reset_all(rpy_deg[0], rpy_deg[1], rpy_deg[2])
            yaw_setpoint = rpy_deg[2]
            action[0, :] = 0
            log_t[i] = i / CTRL_FREQ
            log_pos[i] = pos; log_vel[i] = vel; log_rpy[i] = rpy_deg
            continue

        motor_pwm = power.distribute(thrust_cmd, roll_cmd, pitch_cmd, yaw_cmd)
        rpms = pwm_to_rpm(motor_pwm, KF, truncate_8bit=True)
        rpms = motor_filter.apply(rpms)
        action[0, :] = rpms

        log_t[i] = i / CTRL_FREQ
        log_pos[i] = pos
        log_vel[i] = vel
        log_rpy[i] = rpy_deg
        log_rpms[i] = rpms
        log_thrust[i] = thrust_cmd
        log_nn_cmd[i] = [thrust_cmd, roll_des, pitch_des, yaw_rate_cmd]

        if args.gui:
            sync(i, START, 1.0 / CTRL_FREQ)

    env.close()
    print("[SIM] Done.")

    return dict(t=log_t, pos=log_pos, vel=log_vel, rpy=log_rpy,
                rpms=log_rpms, thrust=log_thrust, nn_cmd=log_nn_cmd,
                target=nn_target, init_pos=init_pos,
                m_extra=m_extra, r_offset=np.array(r_offset),
                hover_thrust_u16=M_total_hover_u16)


# ===================================================================
#  Plotting
# ===================================================================

def plot_results(data, out_path: str, adapt_losses=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[PLOT] matplotlib not found.")
        return

    t = data['t']; pos = data['pos']; vel = data['vel']; rpy = data['rpy']
    rpms = data['rpms']; thrust = data['thrust']; nn_cmd = data['nn_cmd']
    target = data['target']; hover = data['hover_thrust_u16']

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex='col')
    fig.suptitle(
        f"MAML controller — m_extra={data['m_extra']*1e3:.1f}g, "
        f"r_offset=({data['r_offset'][0]*100:+.2f},"
        f"{data['r_offset'][1]*100:+.2f},"
        f"{data['r_offset'][2]*100:+.2f})cm",
        fontsize=13)

    ax = axes[0, 0]
    ax.plot(t, pos[:, 0], label='x'); ax.plot(t, pos[:, 1], label='y')
    ax.plot(t, pos[:, 2], label='z')
    for j, c in enumerate(['C0', 'C1', 'C2']):
        ax.axhline(target[j], ls='--', color=c, alpha=0.5,
                   label=f'tgt {["x","y","z"][j]}={target[j]:.2f}')
    ax.set_ylabel('Position [m]'); ax.legend(fontsize=7, ncol=2)
    ax.set_title('Position'); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, vel[:, 0], label='vx'); ax.plot(t, vel[:, 1], label='vy')
    ax.plot(t, vel[:, 2], label='vz')
    ax.set_ylabel('Velocity [m/s]'); ax.legend(fontsize=7)
    ax.set_title('Velocity'); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, rpy[:, 0], label='roll'); ax.plot(t, rpy[:, 1], label='pitch')
    ax.plot(t, rpy[:, 2], label='yaw', alpha=0.6)
    ax.set_ylabel('Angle [deg]'); ax.legend(fontsize=7)
    ax.set_title('Attitude'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, nn_cmd[:, 1], label='NN roll [deg]', alpha=0.8)
    ax.plot(t, nn_cmd[:, 2], label='NN pitch [deg]', alpha=0.8)
    ax.plot(t, nn_cmd[:, 3], label='NN yaw_rate [deg/s]', alpha=0.6)
    ax.set_ylabel('NN attitude cmd'); ax.legend(fontsize=7)
    ax.set_title('NN outputs'); ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(t, thrust, color='black', label='thrust')
    ax.axhline(hover, ls=':', color='gray', label=f'hover≈{hover:.0f}')
    ax.set_ylabel('Thrust [u16]'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7); ax.set_title('Thrust'); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    for m in range(4):
        ax.plot(t, rpms[:, m], label=f'M{m + 1}', alpha=0.7)
    ax.set_ylabel('RPM'); ax.set_xlabel('Time [s]')
    ax.legend(fontsize=7, ncol=2); ax.set_title('Motor RPMs')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Saved to {out_path}")

    if adapt_losses:
        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(adapt_losses) + 1), adapt_losses, marker='o')
        ax.set_xlabel('Inner-step #'); ax.set_ylabel('Adaptation loss')
        ax.set_title('Few-shot adaptation curve')
        ax.grid(True, alpha=0.3)
        adapt_path = out_path.replace('.png', '_adapt.png')
        fig2.tight_layout()
        fig2.savefig(adapt_path, dpi=150)
        plt.close(fig2)
        print(f"[PLOT] Saved to {adapt_path}")


# ===================================================================
#  CLI
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Test a MAML controller with offset-mass dynamics in PyBullet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--weights', required=True, help='MAML checkpoint (.pt)')

    # target mass override (default = motor centre from checkpoint)
    p.add_argument('--target-dx', type=float, default=None,
                   help='override mass dx [m] (default: target motor centre)')
    p.add_argument('--target-dy', type=float, default=None,
                   help='override mass dy [m] (default: target motor centre)')
    p.add_argument('--target-dz', type=float, default=0.0,
                   help='mass dz [m]')
    p.add_argument('--mass', type=float, default=None,
                   help='override extra mass [kg] (default: training value)')

    # adaptation overrides (default = checkpoint values)
    p.add_argument('--n-inner-steps', type=int, default=None)
    p.add_argument('--lr-inner', type=float, default=None)

    # sim setup
    p.add_argument('--target-pos', nargs=3, type=float, default=[0.0, 0.0, 1.0],
                   metavar=('X', 'Y', 'Z'),
                   help='NN regulation target [m]')
    p.add_argument('--init-pos', nargs=3, type=float, default=[0.0, 0.0, 1.0],
                   metavar=('X', 'Y', 'Z'),
                   help='initial drone position [m] (Phase B starts here)')
    p.add_argument('--init-rpy', nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   metavar=('R', 'P', 'Y'),
                   help='initial roll/pitch/yaw [rad]')
    p.add_argument('--duration', type=float, default=8.0,
                   help='simulation duration [s]')
    p.add_argument('--gui', action='store_true', help='enable PyBullet GUI')
    p.add_argument('--no-plot', dest='plot', action='store_false',
                   help='disable post-flight plot')

    p.add_argument('--device', type=str, default=None,
                   help='cpu or cuda (default: auto)')
    return p.parse_args()


def main():
    args = parse_args()
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[Device] {device}")

    policy, theta_adapt, m_extra, r_offset, adapt_losses = load_and_adapt(args, device)

    data = run_sim(args, policy, theta_adapt, m_extra, r_offset, device)

    if args.plot:
        out_dir = os.path.dirname(os.path.abspath(args.weights))
        base = os.path.splitext(os.path.basename(args.weights))[0]
        out_path = os.path.join(out_dir, base + "_test_phaseB.png")
        plot_results(data, out_path, adapt_losses=adapt_losses)


if __name__ == "__main__":
    main()
