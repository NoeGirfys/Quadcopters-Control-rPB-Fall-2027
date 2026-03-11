#!/usr/bin/env python3
"""
Validate the trained NN controller inside gym-pybullet-drones.

Loads a saved checkpoint, builds the MLP policy, runs the CF2X simulation
with a selectable BaseAviary physics mode, and checks whether the drone
regulates to a chosen target position from various initial positions.

The pipeline at each control step:
    1. Read state from pybullet  →  [pos, vel, rpy, rpy_rates]
    2. Map to Goffin ordering    →  [x,xd,y,yd,z,zd,φ,φd,θ,θd,ψ,ψd]
    3. Forward through MLP       →  [F, τx, τy, τz]
    4. Invert allocation matrix  →  [ω₁², ω₂², ω₃², ω₄²]
    5. Take sqrt, clip           →  RPMs sent to env.step()

Usage:
    python validate_nn_pybullet.py
    python validate_nn_pybullet.py --gui
    python validate_nn_pybullet.py --weights trained_weights_cf2x_nonlinear_h64_ep2000_lr1e-03.pt
    python validate_nn_pybullet.py --physics pyb_drag
"""

import sys, os, argparse
import numpy as np
import torch
import torch.nn as nn

# ─── Make sure gym-pybullet-drones is importable ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYB_DIR = os.path.join(SCRIPT_DIR, "gym-pybullet-drones-main")
if PYB_DIR not in sys.path:
    sys.path.insert(0, PYB_DIR)

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics


def rotation_matrix_zyx_np(phi, theta, psi):
    """Rotation matrix matching the convention used in training."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)
    return np.array([
        [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
        [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
        [-sth,       sphi * cth,                          cphi * cth],
    ], dtype=np.float64)


def world_ang_vel_to_body_rates(rpy, ang_v_world):
    """Convert world-frame angular velocity to body-frame rates [p, q, r].

    In Physics.DYN, gym-pybullet-drones stores body-frame rates in env.rpy_rates.
    In all other modes, only env.ang_v is available; it is world-frame angular
    velocity, so we map it back to the body frame with R^T ω_world.
    """
    R = rotation_matrix_zyx_np(rpy[0], rpy[1], rpy[2])
    return R.T @ np.asarray(ang_v_world, dtype=np.float64)


def physics_from_string(name):
    """Map CLI string like 'dyn' or 'pyb_drag' to the Physics enum."""
    for physics in Physics:
        if physics.value == name:
            return physics
    allowed = ", ".join(p.value for p in Physics)
    raise ValueError(f"Unknown physics mode '{name}'. Allowed: {allowed}")


def resolve_weights_path(script_dir, weights_arg):
    """Resolve checkpoint path from absolute or script-relative input."""
    if os.path.isabs(weights_arg):
        return weights_arg
    candidate = os.path.join(script_dir, weights_arg)
    if os.path.isfile(candidate):
        return candidate
    return weights_arg


def build_plot_filename(weights_path, physics_mode, target_pos, duration):
    """Create a descriptive plot filename from the checkpoint and validation setup."""
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    tz = f"{target_pos[2]:.2f}".replace("-", "m").replace(".", "p")
    dur = f"{duration:.1f}".replace(".", "p")
    return f"validation_{stem}_phys-{physics_mode}_targetz-{tz}_T-{dur}s.png"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. MLP definition (must match training)
# ═══════════════════════════════════════════════════════════════════════════════
class PolicyMLP(nn.Module):
    def __init__(self, x_scale, u_max, hidden=64):
        super().__init__()
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        self.register_buffer("u_max",   torch.tensor(u_max,   dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(12, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        x_n = x / self.x_scale
        raw = self.net(x_n)
        thrust  = torch.sigmoid(raw[..., [0]]) * self.u_max[0]
        torques = torch.tanh(raw[..., 1:])     * self.u_max[1:]
        return torch.cat([thrust, torques], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Wrench → RPM Allocation  (CF2X configuration)
# ═══════════════════════════════════════════════════════════════════════════════
def build_allocation_matrix(KF, KM, L):
    """Build the 4×4 matrix M such that  [F, τx, τy, τz]^T = M @ [ω₁², ω₂², ω₃², ω₄²]^T.

    Motor layout for CF2X (X-configuration):
        Motor 0: front-left   (CCW)
        Motor 1: front-right  (CW)
        Motor 2: rear-right   (CCW)
        Motor 3: rear-left    (CW)

    From pybullet-drones BaseAviary._dynamics() for CF2X:
        x_torque = -(f0 + f1 - f2 - f3) * L/√2
        y_torque = (-f0 + f1 + f2 - f3) * L/√2
        z_torque = -km*ω0² + km*ω1² - km*ω2² + km*ω3²
    where fi = KF * ωi².
    """
    a = KF * L / np.sqrt(2)
    M = np.array([
        [ KF,   KF,   KF,   KF  ],     # F     = KF*(ω0²+ω1²+ω2²+ω3²)
        [ -a,   -a,    a,    a   ],     # τx    (roll)
        [ -a,    a,    a,   -a   ],     # τy    (pitch)
        [ -KM,   KM,  -KM,   KM ],     # τz    (yaw)
    ])
    return M


def wrench_to_rpm(wrench, M_inv, max_rpm):
    """Convert [F, τx, τy, τz] → RPMs using the inverse allocation matrix.

    Parameters
    ----------
    wrench : np.ndarray (4,)
        [F, tau_x, tau_y, tau_z]
    M_inv : np.ndarray (4, 4)
        Inverse of the allocation matrix.
    max_rpm : float
        Maximum allowable RPM.

    Returns
    -------
    np.ndarray (4,)
        RPM values for each motor.
    """
    omega_sq = M_inv @ wrench
    # Clip to [0, max_rpm²] (negative ω² is physically impossible)
    omega_sq = np.clip(omega_sq, 0, max_rpm**2)
    return np.sqrt(omega_sq)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. State extraction helper
# ═══════════════════════════════════════════════════════════════════════════════
def pybullet_state_to_goffin(env, drone_idx=0, target_pos=np.zeros(3)):
    """Extract the 12-dim ERROR state from pybullet in Goffin's ordering.

    Training was done on an error-state model centered at the origin. During
    validation, we can therefore target any absolute hover point by feeding the
    network the state error relative to ``target_pos``.

    Goffin ordering:
        [x, xdot, y, ydot, z, zdot, phi, p, theta, q, psi, r]

    Notes
    -----
    - Position/velocity are in world frame, as in the training script.
    - For Physics.DYN we use env.rpy_rates directly (body-frame rates).
    - For all other BaseAviary physics modes, env.rpy_rates does not exist,
      so we reconstruct [p, q, r] from env.ang_v.
    """
    pos = env.pos[drone_idx]
    vel = env.vel[drone_idx]
    rpy = env.rpy[drone_idx]

    if hasattr(env, "rpy_rates"):
        body_rates = env.rpy_rates[drone_idx]
    else:
        body_rates = world_ang_vel_to_body_rates(rpy, env.ang_v[drone_idx]).astype(np.float32)

    err_pos = pos - target_pos

    state = np.array([
        err_pos[0], vel[0],
        err_pos[1], vel[1],
        err_pos[2], vel[2],
        rpy[0], body_rates[0],
        rpy[1], body_rates[1],
        rpy[2], body_rates[2],
    ], dtype=np.float32)
    return state


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main simulation loop
# ═══════════════════════════════════════════════════════════════════════════════
def run_validation(policy, checkpoint, offset_xyz, gui=False, duration=4.0,
                   target_pos=np.array([0.0, 0.0, 0.5]), physics_mode=Physics.DYN):
    """Run the NN controller in pybullet-drones and record the trajectory.

    Parameters
    ----------
    policy : PolicyMLP
        Trained neural network controller.
    checkpoint : dict
        Saved checkpoint with allocation parameters.
    offset_xyz : np.ndarray (3,)
        Initial offset from target position [dx, dy, dz].
        The drone starts at  target_pos + offset_xyz.
    gui : bool
        Whether to show the 3D viewer.
    duration : float
        Simulation duration in seconds.
    target_pos : np.ndarray (3,)
        Target hover position.  The NN sees error = current - target.

    Returns
    -------
    traj : dict with 'pos', 'vel', 'rpy', 'time', 'wrench', 'rpm' arrays.
    """
    KF  = checkpoint["KF"]
    KM  = checkpoint["KM"]
    L   = checkpoint["L"]
    MAX_RPM = checkpoint["MAX_RPM"]
    CTRL_FREQ = checkpoint["ctrl_freq"]

    # Build allocation
    M_alloc = build_allocation_matrix(KF, KM, L)
    M_inv   = np.linalg.inv(M_alloc)

    # Initial position = target + offset
    init_xyz = target_pos + np.array(offset_xyz)

    # Create environment
    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=np.array([[init_xyz[0], init_xyz[1], init_xyz[2]]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=physics_mode,
        ctrl_freq=CTRL_FREQ,
        gui=gui,
    )

    N_steps = int(duration * CTRL_FREQ)

    # Storage
    pos_log   = np.zeros((N_steps, 3))
    vel_log   = np.zeros((N_steps, 3))
    rpy_log   = np.zeros((N_steps, 3))
    wrench_log = np.zeros((N_steps, 4))
    rpm_log   = np.zeros((N_steps, 4))
    time_log  = np.zeros(N_steps)

    print(f"  Running {N_steps} steps from offset={offset_xyz}, target={target_pos}, physics={physics_mode.value} ...")

    for k in range(N_steps):
        # 1. Get state (error relative to target)
        state_12 = pybullet_state_to_goffin(env, 0, target_pos=target_pos)

        # 2. NN forward pass
        with torch.no_grad():
            state_t = torch.tensor(state_12, dtype=torch.float32).unsqueeze(0)
            wrench = policy(state_t).squeeze(0).numpy()     # [F, τx, τy, τz]

        # 3. Convert wrench → RPM
        rpms = wrench_to_rpm(wrench, M_inv, MAX_RPM)

        # 4. Step the environment
        action = np.array([rpms])  # shape (1, 4) for single drone
        obs, _, _, _, _ = env.step(action)

        # 5. Log
        pos_log[k]    = env.pos[0]
        vel_log[k]    = env.vel[0]
        rpy_log[k]    = env.rpy[0]
        wrench_log[k] = wrench
        rpm_log[k]    = rpms
        time_log[k]   = (k + 1) / CTRL_FREQ

    env.close()

    return {
        "pos": pos_log, "vel": vel_log, "rpy": rpy_log,
        "wrench": wrench_log, "rpm": rpm_log, "time": time_log,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Results display
# ═══════════════════════════════════════════════════════════════════════════════
def print_results(trajs, labels, target_pos):
    """Print a summary table of terminal errors."""
    print(f"\n  Target position: {target_pos}")
    print(f"\n{'Offset (dx,dy,dz)':>25s}  │  {'|pos_err|':>10s}  {'|vel_T|':>10s}  {'max|rpy|':>10s}")
    print("─" * 70)
    for label, traj in zip(labels, trajs):
        pos_T = traj["pos"][-1]
        pos_err = np.linalg.norm(pos_T - target_pos)
        vel_T = traj["vel"][-1]
        rpy_max = np.abs(traj["rpy"]).max()
        print(f"  {label:>23s}  │  {pos_err:10.5f}  "
              f"{np.linalg.norm(vel_T):10.5f}  {np.rad2deg(rpy_max):10.3f}°")


def save_plots(trajs, labels, target_pos, physics_mode, filename="validation_results.png"):
    """Save position trajectories plot (showing error from target)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    fig, axes = plt.subplots(len(trajs), 3, figsize=(14, 3 * len(trajs)),
                              sharex=True, squeeze=False)
    coord_names = ["x (m)", "y (m)", "z (m)"]
    for i, (traj, label) in enumerate(zip(trajs, labels)):
        t = traj["time"]
        for j, name in enumerate(coord_names):
            err = traj["pos"][:, j] - target_pos[j]
            axes[i, j].plot(t, err, linewidth=1.5)
            axes[i, j].axhline(0, color="k", linewidth=0.5, linestyle="--")
            axes[i, j].set_ylabel(f"err {name}")
            axes[i, j].grid(True, alpha=0.3)
            if i == 0:
                axes[i, j].set_title(f"Error in {name}")
        axes[i, 0].annotate(label, xy=(0.02, 0.95), xycoords="axes fraction",
                             fontsize=8, va="top")
    for j in range(3):
        axes[-1, j].set_xlabel("Time (s)")

    fig.suptitle(f"NN Controller Validation in pybullet-drones ({physics_mode.value})", fontsize=13)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"\n[Saved] {filename}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Entry point
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open 3D viewer", default=True)
    parser.add_argument("--weights", type=str, default="trained_weights_cf2x_nonlinear_h64_ep2000_lr1e-03.pt",
                        help="Path to saved weights checkpoint (absolute or script-relative)")
    parser.add_argument("--physics", type=str, default="dyn",
                        choices=[p.value for p in Physics],
                        help="BaseAviary physics mode used for validation")
    parser.add_argument("--duration", type=float, default=4.0,
                        help="Simulation duration (s)")
    parser.add_argument("--target_z", type=float, default=0.5,
                        help="Target hover altitude (m)")
    args = parser.parse_args()

    physics_mode = physics_from_string(args.physics)

    # ── Load weights ──
    ckpt_path = resolve_weights_path(SCRIPT_DIR, args.weights)
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] Weights file not found: {ckpt_path}")
        print("        Run  train_nn_cf2x.py  first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = PolicyMLP(
        x_scale=ckpt["x_scale"],
        u_max=ckpt["u_max"],
        hidden=ckpt["hidden"],
    )
    policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    print(f"[Loaded] {ckpt_path}")
    print(f"  ctrl_freq={ckpt['ctrl_freq']}, Ts={ckpt['Ts']:.5f}")
    if "train_dynamics" in ckpt:
        print(f"  trained_with={ckpt['train_dynamics']}")
    print(f"  validation_physics={physics_mode.value}")

    # ── Target position (hover at this point) ──
    target_pos = np.array([0.0, 0.0, args.target_z])

    # ── Test offsets from target (same cube as training ±0.3m) ──
    test_offsets = [
        (0.0,  0.0,  0.0),     # already at target (trivial)
        (0.5,  0.1,  0.05),     # cube corner
        (-0.28, -0.1, -0.2),    # opposite corner
        (0.05, -0.38,  0.0),     # edge
        (0.0,  0.0,  0.53),     # above target
        (-0.2,  0.1, -0.15),   # random
    ]

    trajs = []
    labels = []
    for offset in test_offsets:
        traj = run_validation(policy, ckpt, np.array(offset),
                              gui=args.gui, duration=args.duration,
                              target_pos=target_pos, physics_mode=physics_mode)
        trajs.append(traj)
        labels.append(f"({offset[0]:+.1f}, {offset[1]:+.1f}, {offset[2]:+.1f})")

    # ── Print & plot ──
    print_results(trajs, labels, target_pos)
    plot_name = build_plot_filename(ckpt_path, physics_mode.value, target_pos, args.duration)
    save_plots(trajs, labels, target_pos, physics_mode,
               os.path.join(SCRIPT_DIR, plot_name))

    print("\nDone!")