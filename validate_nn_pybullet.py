#!/usr/bin/env python3
"""
Validate the trained NN controller inside gym-pybullet-drones.

Supports multiple physics backends:
  --physics DYN   Explicit dynamics (default, matches training)
  --physics PYB   PyBullet physics engine (more realistic)

Usage:
    python validate_nn_pybullet.py --weights weights_nonlinear_quat_h64.pt
    python validate_nn_pybullet.py --weights weights_linear_h64.pt --physics PYB
    python validate_nn_pybullet.py --gui
"""

import sys, os, argparse
import numpy as np
import torch
import torch.nn as nn
import pybullet as p_module

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYB_DIR = os.path.join(SCRIPT_DIR, "gym-pybullet-drones-main")
if PYB_DIR not in sys.path:
    sys.path.insert(0, PYB_DIR)

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# =====================================================================
# 1. MLP (must match training)
# =====================================================================

class PolicyMLP(nn.Module):
    def __init__(self, x_scale, u_max, hidden=64):
        super().__init__()
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        self.register_buffer("u_max",   torch.tensor(u_max,   dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(12, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        raw = self.net(x / self.x_scale)
        thrust  = torch.sigmoid(raw[..., [0]]) * self.u_max[0]
        torques = torch.tanh(raw[..., 1:])     * self.u_max[1:]
        return torch.cat([thrust, torques], dim=-1)

# =====================================================================
# 2. Wrench -> RPM allocation (CF2X)
# =====================================================================

def build_allocation_matrix(KF, KM, L):
    a = KF * L / np.sqrt(2)
    return np.array([
        [ KF,  KF,  KF,  KF],
        [ -a,  -a,   a,   a],
        [ -a,   a,   a,  -a],
        [-KM,  KM, -KM,  KM],
    ])

def wrench_to_rpm(wrench, M_inv, max_rpm):
    omega_sq = np.clip(M_inv @ wrench, 0, max_rpm**2)
    return np.sqrt(omega_sq)

# =====================================================================
# 3. State extraction — works for both DYN and PYB modes
# =====================================================================

def get_body_angular_rates(env, drone_idx=0):
    """Get body-frame angular rates [p, q, r].

    - DYN mode:  uses env.rpy_rates (maintained by _dynamics)
    - PYB mode:  computes R^T @ ang_v  (world-frame -> body-frame)
    """
    if hasattr(env, 'rpy_rates'):
        return env.rpy_rates[drone_idx].copy()
    else:
        # ang_v is world-frame angular velocity from getBaseVelocity
        ang_v_world = env.ang_v[drone_idx]
        # Get rotation matrix from quaternion
        quat = env.quat[drone_idx]
        R = np.array(p_module.getMatrixFromQuaternion(quat)).reshape(3, 3)
        # Body-frame rates = R^T @ world-frame rates
        return R.T @ ang_v_world


def pybullet_state_to_12(env, drone_idx=0, target_pos=np.zeros(3)):
    """Extract 12D error state: [x,vx, y,vy, z,vz, phi,p, theta,q, psi,r]."""
    pos = env.pos[drone_idx]
    vel = env.vel[drone_idx]
    rpy = env.rpy[drone_idx]
    ang_rates = get_body_angular_rates(env, drone_idx)
    err = pos - target_pos
    return np.array([
        err[0], vel[0], err[1], vel[1], err[2], vel[2],
        rpy[0], ang_rates[0], rpy[1], ang_rates[1], rpy[2], ang_rates[2],
    ], dtype=np.float32)

# =====================================================================
# 4. Simulation loop
# =====================================================================

def run_validation(policy, ckpt, offset_xyz, physics_mode, gui=False,
                   duration=4.0, target_pos=np.array([0.,0.,0.5])):
    KF = ckpt["KF"]; KM = ckpt["KM"]; L = ckpt["L"]
    MAX_RPM = ckpt["MAX_RPM"]; CF = ckpt["ctrl_freq"]

    M_alloc = build_allocation_matrix(KF, KM, L)
    M_inv = np.linalg.inv(M_alloc)
    init_xyz = target_pos + np.array(offset_xyz)

    physics = {"DYN": Physics.DYN, "PYB": Physics.PYB}[physics_mode]

    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=np.array([init_xyz]),
        initial_rpys=np.array([[0.,0.,0.]]),
        physics=physics, ctrl_freq=CF, gui=gui,
    )

    N = int(duration * CF)
    pos_log=np.zeros((N,3)); vel_log=np.zeros((N,3))
    rpy_log=np.zeros((N,3)); wrench_log=np.zeros((N,4))
    rpm_log=np.zeros((N,4)); time_log=np.zeros(N)

    for k in range(N):
        s12 = pybullet_state_to_12(env, 0, target_pos)
        with torch.no_grad():
            w = policy(torch.tensor(s12).unsqueeze(0)).squeeze(0).numpy()
        rpms = wrench_to_rpm(w, M_inv, MAX_RPM)
        env.step(np.array([rpms]))

        pos_log[k]=env.pos[0]; vel_log[k]=env.vel[0]
        rpy_log[k]=env.rpy[0]; wrench_log[k]=w
        rpm_log[k]=rpms; time_log[k]=(k+1)/CF

    env.close()
    return {"pos":pos_log, "vel":vel_log, "rpy":rpy_log,
            "wrench":wrench_log, "rpm":rpm_log, "time":time_log}

# =====================================================================
# 5. Display results
# =====================================================================

def print_results(trajs, labels, target_pos):
    print(f"\n  Target: {target_pos}")
    print(f"{'Offset':>25s}  |  {'|pos_err|':>10s}  {'|vel_T|':>10s}  {'max|rpy|':>10s}")
    print("-"*70)
    for lb, tr in zip(labels, trajs):
        pe = np.linalg.norm(tr["pos"][-1] - target_pos)
        ve = np.linalg.norm(tr["vel"][-1])
        rm = np.rad2deg(np.abs(tr["rpy"]).max())
        print(f"  {lb:>23s}  |  {pe:10.5f}  {ve:10.5f}  {rm:10.3f} deg")


def save_plots(trajs, labels, target_pos, filename):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available"); return

    fig, axes = plt.subplots(len(trajs), 3, figsize=(14, 3*len(trajs)),
                              sharex=True, squeeze=False)
    names = ["x (m)", "y (m)", "z (m)"]
    for i,(tr,lb) in enumerate(zip(trajs,labels)):
        for j,nm in enumerate(names):
            axes[i,j].plot(tr["time"], tr["pos"][:,j]-target_pos[j], lw=1.5)
            axes[i,j].axhline(0, color="k", lw=0.5, ls="--")
            axes[i,j].set_ylabel(f"err {nm}"); axes[i,j].grid(True, alpha=0.3)
            if i==0: axes[i,j].set_title(f"Error {nm}")
        axes[i,0].annotate(lb, xy=(0.02,0.95), xycoords="axes fraction", fontsize=8, va="top")
    for j in range(3): axes[-1,j].set_xlabel("Time (s)")
    fig.suptitle(os.path.basename(filename).replace(".png",""), fontsize=13)
    fig.tight_layout(); fig.savefig(filename, dpi=150); plt.close(fig)
    print(f"[Saved] {filename}")

# =====================================================================
# 6. Main
# =====================================================================

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--gui", action="store_true")
    pa.add_argument("--weights", type=str, default=None,
                    help="Path to weights checkpoint (auto-detected if omitted)")
    pa.add_argument("--physics", type=str, default="DYN", choices=["DYN","PYB"])
    pa.add_argument("--duration", type=float, default=4.0)
    pa.add_argument("--target_z", type=float, default=0.5)
    args = pa.parse_args()

    # ---- Find weights file ----
    if args.weights is None:
        # Auto-detect: find first weights_*.pt in script dir
        candidates = sorted([f for f in os.listdir(SCRIPT_DIR) if f.startswith("weights_") and f.endswith(".pt")])
        if not candidates:
            print("[ERROR] No weights_*.pt found. Run train_nn_cf2x.py first."); sys.exit(1)
        args.weights = candidates[0]
        print(f"[Auto] Using {args.weights}")

    ckpt_path = os.path.join(SCRIPT_DIR, args.weights)
    if not os.path.isfile(ckpt_path):
        print(f"[ERROR] {ckpt_path} not found"); sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = PolicyMLP(x_scale=ckpt["x_scale"], u_max=ckpt["u_max"], hidden=ckpt["hidden"])
    policy.load_state_dict(ckpt["state_dict"]); policy.eval()

    train_dyn = ckpt.get("dynamics", "unknown")
    print(f"[Loaded] {args.weights}  (trained: {train_dyn})")
    print(f"[Validate] physics={args.physics}")

    target_pos = np.array([0., 0., args.target_z])

    offsets = [
        (0.0, 0.0, 0.0), (0.3, 0.3, 0.3), (-0.3,-0.3,-0.3),
        (0.3,-0.3, 0.0), (0.0, 0.0, 0.3), (-0.2, 0.1,-0.15),
    ]

    trajs, labels = [], []
    for off in offsets:
        tr = run_validation(policy, ckpt, np.array(off), args.physics,
                            gui=args.gui, duration=args.duration,
                            target_pos=target_pos)
        trajs.append(tr)
        labels.append(f"({off[0]:+.1f},{off[1]:+.1f},{off[2]:+.1f})")

    print_results(trajs, labels, target_pos)

    # ---- Plot with descriptive filename ----
    # e.g. validation_nonlinear_quat_h64_PYB.png
    tag = os.path.basename(args.weights).replace("weights_","").replace(".pt","")
    plot_name = f"validation_{tag}_{args.physics}.png"
    save_plots(trajs, labels, target_pos, os.path.join(SCRIPT_DIR, plot_name))
    print("\nDone!")