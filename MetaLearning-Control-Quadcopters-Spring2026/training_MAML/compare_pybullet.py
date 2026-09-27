"""Compare three controllers in PyBullet over a sweep of test masses.

Given a MAML checkpoint and a baseline checkpoint, a start position, a target
position and a list of test masses, this runs the closed-loop PyBullet
simulation (the firmware DYN_OFFSET pipeline of ``test_maml_pybullet.py``) for
three controllers at each mass:

  * ``baseline``        : the joint-trained baseline, deployed as-is (no adapt),
  * ``baseline adapted``: the baseline after K-shot few-shot adaptation,
  * ``MAML adapted``    : the MAML meta-init after the same K-shot adaptation,

and plots the resulting position trajectories (x, y, z) so the three can be
compared at a glance, one row per mass.

The few-shot budget (K), inner learning rate, number of gradient steps,
grad-clip, roll-out horizon, curriculum threshold and observation-noise scale
are all taken from the MAML checkpoint. For each test mass the adaptation
support is sampled FRESH at that mass (and the requested attachment offset),
so each controller is adapted to the very task it is then flown on.

By default the mass is centred (``--target-dx/dy/dz`` = 0), which lets heavier
masses be tested without the offset torque driving a motor into saturation.

Usage:
    cd training_MAML
    python compare_pybullet.py \\
        --maml-ckpt     maml_nonlinear_h64_o1_n6_izar_offdirmag_ep500.pt \\
        --baseline-ckpt baseline_nonlinear_h64_n6_izar_offdirmag_ep500.pt \\
        --masses 4 8 12 16 \\
        --init-pos 0 0 1 --target-pos 0 0 1.5
"""
import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import csv
import math

from maml_lib import (
    config as C,
    PolicyMLP, compute_mass_params_batched, maml_adapt, DYNAMICS,
)
from maml_lib.cost import trajectory_cost
# Reuse the exact closed-loop NN simulation + state extraction.
from test_maml_pybullet import run_sim, obs_to_nn_state

# For the pure-firmware (4-PID, no NN) reference controller.
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from crazyflie_firmware.firmware import CrazyflieFirmwarePID
from circle_comparison_simu_and_real.cf_firmware_pid_sim import (
    MotorDynamicsFilter, obs_to_firmware_state, pwm_to_rpm,
)


# ===================================================================
#  Adaptation set-up (hyper-parameters from the MAML checkpoint)
# ===================================================================

def adapt_hparams(maml_ckpt: dict) -> dict:
    a = maml_ckpt["args"]
    ts = maml_ckpt["task_set"]
    return dict(
        n_inner   = int(a.get("n_inner_steps", 1)),
        lr_inner  = float(a.get("lr_inner", 0.05)),
        clip      = float(a.get("inner_grad_clip", 1.0)),
        K         = int(a.get("k_samples", 5) or 5),
        term_w    = float(a.get("terminal_weight", 50.0)),
        pos_w     = float(a.get("pos_weight", 10.0)),
        z_w       = float(a.get("z_weight", 1.0)),
        t_sim     = float(a.get("t_sim", 3.0)),
        tau       = (a.get("tau_end") or a.get("tau_div") or None),
        half_side = float(ts.get("half_side", 0.2)),
        hidden    = int(a.get("hidden", 64)),
        dyn       = DYNAMICS[a.get("dynamics", "nonlinear")],
        seed      = int(a.get("seed", 0)),
        noise_scale = float(a.get("obs_noise_scale", 0.0)),
    )


def load_policy(ckpt_path: str, hidden: int, device: str) -> PolicyMLP:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    pol = PolicyMLP(hidden=hidden).to(device)
    pol.load_state_dict(ck["state_dict"])
    pol.eval()
    return pol


def identity_theta(policy: PolicyMLP) -> dict:
    """Parameter dict of the un-adapted policy (for the no-adaptation case)."""
    return {n: p.detach().clone() for n, p in policy.named_parameters()}


def sample_support(K: int, half_side: float, m_kg: float, r_offset, rng):
    """K regulation-to-origin support points at the given (centred/offset) mass."""
    x0 = torch.zeros(K, 12)
    p = torch.from_numpy(rng.uniform(-half_side, half_side, (K, 3)).astype(np.float32))
    x0[:, 0], x0[:, 2], x0[:, 4] = p[:, 0], p[:, 1], p[:, 2]
    mass = compute_mass_params_batched(
        np.full(K, m_kg, dtype=np.float32),
        np.tile(np.asarray(r_offset, dtype=np.float32), (K, 1)))
    return x0, mass


def adapt_to_mass(policy, m_kg, r_offset, hp, device):
    """K-shot adaptation of ``policy`` to a (mass, offset) task; returns theta."""
    rng = np.random.default_rng(hp["seed"])
    x0, mass = sample_support(hp["K"], hp["half_side"], m_kg, r_offset, rng)
    ons = (C.OBS_NOISE_STD * hp["noise_scale"]) if hp["noise_scale"] > 0 else None
    n_steps = int(hp["t_sim"] * C.NN_FREQ)
    theta, _ = maml_adapt(
        policy, mass.to(device), x0.to(device),
        dynamics_step=hp["dyn"], n_steps=n_steps, n_steps_adapt=hp["n_inner"],
        lr_inner=hp["lr_inner"], terminal_weight=hp["term_w"],
        pos_weight=hp["pos_w"], z_weight=hp["z_w"], obs_noise_std=ons,
        tau_div=hp["tau"], inner_grad_clip=hp["clip"], device=device,
        seed=hp["seed"], verbose=False)
    return theta


# ===================================================================
#  Pure-firmware reference (4-PID cascade, no neural network)
# ===================================================================

def run_sim_pid(sim_args, m_extra, r_offset, device):
    """Closed-loop sim driven by the FULL firmware cascade PID (no NN).

    The complete controller (position -> velocity -> attitude -> rate ->
    motors) flies the drone from ``init_pos`` to ``target_pos``. Unlike the
    NN controllers, this one keeps the firmware's velocity-Z integrator, so
    it rejects the payload weight with little steady-state error on its own.
    Returns the same log dict shape as ``run_sim`` (keys ``t``, ``pos``...).
    """
    target   = np.array(sim_args.target_pos, dtype=np.float64)
    init_pos = np.array(sim_args.init_pos,   dtype=np.float64)
    init_rpy = np.array(sim_args.init_rpy,   dtype=np.float64)

    PYB_FREQ  = 1000
    CTRL_FREQ = C.ATTITUDE_RATE          # 500 Hz (firmware schedules pos@100Hz inside)

    env = CtrlAviary(
        drone_model=DroneModel.CF2X, num_drones=1,
        initial_xyzs=init_pos.reshape(1, 3), initial_rpys=init_rpy.reshape(1, 3),
        physics=Physics.DYN_OFFSET, pyb_freq=PYB_FREQ, ctrl_freq=CTRL_FREQ,
        gui=False, record=False, obstacles=False, user_debug_gui=False)
    env.set_offset_mass(m_extra, r_offset)

    KF = env.KF
    M_total = C.M_BASE + m_extra
    HOVER_RPM = math.sqrt(M_total * C.G / (4 * C.KF))

    ctrl = CrazyflieFirmwarePID()
    ctrl.reset(tuple(np.rad2deg(init_rpy)), tuple(init_pos))
    motor_filter = MotorDynamicsFilter(n_motors=4, tau=0.02, dt=1.0 / CTRL_FREQ)
    motor_filter.rpm = np.full(4, HOVER_RPM)

    n_steps = int(CTRL_FREQ * sim_args.duration)
    print(f"[SIM-PID] CTRL={CTRL_FREQ}Hz  duration={sim_args.duration}s  "
          f"setpoint={target}  m_extra={m_extra*1e3:.1f}g")
    log_t   = np.zeros(n_steps)
    log_pos = np.zeros((n_steps, 3))
    log_vel = np.zeros((n_steps, 3))
    log_rpy = np.zeros((n_steps, 3))
    log_state12 = np.zeros((n_steps, 12))

    action = np.full((1, 4), HOVER_RPM)
    for i in range(n_steps):
        obs, _, _, _, _ = env.step(action)
        pos, vel, rpy_deg, gyro_deg = obs_to_firmware_state(obs[0])
        log_state12[i] = obs_to_nn_state(obs[0])
        motor_pwm = ctrl.update(target, 0.0, pos, vel, rpy_deg, gyro_deg)
        rpms = pwm_to_rpm(np.asarray(motor_pwm, dtype=float), KF, truncate_8bit=True)
        rpms = motor_filter.apply(rpms)
        action[0, :] = rpms
        log_t[i] = i / CTRL_FREQ
        log_pos[i] = pos
        log_vel[i] = vel
        log_rpy[i] = rpy_deg
    env.close()
    print("[SIM-PID] Done.")
    return dict(t=log_t, pos=log_pos, vel=log_vel, rpy=log_rpy,
                state12=log_state12,
                target=target, init_pos=init_pos,
                m_extra=m_extra, r_offset=np.array(r_offset))


# ===================================================================
#  Plot
# ===================================================================

CTRL_ORDER = ["pid", "base", "base_adapt", "maml_adapt"]
CTRL_STYLE = {
    "pid":        dict(color="C2",   ls="-.", label="firmware PID (no NN)",   short="firmwarePID"),
    "base":       dict(color="0.55", ls="--", label="baseline NN (no adapt)", short="baseNN"),
    "base_adapt": dict(color="C0",   ls="-",  label="baseline NN adapted",    short="baseNN-adapt"),
    "maml_adapt": dict(color="C3",   ls="-",  label="MAML adapted",           short="MAML-adapt"),
}


def plot_compare(results, masses_g, target, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nrows = len(masses_g)
    fig, axes = plt.subplots(nrows, 3, figsize=(15, 3.0 * nrows + 0.5),
                             squeeze=False, sharex=True)
    axis_names = ["x", "y", "z"]
    for i, mg in enumerate(masses_g):
        res = results[mg]
        t = res["base"]["t"]
        for j in range(3):
            ax = axes[i][j]
            for key in CTRL_ORDER:
                st = CTRL_STYLE[key]
                ax.plot(t, res[key]["pos"][:, j], color=st["color"],
                        ls=st["ls"], lw=1.5,
                        label=(st["label"] if (i == 0 and j == 0) else None))
            ax.axhline(target[j], color="k", ls=":", alpha=0.5)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(f"{axis_names[j]}  (target {target[j]:.2f} m)")
            if j == 0:
                ax.set_ylabel(f"{mg:.0f} g\nposition [m]")
            if i == nrows - 1:
                ax.set_xlabel("time [s]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("PyBullet position: baseline vs baseline-adapted vs MAML-adapted",
                 y=0.998, fontsize=13)
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[PLOT] saved {out_path}")


def steady_state_error(data, target, last_s=1.0):
    """Mean position error over the last ``last_s`` seconds [m]."""
    t = data["t"]; pos = data["pos"]
    mask = t >= (t[-1] - last_s)
    return float(np.linalg.norm(pos[mask].mean(axis=0) - np.asarray(target)))


def trajectory_loss(data, target, hp):
    """Training-style cumulative cost over the whole flight.

    Uses the same quadratic cost as training (``maml_lib.cost.trajectory_cost``
    with the checkpoint's ``pos_weight``/``z_weight``/``terminal_weight``), on
    the target-shifted 12-D state, downsampled from the control rate to NN_FREQ
    so the magnitude is on the same footing as the training loss. Absolute
    values are not comparable to the training run (different physics/horizon)
    but ARE comparable across the four controllers at a given mass.
    """
    div = max(1, C.ATTITUDE_RATE // C.NN_FREQ)            # 500 Hz -> 100 Hz
    X = np.asarray(data["state12"], dtype=np.float32).copy()
    X[:, 0] -= target[0]; X[:, 2] -= target[1]; X[:, 4] -= target[2]
    Xt = torch.from_numpy(X[::div]).unsqueeze(0)          # (1, n, 12)
    L = trajectory_cost(Xt, hp["term_w"],
                        pos_weight=hp["pos_w"], z_weight=hp["z_w"])
    return float(L)


def save_metrics_csv(path, masses_g, sse, tloss):
    """Write both metrics (per mass x controller) to a CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mass_g", "controller",
                    "steady_state_err_cm", "trajectory_loss"])
        for mg in masses_g:
            for k in CTRL_ORDER:
                w.writerow([f"{mg:g}", CTRL_STYLE[k]["short"],
                            f"{sse[mg][k] * 100:.4f}", f"{tloss[mg][k]:.4f}"])
    print(f"[CSV] saved {path}")


# ===================================================================
#  CLI / main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maml-ckpt", required=True, help="MAML checkpoint (.pt)")
    p.add_argument("--baseline-ckpt", required=True, help="baseline checkpoint (.pt)")
    p.add_argument("--masses", type=float, nargs="+", required=True,
                   help="test masses [g] (e.g. 4 8 12 16)")
    p.add_argument("--target-dx", type=float, default=0.0,
                   help="mass attachment dx [m] (default 0 = centred)")
    p.add_argument("--target-dy", type=float, default=0.0)
    p.add_argument("--target-dz", type=float, default=0.0)
    p.add_argument("--init-pos", nargs=3, type=float, default=[0.0, 0.0, 1.0],
                   metavar=("X", "Y", "Z"), help="start position [m]")
    p.add_argument("--target-pos", nargs=3, type=float, default=[0.0, 0.0, 1.0],
                   metavar=("X", "Y", "Z"), help="arrival / regulation target [m]")
    p.add_argument("--init-rpy", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   metavar=("R", "P", "Y"))
    p.add_argument("--duration", type=float, default=8.0, help="sim duration [s]")
    p.add_argument("--out", type=str, default=None, help="output figure path")
    p.add_argument("--metrics-out", type=str, default=None,
                   help="output CSV path for the metrics (default: next to "
                        "--out, '<out>_metrics.csv')")
    p.add_argument("--device", type=str, default=None, help="cpu or cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    maml_ckpt = torch.load(args.maml_ckpt, map_location=device, weights_only=False)
    hp = adapt_hparams(maml_ckpt)
    r_offset = (args.target_dx, args.target_dy, args.target_dz)
    centred = (abs(args.target_dx) + abs(args.target_dy) + abs(args.target_dz)) < 1e-9
    print(f"[Adapt] K={hp['K']}  n_inner={hp['n_inner']}  lr_inner={hp['lr_inner']}  "
          f"clip={hp['clip']}  t_sim={hp['t_sim']}s  noise_scale={hp['noise_scale']}  "
          f"half_side={hp['half_side']}  ({'centred' if centred else f'offset {r_offset}'})")

    pol_m = load_policy(args.maml_ckpt, hp["hidden"], device)
    pol_b = load_policy(args.baseline_ckpt, hp["hidden"], device)

    # run_sim reads these fields off an args-like object.
    sim_args = SimpleNamespace(
        target_pos=args.target_pos, init_pos=args.init_pos,
        init_rpy=args.init_rpy, duration=args.duration, gui=False)

    results = {}
    sse = {}
    tloss = {}
    for mg in args.masses:
        m_kg = mg * 1e-3
        print(f"\n========== mass {mg:.1f} g ==========")
        theta = {
            "base":       identity_theta(pol_b),
            "base_adapt": adapt_to_mass(pol_b, m_kg, r_offset, hp, device),
            "maml_adapt": adapt_to_mass(pol_m, m_kg, r_offset, hp, device),
        }
        runs = {}
        for key in CTRL_ORDER:
            print(f"  -- {key} --")
            if key == "pid":
                runs[key] = run_sim_pid(sim_args, m_kg, r_offset, device)
            else:
                pol = pol_b if key.startswith("base") else pol_m
                runs[key] = run_sim(sim_args, pol, theta[key], m_kg, r_offset, device)
        results[mg] = runs
        sse[mg]   = {k: steady_state_error(runs[k], args.target_pos) for k in CTRL_ORDER}
        tloss[mg] = {k: trajectory_loss(runs[k], args.target_pos, hp) for k in CTRL_ORDER}

    # --- console summaries ---
    hdr = f"{'mass[g]':>7} | " + " | ".join(f"{CTRL_STYLE[k]['short']:>13}" for k in CTRL_ORDER)
    print("\n=== steady-state position error [cm] (last 1 s) ===")
    print(hdr)
    for mg in args.masses:
        print(f"{mg:7.1f} | " + " | ".join(f"{sse[mg][k]*100:13.2f}" for k in CTRL_ORDER))
    print("\n=== trajectory loss (training cost, whole flight @ NN_FREQ) ===")
    print(hdr)
    for mg in args.masses:
        print(f"{mg:7.1f} | " + " | ".join(f"{tloss[mg][k]:13.1f}" for k in CTRL_ORDER))

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.maml_ckpt)), "compare_pybullet.png")
    plot_compare(results, args.masses, args.target_pos, out_path)

    metrics_path = args.metrics_out or (os.path.splitext(out_path)[0] + "_metrics.csv")
    save_metrics_csv(metrics_path, args.masses, sse, tloss)
    print("Done.")


if __name__ == "__main__":
    main()
