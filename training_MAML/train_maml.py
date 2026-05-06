"""Train a Crazyflie controller with MAML over offset-mass tasks.

The policy has the same role as
``training_regulation_simu_and_real/train_nn_cf_pid.py``: it replaces the
firmware position+velocity PIDs and outputs
``[thrust_u16, roll_deg, pitch_deg, yaw_rate_deg/s]`` at NN_FREQ. The
firmware attitude+rate PIDs run downstream at ATTITUDE_RATE.

A *task* is a drone with an extra mass attached at an arbitrary body
offset. This perturbs total mass, centre of mass, and the inertia tensor
(full Steiner correction). MAML learns an initialisation that adapts in
a few gradient steps to any specific configuration.

The policy is trained to bring the drone to the origin; to fly to any
other point at deployment, shift the input the same way as in
``fly_nn_cf_pid.py``.

Usage examples:

    cd training_MAML
    python train_maml.py
    python train_maml.py --dynamics linearized --maml-order 1
    python train_maml.py --epochs 1000 --meta-batch 8 --k-inner 8 \\
                         --n-inner-steps 1 --lr-outer 1e-3 --lr-inner 5e-2 \\
                         --t-sim 2.0 --hidden 64 --tag exp1
"""
import argparse
import os
import sys

import torch

# Add project root so maml_lib resolves.
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from maml_lib import (
    config as C,
    PolicyMLP, OffsetMassTask, meta_train,
    make_x0_batch, generate_cube_points, GracefulKiller,
    DYNAMICS,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="MAML training for CF2 with offset-mass tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Core training
    p.add_argument("--epochs",        type=int,   default=500)
    p.add_argument("--lr-outer",      type=float, default=1e-3,
                   help="meta (outer) learning rate")
    p.add_argument("--lr-inner",      type=float, default=1e-2,
                   help="inner-loop (per-task) learning rate")
    p.add_argument("--meta-batch",    type=int,   default=4,
                   help="number of tasks sampled per meta update")
    p.add_argument("--k-inner",       type=int,   default=8,
                   help="number of trajectories used to compute the inner loss")
    p.add_argument("--n-inner-steps", type=int,   default=1,
                   help="number of inner-loop gradient steps per task "
                        "(MAML paper uses 1)")
    p.add_argument("--k-outer",       type=int,   default=8,
                   help="number of trajectories used to compute the outer loss")
    p.add_argument("--maml-order",    type=int,   default=2, choices=[1, 2],
                   help="1 = FOMAML, 2 = full second-order MAML")
    p.add_argument("--hidden",        type=int,   default=64)

    # Rollout
    p.add_argument("--t-sim",         type=float, default=2.0,
                   help="rollout duration [s]")
    p.add_argument("--half-side",     type=float, default=0.3,
                   help="initial-position cube half-side [m]")
    p.add_argument("--tau-div",       type=float, default=2.0,
                   help="curriculum divergence threshold [m]; "
                        "set <=0 to disable")
    p.add_argument("--obs-noise-scale", type=float, default=1.0,
                   help="0 = off, 1 = default, 2 = double")
    p.add_argument("--terminal-weight", type=float, default=50.0)
    p.add_argument("--z-weight",        type=float, default=1.0)

    # Dynamics
    p.add_argument("--dynamics", choices=list(DYNAMICS.keys()),
                   default="nonlinear")

    # Task ranges
    p.add_argument("--m-max",   type=float, default=0.015,
                   help="max extra mass [kg] (m_min is fixed at 0)")
    p.add_argument("--dxy-max", type=float, default=0.03,
                   help="max in-plane offset (|dx|, |dy|) [m]")
    p.add_argument("--dz-min",  type=float, default=-0.03)
    p.add_argument("--dz-max",  type=float, default=+0.05)

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag",  type=str, default="")
    p.add_argument("--verbose-every", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Device] {device}")
    print(f"[Config] dynamics={args.dynamics}  maml_order={args.maml_order}")
    print(f"[Config] epochs={args.epochs}  meta_batch={args.meta_batch}  "
          f"k_inner={args.k_inner}  n_inner_steps={args.n_inner_steps}  "
          f"k_outer={args.k_outer}")
    print(f"[Config] lr_outer={args.lr_outer}  lr_inner={args.lr_inner}")
    print(f"[Config] hover thrust (baseline) ≈ {C.HOVER_THRUST_U16_BASE:.0f}")

    n_steps = int(args.t_sim * C.NN_FREQ)
    print(f"[Config] n_steps_per_rollout={n_steps} ({args.t_sim}s @ {C.NN_FREQ}Hz)")

    # Pool of initial states (cube vertices/edges/faces/center).
    cube = generate_cube_points(args.half_side)
    x0_pool = make_x0_batch(cube, device=device)
    print(f"[Train] {len(cube)} initial points, ±{args.half_side}m cube")

    # Task generator.
    task = OffsetMassTask(
        m_min=0.0, m_max=args.m_max,
        dx_max=args.dxy_max, dy_max=args.dxy_max,
        dz_min=args.dz_min,  dz_max=args.dz_max,
    )
    print(f"[Task] OffsetMass  m∈[0,{args.m_max*1e3:.0f}]g  "
          f"dxy∈±{args.dxy_max*100:.1f}cm  "
          f"dz∈[{args.dz_min*100:+.1f},{args.dz_max*100:+.1f}]cm")

    # Policy.
    policy = PolicyMLP(hidden=args.hidden).to(device)

    # Sim-to-real obs noise.
    obs_noise = ((C.OBS_NOISE_STD * args.obs_noise_scale).to(device)
                 if args.obs_noise_scale > 0 else None)

    tau_div = args.tau_div if args.tau_div > 0 else None

    killer = GracefulKiller()

    policy, history = meta_train(
        policy, task, x0_pool,
        dynamics_step=DYNAMICS[args.dynamics],
        n_steps=n_steps,
        epochs=args.epochs,
        meta_batch=args.meta_batch,
        k_inner=args.k_inner,
        n_inner_steps=args.n_inner_steps,
        k_outer=args.k_outer,
        lr_outer=args.lr_outer,
        lr_inner=args.lr_inner,
        maml_order=args.maml_order,
        tau_div=tau_div,
        obs_noise_std=obs_noise,
        terminal_weight=args.terminal_weight,
        z_weight=args.z_weight,
        device=device,
        verbose_every=args.verbose_every,
        seed=args.seed,
        killer=killer,
    )

    # Save checkpoint.
    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag = f"_{args.tag}" if args.tag else ""
    ckpt_name = (f"maml_{args.dynamics}_h{args.hidden}_"
                 f"ep{args.epochs}_mb{args.meta_batch}_"
                 f"o{args.maml_order}{tag}.pt")
    ckpt_path = os.path.join(out_dir, ckpt_name)
    torch.save({
        "state_dict": policy.cpu().state_dict(),
        "args":       vars(args),
        "history":    history,
        "x_scale":    C.X_SCALE,
        "pos_scale":  C.POS_SCALE,
        "nn_freq":    C.NN_FREQ,
        "att_rate":   C.ATTITUDE_RATE,
    }, ckpt_path)
    print(f"[Saved] {ckpt_path}")
    print("Done.")


if __name__ == "__main__":
    main()
