"""Train a Crazyflie controller with MAML over Gaussian-motor tasks.

The policy has the same role as
``training_regulation_simu_and_real/train_nn_cf_pid.py``: it replaces the
firmware position+velocity PIDs and outputs
``[thrust_u16, roll_deg, pitch_deg, yaw_rate_deg/s]`` at NN_FREQ. The
firmware attitude+rate PIDs run downstream at ATTITUDE_RATE.

Setup
-----
Four task distributions, one centred on each motor's body-frame xy.
Three are used for training, one is held out as the *target* task for
later few-shot adaptation. The extra mass is fixed (10 g default) and
attached at z = 0; only (dx, dy) is sampled from a 2D Gaussian.

Each epoch:
  * one (dx, dy) is drawn from each of the 3 training distributions
    -> 3 tasks per meta-update;
  * adaptation (inner loop) uses a *fixed* hover x0 batch ``x0_train``
    drawn uniformly in the cube at start-up;
  * the meta loss is computed on a *fixed* disjoint batch ``x0_eval``,
    same hover convention.

Both x0 batches and the chronological list of sampled task offsets are
saved into the checkpoint after every epoch, so a Ctrl+C preserves the
full training map.

Usage examples:

    cd training_MAML
    python train_maml.py
    python train_maml.py --dynamics linearized --maml-order 1
    python train_maml.py --epochs 1000 --target-motor 2 --sigma 0.005 \\
                         --n-x0-train 64 --n-x0-eval 64 --t-sim 2.0 --tag exp1
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
    PolicyMLP, GaussianMotorTask, meta_train,
    sample_hover_x0, plot_training_map, GracefulKiller,
    DYNAMICS,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="MAML training for CF2 with Gaussian-motor tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Core training
    p.add_argument("--epochs",        type=int,   default=500)
    p.add_argument("--lr-outer",      type=float, default=1e-3,
                   help="meta (outer) learning rate")
    p.add_argument("--lr-inner",      type=float, default=1e-2,
                   help="inner-loop (per-task) learning rate")
    p.add_argument("--n-inner-steps", type=int,   default=1,
                   help="number of inner-loop gradient steps per task "
                        "(MAML paper uses 1)")
    p.add_argument("--maml-order",    type=int,   default=2, choices=[1, 2],
                   help="1 = FOMAML, 2 = full second-order MAML")
    p.add_argument("--hidden",        type=int,   default=64)

    # x0 batches (drawn once at start)
    p.add_argument("--n-x0-train", type=int, default=32,
                   help="number of fixed initial states for the inner loss")
    p.add_argument("--n-x0-eval",  type=int, default=32,
                   help="number of fixed initial states for the meta loss "
                        "(disjoint from train)")
    p.add_argument("--half-side",  type=float, default=0.3,
                   help="cube half-side for x0 sampling [m]")

    # Rollout
    p.add_argument("--t-sim",         type=float, default=2.0,
                   help="rollout duration [s]")
    p.add_argument("--tau-div",       type=float, default=2.0,
                   help="curriculum divergence threshold [m]; "
                        "set <=0 to disable")
    p.add_argument("--obs-noise-scale", type=float, default=1.0,
                   help="0 = off, 1 = default, 2 = double")
    p.add_argument("--terminal-weight", type=float, default=50.0)
    p.add_argument("--pos-weight",      type=float, default=10.0,
                   help="weight on (x, y, z) in the running cost")
    p.add_argument("--z-weight",        type=float, default=1.0)
    p.add_argument("--inner-grad-clip", type=float, default=1.0,
                   help="max-norm clip on inner-loop gradients; "
                        "set <=0 to disable")

    # Dynamics
    p.add_argument("--dynamics", choices=list(DYNAMICS.keys()),
                   default="nonlinear")

    # Task definition (Gaussian-motor)
    p.add_argument("--mass",   type=float, default=0.010,
                   help="fixed extra mass for every task [kg]")
    p.add_argument("--sigma",  type=float, default=0.005,
                   help="std of the 2D Gaussian over (dx, dy) [m]")
    p.add_argument("--target-motor", type=int, default=0, choices=[0, 1, 2, 3],
                   help="motor index held out from training "
                        "(0=front-right, 1=rear-right, 2=rear-left, 3=front-left)")

    # Logging
    p.add_argument("--plot-every", type=int, default=25,
                   help="regenerate the position map every N epochs "
                        "(set <=0 to only plot at the end)")

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
    print(f"[Config] epochs={args.epochs}  n_inner_steps={args.n_inner_steps}")
    print(f"[Config] lr_outer={args.lr_outer}  lr_inner={args.lr_inner}")
    print(f"[Config] hover thrust (baseline) ≈ {C.HOVER_THRUST_U16_BASE:.0f}")

    n_steps = int(args.t_sim * C.NN_FREQ)
    print(f"[Config] n_steps_per_rollout={n_steps} ({args.t_sim}s @ {C.NN_FREQ}Hz)")

    # --- Task setup: 4 motor distributions, hold out args.target_motor -----
    motor_centers = [(float(C.MOTOR_POS[i, 0]), float(C.MOTOR_POS[i, 1]))
                     for i in range(4)]
    training_motor_indices = [i for i in range(4) if i != args.target_motor]
    training_tasks = [
        GaussianMotorTask(motor_xy=motor_centers[i],
                          sigma=args.sigma,
                          m_extra=args.mass,
                          motor_idx=i)
        for i in training_motor_indices
    ]
    target_task = GaussianMotorTask(motor_xy=motor_centers[args.target_motor],
                                    sigma=args.sigma,
                                    m_extra=args.mass,
                                    motor_idx=args.target_motor)
    print(f"[Tasks]  3 training motors {training_motor_indices}, "
          f"target = motor #{args.target_motor}  "
          f"σ={args.sigma*100:.2f}cm  m={args.mass*1e3:.0f}g")

    # --- Fixed x0 batches (sampled once, reproducible from seed) -----------
    x0_gen = torch.Generator(device=device).manual_seed(args.seed)
    n_total = args.n_x0_train + args.n_x0_eval
    x0_all = sample_hover_x0(n_total, args.half_side, x0_gen, device=device)
    x0_train = x0_all[:args.n_x0_train].contiguous()
    x0_eval  = x0_all[args.n_x0_train:].contiguous()
    print(f"[x0]    n_train={args.n_x0_train}  n_eval={args.n_x0_eval}  "
          f"cube ±{args.half_side}m (hover)")

    # --- Policy ------------------------------------------------------------
    # Initialise the thrust bias for the hover of the *total* task mass
    # (M_base + args.mass), assumed centred. The offset torque is what MAML
    # adaptation has to learn to cancel via the roll/pitch outputs.
    M_total = C.M_BASE + args.mass
    hover_thrust_u16 = (M_total * C.G * C.UINT16_MAX) / (4 * C.CF2_THRUST_MAX_PER_MOTOR)
    print(f"[Policy] hover thrust (M_total={M_total*1e3:.1f}g) "
          f"≈ {hover_thrust_u16:.0f}  "
          f"(baseline was {C.HOVER_THRUST_U16_BASE:.0f})")
    policy = PolicyMLP(hidden=args.hidden,
                       hover_thrust_u16=hover_thrust_u16).to(device)

    # --- Sim-to-real obs noise --------------------------------------------
    obs_noise = ((C.OBS_NOISE_STD * args.obs_noise_scale).to(device)
                 if args.obs_noise_scale > 0 else None)

    tau_div = args.tau_div if args.tau_div > 0 else None

    # --- Output paths ------------------------------------------------------
    # During training we save to a stable "in-progress" name so a Ctrl+C
    # never leaves a half-written file. After meta_train returns, we rename
    # the working files to include the *actual* number of completed epochs
    # (which may be less than args.epochs if the user interrupted).
    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag = f"_{args.tag}" if args.tag else ""
    base_no_ep = (f"maml_{args.dynamics}_h{args.hidden}_"
                  f"o{args.maml_order}_tm{args.target_motor}{tag}")
    working_ckpt = os.path.join(out_dir, base_no_ep + "_inprogress.pt")
    working_map  = os.path.join(out_dir, base_no_ep + "_inprogress_map.png")

    common_save_payload = {
        "args":               vars(args),
        "x_scale":            C.X_SCALE,
        "pos_scale":          C.POS_SCALE,
        "nn_freq":            C.NN_FREQ,
        "att_rate":           C.ATTITUDE_RATE,
        "motor_centers":      motor_centers,
        "target_motor":       args.target_motor,
        "training_motor_indices": training_motor_indices,
        "sigma":              args.sigma,
        "m_extra":            args.mass,
        "x0_train":           x0_train.detach().cpu(),
        "x0_eval":            x0_eval.detach().cpu(),
    }

    def save_checkpoint(policy_module, history, path):
        payload = dict(common_save_payload)
        payload["state_dict"]    = {k: v.detach().cpu()
                                    for k, v in policy_module.state_dict().items()}
        payload["history"]       = history
        payload["actual_epochs"] = len(history.get("epoch", []))
        # atomic write: temp file then rename, so a Ctrl+C mid-save won't
        # leave a half-written checkpoint.
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def save_plot(history, path):
        try:
            plot_training_map(
                path,
                sampled_positions=history["sampled_positions"],
                motor_centers=motor_centers,
                target_motor=args.target_motor,
                training_motor_indices=training_motor_indices,
                sigma=args.sigma,
                x0_train=x0_train,
                x0_eval=x0_eval,
                half_side=args.half_side,
            )
        except Exception as e:  # plotting must never kill training
            print(f"[Plot] failed: {e}")

    def on_epoch_end(ep, policy_module, history):
        # always save the checkpoint so Ctrl+C preserves the full map
        save_checkpoint(policy_module, history, working_ckpt)
        if args.plot_every > 0 and ((ep + 1) % args.plot_every == 0):
            save_plot(history, working_map)

    killer = GracefulKiller()

    policy, history = meta_train(
        policy, training_tasks, x0_train, x0_eval,
        dynamics_step=DYNAMICS[args.dynamics],
        n_steps=n_steps,
        epochs=args.epochs,
        n_inner_steps=args.n_inner_steps,
        lr_outer=args.lr_outer,
        lr_inner=args.lr_inner,
        maml_order=args.maml_order,
        tau_div=tau_div,
        obs_noise_std=obs_noise,
        terminal_weight=args.terminal_weight,
        pos_weight=args.pos_weight,
        z_weight=args.z_weight,
        inner_grad_clip=args.inner_grad_clip,
        device=device,
        verbose_every=args.verbose_every,
        seed=args.seed,
        on_epoch_end=on_epoch_end,
        killer=killer,
    )

    # Final save under the "actual epochs" name (matches len(history)).
    actual_epochs = len(history["epoch"])
    final_ckpt = os.path.join(out_dir, f"{base_no_ep}_ep{actual_epochs}.pt")
    final_map  = os.path.join(out_dir, f"{base_no_ep}_ep{actual_epochs}_map.png")
    save_checkpoint(policy, history, final_ckpt)
    save_plot(history, final_map)
    # Clean up the in-progress files (they are now superseded by the final ones).
    for stale in (working_ckpt, working_map):
        if os.path.exists(stale):
            try:
                os.remove(stale)
            except OSError:
                pass
    print(f"[Saved] {final_ckpt}")
    print(f"[Saved] {final_map}")
    print("Done.")


if __name__ == "__main__":
    main()
