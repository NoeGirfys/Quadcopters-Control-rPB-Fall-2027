"""Train a baseline (non-MAML) NN over the 3 Gaussian-motor training tasks.

Same task setup as ``train_maml.py``: 4 motor-centred Gaussian
distributions; the motor index given by ``--target-motor`` is held out,
the other 3 are mixed and used for training. At each epoch one
MassParams is sampled from each of the 3 training distributions; the
policy is rolled out on the fixed ``x0_train`` batch and the average of
the 3 per-task losses is back-propagated directly into the network
parameters — no inner-loop adaptation. The resulting controller is
*the* baseline that ``test_maml_pybullet.py`` should be compared
against.

Saved checkpoints carry the same payload structure as the MAML ones
(``state_dict``, ``args``, ``motor_centers``, ``target_motor``,
``training_motor_indices``, ``sigma``, ``m_extra``, ``x0_train``,
``x0_eval``, ``history``), with ``args["n_inner_steps"] = 0`` and
``args["lr_inner"] = 0`` so ``test_maml_pybullet.py`` can load them and
adapt 0 steps (i.e. inference straight from the saved weights).

Usage:

    cd training_MAML
    python train_baseline.py
    python train_baseline.py --epochs 1000 --target-motor 2 --tag baseline-tm2
"""
import argparse
import os
import sys

import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from maml_lib import (
    config as C,
    PolicyMLP, GaussianMotorTask, baseline_train,
    sample_hover_x0, plot_training_map, GracefulKiller,
    DYNAMICS,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Baseline training (no MAML) for CF2 with Gaussian-motor tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Core training
    p.add_argument("--epochs",        type=int,   default=500)
    p.add_argument("--lr-outer",      type=float, default=1e-3,
                   help="learning rate (Adam)")
    p.add_argument("--hidden",        type=int,   default=64)

    # x0 batches (drawn once at start)
    p.add_argument("--n-x0-train", type=int, default=32,
                   help="number of fixed initial states for the training loss")
    p.add_argument("--n-x0-eval",  type=int, default=32,
                   help="number of fixed initial states for the held-out eval loss")
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
    p.add_argument("--pos-weight",      type=float, default=10.0)
    p.add_argument("--z-weight",        type=float, default=1.0)

    # Dynamics
    p.add_argument("--dynamics", choices=list(DYNAMICS.keys()),
                   default="nonlinear")

    # Task definition (mirror train_maml.py)
    p.add_argument("--mass",   type=float, default=0.010,
                   help="fixed extra mass for every task [kg]")
    p.add_argument("--sigma",  type=float, default=0.005,
                   help="std of the 2D Gaussian over (dx, dy) [m]")
    p.add_argument("--target-motor", type=int, default=0, choices=[0, 1, 2, 3],
                   help="motor index held out from training")

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
    print(f"[Config] dynamics={args.dynamics}  epochs={args.epochs}")
    print(f"[Config] lr_outer={args.lr_outer}  hidden={args.hidden}")

    n_steps = int(args.t_sim * C.NN_FREQ)
    print(f"[Config] n_steps_per_rollout={n_steps} ({args.t_sim}s @ {C.NN_FREQ}Hz)")

    # --- Same task setup as MAML training ----------------------------------
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
    print(f"[Tasks]  3 training motors {training_motor_indices}, "
          f"target = motor #{args.target_motor}  "
          f"σ={args.sigma*100:.2f}cm  m={args.mass*1e3:.0f}g")

    # --- Fixed x0 batches (sampled once, reproducible from seed) -----------
    x0_gen = torch.Generator(device=device).manual_seed(args.seed)
    n_total = args.n_x0_train + args.n_x0_eval
    x0_all = sample_hover_x0(n_total, args.half_side, x0_gen, device=device)
    x0_train = x0_all[:args.n_x0_train].contiguous()
    x0_eval  = x0_all[args.n_x0_train:].contiguous()
    print(f"[x0]    n_train={args.n_x0_train}  n_eval={args.n_x0_eval}")

    # --- Policy (same hover bias trick as MAML) ----------------------------
    M_total = C.M_BASE + args.mass
    hover_thrust_u16 = (M_total * C.G * C.UINT16_MAX) / (4 * C.CF2_THRUST_MAX_PER_MOTOR)
    print(f"[Policy] hover thrust (M_total={M_total*1e3:.1f}g) "
          f"≈ {hover_thrust_u16:.0f}")
    policy = PolicyMLP(hidden=args.hidden,
                       hover_thrust_u16=hover_thrust_u16).to(device)

    # --- Sim-to-real obs noise --------------------------------------------
    obs_noise = ((C.OBS_NOISE_STD * args.obs_noise_scale).to(device)
                 if args.obs_noise_scale > 0 else None)
    tau_div = args.tau_div if args.tau_div > 0 else None

    # --- Output paths (final name reflects ACTUAL epoch count) -------------
    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag = f"_{args.tag}" if args.tag else ""
    base_no_ep = (f"baseline_{args.dynamics}_h{args.hidden}_"
                  f"tm{args.target_motor}{tag}")
    working_ckpt = os.path.join(out_dir, base_no_ep + "_inprogress.pt")
    working_map  = os.path.join(out_dir, base_no_ep + "_inprogress_map.png")

    # Augment args so test_maml_pybullet.py can load this checkpoint with
    # n_inner_steps=0 by default (i.e. inference straight from the weights).
    args_dict = vars(args)
    args_dict["n_inner_steps"]  = 0
    args_dict["lr_inner"]       = 0.0
    args_dict["maml_order"]     = 0
    args_dict["inner_grad_clip"] = 0.0

    common_save_payload = {
        "args":               args_dict,
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
        payload["state_dict"] = {k: v.detach().cpu()
                                 for k, v in policy_module.state_dict().items()}
        payload["history"]    = history
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
        except Exception as e:
            print(f"[Plot] failed: {e}")

    def on_epoch_end(ep, policy_module, history):
        save_checkpoint(policy_module, history, working_ckpt)
        if args.plot_every > 0 and ((ep + 1) % args.plot_every == 0):
            save_plot(history, working_map)

    killer = GracefulKiller()

    policy, history = baseline_train(
        policy, training_tasks, x0_train, x0_eval,
        dynamics_step=DYNAMICS[args.dynamics],
        n_steps=n_steps,
        epochs=args.epochs,
        lr_outer=args.lr_outer,
        tau_div=tau_div,
        obs_noise_std=obs_noise,
        terminal_weight=args.terminal_weight,
        pos_weight=args.pos_weight,
        z_weight=args.z_weight,
        device=device,
        verbose_every=args.verbose_every,
        seed=args.seed,
        on_epoch_end=on_epoch_end,
        killer=killer,
    )

    # Final save under the "actual epochs" name.
    actual_epochs = len(history["epoch"])
    final_ckpt = os.path.join(out_dir, f"{base_no_ep}_ep{actual_epochs}.pt")
    final_map  = os.path.join(out_dir, f"{base_no_ep}_ep{actual_epochs}_map.png")
    save_checkpoint(policy, history, final_ckpt)
    save_plot(history, final_map)
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
