"""Train a Crazyflie controller with MAML over a fixed uniform mass task set.

Task distribution
-----------------
N tasks are sampled *once at startup* from a uniform box:
    dx ~ Uniform(xy_min, xy_max)
    dy ~ Uniform(xy_min, xy_max)
    dz ~ Uniform(z_min,  z_max)
and reused at every epoch.  This makes the meta-gradient estimate stable and
lets the baseline be trained on the exact same tasks for a fair comparison.

GPU parallelism
---------------
MAML inner loop  : all N tasks are batched into one rollout (N*B_train, 12)
                   for the first adaptation step; a single GPU forward pass
                   replaces N sequential calls.
Baseline         : all N tasks are always batched (N*B_train, 12).

Usage examples:

    cd training_MAML
    python train_maml.py
    python train_maml.py --n-tasks 50 --xy-min -0.04 --xy-max 0.04 \\
                         --dynamics linearized --maml-order 1 --epochs 500
    python train_maml.py --resume maml_..._ep200.pt --epochs 500
"""
import argparse
import os
import sys

import numpy as np
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from maml_lib import (
    config as C,
    PolicyMLP, FixedUniformMassSet, meta_train,
    sample_hover_x0, GracefulKiller,
    DYNAMICS,
)


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_resume_defaults(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    actual_epochs = ckpt.get("actual_epochs",
                             len(ckpt.get("history", {}).get("epoch", []))) \
                    or saved.get("epochs", 500)
    out = dict(saved)
    out["epochs"] = max(int(saved.get("epochs", 500)), int(actual_epochs))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--resume", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    if pre_args.resume is not None:
        d = _load_resume_defaults(pre_args.resume)
        print(f"[Init] Resuming from {pre_args.resume}")
        print(f"[Init]   inherited hyperparameters from checkpoint; "
              "override any CLI arg to deviate.")
    else:
        d = {}

    p = argparse.ArgumentParser(
        description="MAML training for CF2 with fixed uniform mass tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--resume", type=str, default=None,
                   help="resume training from a .pt checkpoint")

    # Training
    p.add_argument("--epochs",          type=int,   default=d.get("epochs",        500))
    p.add_argument("--lr-outer",        type=float, default=d.get("lr_outer",      3e-4))
    p.add_argument("--lr-inner",        type=float, default=d.get("lr_inner",      5e-2))
    p.add_argument("--n-inner-steps",   type=int,   default=d.get("n_inner_steps", 1))
    p.add_argument("--maml-order",      type=int,   default=d.get("maml_order",    1),
                   choices=[1, 2])
    p.add_argument("--hidden",          type=int,   default=d.get("hidden",        64))

    # Task set
    p.add_argument("--n-tasks",  type=int,   default=d.get("n_tasks",  50),
                   help="number of fixed tasks sampled at startup")
    p.add_argument("--mass-min", type=float, default=d.get("mass_min", 0.010),
                   help="lower bound for the per-task extra mass [kg]")
    p.add_argument("--mass-max", type=float, default=d.get("mass_max", 0.010),
                   help="upper bound for the per-task extra mass [kg] "
                        "(set equal to --mass-min for a constant mass)")
    p.add_argument("--xy-min",   type=float, default=d.get("xy_min",  -0.04),
                   help="lower bound for dx and dy [m]")
    p.add_argument("--xy-max",   type=float, default=d.get("xy_max",   0.04),
                   help="upper bound for dx and dy [m]")
    p.add_argument("--z-min",    type=float, default=d.get("z_min",   -0.01),
                   help="lower bound for dz [m]")
    p.add_argument("--z-max",    type=float, default=d.get("z_max",    0.01),
                   help="upper bound for dz [m]")

    # x0 batches
    p.add_argument("--n-x0-train", type=int,   default=d.get("n_x0_train", 64))
    p.add_argument("--n-x0-eval",  type=int,   default=d.get("n_x0_eval",  32))
    p.add_argument("--half-side",  type=float, default=d.get("half_side",   0.3))

    # Rollout
    p.add_argument("--t-sim",           type=float, default=d.get("t_sim",           2.0))
    p.add_argument("--tau-div",         type=float, default=d.get("tau_div",         1.0),
                   help="divergence threshold [m]; <=0 to disable")
    p.add_argument("--obs-noise-scale", type=float, default=d.get("obs_noise_scale", 1.0))
    p.add_argument("--terminal-weight", type=float, default=d.get("terminal_weight", 50.0))
    p.add_argument("--pos-weight",      type=float, default=d.get("pos_weight",      10.0))
    p.add_argument("--z-weight",        type=float, default=d.get("z_weight",        1.0))
    p.add_argument("--inner-grad-clip", type=float, default=d.get("inner_grad_clip", 1.0))

    # Dynamics
    p.add_argument("--dynamics", choices=list(DYNAMICS.keys()),
                   default=d.get("dynamics", "nonlinear"))

    # Misc
    p.add_argument("--seed",         type=int, default=d.get("seed",         42))
    p.add_argument("--tag",          type=str, default=d.get("tag",          ""))
    p.add_argument("--verbose-every",type=int, default=d.get("verbose_every", 1))
    p.add_argument("--profile", action="store_true",
                   help="print a per-epoch timing breakdown of the meta-loop")
    p.add_argument("--grad-chunk", type=int, default=d.get("grad_chunk", 0),
                   help="task-chunk size for the batched inner-grad VJP; "
                        "0 = all tasks at once (lower it on CUDA OOM)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Device] {device}")
    print(f"[Config] dynamics={args.dynamics}  maml_order={args.maml_order}")
    print(f"[Config] epochs={args.epochs}  n_inner_steps={args.n_inner_steps}")
    print(f"[Config] lr_outer={args.lr_outer}  lr_inner={args.lr_inner}")
    n_steps = int(args.t_sim * C.NN_FREQ)
    print(f"[Config] n_steps={n_steps} ({args.t_sim}s @ {C.NN_FREQ}Hz)")

    # ── Load checkpoint if resuming ────────────────────────────────────
    ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

    # ── Fixed task set ─────────────────────────────────────────────────
    if ckpt is not None:
        # Reconstruct the exact task set (positions + per-task masses) so
        # the resumed run is faithful. Old checkpoints store only a scalar
        # m_extra; broadcast it for backward compatibility.
        if "m_extras" in ckpt:
            m_extras = np.asarray(ckpt["m_extras"], dtype=np.float32)
        else:
            m_extras = np.full(len(ckpt["task_positions"]),
                               float(ckpt["m_extra"]), dtype=np.float32)
        task_set = FixedUniformMassSet.from_arrays(
            ckpt["task_positions"], m_extras)
        print(f"[Tasks]  {len(task_set)} fixed tasks restored from checkpoint")
    else:
        rng_tasks = np.random.default_rng(args.seed)
        task_set = FixedUniformMassSet.sample(
            n_tasks=args.n_tasks,
            xy_min=args.xy_min, xy_max=args.xy_max,
            z_min=args.z_min,   z_max=args.z_max,
            mass_min=args.mass_min, mass_max=args.mass_max,
            rng=rng_tasks,
        )
        print(f"[Tasks]  {args.n_tasks} tasks sampled  "
              f"xy=[{args.xy_min*100:.1f},{args.xy_max*100:.1f}]cm  "
              f"z=[{args.z_min*100:.1f},{args.z_max*100:.1f}]cm  "
              f"m=[{args.mass_min*1e3:.1f},{args.mass_max*1e3:.1f}]g")

    # ── x0 batches ────────────────────────────────────────────────────
    if ckpt is not None:
        x0_train = ckpt["x0_train"].to(device)
        x0_eval  = ckpt["x0_eval"].to(device)
        print(f"[x0]    n_train={x0_train.shape[0]}  n_eval={x0_eval.shape[0]}"
              f"  (restored)")
    else:
        x0_gen = torch.Generator(device=device).manual_seed(args.seed + 1)
        n_total = args.n_x0_train + args.n_x0_eval
        x0_all  = sample_hover_x0(n_total, args.half_side, x0_gen, device=device)
        x0_train = x0_all[:args.n_x0_train].contiguous()
        x0_eval  = x0_all[args.n_x0_train:].contiguous()
        print(f"[x0]    n_train={args.n_x0_train}  n_eval={args.n_x0_eval}"
              f"  cube ±{args.half_side}m")

    # ── Policy ────────────────────────────────────────────────────────
    # Seed the global torch RNG *before* constructing the network so the
    # weight initialisation is reproducible across runs and identical to the
    # baseline (which seeds the same way). On resume this is harmless: the
    # init weights are immediately overwritten by load_state_dict, and
    # meta_train restores the saved RNG state anyway.
    torch.manual_seed(args.seed)
    M_total = C.M_BASE + task_set.m_extra          # mean extra mass over the task set
    hover_thrust_u16 = (M_total * C.G * C.UINT16_MAX) / (4 * C.CF2_THRUST_MAX_PER_MOTOR)
    policy = PolicyMLP(hidden=args.hidden, hover_thrust_u16=hover_thrust_u16).to(device)
    if ckpt is not None:
        policy.load_state_dict(ckpt["state_dict"])
        print(f"[Policy] restored weights  (actual_epochs={ckpt.get('actual_epochs','?')})")
    print(f"[Policy] hover thrust (mean M={M_total*1e3:.1f}g) ≈ {hover_thrust_u16:.0f}")

    # ── Ancillary ─────────────────────────────────────────────────────
    obs_noise = (C.OBS_NOISE_STD * args.obs_noise_scale).to(device) \
                if args.obs_noise_scale > 0 else None
    tau_div = args.tau_div if args.tau_div > 0 else None

    # ── Output paths ──────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag = f"_{args.tag}" if args.tag else ""
    base = (f"maml_{args.dynamics}_h{args.hidden}_"
            f"o{args.maml_order}_n{len(task_set)}{tag}")
    working_ckpt = os.path.join(out_dir, base + "_inprogress.pt")

    # Payload fields that never change between epochs.
    common_payload = {
        "args":          vars(args),
        "x_scale":       C.X_SCALE,
        "pos_scale":     C.POS_SCALE,
        "nn_freq":       C.NN_FREQ,
        "att_rate":      C.ATTITUDE_RATE,
        "task_positions": task_set.positions,   # (N, 3) offsets
        "m_extras":      task_set.m_extras,      # (N,) per-task extra mass
        "m_extra":       task_set.m_extra,       # mean — kept for old readers
        "x0_train":      x0_train.detach().cpu(),
        "x0_eval":       x0_eval.detach().cpu(),
    }

    def save_checkpoint(policy_module, history, path,
                        optimizer=None, torch_gen=None):
        payload = dict(common_payload)
        payload["state_dict"]    = {k: v.detach().cpu()
                                    for k, v in policy_module.state_dict().items()}
        payload["history"]       = history
        payload["actual_epochs"] = len(history.get("epoch", []))
        if optimizer  is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if torch_gen  is not None:
            payload["torch_gen_state"] = torch_gen.get_state().cpu()
        payload["torch_rng_state"] = torch.get_rng_state().cpu()
        if torch.cuda.is_available():
            payload["torch_cuda_rng_state"] = torch.cuda.get_rng_state().cpu()
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    def on_epoch_end(ep, policy_module, history,
                     optimizer=None, torch_gen=None, **_):
        save_checkpoint(policy_module, history, working_ckpt,
                        optimizer=optimizer, torch_gen=torch_gen)

    killer = GracefulKiller()

    # ── Resume state ──────────────────────────────────────────────────
    resume_state = None
    start_epoch  = 0
    if ckpt is not None:
        required = ("optimizer_state", "torch_gen_state",
                    "torch_rng_state", "history")
        missing = [k for k in required if k not in ckpt]
        if missing:
            raise RuntimeError(
                f"Cannot resume from {args.resume}: missing keys {missing}. "
                "Re-train from scratch.")
        resume_state = {
            "optimizer_state":  ckpt["optimizer_state"],
            "torch_gen_state":  ckpt["torch_gen_state"],
            "torch_rng_state":  ckpt["torch_rng_state"],
            "history":          ckpt["history"],
        }
        if "torch_cuda_rng_state" in ckpt:
            resume_state["torch_cuda_rng_state"] = ckpt["torch_cuda_rng_state"]
        start_epoch = len(ckpt["history"].get("epoch", []))
        if start_epoch >= args.epochs:
            print(f"[Resume] checkpoint already has {start_epoch} epochs "
                  f">= --epochs {args.epochs}; nothing to do. "
                  f"Pass --epochs > {start_epoch} to extend.")

    # ── Training ──────────────────────────────────────────────────────
    policy, history = meta_train(
        policy, task_set, x0_train, x0_eval,
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
        resume_state=resume_state,
        profile=args.profile,
        grad_chunk=(args.grad_chunk if args.grad_chunk > 0 else None),
    )

    # ── Save final checkpoint ─────────────────────────────────────────
    actual_epochs = len(history["epoch"])
    newly_trained = actual_epochs - start_epoch

    if newly_trained == 0:
        print("[Info] No new epochs trained; existing checkpoint unchanged.")
        print("Done.")
        return

    final_ckpt = os.path.join(out_dir, f"{base}_ep{actual_epochs}.pt")

    if os.path.exists(working_ckpt):
        os.replace(working_ckpt, final_ckpt)
    else:
        save_checkpoint(policy, history, final_ckpt)

    print(f"[Saved] {final_ckpt}")
    print(f"[Plots] run  python plot_ckpt.py --ckpt {final_ckpt}  "
          "to generate the task map and loss curve.")
    print("Done.")


if __name__ == "__main__":
    main()
