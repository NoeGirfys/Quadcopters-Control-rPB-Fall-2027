"""Train a baseline (non-MAML) controller over a fixed uniform mass task set.

The task set is either loaded from an existing MAML checkpoint (recommended,
so that the baseline is trained on exactly the same tasks) or sampled fresh
from a uniform box.

Usage:

    cd training_MAML
    # Preferred: mirror the MAML run's task set
    python train_baseline.py --from-maml-ckpt maml_linearized_h64_o1_n50_ep500.pt

    # Standalone (same CLI as train_maml.py without MAML-specific args)
    python train_baseline.py --n-tasks 50 --epochs 500

    # Resume a baseline run
    python train_baseline.py --resume baseline_linearized_h64_n50_inprogress.pt
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
    PolicyMLP, FixedUniformMassSet, baseline_train,
    sample_hover_x0, GracefulKiller,
    DYNAMICS,
)


# ---------------------------------------------------------------------------
# Resume / inherit helpers
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


def _load_maml_defaults(ckpt_path: str) -> dict:
    """Pull hyperparameters from a MAML checkpoint (skip MAML-specific ones)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("args", {})
    actual_epochs = ckpt.get("actual_epochs",
                             len(ckpt.get("history", {}).get("epoch", []))) \
                    or saved.get("epochs", 500)
    return {
        "epochs":          int(actual_epochs),
        "lr_outer":        float(saved.get("lr_outer", 3e-4)),
        "hidden":          int(saved.get("hidden", 64)),
        "n_x0_train":      int(saved.get("n_x0_train", 64)),
        "n_x0_eval":       int(saved.get("n_x0_eval", 32)),
        "half_side":       float(saved.get("half_side", 0.3)),
        "t_sim":           float(saved.get("t_sim", 2.0)),
        "tau_div":         float(saved.get("tau_div", 1.0)),
        "obs_noise_scale": float(saved.get("obs_noise_scale", 1.0)),
        "terminal_weight": float(saved.get("terminal_weight", 50.0)),
        "pos_weight":      float(saved.get("pos_weight", 10.0)),
        "z_weight":        float(saved.get("z_weight", 1.0)),
        "dynamics":        str(saved.get("dynamics", "nonlinear")),
        "mass":            float(saved.get("mass", 0.010)),
        "n_tasks":         int(saved.get("n_tasks", 50)),
        "xy_min":          float(saved.get("xy_min", -0.04)),
        "xy_max":          float(saved.get("xy_max",  0.04)),
        "z_min":           float(saved.get("z_min",  -0.01)),
        "z_max":           float(saved.get("z_max",   0.01)),
        "seed":            int(saved.get("seed", 42)),
        "verbose_every":   int(saved.get("verbose_every", 1)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--resume",        type=str, default=None)
    pre.add_argument("--from-maml-ckpt", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    if pre_args.resume is not None:
        d = _load_resume_defaults(pre_args.resume)
        print(f"[Init] Resuming from {pre_args.resume}")
    elif pre_args.from_maml_ckpt is not None:
        d = _load_maml_defaults(pre_args.from_maml_ckpt)
        print(f"[Init] Inheriting hyperparameters from MAML checkpoint: "
              f"{pre_args.from_maml_ckpt}")
    else:
        d = {}

    p = argparse.ArgumentParser(
        description="Baseline (no MAML) training for CF2 with fixed uniform tasks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--resume", type=str, default=None,
                   help="resume baseline training from a .pt checkpoint")
    p.add_argument("--from-maml-ckpt", type=str, default=None,
                   help="load task set + hyperparameters from a MAML checkpoint "
                        "to train the baseline on identical data")

    # Training
    p.add_argument("--epochs",    type=int,   default=d.get("epochs",    500))
    p.add_argument("--lr-outer",  type=float, default=d.get("lr_outer",  3e-4))
    p.add_argument("--hidden",    type=int,   default=d.get("hidden",    64))

    # Task set
    p.add_argument("--n-tasks",  type=int,   default=d.get("n_tasks",  50))
    p.add_argument("--mass",     type=float, default=d.get("mass",     0.010),
                   help="extra point mass for every task [kg]")
    p.add_argument("--xy-min",   type=float, default=d.get("xy_min",  -0.04))
    p.add_argument("--xy-max",   type=float, default=d.get("xy_max",   0.04))
    p.add_argument("--z-min",    type=float, default=d.get("z_min",   -0.01))
    p.add_argument("--z-max",    type=float, default=d.get("z_max",    0.01))

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

    # Dynamics
    p.add_argument("--dynamics", choices=list(DYNAMICS.keys()),
                   default=d.get("dynamics", "nonlinear"))

    # Misc
    p.add_argument("--seed",          type=int, default=d.get("seed",         42))
    p.add_argument("--tag",           type=str, default=d.get("tag",          ""))
    p.add_argument("--verbose-every", type=int, default=d.get("verbose_every", 1))

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Device] {device}")
    print(f"[Config] dynamics={args.dynamics}  epochs={args.epochs}")
    print(f"[Config] lr_outer={args.lr_outer}  hidden={args.hidden}")
    n_steps = int(args.t_sim * C.NN_FREQ)
    print(f"[Config] n_steps={n_steps} ({args.t_sim}s @ {C.NN_FREQ}Hz)")

    # ── Load checkpoint if resuming ────────────────────────────────────
    ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

    # ── Fixed task set ─────────────────────────────────────────────────
    if ckpt is not None:
        ckpt_m_extra = float(ckpt["m_extra"])
        if abs(args.mass - ckpt_m_extra) > 1e-9:
            print(f"[WARNING] --mass {args.mass} differs from the checkpoint's "
                  f"m_extra {ckpt_m_extra}. Resuming keeps the saved task "
                  f"positions and optimizer state but changes the extra mass, "
                  f"which is inconsistent with the run so far. Use --mass "
                  f"{ckpt_m_extra} to resume faithfully.")
        task_set = FixedUniformMassSet.from_positions(
            ckpt["task_positions"], m_extra=args.mass)
        print(f"[Tasks]  {len(task_set)} tasks restored from checkpoint")
    elif args.from_maml_ckpt is not None:
        maml_ckpt = torch.load(args.from_maml_ckpt, map_location="cpu",
                               weights_only=False)
        task_set = FixedUniformMassSet.from_positions(
            maml_ckpt["task_positions"], m_extra=maml_ckpt["m_extra"])
        print(f"[Tasks]  {len(task_set)} tasks loaded from MAML checkpoint")
    else:
        rng_tasks = np.random.default_rng(args.seed)
        task_set = FixedUniformMassSet.sample(
            n_tasks=args.n_tasks,
            xy_min=args.xy_min, xy_max=args.xy_max,
            z_min=args.z_min,   z_max=args.z_max,
            m_extra=args.mass,
            rng=rng_tasks,
        )
        print(f"[Tasks]  {args.n_tasks} tasks sampled  "
              f"xy=[{args.xy_min*100:.1f},{args.xy_max*100:.1f}]cm  "
              f"z=[{args.z_min*100:.1f},{args.z_max*100:.1f}]cm  "
              f"m={args.mass*1e3:.0f}g")

    # ── x0 batches ────────────────────────────────────────────────────
    if ckpt is not None:
        x0_train = ckpt["x0_train"].to(device)
        x0_eval  = ckpt["x0_eval"].to(device)
        print(f"[x0]    n_train={x0_train.shape[0]}  n_eval={x0_eval.shape[0]}"
              f"  (restored)")
    elif args.from_maml_ckpt is not None:
        x0_train = maml_ckpt["x0_train"].to(device)
        x0_eval  = maml_ckpt["x0_eval"].to(device)
        print(f"[x0]    n_train={x0_train.shape[0]}  n_eval={x0_eval.shape[0]}"
              f"  (from MAML checkpoint)")
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
    # MAML policy (which seeds the same way). On resume this is harmless: the
    # init weights are immediately overwritten by load_state_dict, and
    # baseline_train restores the saved RNG state anyway.
    torch.manual_seed(args.seed)
    M_total = C.M_BASE + args.mass
    hover_thrust_u16 = (M_total * C.G * C.UINT16_MAX) / (4 * C.CF2_THRUST_MAX_PER_MOTOR)
    policy = PolicyMLP(hidden=args.hidden, hover_thrust_u16=hover_thrust_u16).to(device)
    if ckpt is not None:
        policy.load_state_dict(ckpt["state_dict"])
        print(f"[Policy] restored weights  (actual_epochs={ckpt.get('actual_epochs','?')})")
    print(f"[Policy] hover thrust (M={M_total*1e3:.1f}g) ≈ {hover_thrust_u16:.0f}")

    # ── Ancillary ─────────────────────────────────────────────────────
    obs_noise = (C.OBS_NOISE_STD * args.obs_noise_scale).to(device) \
                if args.obs_noise_scale > 0 else None
    tau_div = args.tau_div if args.tau_div > 0 else None

    # ── Output paths ──────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag = f"_{args.tag}" if args.tag else ""
    base = (f"baseline_{args.dynamics}_h{args.hidden}_"
            f"n{len(task_set)}{tag}")
    working_ckpt = os.path.join(out_dir, base + "_inprogress.pt")

    common_payload = {
        "args":           vars(args),
        "x_scale":        C.X_SCALE,
        "pos_scale":      C.POS_SCALE,
        "nn_freq":        C.NN_FREQ,
        "att_rate":       C.ATTITUDE_RATE,
        "task_positions": task_set.positions,
        "m_extra":        task_set.m_extra,
        "x0_train":       x0_train.detach().cpu(),
        "x0_eval":        x0_eval.detach().cpu(),
    }

    def save_checkpoint(policy_module, history, path,
                        optimizer=None, torch_gen=None):
        payload = dict(common_payload)
        payload["state_dict"]    = {k: v.detach().cpu()
                                    for k, v in policy_module.state_dict().items()}
        payload["history"]       = history
        payload["actual_epochs"] = len(history.get("epoch", []))
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if torch_gen is not None:
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
            "optimizer_state": ckpt["optimizer_state"],
            "torch_gen_state": ckpt["torch_gen_state"],
            "torch_rng_state": ckpt["torch_rng_state"],
            "history":         ckpt["history"],
        }
        if "torch_cuda_rng_state" in ckpt:
            resume_state["torch_cuda_rng_state"] = ckpt["torch_cuda_rng_state"]
        start_epoch = len(ckpt["history"].get("epoch", []))
        if start_epoch >= args.epochs:
            print(f"[Resume] checkpoint already has {start_epoch} epochs "
                  f">= --epochs {args.epochs}; nothing to do. "
                  f"Pass --epochs > {start_epoch} to extend.")

    # ── Training ──────────────────────────────────────────────────────
    policy, history = baseline_train(
        policy, task_set, x0_train, x0_eval,
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
        resume_state=resume_state,
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
