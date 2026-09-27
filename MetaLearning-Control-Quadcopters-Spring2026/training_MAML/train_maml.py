"""Train a Crazyflie controller with MAML over a composite task set.

Task structure (see ``maml_lib.tasks.CompositeTaskSet``):
  * each task is a row of a YAML/JSON config, selecting one or more of
    4 motor-centered isotropic Gaussians (mass attachment position) and
    one or more of 8 cube octants (drone start position).
  * extra-mass magnitude is a shared 1-D uniform ``[mass_min, mass_max]``.
  * ``M_train`` support + ``M_eval`` query points per task are sampled
    once at startup and reused every epoch (fixed task set philosophy).

An optional ``--target-config`` defines a held-out task set: its loss
after few-shot adaptation is logged every epoch as a generalisation
metric but **never** feeds the optimiser.

Usage examples:

    cd training_MAML
    python train_maml.py --tasks-config tasks.yaml \\
                         --target-config target.yaml \\
                         --dynamics nonlinear --epochs 500
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
    PolicyMLP, CompositeTaskSet, meta_train,
    GracefulKiller,
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
        description="MAML training for CF2 with a composite task set.",
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
    p.add_argument("--tasks-config",  type=str,   default=d.get("tasks_config", None),
                   help="path to the training tasks YAML/JSON (required on a "
                        "fresh run; ignored on resume — tasks come from the ckpt)")
    p.add_argument("--target-config", type=str,   default=d.get("target_config", None),
                   help="optional held-out target task YAML/JSON; if given, "
                        "the policy's loss on it is logged every epoch")
    # Task-distribution parameters (mass magnitude pool, position σ, start
    # cube half-side) now live in the YAML config — it is the single source
    # of truth for the task distribution. Only the sampling counts and seed
    # stay on the CLI.
    p.add_argument("--n-points-train", type=int,  default=d.get("n_points_train", 100),
                   help="number of support points sampled per task")
    p.add_argument("--n-points-eval",  type=int,  default=d.get("n_points_eval", 50),
                   help="number of query points sampled per task")
    p.add_argument("--k-samples",      type=int,  default=d.get("k_samples", None),
                   help="few-shot adaptation budget on the held-out target "
                        "task(s). If set, the target support set is sized "
                        "to exactly K points (saving memory and making the "
                        "few-shot semantics explicit); the per-epoch target "
                        "eval and the test-time adaptation then use all K. "
                        "None = target support sized to --n-points-train "
                        "(legacy behavior, used as full support).")

    # Rollout
    p.add_argument("--t-sim",           type=float, default=d.get("t_sim",           2.0))
    p.add_argument("--tau-div",         type=float, default=d.get("tau_div",         1.0),
                   help="fixed divergence threshold [m]; <=0 to disable. "
                        "Ignored if --tau-start/--tau-end are both > 0.")
    p.add_argument("--tau-start",       type=float, default=d.get("tau_start",       0.0),
                   help="curriculum: initial divergence threshold [m]. If "
                        "this and --tau-end are both > 0, tau grows linearly "
                        "tau_start -> tau_end over the run (short BPTT horizon "
                        "early -> clean gradients for base regulation incl. z).")
    p.add_argument("--tau-end",         type=float, default=d.get("tau_end",         0.0),
                   help="curriculum: final divergence threshold [m].")
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

    # ── Task sets ──────────────────────────────────────────────────────
    if ckpt is not None:
        task_set = CompositeTaskSet.from_dict(ckpt["task_set"])
        target_set = (CompositeTaskSet.from_dict(ckpt["target_set"])
                      if ckpt.get("target_set") is not None else None)
        print(f"[Tasks]  {len(task_set)} tasks restored from checkpoint "
              f"(M_train={task_set.M_train}, M_eval={task_set.M_eval})")
        if target_set is not None:
            print(f"[Target] {len(target_set)} target task(s) restored "
                  f"from checkpoint")
    else:
        if args.tasks_config is None:
            raise SystemExit(
                "--tasks-config is required for a fresh run. "
                "Provide a YAML/JSON describing the training tasks.")
        rng_tasks = np.random.default_rng(args.seed)
        task_set = CompositeTaskSet.from_config(
            args.tasks_config,
            M_train=args.n_points_train, M_eval=args.n_points_eval,
            rng=rng_tasks)
        print(f"[Tasks]  {len(task_set)} tasks from {args.tasks_config}  "
              f"M_train={task_set.M_train}  M_eval={task_set.M_eval}  "
              f"σ={task_set.mass_pos_sigma*100:.1f}cm  "
              f"half_side={task_set.half_side}m  "
              f"m∈[{task_set.mass_min*1e3:.1f},{task_set.mass_max*1e3:.1f}]g")
        if args.target_config is not None:
            # Independent RNG stream so target sampling does not perturb
            # the training-task sampling stream.
            rng_target = np.random.default_rng(args.seed + 1)
            # Few-shot K: if --k-samples is set, the target support set is
            # sized to exactly K (no point storing more than what
            # adaptation will use). Otherwise fall back to n_points_train
            # for backward compatibility.
            target_M_train = (args.k_samples
                              if args.k_samples and args.k_samples > 0
                              else args.n_points_train)
            target_set = CompositeTaskSet.from_config(
                args.target_config,
                M_train=target_M_train, M_eval=args.n_points_eval,
                rng=rng_target)
            kspec = (f"K={target_M_train} (few-shot)"
                     if args.k_samples and args.k_samples > 0
                     else f"M_train={target_M_train} (full)")
            print(f"[Target] {len(target_set)} target task(s) from "
                  f"{args.target_config}  {kspec}")
        else:
            target_set = None

    # ── Policy ────────────────────────────────────────────────────────
    # Hover bias = mean extra mass over the *sampled* training points,
    # i.e. ≈ (mass_min + mass_max)/2 for large enough samples. This is
    # just the optimisation starting point — meta-training adjusts it.
    torch.manual_seed(args.seed)
    M_total = C.M_BASE + task_set.m_extra
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
    tau_start = args.tau_start if args.tau_start > 0 else None
    tau_end   = args.tau_end   if args.tau_end   > 0 else None
    if (tau_start is None) != (tau_end is None):
        raise SystemExit("--tau-start and --tau-end must both be > 0 to "
                         "enable the curriculum (or both omitted/<=0 to use "
                         "the fixed --tau-div).")
    if tau_start is not None:
        print(f"[Config] tau curriculum: {tau_start} -> {tau_end} m "
              f"(fixed tau_div ignored; target eval uses tau={tau_end})")
    else:
        print(f"[Config] tau fixed: {tau_div} m (no curriculum)")

    # ── Output paths ──────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag = f"_{args.tag}" if args.tag else ""
    base = (f"maml_{args.dynamics}_h{args.hidden}_"
            f"o{args.maml_order}_n{len(task_set)}{tag}")
    working_ckpt = os.path.join(out_dir, base + "_inprogress.pt")

    # Payload fields that never change between epochs.
    common_payload = {
        "args":       vars(args),
        "x_scale":    C.X_SCALE,
        "pos_scale":  C.POS_SCALE,
        "nn_freq":    C.NN_FREQ,
        "att_rate":   C.ATTITUDE_RATE,
        "task_set":   task_set.to_dict(),
        "target_set": target_set.to_dict() if target_set is not None else None,
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
        policy, task_set,
        target_set=target_set,
        dynamics_step=DYNAMICS[args.dynamics],
        n_steps=n_steps,
        epochs=args.epochs,
        n_inner_steps=args.n_inner_steps,
        lr_outer=args.lr_outer,
        lr_inner=args.lr_inner,
        maml_order=args.maml_order,
        tau_div=tau_div,
        tau_start=tau_start,
        tau_end=tau_end,
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
        k_samples=(args.k_samples if args.k_samples and args.k_samples > 0
                   else None),
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
