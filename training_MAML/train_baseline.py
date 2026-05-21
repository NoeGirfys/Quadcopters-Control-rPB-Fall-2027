"""Train a baseline (non-MAML) controller over a composite task set.

The task set is either:
  * **resumed** from a baseline checkpoint (``--resume``),
  * **mirrored** from a MAML checkpoint (``--from-maml-ckpt``, recommended:
    the baseline then sees *exactly* the same per-task support and query
    points as the MAML run), or
  * **sampled fresh** from a YAML/JSON config (``--tasks-config``).

Usage:

    cd training_MAML
    # Preferred: mirror the MAML run's task set
    python train_baseline.py --from-maml-ckpt maml_..._ep500.pt

    # Standalone (same defaults as train_maml.py)
    python train_baseline.py --tasks-config tasks.yaml --epochs 500

    # Resume
    python train_baseline.py --resume baseline_..._inprogress.pt
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
    PolicyMLP, CompositeTaskSet, baseline_train,
    GracefulKiller,
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
        "n_points_train":  int(saved.get("n_points_train", 100)),
        "n_points_eval":   int(saved.get("n_points_eval",  50)),
        "half_side":       float(saved.get("half_side", 0.3)),
        "mass_pos_sigma":  float(saved.get("mass_pos_sigma", 0.01)),
        "mass_min":        float(saved.get("mass_min", 0.002)),
        "mass_max":        float(saved.get("mass_max", 0.014)),
        "t_sim":           float(saved.get("t_sim", 2.0)),
        "tau_div":         float(saved.get("tau_div", 1.0)),
        "obs_noise_scale": float(saved.get("obs_noise_scale", 1.0)),
        "terminal_weight": float(saved.get("terminal_weight", 50.0)),
        "pos_weight":      float(saved.get("pos_weight", 10.0)),
        "z_weight":        float(saved.get("z_weight",  1.0)),
        "dynamics":        str(saved.get("dynamics", "nonlinear")),
        "seed":            int(saved.get("seed", 42)),
        "verbose_every":   int(saved.get("verbose_every", 1)),
        "k_samples":       saved.get("k_samples", None),
        # Adaptation hyperparameters: inherited so the baseline's per-epoch
        # target_loss uses the EXACT same K-shot protocol as the MAML run
        # and as the test-time protocol in test_maml_pybullet.py.
        "n_inner_steps":   int(saved.get("n_inner_steps", 0)),
        "lr_inner":        float(saved.get("lr_inner", 0.0)),
        "inner_grad_clip": float(saved.get("inner_grad_clip", 1.0)),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--resume",         type=str, default=None)
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
        description="Baseline (no MAML) training for CF2 with a composite task set.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--resume", type=str, default=None,
                   help="resume baseline training from a .pt checkpoint")
    p.add_argument("--from-maml-ckpt", type=str, default=None,
                   help="load task set (positions + masses + x0) and "
                        "hyperparameters from a MAML checkpoint, so the "
                        "baseline trains on identical data")

    # Training
    p.add_argument("--epochs",    type=int,   default=d.get("epochs",    500))
    p.add_argument("--lr-outer",  type=float, default=d.get("lr_outer",  3e-4))
    p.add_argument("--hidden",    type=int,   default=d.get("hidden",    64))

    # Task set (used only on a fresh standalone run)
    p.add_argument("--tasks-config",  type=str,   default=d.get("tasks_config", None),
                   help="path to the training tasks YAML/JSON (required for "
                        "a fresh standalone run; ignored when resuming or "
                        "using --from-maml-ckpt)")
    p.add_argument("--target-config", type=str,   default=d.get("target_config", None),
                   help="optional held-out target task YAML/JSON")
    p.add_argument("--mass-min",      type=float, default=d.get("mass_min", 0.002))
    p.add_argument("--mass-max",      type=float, default=d.get("mass_max", 0.014))
    p.add_argument("--mass-pos-sigma", type=float, default=d.get("mass_pos_sigma", 0.01))
    p.add_argument("--half-side",     type=float, default=d.get("half_side", 0.3))
    p.add_argument("--n-points-train", type=int,  default=d.get("n_points_train", 100))
    p.add_argument("--n-points-eval",  type=int,  default=d.get("n_points_eval",  50))
    p.add_argument("--k-samples",      type=int,  default=d.get("k_samples", None),
                   help="few-shot adaptation budget used both for the "
                        "per-epoch target_loss (if --n-inner-steps > 0) "
                        "and for the test-time adaptation in "
                        "test_maml_pybullet.py.")

    # Adaptation hyperparameters: if n_inner_steps > 0 and lr_inner > 0,
    # the per-epoch target_loss is computed *after* adapting the baseline
    # on K target support points — same protocol as MAML, so the two
    # histories are directly comparable. Default (0 / 0.0) keeps the
    # legacy raw-policy eval.
    p.add_argument("--n-inner-steps",   type=int,
                   default=d.get("n_inner_steps", 0),
                   help="adaptation steps for the per-epoch target_loss. "
                        "0 = raw-policy eval (legacy). Inherited from "
                        "--from-maml-ckpt by default.")
    p.add_argument("--lr-inner",        type=float,
                   default=d.get("lr_inner", 0.0),
                   help="inner-loop LR for the per-epoch target adaptation.")
    p.add_argument("--inner-grad-clip", type=float,
                   default=d.get("inner_grad_clip", 1.0),
                   help="grad-norm clip on the inner adaptation steps.")

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
    if args.n_inner_steps > 0 and args.lr_inner > 0:
        print(f"[Config] target eval: K-shot adaptation "
              f"(n_inner_steps={args.n_inner_steps}, lr_inner={args.lr_inner}, "
              f"clip={args.inner_grad_clip})  — symmetric with MAML")
    else:
        print(f"[Config] target eval: raw policy (no adaptation) — "
              f"NOT comparable to MAML's target_loss")

    # ── Load checkpoint if resuming ────────────────────────────────────
    ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)

    # ── Task sets ──────────────────────────────────────────────────────
    if ckpt is not None:
        task_set = CompositeTaskSet.from_dict(ckpt["task_set"])
        target_set = (CompositeTaskSet.from_dict(ckpt["target_set"])
                      if ckpt.get("target_set") is not None else None)
        print(f"[Tasks]  {len(task_set)} tasks restored from checkpoint")
    elif args.from_maml_ckpt is not None:
        maml_ckpt = torch.load(args.from_maml_ckpt, map_location="cpu",
                               weights_only=False)
        task_set = CompositeTaskSet.from_dict(maml_ckpt["task_set"])
        target_set = (CompositeTaskSet.from_dict(maml_ckpt["target_set"])
                      if maml_ckpt.get("target_set") is not None else None)
        print(f"[Tasks]  {len(task_set)} tasks loaded from MAML checkpoint  "
              f"M_train={task_set.M_train}  M_eval={task_set.M_eval}")
        if target_set is not None:
            print(f"[Target] {len(target_set)} target task(s) loaded from "
                  f"MAML checkpoint")
    else:
        if args.tasks_config is None:
            raise SystemExit(
                "--tasks-config is required for a fresh standalone run "
                "(or use --from-maml-ckpt to mirror a MAML run).")
        rng_tasks = np.random.default_rng(args.seed)
        task_set = CompositeTaskSet.from_config(
            args.tasks_config,
            M_train=args.n_points_train, M_eval=args.n_points_eval,
            mass_pos_sigma=args.mass_pos_sigma, half_side=args.half_side,
            mass_min=args.mass_min, mass_max=args.mass_max,
            rng=rng_tasks)
        print(f"[Tasks]  {len(task_set)} tasks from {args.tasks_config}")
        if args.target_config is not None:
            rng_target = np.random.default_rng(args.seed + 1)
            # Mirror train_maml.py: size target support to K if set, so the
            # resulting checkpoint is consistent with a MAML run started
            # with the same --k-samples.
            target_M_train = (args.k_samples
                              if args.k_samples and args.k_samples > 0
                              else args.n_points_train)
            target_set = CompositeTaskSet.from_config(
                args.target_config,
                M_train=target_M_train, M_eval=args.n_points_eval,
                mass_pos_sigma=args.mass_pos_sigma, half_side=args.half_side,
                mass_min=args.mass_min, mass_max=args.mass_max,
                rng=rng_target)
            print(f"[Target] {len(target_set)} target task(s) from "
                  f"{args.target_config}  M_train={target_M_train}")
        else:
            target_set = None

    # ── Policy ────────────────────────────────────────────────────────
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

    # ── Output paths ──────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(__file__))
    tag = f"_{args.tag}" if args.tag else ""
    base = (f"baseline_{args.dynamics}_h{args.hidden}_"
            f"n{len(task_set)}{tag}")
    working_ckpt = os.path.join(out_dir, base + "_inprogress.pt")

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
        policy, task_set,
        target_set=target_set,
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
        n_inner_steps=int(args.n_inner_steps),
        lr_inner=float(args.lr_inner),
        inner_grad_clip=float(args.inner_grad_clip),
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
    print(f"[Plots] run  python plot_ckpt.py --ckpt {final_ckpt}")
    print("Done.")


if __name__ == "__main__":
    main()
