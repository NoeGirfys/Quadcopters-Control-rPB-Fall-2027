"""Two report figures comparing the MAML and baseline checkpoints.

(1) Convergence  -- ``<prefix>_convergence.png``
    MAML inner (pre-adaptation) and outer (post-adaptation / meta) losses,
    together with the baseline train (and eval) loss, versus epoch.

(2) Adaptation curve -- ``<prefix>_adaptation.png``
    Held-out task loss as a function of the number of K-shot gradient
    updates (0, 1, 2, ... N), using the MAML checkpoint's inner learning
    rate, for both the MAML meta-initialisation and the joint baseline.
    This generalises the 0- and 1-step comparison to a full curve: MAML
    typically starts high (its meta-init is a launch-point, not a good
    zero-shot policy) and drops fast, while the baseline starts low (good
    zero-shot) and may stagnate or over-fit as steps accumulate.

The few-shot task is, by default, a CENTERED payload of the MAML
checkpoint's target magnitude; ``--mass`` / ``--offset-dir`` / ``--offset-dist``
override it. K, lr_inner and grad-clip come from the MAML checkpoint.

Usage:
    cd training_MAML
    python plot_report_curves.py \\
        --maml-ckpt     maml_..._offdirmag_ep500.pt \\
        --baseline-ckpt baseline_..._offdirmag_ep500.pt \\
        --max-steps 10 --mass 10
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
    PolicyMLP, compute_mass_params_batched, DYNAMICS,
)
from maml_lib.cost import trajectory_cost
from maml_lib.rollout import rollout
from maml_lib.maml import _make_fn


# ===================================================================
#  (1) Convergence figure
# ===================================================================

def plot_convergence(maml_ckpt, base_ckpt, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mh = maml_ckpt["history"]
    bh = base_ckpt["history"]
    ep_m = mh.get("epoch", [])
    ep_b = bh.get("epoch", [])

    fig, ax = plt.subplots(figsize=(9, 5))
    if ep_m:
        ax.plot(ep_m, mh["inner_pre"], color="C0", label="MAML inner (pre-adaptation)")
        ax.plot(ep_m, mh["meta_loss"], color="C3", label="MAML outer (post-adaptation / meta)")
    if ep_b:
        ax.plot(ep_b, bh["train_loss"], color="C2", label="baseline train")
        if bh.get("eval_loss"):
            ax.plot(ep_b, bh["eval_loss"], color="C2", ls="--", alpha=0.6,
                    label="baseline eval")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training convergence: MAML (inner / outer) vs baseline")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ===================================================================
#  (2) Adaptation curve: loss vs number of K-shot gradient updates
# ===================================================================

def _clip_grads(grads, max_norm):
    if max_norm <= 0:
        return grads
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    coef = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    return [g * coef for g in grads]


def _make_x0(n, half_side, rng):
    x0 = torch.zeros(n, 12)
    p = torch.from_numpy(rng.uniform(-half_side, half_side, (n, 3)).astype(np.float32))
    x0[:, 0], x0[:, 2], x0[:, 4] = p[:, 0], p[:, 1], p[:, 2]
    return x0


def _mass(m_kg, n, r_offset):
    return compute_mass_params_batched(
        np.full(n, m_kg, dtype=np.float32),
        np.tile(np.asarray(r_offset, dtype=np.float32), (n, 1)))


def adaptation_curve(policy, x0K, massK, x0Q, massQ, cfg, max_steps):
    """Return [L_0, L_1, ..., L_max] = query loss after s gradient steps."""
    theta = {n: p.detach().clone().requires_grad_(True)
             for n, p in policy.named_parameters()}
    gen = (torch.Generator().manual_seed(cfg["seed"]) if cfg["obs_noise"] is not None
           else None)

    def query_loss(th):
        with torch.no_grad():
            X = rollout(_make_fn(policy, th), x0Q, cfg["n_steps"], massQ, cfg["dyn"],
                        tau_div=cfg["tau"], obs_noise_std=None, gen=None)
            return float(trajectory_cost(X, cfg["term_w"],
                                         pos_weight=cfg["pos_w"], z_weight=cfg["z_w"]).mean())

    losses = [query_loss(theta)]
    for _ in range(max_steps):
        X = rollout(_make_fn(policy, theta), x0K, cfg["n_steps"], massK, cfg["dyn"],
                    tau_div=cfg["tau"], obs_noise_std=cfg["obs_noise"], gen=gen)
        L = trajectory_cost(X, cfg["term_w"],
                            pos_weight=cfg["pos_w"], z_weight=cfg["z_w"]).mean()
        grads = torch.autograd.grad(L, tuple(theta.values()), create_graph=False)
        grads = _clip_grads(grads, cfg["clip"])
        theta = {n: (p - cfg["lr_inner"] * g).detach().requires_grad_(True)
                 for (n, p), g in zip(theta.items(), grads)}
        losses.append(query_loss(theta))
    return losses


def plot_adaptation(maml_ckpt, base_ckpt, cfg, m_kg, r_offset, max_steps,
                    n_query, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pol_m = PolicyMLP(hidden=cfg["hidden"]); pol_m.load_state_dict(maml_ckpt["state_dict"]); pol_m.eval()
    pol_b = PolicyMLP(hidden=cfg["hidden"]); pol_b.load_state_dict(base_ckpt["state_dict"]); pol_b.eval()

    rng = np.random.default_rng(cfg["seed"])
    x0K = _make_x0(cfg["K"], cfg["half_side"], rng); massK = _mass(m_kg, cfg["K"], r_offset)
    x0Q = _make_x0(n_query, cfg["half_side"], rng);  massQ = _mass(m_kg, n_query, r_offset)

    L_m = adaptation_curve(pol_m, x0K, massK, x0Q, massQ, cfg, max_steps)
    L_b = adaptation_curve(pol_b, x0K, massK, x0Q, massQ, cfg, max_steps)
    steps = np.arange(max_steps + 1)

    print(f"  step:  " + " ".join(f"{s:6d}" for s in steps))
    print(f"  MAML:  " + " ".join(f"{v:6.1f}" for v in L_m))
    print(f"  base:  " + " ".join(f"{v:6.1f}" for v in L_b))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, L_m, color="C3", marker="o", label="MAML meta-init")
    ax.plot(steps, L_b, color="C0", marker="s", label="joint baseline")
    ax.axvline(cfg["n_inner_trained"], color="gray", ls=":", alpha=0.7,
               label=f"trained K-shot = {cfg['n_inner_trained']}")
    ax.set_xlabel(f"number of K-shot gradient updates (lr_inner = {cfg['lr_inner']})")
    ax.set_ylabel("held-out task loss (query)")
    ax.set_yscale("log")
    tag = ("centred" if abs(np.sum(np.abs(r_offset))) < 1e-9
           else f"offset {tuple(np.round(r_offset,3))}")
    ax.set_title(f"Few-shot adaptation curve — {m_kg*1e3:.0f} g payload ({tag})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ===================================================================
#  CLI / main
# ===================================================================

def build_cfg(maml_ckpt):
    a = maml_ckpt["args"]
    ts = maml_ckpt["task_set"]
    return dict(
        hidden    = int(a.get("hidden", 64)),
        n_inner_trained = int(a.get("n_inner_steps", 1)),
        lr_inner  = float(a.get("lr_inner", 0.05)),
        clip      = float(a.get("inner_grad_clip", 1.0)),
        K         = int(a.get("k_samples", 5) or 5),
        term_w    = float(a.get("terminal_weight", 50.0)),
        pos_w     = float(a.get("pos_weight", 10.0)),
        z_w       = float(a.get("z_weight", 1.0)),
        n_steps   = int(float(a.get("t_sim", 3.0)) * C.NN_FREQ),
        tau       = (a.get("tau_end") or a.get("tau_div") or None),
        half_side = float(ts.get("half_side", 0.2)),
        dyn       = DYNAMICS[a.get("dynamics", "nonlinear")],
        seed      = int(a.get("seed", 0)),
        obs_noise = None,
        target_mass_g = (float(np.asarray(maml_ckpt["target_set"]["m_extras_train"]).mean() * 1e3)
                         if maml_ckpt.get("target_set") is not None else 6.0),
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maml-ckpt", required=True)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--max-steps", type=int, default=10,
                   help="max number of K-shot gradient updates for the curve")
    p.add_argument("--adapt-lr", type=float, default=None,
                   help="override inner lr for the adaptation curve (default: "
                        "MAML ckpt value, 0.05). The ckpt lr is tuned for a "
                        "single step and oscillates when iterated; a smaller "
                        "lr (e.g. 0.01) gives a monotone descent curve.")
    p.add_argument("--mass", type=float, default=None,
                   help="adaptation task payload [g] (default: MAML target mass)")
    p.add_argument("--offset-dir", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                   metavar=("X", "Y", "Z"), help="attachment direction (default centred)")
    p.add_argument("--offset-dist", type=float, default=None,
                   help="attachment lever [m] when offset-dir != 0 (default motor arm)")
    p.add_argument("--n-query", type=int, default=40)
    p.add_argument("--obs-noise-scale", type=float, default=0.0,
                   help="observation noise during adaptation (eval stays noise-free)")
    p.add_argument("--out-prefix", type=str, default=None,
                   help="output prefix (default: next to the MAML ckpt)")
    return p.parse_args()


def main():
    args = parse_args()
    maml_ckpt = torch.load(args.maml_ckpt, map_location="cpu", weights_only=False)
    base_ckpt = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    cfg = build_cfg(maml_ckpt)
    if args.obs_noise_scale > 0:
        cfg["obs_noise"] = C.OBS_NOISE_STD * args.obs_noise_scale
    if args.adapt_lr is not None:
        cfg["lr_inner"] = args.adapt_lr

    prefix = args.out_prefix or os.path.join(
        os.path.dirname(os.path.abspath(args.maml_ckpt)),
        os.path.splitext(os.path.basename(args.maml_ckpt))[0])

    # (1) convergence
    plot_convergence(maml_ckpt, base_ckpt, prefix + "_convergence.png")

    # (2) adaptation curve
    m_g = args.mass if args.mass is not None else cfg["target_mass_g"]
    d = np.asarray(args.offset_dir, dtype=np.float32)
    if np.linalg.norm(d) < 1e-9:
        r_offset = np.zeros(3, dtype=np.float32)
    else:
        d = d / np.linalg.norm(d)
        dist = args.offset_dist if args.offset_dist is not None else float(C.L / np.sqrt(2))
        r_offset = d * dist
    print(f"[Adapt curve] mass={m_g:.1f} g  r_offset={r_offset*100} cm  "
          f"K={cfg['K']}  lr_inner={cfg['lr_inner']}  max_steps={args.max_steps}  "
          f"noise={args.obs_noise_scale}")
    plot_adaptation(maml_ckpt, base_ckpt, cfg, m_g * 1e-3, r_offset,
                    args.max_steps, args.n_query, prefix + "_adaptation.png")
    print("Done.")


if __name__ == "__main__":
    main()
