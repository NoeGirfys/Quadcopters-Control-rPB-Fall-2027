"""Evaluate the MAML-vs-baseline few-shot gap and plot graceful degradation.

For a sweep of payload masses, and averaged over ``--n-draws`` independent
few-shot support draws, this measures three controllers on a fresh
(noise-free) query set:

  * ``base_0shot`` : the joint-trained baseline, deployed as-is (no adapt).
  * ``base_adapt`` : the baseline after K-shot adaptation (standard
    "pretrain + fine-tune" MAML comparison).
  * ``maml_adapt`` : the MAML meta-init after the same K-shot adaptation.

It prints a mean±std table with the per-mass win-rate (MAML < base_adapt)
and saves a "graceful degradation" figure: cost vs mass for the three
controllers, with ±1σ bands, training masses and the target mass marked.

Two task modes:
  * ``--mode centered`` : payload at the drone centre (pure weight change;
    the firmware's integral attitude loop is irrelevant, the un-integrated
    collective-thrust channel is what differs per task).
  * ``--mode offset``   : payload pinned on motor M1 at ``--offset-dist``
    along ``--offset-dir`` (default the motor arm) — weight AND a gravity
    torque grow with the mass, probing the "offset that saturates" regime
    (attitude correction steals collective via the motor cap).

NB: a checkpoint trained on CENTERED tasks is OUT-OF-DISTRIBUTION under
``--mode offset`` for *both* controllers — read that sweep as "how does a
centered-trained policy transfer to an offset task", not as a fair
offset-vs-offset MAML benchmark (for that, train on an offset task set).

Usage:
    cd training_MAML
    python eval_gap.py \
        --maml-ckpt     maml_nonlinear_h64_o1_n3_izar_payload_ep500.pt \
        --baseline-ckpt baseline_nonlinear_h64_n3_izar_payload_ep500.pt \
        --mode centered
"""
import argparse
import os
import sys

import numpy as np
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from maml_lib import (config as C, PolicyMLP, compute_mass_params_batched,
                      maml_adapt, DYNAMICS)
from maml_lib.rollout import rollout
from maml_lib.cost import trajectory_cost
from maml_lib.maml import _make_fn


# ---------------------------------------------------------------------------
# Config extracted from the MAML checkpoint (single source of truth)
# ---------------------------------------------------------------------------

class EvalCfg:
    def __init__(self, maml_ckpt: dict):
        a = maml_ckpt["args"]
        self.hidden     = int(a.get("hidden", 64))
        self.n_inner    = int(a.get("n_inner_steps", 1))
        self.lr_inner   = float(a.get("lr_inner", 0.05))
        self.clip       = float(a.get("inner_grad_clip", 1.0))
        self.K          = int(a.get("k_samples", 5) or 5)
        self.term_w     = float(a.get("terminal_weight", 50.0))
        self.pos_w      = float(a.get("pos_weight", 10.0))
        self.z_w        = float(a.get("z_weight", 1.0))
        self.t_sim      = float(a.get("t_sim", 3.0))
        self.dyn        = DYNAMICS[a.get("dynamics", "nonlinear")]
        # curriculum-aware threshold used for the (held-out) metric
        self.tau        = (a.get("tau_end") or a.get("tau_div") or 1.0)
        self.n_steps    = int(self.t_sim * C.NN_FREQ)
        ts = maml_ckpt["task_set"]
        self.half_side  = float(ts.get("half_side", 0.1))
        self.train_masses_g = np.unique(
            np.round(np.asarray(ts["m_extras_train"]).mean(1) * 1e3, 1))
        tg = maml_ckpt.get("target_set")
        self.target_mass_g = (float(np.asarray(tg["m_extras_train"]).mean() * 1e3)
                              if tg is not None else None)
        # Observation noise for the final-model test (set from the CLI). When
        # > 0, both the few-shot adaptation and the query eval roll-outs are
        # run on a noisy state estimate — realistic (= deployment conditions),
        # and on the same footing as the (noisy) training roll-outs.
        self.obs_noise = None


def _policy(ckpt: dict, hidden: int) -> PolicyMLP:
    pol = PolicyMLP(hidden=hidden)
    pol.load_state_dict(ckpt["state_dict"])
    pol.eval()
    return pol


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _make_x0(n: int, half: float, rng: np.random.Generator) -> torch.Tensor:
    x0 = torch.zeros(n, 12)
    p = torch.from_numpy(rng.uniform(-half, half, (n, 3)).astype(np.float32))
    x0[:, 0], x0[:, 2], x0[:, 4] = p[:, 0], p[:, 1], p[:, 2]
    return x0


def _mass(m_kg: float, n: int, r_offset: np.ndarray):
    me = np.full(n, m_kg, dtype=np.float32)
    pos = np.tile(r_offset.astype(np.float32), (n, 1))
    return compute_mass_params_batched(me, pos)


def _cost(fn, x0, mass, cfg: EvalCfg, noise_seed=None) -> float:
    gen = None
    if cfg.obs_noise is not None and noise_seed is not None:
        gen = torch.Generator().manual_seed(int(noise_seed))
    with torch.no_grad():
        X = rollout(fn, x0, cfg.n_steps, mass, cfg.dyn,
                    tau_div=cfg.tau, obs_noise_std=cfg.obs_noise, gen=gen)
        return float(trajectory_cost(X, cfg.term_w,
                                     pos_weight=cfg.pos_w, z_weight=cfg.z_w).mean())


def _adapt(pol, x0K, mK, cfg: EvalCfg, seed: int = 0):
    theta, _ = maml_adapt(pol, mK, x0K, dynamics_step=cfg.dyn,
                          n_steps=cfg.n_steps, n_steps_adapt=cfg.n_inner,
                          lr_inner=cfg.lr_inner, terminal_weight=cfg.term_w,
                          pos_weight=cfg.pos_w, z_weight=cfg.z_w,
                          obs_noise_std=cfg.obs_noise, tau_div=cfg.tau,
                          inner_grad_clip=cfg.clip, device="cpu",
                          seed=seed, verbose=False)
    return theta


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(pol_m, pol_b, cfg: EvalCfg, masses_g, r_offset, n_draws, n_query):
    """Return dict mass_g -> {key: (mean, std, vals)} for the 3 controllers."""
    out = {}
    for m_g in masses_g:
        m = m_g * 1e-3
        b0, ba, ma = [], [], []
        for s in range(n_draws):
            rng = np.random.default_rng(7000 + int(round(m_g * 10)) * 17 + s)
            x0K = _make_x0(cfg.K, cfg.half_side, rng)
            mK  = _mass(m, cfg.K, r_offset)
            x0Q = _make_x0(n_query, cfg.half_side, rng)
            mQ  = _mass(m, n_query, r_offset)
            # Same adaptation-noise realisation (seed=s) and same eval-noise
            # realisation (noise_seed) for all three controllers -> fair.
            nseed = 9000 + int(round(m_g * 10)) * 17 + s
            b0.append(_cost(pol_b, x0Q, mQ, cfg, noise_seed=nseed))
            ba.append(_cost(_make_fn(pol_b, _adapt(pol_b, x0K, mK, cfg, seed=s)), x0Q, mQ, cfg, noise_seed=nseed))
            ma.append(_cost(_make_fn(pol_m, _adapt(pol_m, x0K, mK, cfg, seed=s)), x0Q, mQ, cfg, noise_seed=nseed))
        b0, ba, ma = map(np.asarray, (b0, ba, ma))
        out[m_g] = {
            "base_0shot": (b0.mean(), b0.std(), b0),
            "base_adapt": (ba.mean(), ba.std(), ba),
            "maml_adapt": (ma.mean(), ma.std(), ma),
            "winrate":    float(np.mean(ma < ba) * 100.0),
        }
    return out


def print_table(results, mode_label: str, cfg: EvalCfg) -> None:
    print(f"\n=== {mode_label} | {cfg.K}-shot, {len(next(iter(results.values()))['base_0shot'][2])} draws, "
          f"n_inner={cfg.n_inner}, lr_inner={cfg.lr_inner}, t_sim={cfg.t_sim}s ===")
    print(f"{'mass[g]':>7} | {'base_0shot':>16} | {'base_adapt':>16} | "
          f"{'maml_adapt':>16} | win% (MAML<base_adapt)")
    for m_g, r in results.items():
        b0m, b0s, _ = r["base_0shot"]; bam, bas, _ = r["base_adapt"]
        mam, mas, _ = r["maml_adapt"]
        print(f"{m_g:7.1f} | {b0m:7.1f} ± {b0s:5.1f} | {bam:7.1f} ± {bas:5.1f} | "
              f"{mam:7.1f} ± {mas:5.1f} | {r['winrate']:5.0f}")


def plot_degradation(results, out_path: str, mode_label: str, cfg: EvalCfg) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = np.asarray(sorted(results.keys()))
    def band(key, color, label, ls="-"):
        mean = np.array([results[m][key][0] for m in ms])
        std  = np.array([results[m][key][1] for m in ms])
        ax.plot(ms, mean, ls, color=color, marker="o", ms=4, label=label)
        ax.fill_between(ms, mean - std, mean + std, color=color, alpha=0.15)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    band("base_0shot", "C7", "baseline (deploy as-is, no adapt)", ls="--")
    band("base_adapt", "C0", "baseline + K-shot fine-tune")
    band("maml_adapt", "C3", "MAML + K-shot adapt")

    for tm in cfg.train_masses_g:
        ax.axvline(tm, color="gray", ls=":", lw=0.8, alpha=0.6)
    ax.plot([], [], color="gray", ls=":", lw=0.8, label="training masses")
    if cfg.target_mass_g is not None:
        ax.axvline(cfg.target_mass_g, color="green", ls="-.", lw=1.2, alpha=0.7,
                   label=f"target ({cfg.target_mass_g:.0f} g)")

    ax.set_xlabel("Payload mass [g]")
    ax.set_ylabel("Trajectory cost (noise-free eval, lower = better)")
    ax.set_yscale("log")
    ax.set_title(f"Few-shot graceful degradation — {mode_label}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--maml-ckpt", required=True)
    p.add_argument("--baseline-ckpt", required=True)
    p.add_argument("--mode", choices=["centered", "offset"], default="centered")
    p.add_argument("--masses", type=float, nargs="+", default=None,
                   help="payload masses [g] to sweep (default: 2..16 step 2)")
    p.add_argument("--offset-dir", type=float, nargs=3, default=[1.0, -1.0, 0.0],
                   metavar=("X", "Y", "Z"),
                   help="offset-mode attachment direction (will be normalised)")
    p.add_argument("--offset-dist", type=float, default=None,
                   help="offset-mode lever [m] (default: motor arm L/sqrt2)")
    p.add_argument("--n-draws", type=int, default=10,
                   help="independent few-shot support draws to average over")
    p.add_argument("--n-query", type=int, default=40)
    p.add_argument("--adapt-lr", type=float, default=None,
                   help="override the few-shot adaptation lr_inner (default: "
                        "the MAML checkpoint's value). Lower it to check the "
                        "'pretrain+finetune' baseline's lr sensitivity.")
    p.add_argument("--adapt-steps", type=int, default=None,
                   help="override the number of adaptation steps n_inner "
                        "(default: the MAML checkpoint's value).")
    p.add_argument("--obs-noise-scale", type=float, default=0.0,
                   help="observation-noise scale for the final-model test "
                        "(0 = noise-free, clean structural comparison; e.g. "
                        "2.0 = same noisy footing as training / deployment). "
                        "Applied to both adaptation and eval roll-outs, with "
                        "identical noise realisations across the 3 controllers.")
    p.add_argument("--out", type=str, default=None,
                   help="output figure path (default: next to the MAML ckpt)")
    return p.parse_args()


def main():
    args = parse_args()
    mck = torch.load(args.maml_ckpt, map_location="cpu", weights_only=False)
    bck = torch.load(args.baseline_ckpt, map_location="cpu", weights_only=False)
    cfg = EvalCfg(mck)
    if args.adapt_lr is not None:
        cfg.lr_inner = args.adapt_lr
    if args.adapt_steps is not None:
        cfg.n_inner = args.adapt_steps
    if args.obs_noise_scale > 0:
        cfg.obs_noise = C.OBS_NOISE_STD * args.obs_noise_scale
    print(f"[cfg] obs_noise_scale={args.obs_noise_scale} "
          f"({'NOISY test' if args.obs_noise_scale > 0 else 'noise-free test'})")
    pol_m = _policy(mck, cfg.hidden)
    pol_b = _policy(bck, cfg.hidden)

    masses_g = args.masses if args.masses is not None else list(range(2, 17, 2))

    if args.mode == "centered":
        r_offset = np.zeros(3, dtype=np.float32)
        mode_label = "centered payload (pure weight)"
        suffix = "centered"
    else:
        d = np.asarray(args.offset_dir, dtype=np.float32)
        d = d / (np.linalg.norm(d) + 1e-9)
        dist = args.offset_dist if args.offset_dist is not None else float(C.L / np.sqrt(2))
        r_offset = d * dist
        mode_label = (f"offset payload @ {dist*100:.1f} cm "
                      f"dir=({d[0]:+.2f},{d[1]:+.2f},{d[2]:+.2f})  [OOD for centered-trained ckpts]")
        suffix = "offset"

    print(f"[cfg] mode={args.mode}  masses={masses_g} g  r_offset={r_offset*100} cm  "
          f"half_side={cfg.half_side} m  train_masses={cfg.train_masses_g} g  "
          f"target={cfg.target_mass_g} g")

    results = run_sweep(pol_m, pol_b, cfg, masses_g, r_offset,
                        args.n_draws, args.n_query)
    print_table(results, mode_label, cfg)

    if args.out is not None:
        out_path = args.out
    else:
        out_dir = os.path.dirname(os.path.abspath(args.maml_ckpt))
        base = os.path.splitext(os.path.basename(args.maml_ckpt))[0]
        out_path = os.path.join(out_dir, f"{base}_gap_{suffix}.png")
    plot_degradation(results, out_path, mode_label, cfg)
    print("Done.")


if __name__ == "__main__":
    main()
