"""Equivalence test for the batched MAML outer loop (optimisation A).

Checks that the new single batched outer rollout
(``maml._make_batched_fn``) produces the same meta-loss and the same
gradients w.r.t. the meta-parameters as the old per-task sequential
loop it replaces.

Run (CPU, a few seconds):

    cd training_MAML
    python test_batched_outer.py
"""
import os
import sys

import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from maml_lib import PolicyMLP, DYNAMICS, sample_hover_x0
from maml_lib.rollout import rollout
from maml_lib.cost import trajectory_cost
from maml_lib.mass_params import compute_mass_params, batch_mass_params
from maml_lib.maml import _make_fn, _make_batched_fn

TERMINAL_W, POS_W, Z_W = 50.0, 10.0, 1.0


def _meta_old(policy, adapted, masses_dev, x0_eval, n_steps, dyn, tau_div):
    """Old path: N sequential rollouts, summed then averaged."""
    N = len(adapted)
    meta = torch.zeros((), dtype=x0_eval.dtype)
    for i in range(N):
        X = rollout(_make_fn(policy, adapted[i]), x0_eval, n_steps,
                    masses_dev[i], dyn, tau_div=tau_div,
                    obs_noise_std=None, gen=None)
        meta = meta + trajectory_cost(X, TERMINAL_W,
                                      pos_weight=POS_W, z_weight=Z_W).mean()
    return meta / N


def _meta_new(policy, adapted, mass_big, x0_big, n_steps, dyn, tau_div,
              N, B_eval):
    """New path: one batched rollout over all N tasks."""
    stacked = {name: torch.stack([adapted[i][name] for i in range(N)], dim=0)
               for name in adapted[0]}
    X = rollout(_make_batched_fn(policy, stacked, N, B_eval),
                x0_big, n_steps, mass_big, dyn, tau_div=tau_div,
                obs_noise_std=None, gen=None)
    return trajectory_cost(X, TERMINAL_W,
                           pos_weight=POS_W, z_weight=Z_W).mean()


def run_case(tau_div):
    torch.manual_seed(0)
    device = "cpu"
    N, B_eval, n_steps = 4, 8, 30
    dyn = DYNAMICS["linearized"]

    policy = PolicyMLP(hidden=64)

    masses = [compute_mass_params(0.010, (0.01 * i, -0.01 * i, 0.005))
              for i in range(N)]
    mass_big = batch_mass_params(masses, B_eval, device)
    masses_dev = [m.to(device) for m in masses]

    gen = torch.Generator(device=device).manual_seed(1)
    x0_eval = sample_hover_x0(B_eval, 0.3, gen, device=device)
    x0_big = x0_eval.repeat(N, 1)

    # Meta-parameters as leaf tensors; per-task constant perturbations stand
    # in for the inner-loop adaptation (theta_i = base - lr * pert_i).
    base = {n: p.detach().clone().requires_grad_(True)
            for n, p in policy.named_parameters()}
    perts = [{n: 0.01 * torch.randn_like(p) for n, p in base.items()}
             for _ in range(N)]

    def make_adapted():
        return [{n: base[n] - 0.05 * perts[i][n] for n in base}
                for i in range(N)]

    # ── Old path ──
    meta_old = _meta_old(policy, make_adapted(), masses_dev,
                         x0_eval, n_steps, dyn, tau_div)
    meta_old.backward()
    g_old = {n: base[n].grad.detach().clone() for n in base}
    for n in base:
        base[n].grad = None

    # ── New path ──
    meta_new = _meta_new(policy, make_adapted(), mass_big, x0_big,
                         n_steps, dyn, tau_div, N, B_eval)
    meta_new.backward()
    g_new = {n: base[n].grad.detach().clone() for n in base}

    loss_err = (meta_new - meta_old).abs().item() / (meta_old.abs().item() + 1e-12)
    grad_err = max(
        ((g_new[n] - g_old[n]).abs().max()
         / (g_old[n].abs().max() + 1e-12)).item()
        for n in base)

    tag = f"tau_div={tau_div}"
    print(f"[{tag}]  meta_old={meta_old.item():.6e}  "
          f"meta_new={meta_new.item():.6e}  rel_err={loss_err:.2e}")
    print(f"[{tag}]  max relative gradient error = {grad_err:.2e}")

    tol = 1e-4
    ok = loss_err < tol and grad_err < tol
    print(f"[{tag}]  {'PASS' if ok else 'FAIL'} (tol={tol:.0e})\n")
    return ok


if __name__ == "__main__":
    all_ok = True
    for tau in (None, 1.0):
        all_ok &= run_case(tau)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
