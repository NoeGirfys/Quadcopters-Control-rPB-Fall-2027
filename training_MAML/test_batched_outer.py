"""Equivalence test for the batched MAML meta-loop (optimisations A + B).

Compares, on a small CPU setup:

  * OLD path - N sequential ``autograd.grad`` inner gradients followed by
    N sequential outer rollouts.
  * NEW path - one vmapped VJP for all inner gradients (B) followed by a
    single batched outer rollout (A).

Both the meta-loss and its gradient w.r.t. the meta-parameters must match.
Also checks that chunked and full batched inner grads agree.

Run (a few seconds):

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
from maml_lib.maml import (
    _make_fn, _make_batched_fn,
    _clip_grads, _clip_grads_batched, _batched_inner_grads,
)

TERMINAL_W, POS_W, Z_W = 50.0, 10.0, 1.0
LR_INNER, CLIP = 0.05, 1.0
N, B_TRAIN, B_EVAL, N_STEPS = 4, 8, 8, 30
TOL = 1e-4


def _cost(X):
    return trajectory_cost(X, TERMINAL_W, pos_weight=POS_W, z_weight=Z_W)


def _rel_err(a, b):
    return ((a - b).abs().max() / (b.abs().max() + 1e-12)).item()


def _build(tau_div):
    """Fixed small problem; returns everything both paths need."""
    torch.manual_seed(0)
    dyn = DYNAMICS["linearized"]
    policy = PolicyMLP(hidden=64)

    masses = [compute_mass_params(0.010, (0.01 * i, -0.01 * i, 0.005))
              for i in range(N)]
    gen = torch.Generator(device="cpu").manual_seed(1)
    x0_tr = sample_hover_x0(B_TRAIN, 0.3, gen, device="cpu")
    x0_ev = sample_hover_x0(B_EVAL, 0.3, gen, device="cpu")

    ctx = dict(
        policy=policy, dyn=dyn, tau_div=tau_div,
        masses_dev=masses,
        mass_in_big=batch_mass_params(masses, B_TRAIN, "cpu"),
        mass_out_big=batch_mass_params(masses, B_EVAL, "cpu"),
        x0_ev=x0_ev,
        x0_in_big=x0_tr.repeat(N, 1),
        x0_out_big=x0_ev.repeat(N, 1),
    )
    return ctx


def _inner_costs(ctx, theta):
    """Fresh inner rollout -> per-task costs (N,) with a fresh graph."""
    X = rollout(_make_fn(ctx["policy"], theta), ctx["x0_in_big"], N_STEPS,
                ctx["mass_in_big"], ctx["dyn"], tau_div=ctx["tau_div"],
                obs_noise_std=None, gen=None)
    return _cost(X).view(N, B_TRAIN).mean(dim=1)


def _meta_old(ctx, base):
    costs = _inner_costs(ctx, base)
    adapted = []
    for i in range(N):
        g = torch.autograd.grad(costs[i], tuple(base.values()),
                                retain_graph=True, create_graph=False)
        g = _clip_grads(g, CLIP)
        adapted.append({n: p - LR_INNER * gg
                        for (n, p), gg in zip(base.items(), g)})
    meta = torch.zeros((), dtype=torch.float32)
    for i in range(N):
        X = rollout(_make_fn(ctx["policy"], adapted[i]), ctx["x0_ev"], N_STEPS,
                    ctx["masses_dev"][i], ctx["dyn"], tau_div=ctx["tau_div"],
                    obs_noise_std=None, gen=None)
        meta = meta + _cost(X).mean()
    return meta / N


def _meta_new(ctx, base):
    costs = _inner_costs(ctx, base)
    batched = _batched_inner_grads(costs, tuple(base.values()), N,
                                   create_graph=False, chunk=None)
    batched = _clip_grads_batched(batched, CLIP)
    stacked = {name: p.unsqueeze(0) - LR_INNER * g
               for (name, p), g in zip(base.items(), batched)}
    X = rollout(_make_batched_fn(ctx["policy"], stacked, N, B_EVAL),
                ctx["x0_out_big"], N_STEPS, ctx["mass_out_big"], ctx["dyn"],
                tau_div=ctx["tau_div"], obs_noise_std=None, gen=None)
    return _cost(X).mean()


def run_case(tau_div):
    ctx = _build(tau_div)
    policy = ctx["policy"]
    base = {n: p.detach().clone().requires_grad_(True)
            for n, p in policy.named_parameters()}

    meta_old = _meta_old(ctx, base)
    meta_old.backward()
    g_old = {n: base[n].grad.detach().clone() for n in base}
    for n in base:
        base[n].grad = None

    meta_new = _meta_new(ctx, base)
    meta_new.backward()
    g_new = {n: base[n].grad.detach().clone() for n in base}

    loss_err = _rel_err(meta_new, meta_old)
    grad_err = max(_rel_err(g_new[n], g_old[n]) for n in base)

    # Chunked vs full batched inner grads (fresh graphs each).
    full = _batched_inner_grads(_inner_costs(ctx, base), tuple(base.values()),
                                N, create_graph=False, chunk=None)
    chk  = _batched_inner_grads(_inner_costs(ctx, base), tuple(base.values()),
                                N, create_graph=False, chunk=3)
    chunk_err = max(_rel_err(c, f) for c, f in zip(chk, full))

    tag = f"tau_div={tau_div}"
    print(f"[{tag}]  meta_old={meta_old.item():.6e}  "
          f"meta_new={meta_new.item():.6e}  rel_err={loss_err:.2e}")
    print(f"[{tag}]  max relative gradient error = {grad_err:.2e}")
    print(f"[{tag}]  chunked vs full inner-grad error = {chunk_err:.2e}")

    ok = loss_err < TOL and grad_err < TOL and chunk_err < TOL
    print(f"[{tag}]  {'PASS' if ok else 'FAIL'} (tol={TOL:.0e})\n")
    return ok


if __name__ == "__main__":
    all_ok = True
    for tau in (None, 1.0):
        all_ok &= run_case(tau)
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)
