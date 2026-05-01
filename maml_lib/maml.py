"""MAML (Finn et al. 2017, https://arxiv.org/abs/1703.03400) for the
Crazyflie offset-mass setting.

Inner loop:
  Sample one task (MassParams), do ``n_inner_steps`` differentiable
  gradient updates on a batch of ``k_inner`` initial states. Each update
  uses the current adapted ``theta`` (parameter dict) via
  ``functional_call``.

Outer loop:
  After the inner steps, evaluate the adapted ``theta`` on a fresh batch
  of ``k_outer`` initial states and accumulate the loss into the meta
  loss. Average over ``meta_batch`` tasks per epoch, then back-prop into
  the original parameters.

``maml_order=2`` (default) propagates gradients through the inner-loop
gradients (full MAML). ``maml_order=1`` is FOMAML: cheaper, no
second-order graph.
"""
from typing import Callable, List

import numpy as np
import torch
try:
    from torch.func import functional_call
except ImportError:                                # PyTorch < 2.0
    from torch.nn.utils.stateless import functional_call

from .cost import trajectory_cost
from .rollout import rollout
from .mass_params import MassParams


def _sample_x0(x0_pool: torch.Tensor, k: int,
               gen: torch.Generator) -> torch.Tensor:
    B0 = x0_pool.shape[0]
    if k >= B0:
        return x0_pool
    idx = torch.randperm(B0, generator=gen, device=x0_pool.device)[:k]
    return x0_pool[idx]


def _make_fn(policy, theta):
    """Stateless forward closure capturing a parameter dict."""
    def _fn(state, target_rel):
        return functional_call(policy, theta, (state, target_rel))
    return _fn


def _clip_grads(grads, max_norm: float = 1.0):
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    coef  = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    return [g * coef for g in grads]


def meta_train(
    policy: torch.nn.Module,
    task,
    x0_pool: torch.Tensor,
    *,
    dynamics_step: Callable,
    n_steps: int,
    epochs: int,
    meta_batch: int,
    k_inner: int,
    n_inner_steps: int,
    k_outer: int,
    lr_outer: float,
    lr_inner: float,
    maml_order: int = 2,
    tau_div: float = None,
    obs_noise_std: torch.Tensor = None,
    terminal_weight: float = 50.0,
    z_weight: float = 1.0,
    device: str = "cpu",
    verbose_every: int = 1,
    seed: int = 42,
    on_epoch_end: Callable = None,
    killer=None,
):
    """Meta-train ``policy`` with MAML.

    Args:
        on_epoch_end: optional ``callable(ep, policy, history)`` called once
                      per epoch, e.g. for plotting / checkpointing.
        killer:       optional ``GracefulKiller``; if its ``kill_now`` flag
                      is set after an epoch the loop exits cleanly.
    """
    torch.manual_seed(seed)
    np_rng    = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=device).manual_seed(seed)

    opt = torch.optim.Adam(policy.parameters(), lr=lr_outer)
    create_graph = (maml_order == 2)

    history = {"epoch": [], "meta_loss": [],
               "inner_pre": [], "outer_loss": []}

    for ep in range(epochs):
        opt.zero_grad()
        meta_loss   = 0.0
        inner_pre_vals: List[float] = []
        outer_vals:     List[float] = []

        for _ in range(meta_batch):
            mass = task.sample(np_rng).to(device)

            # --- Inner loop: differentiable adaptation -------------------
            theta = {n: p for n, p in policy.named_parameters()}

            for step in range(n_inner_steps):
                x0_in = _sample_x0(x0_pool, k_inner, torch_gen)
                X_in = rollout(_make_fn(policy, theta), x0_in, n_steps,
                               mass, dynamics_step,
                               tau_div=tau_div, obs_noise_std=obs_noise_std)
                L_in = trajectory_cost(X_in, terminal_weight,
                                       z_weight=z_weight).mean()
                if step == 0:
                    inner_pre_vals.append(L_in.item())

                grads = torch.autograd.grad(
                    L_in, tuple(theta.values()),
                    create_graph=create_graph)
                grads = _clip_grads(grads, max_norm=1.0)
                theta = {n: p - lr_inner * g
                         for (n, p), g in zip(theta.items(), grads)}

            # --- Outer loss on a fresh batch -----------------------------
            x0_out = _sample_x0(x0_pool, k_outer, torch_gen)
            X_out = rollout(_make_fn(policy, theta), x0_out, n_steps,
                            mass, dynamics_step,
                            tau_div=tau_div, obs_noise_std=obs_noise_std)
            L_out = trajectory_cost(X_out, terminal_weight,
                                    z_weight=z_weight).mean()
            outer_vals.append(L_out.item())
            meta_loss = meta_loss + L_out

        meta_loss = meta_loss / meta_batch
        meta_loss.backward()

        # NaN firewall before stepping
        has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                      for p in policy.parameters())
        if has_nan:
            print(f"[ep {ep+1}] NaN in grads — skipping optimizer step.")
        else:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()

        history["epoch"].append(ep + 1)
        history["meta_loss"].append(float(meta_loss.item()))
        history["inner_pre"].append(float(np.mean(inner_pre_vals)))
        history["outer_loss"].append(float(np.mean(outer_vals)))

        if verbose_every and ((ep + 1) % verbose_every == 0):
            print(f"  [ep {ep+1:4d}/{epochs}] "
                  f"meta={meta_loss.item():.4e}  "
                  f"inner_pre={np.mean(inner_pre_vals):.4e}  "
                  f"outer={np.mean(outer_vals):.4e}")

        if on_epoch_end is not None:
            on_epoch_end(ep, policy, history)

        if killer is not None and killer.kill_now:
            print(f"[Arrêt propre] à l'epoch {ep+1}.")
            break

    return policy, history


def maml_adapt(
    policy: torch.nn.Module,
    mass: MassParams,
    x0_pool: torch.Tensor,
    *,
    dynamics_step: Callable,
    n_steps: int,
    k_samples: int,
    n_steps_adapt: int,
    lr_inner: float,
    terminal_weight: float = 50.0,
    z_weight: float = 1.0,
    device: str = "cpu",
    verbose: bool = True,
):
    """Few-shot adaptation on a single test task.

    Returns:
        theta_adapt : dict[name -> Tensor]  adapted parameters
        losses      : list[float]            inner-loop losses per step
    """
    theta = {n: p.detach().clone().requires_grad_(True)
             for n, p in policy.named_parameters()}
    gen = torch.Generator(device=device).manual_seed(0)
    losses = []
    mass = mass.to(device)
    for s in range(n_steps_adapt):
        x0 = _sample_x0(x0_pool, k_samples, gen)
        X = rollout(_make_fn(policy, theta), x0, n_steps, mass, dynamics_step)
        L = trajectory_cost(X, terminal_weight, z_weight=z_weight).mean()
        grads = torch.autograd.grad(L, tuple(theta.values()), create_graph=False)
        grads = _clip_grads(grads, max_norm=1.0)
        theta = {n: (p - lr_inner * g).detach().requires_grad_(True)
                 for (n, p), g in zip(theta.items(), grads)}
        losses.append(float(L.item()))
        if verbose:
            print(f"  [adapt {s+1}/{n_steps_adapt}] loss={L.item():.4e}")
    return theta, losses
