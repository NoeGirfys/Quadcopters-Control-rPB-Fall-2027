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

Task pool:
  ``meta_train`` accepts either a single ``Task`` (fresh sampling at
  every meta-iteration) or a list of ``Task`` (fixed pool — at each
  epoch we draw ``meta_batch`` of them). The pool form is preferred for
  smoother training curves and structured task families (e.g. one task
  per motor).
"""
from typing import Callable, List, Optional, Sequence, Union

import numpy as np
import torch
try:
    from torch.func import functional_call
    """
    functional_call(module, params_dict, args) runs module.forward(*args)
    but it replaces its parameters by those provided in params_dict.
    It is necessary for MAML: it enables to make a foward pass with another
    set of weights, without mutating the module.
    """
except ImportError:                                # PyTorch < 2.0
    from torch.nn.utils.stateless import functional_call

from .cost import trajectory_cost
from .rollout import rollout
from .mass_params import MassParams
from .tasks import Task


def _sample_x0(x0_pool: torch.Tensor, k: int,
               gen: torch.Generator) -> torch.Tensor:
    """
    Draw k initial points without puting them back in the pool ( => can not select
    twice the same initial point). If one asks more than available,
    returns all the pool. Enables to make the batchs inner and outer.
    """
    B0 = x0_pool.shape[0]
    if k >= B0:
        return x0_pool
    idx = torch.randperm(B0, generator=gen, device=x0_pool.device)[:k]
    return x0_pool[idx]


def _make_fn(policy, theta):
    """
    Stateless forward closure capturing a parameter dict.
    We lock (policy, theta) in a function that has the same signature
    than policy.forward(state, target_rel). The rollout takes this function
    without knowing that it uses modified weights. It enables to make the
    rollout agnostic to the MAML mecanism.
    """
    def _fn(state, target_rel):
        return functional_call(policy, theta, (state, target_rel))
    return _fn


def _clip_grads(grads, max_norm: float = 1.0):
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    coef  = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    return [g * coef for g in grads]


def _pick_task_indices(n_tasks: int, meta_batch: int,
                       gen: torch.Generator) -> List[int]:
    """Indices of the tasks to use in this meta-update.

    * meta_batch <= n_tasks  ->  random subset, without replacement
    * meta_batch  > n_tasks  ->  with replacement (rarely needed)
    """
    if meta_batch <= n_tasks:
        return torch.randperm(n_tasks, generator=gen)[:meta_batch].tolist()
    return torch.randint(0, n_tasks, (meta_batch,), generator=gen).tolist()


def _as_task_list(tasks: Union[Task, Sequence[Task]]) -> List[Task]:
    """Normalise a single Task or a list/tuple of Tasks into a list."""
    if isinstance(tasks, Task):
        return [tasks]
    return list(tasks)


def _evaluate_on_masses(
    policy: torch.nn.Module,
    eval_masses: Sequence[MassParams],
    x0_pool: torch.Tensor,
    *,
    dynamics_step: Callable,
    n_steps: int,
    n_inner_steps: int,
    k_inner: int,
    lr_inner: float,
    terminal_weight: float,
    z_weight: float,
    gen: torch.Generator,
    device: str,
) -> dict:
    """No-grad evaluation: for each held-out task, measure the loss before
    and after `n_inner_steps` of inner adaptation. Returns aggregate stats.

    Used to monitor true generalisation during training.
    """
    pre_losses, post_losses = [], []
    for mass in eval_masses:
        mass = mass.to(device)

        # ---- Pre-adapt: original policy --------------------------------
        with torch.no_grad():
            X = rollout(lambda s, t: policy(s, t), x0_pool, n_steps,
                        mass, dynamics_step)
            L_pre = trajectory_cost(X, terminal_weight,
                                    z_weight=z_weight).mean().item()
        pre_losses.append(L_pre)

        # ---- Inner adaptation (no graph beyond each step) --------------
        theta = {n: p.detach().clone().requires_grad_(True)
                 for n, p in policy.named_parameters()}
        for _ in range(n_inner_steps):
            x0 = _sample_x0(x0_pool, k_inner, gen)
            X = rollout(_make_fn(policy, theta), x0, n_steps,
                        mass, dynamics_step)
            L = trajectory_cost(X, terminal_weight,
                                z_weight=z_weight).mean()
            grads = torch.autograd.grad(L, tuple(theta.values()),
                                        create_graph=False)
            grads = _clip_grads(grads, max_norm=1.0)
            theta = {n: (p - lr_inner * g).detach().requires_grad_(True)
                     for (n, p), g in zip(theta.items(), grads)}

        # ---- Post-adapt loss on the FULL pool --------------------------
        with torch.no_grad():
            X = rollout(_make_fn(policy, theta), x0_pool, n_steps,
                        mass, dynamics_step)
            L_post = trajectory_cost(X, terminal_weight,
                                     z_weight=z_weight).mean().item()
        post_losses.append(L_post)

    return {
        "pre_mean":  float(np.mean(pre_losses)),
        "pre_max":   float(np.max(pre_losses)),
        "post_mean": float(np.mean(post_losses)),
        "post_max":  float(np.max(post_losses)),
        "pre_per_task":  pre_losses,
        "post_per_task": post_losses,
    }


def meta_train(
    policy: torch.nn.Module,
    tasks: Union[Task, Sequence[Task]],
    x0_pool: torch.Tensor,
    *, # forces every next arguments to be named explicitely during the calling
    dynamics_step: Callable, # nonlinear or linearized
    n_steps: int, # length of a rollout (NN steps at 100 Hz)
    epochs: int, # total number of meta-updates
    meta_batch: int, # nb tasks per meta-update
    k_inner: int,
    n_inner_steps: int, # nb inner updates
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
    eval_masses: Optional[Sequence[MassParams]] = None,
    eval_every: int = 0,
):
    """Meta-train ``policy`` with MAML.

    Args:
        tasks:        a single ``Task`` (fresh sampling) or a list/tuple of
                      ``Task`` (fixed pool — recommended for stability).
        eval_masses:  optional list of *fixed* held-out ``MassParams`` used
                      to track true generalisation. Evaluation is done with
                      no-grad rollouts, per-task inner adaptation, then a
                      post-adapt loss measurement on the full ``x0_pool``.
        eval_every:   eval cadence in epochs; ``0`` (default) disables eval.
        on_epoch_end: optional ``callable(ep, policy, history)`` called once
                      per epoch, e.g. for plotting / checkpointing.
        killer:       optional ``GracefulKiller``; if its ``kill_now`` flag
                      is set after an epoch the loop exits cleanly.
    """
    torch.manual_seed(seed) # global pytorch seed (init weights, dropout, ...)
    np_rng    = np.random.default_rng(seed) # for task.sample(np_rng) that draws the added mass parameters
    torch_gen = torch.Generator(device=device).manual_seed(seed) # for _sample_x0 that draws the indices in the pool
    eval_gen  = torch.Generator(device=device).manual_seed(seed + 1)

    task_list = _as_task_list(tasks)
    n_tasks = len(task_list)

    opt = torch.optim.Adam(policy.parameters(), lr=lr_outer)
    # opt = outer optimizer, for the meta step.
    # The inner loop does not update policy.parameters(),
    # it updates a dict theta separated.

    create_graph = (maml_order == 2)

    history = {"epoch": [], "meta_loss": [],
               "inner_pre": [], "outer_loss": [],
               # per-task averaged outer loss across the run; one list per task
               "outer_loss_per_task": [[] for _ in range(n_tasks)],
               # held-out eval; only filled if eval_masses is provided
               "eval_epoch": [], "eval_pre_mean": [], "eval_post_mean": [],
               "eval_pre_max":  [], "eval_post_max":  []}
    # inner_pre = loss before first inner step = what the NN meta
    # does at first on the new task, without update.
    # outer_loss = loss after adaptation.

    for ep in range(epochs):
        opt.zero_grad()
        meta_loss   = 0.0
        inner_pre_vals: List[float] = []
        outer_vals:     List[float] = []
        # accumulator: outer loss per task index, this epoch
        outer_per_task_this_epoch: List[List[float]] = [[] for _ in range(n_tasks)]

        # Decide which task indices to use for this meta-update.
        task_indices = _pick_task_indices(n_tasks, meta_batch, torch_gen)

        for ti in task_indices: # loop on the tasks of this meta-batch
            mass = task_list[ti].sample(np_rng).to(device) # outputs a MassParams object

            # --- Inner loop: differentiable adaptation -------------------
            theta = {n: p for n, p in policy.named_parameters()}
            # theta = new dict pointing towards the current policy params
            # at this point, theta["net.0.weight"] = policy.net[0].weight
            # (same tensor in memory). theta will be updated in the inner
            # loop without touching policy

            for step in range(n_inner_steps):
                x0_in = _sample_x0(x0_pool, k_inner, torch_gen) # draw k_inner initial points in x0_pool
                X_in = rollout(_make_fn(policy, theta), x0_in, n_steps,
                               mass, dynamics_step,
                               tau_div=tau_div, obs_noise_std=obs_noise_std)
                # X_in is (B, n_steps, 12)
                L_in = trajectory_cost(X_in, terminal_weight,
                                       z_weight=z_weight).mean()
                if step == 0:
                    inner_pre_vals.append(L_in.item())

                grads = torch.autograd.grad(
                    L_in, tuple(theta.values()),
                    create_graph=create_graph)
                # compute the ∂L_in/∂theta for each param in theta. It is here that
                # create_graph acts : if True (MAML 2nd order), the grads are attached
                # to the graph, theta - lr*grad stays differentiable with respect to policy.parameters().
                # If False (FOMAML), the grads are detached, theta - lr*grad is just some numerical values.
                # Important: we do not use L_in.backward() because it would use the parameters of policy for .grad,
                # here autograd.grad returns the gradients without storing them.
                grads = _clip_grads(grads, max_norm=1.0)
                theta = {n: p - lr_inner * g
                         for (n, p), g in zip(theta.items(), grads)}
                # no inner gradient: we create a new dict theta where each param has done
                # a step in the direction -grad. Now theta is different of policy.

            # --- Outer loss on a fresh batch -----------------------------
            x0_out = _sample_x0(x0_pool, k_outer, torch_gen)
            X_out = rollout(_make_fn(policy, theta), x0_out, n_steps,
                            mass, dynamics_step,
                            tau_div=tau_div, obs_noise_std=obs_noise_std)
            L_out = trajectory_cost(X_out, terminal_weight,
                                    z_weight=z_weight).mean()
            outer_vals.append(L_out.item())
            outer_per_task_this_epoch[ti].append(L_out.item())
            meta_loss = meta_loss + L_out

        meta_loss = meta_loss / len(task_indices)
        meta_loss.backward()
        # if create_graph=True, the graph goes back from θ' = θ - α∇L_in until
        # the original parameters. The gradients in policy.parameters().grad include
        # the second derivatives.
        # if create_graph=False, the graph goes back only from L_out to theta (the
        # adapted values). Because theta is not linked anymore to policy.parameters()
        # for the graph, the gradients propagated to policy.parameters() are now
        # just an approximation at the first order (FOMAML).

        # NaN firewall before stepping
        has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                      for p in policy.parameters())
        if has_nan:
            print(f"[ep {ep+1}] NaN in grads — skipping optimizer step.")
        else:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step() # meta-update

        history["epoch"].append(ep + 1)
        history["meta_loss"].append(float(meta_loss.item()))
        history["inner_pre"].append(float(np.mean(inner_pre_vals)))
        history["outer_loss"].append(float(np.mean(outer_vals)))
        for ti in range(n_tasks):
            vals = outer_per_task_this_epoch[ti]
            history["outer_loss_per_task"][ti].append(
                float(np.mean(vals)) if vals else float("nan"))

        if verbose_every and ((ep + 1) % verbose_every == 0):
            print(f"  [ep {ep+1:4d}/{epochs}] "
                  f"meta={meta_loss.item():.4e}  "
                  f"inner_pre={np.mean(inner_pre_vals):.4e}  " # can oscillate
                  f"outer={np.mean(outer_vals):.4e}")          # should decrease
                                                               # the diff between inner_pre and outer should decrease

        # ---- Held-out evaluation (no-grad on a fixed list of MassParams)
        if eval_masses is not None and eval_every > 0 \
                and (ep + 1) % eval_every == 0:
            stats = _evaluate_on_masses(
                policy, eval_masses, x0_pool,
                dynamics_step=dynamics_step,
                n_steps=n_steps,
                n_inner_steps=n_inner_steps,
                k_inner=k_inner,
                lr_inner=lr_inner,
                terminal_weight=terminal_weight,
                z_weight=z_weight,
                gen=eval_gen,
                device=device,
            )
            history["eval_epoch"].append(ep + 1)
            history["eval_pre_mean"].append(stats["pre_mean"])
            history["eval_post_mean"].append(stats["post_mean"])
            history["eval_pre_max"].append(stats["pre_max"])
            history["eval_post_max"].append(stats["post_max"])
            print(f"    [eval @ ep {ep+1}]  "
                  f"pre_mean={stats['pre_mean']:.4e}  "
                  f"post_mean={stats['post_mean']:.4e}  "
                  f"(max post={stats['post_max']:.4e})")

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
