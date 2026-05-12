"""MAML (Finn et al. 2017, https://arxiv.org/abs/1703.03400) for the
Crazyflie offset-mass setting.

This module is now specialised to the *Gaussian-motor* task setup:

  * a fixed list of ``Task`` objects (typically 3 ``GaussianMotorTask``)
    is shared across all epochs;
  * each epoch samples one ``MassParams`` per task, runs the inner-loop
    adaptation on a fixed ``x0_train`` batch, and computes the meta loss
    on a fixed ``x0_eval`` batch (disjoint from x0_train);
  * sampled mass positions are recorded in ``history`` for plotting and
    are preserved on early Ctrl+C through ``on_epoch_end``.
"""
from typing import Callable, List

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


def meta_train(
    policy: torch.nn.Module,
    tasks: List[Task],
    x0_train: torch.Tensor,
    x0_eval: torch.Tensor,
    *, # forces every next argument to be named explicitly
    dynamics_step: Callable, # nonlinear or linearized
    n_steps: int, # length of a rollout (NN steps at NN_FREQ)
    epochs: int, # total number of meta-updates
    n_inner_steps: int, # nb inner updates
    lr_outer: float,
    lr_inner: float,
    maml_order: int = 2,
    tau_div: float = None,
    obs_noise_std: torch.Tensor = None,
    terminal_weight: float = 50.0,
    pos_weight: float = 10.0,
    z_weight: float = 1.0,
    inner_grad_clip: float = 1.0,
    device: str = "cpu",
    verbose_every: int = 1,
    seed: int = 42,
    on_epoch_end: Callable = None,
    killer=None,
):
    """Meta-train ``policy`` with MAML over a fixed list of tasks.

    Args:
        tasks         : meta_batch is implicitly ``len(tasks)``; one
                        MassParams is sampled per task per epoch.
        x0_train      : (N_train, 12) fixed support batch, reused across
                        all tasks and epochs (inner loss).
        x0_eval       : (N_eval,  12) fixed query batch, disjoint from
                        x0_train (outer loss).
        on_epoch_end  : optional ``callable(ep, policy, history)`` called
                        once per epoch, e.g. for checkpointing/plotting.
        killer        : optional ``GracefulKiller``; if its ``kill_now``
                        flag is set after an epoch the loop exits cleanly.

    Returns:
        policy, history
        history["sampled_positions"] is a list of length ``len(tasks)``.
        Element k is the chronological list of (dx, dy, dz) tuples
        sampled from ``tasks[k]``.
    """
    torch.manual_seed(seed) # global pytorch seed (init weights, dropout, ...)
    np_rng    = np.random.default_rng(seed) # for task.sample(np_rng)
    torch_gen = torch.Generator(device=device).manual_seed(seed) # for obs noise

    opt = torch.optim.Adam(policy.parameters(), lr=lr_outer)
    # opt = outer optimizer, for the meta step.
    # The inner loop does not update policy.parameters(),
    # it updates a dict theta separated.

    create_graph = (maml_order == 2)
    M = len(tasks)
    if M == 0:
        raise ValueError("`tasks` must contain at least one Task.")

    x0_train = x0_train.to(device)
    x0_eval  = x0_eval.to(device)

    history = {
        "epoch": [], "meta_loss": [],
        "inner_pre": [], "outer_loss": [],
        "sampled_positions": [[] for _ in range(M)],
    }
    # inner_pre = loss before the first inner step (oscillates).
    # outer_loss = loss after adaptation (should decrease).
    # sampled_positions[k] = chronological list of (dx, dy, dz) for tasks[k].

    for ep in range(epochs):
        opt.zero_grad()
        meta_loss = 0.0
        inner_pre_vals: List[float] = []
        outer_vals:     List[float] = []

        for t_idx, task in enumerate(tasks):
            mass = task.sample(np_rng).to(device)
            history["sampled_positions"][t_idx].append(tuple(mass.r_offset))

            # --- Inner loop: differentiable adaptation -------------------
            theta = {n: p for n, p in policy.named_parameters()}
            # theta = new dict pointing towards the current policy params.
            # theta will be updated in the inner loop without touching policy.

            for step in range(n_inner_steps):
                X_in = rollout(_make_fn(policy, theta), x0_train, n_steps,
                               mass, dynamics_step,
                               tau_div=tau_div, obs_noise_std=obs_noise_std,
                               gen=torch_gen)
                L_in = trajectory_cost(X_in, terminal_weight,
                                       pos_weight=pos_weight,
                                       z_weight=z_weight).mean()
                if step == 0:
                    inner_pre_vals.append(L_in.item())

                grads = torch.autograd.grad(
                    L_in, tuple(theta.values()),
                    create_graph=create_graph)
                # create_graph=True (MAML 2nd order) keeps the inner grads
                # attached to the graph so theta - lr*grad stays differentiable
                # w.r.t. policy.parameters(). False = FOMAML (1st-order approx).
                if inner_grad_clip > 0:
                    grads = _clip_grads(grads, max_norm=inner_grad_clip)
                theta = {n: p - lr_inner * g
                         for (n, p), g in zip(theta.items(), grads)}

            # --- Outer loss on the fixed disjoint query batch ------------
            X_out = rollout(_make_fn(policy, theta), x0_eval, n_steps,
                            mass, dynamics_step,
                            tau_div=tau_div, obs_noise_std=obs_noise_std,
                            gen=torch_gen)
            L_out = trajectory_cost(X_out, terminal_weight,
                                    pos_weight=pos_weight,
                                    z_weight=z_weight).mean()
            outer_vals.append(L_out.item())
            meta_loss = meta_loss + L_out

        meta_loss = meta_loss / M
        meta_loss.backward()
        # if create_graph=True, the graph goes back from θ' = θ - α∇L_in until
        # the original parameters; policy.parameters().grad include the second
        # derivatives. If False, gradients into policy.parameters() are the
        # FOMAML first-order approximation.

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


def baseline_train(
    policy: torch.nn.Module,
    tasks: List[Task],
    x0_train: torch.Tensor,
    x0_eval: torch.Tensor,
    *,
    dynamics_step: Callable,
    n_steps: int,
    epochs: int,
    lr_outer: float,
    tau_div: float = None,
    obs_noise_std: torch.Tensor = None,
    terminal_weight: float = 50.0,
    pos_weight: float = 10.0,
    z_weight: float = 1.0,
    device: str = "cpu",
    verbose_every: int = 1,
    seed: int = 42,
    on_epoch_end: Callable = None,
    killer=None,
):
    """Multi-task baseline training (no MAML adaptation).

    Each epoch samples one MassParams per task, runs ``policy`` on the
    fixed ``x0_train`` batch, and averages the per-task losses into a
    single objective whose gradient flows directly into
    ``policy.parameters()``. Mirror of ``meta_train`` minus the inner
    loop, so a baseline NN is trained on exactly the same task
    distributions, x0 sets, and number of epochs.

    history["train_loss"] : averaged per-task training loss (on x0_train)
    history["eval_loss"]  : same but evaluated on x0_eval (held-out, no grad)
    history["sampled_positions"][k] : chronological r_offset for tasks[k]
    """
    torch.manual_seed(seed)
    np_rng    = np.random.default_rng(seed)
    torch_gen = torch.Generator(device=device).manual_seed(seed)

    opt = torch.optim.Adam(policy.parameters(), lr=lr_outer)
    M = len(tasks)
    if M == 0:
        raise ValueError("`tasks` must contain at least one Task.")

    x0_train = x0_train.to(device)
    x0_eval  = x0_eval.to(device)

    history = {
        "epoch": [], "train_loss": [], "eval_loss": [],
        "sampled_positions": [[] for _ in range(M)],
    }

    for ep in range(epochs):
        opt.zero_grad()
        train_loss = 0.0
        train_vals: List[float] = []
        eval_vals:  List[float] = []

        for t_idx, task in enumerate(tasks):
            mass = task.sample(np_rng).to(device)
            history["sampled_positions"][t_idx].append(tuple(mass.r_offset))

            X_tr = rollout(policy, x0_train, n_steps, mass, dynamics_step,
                           tau_div=tau_div, obs_noise_std=obs_noise_std,
                           gen=torch_gen)
            L_tr = trajectory_cost(X_tr, terminal_weight,
                                   pos_weight=pos_weight,
                                   z_weight=z_weight).mean()
            train_loss = train_loss + L_tr
            train_vals.append(L_tr.item())

            with torch.no_grad():
                X_ev = rollout(policy, x0_eval, n_steps, mass, dynamics_step,
                               tau_div=tau_div, obs_noise_std=obs_noise_std,
                               gen=torch_gen)
                L_ev = trajectory_cost(X_ev, terminal_weight,
                                       pos_weight=pos_weight,
                                       z_weight=z_weight).mean()
            eval_vals.append(L_ev.item())

        train_loss = train_loss / M
        train_loss.backward()

        has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                      for p in policy.parameters())
        if has_nan:
            print(f"[ep {ep+1}] NaN in grads — skipping optimizer step.")
        else:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()

        history["epoch"].append(ep + 1)
        history["train_loss"].append(float(train_loss.item()))
        history["eval_loss"].append(float(np.mean(eval_vals)))

        if verbose_every and ((ep + 1) % verbose_every == 0):
            print(f"  [ep {ep+1:4d}/{epochs}] "
                  f"train={train_loss.item():.4e}  "
                  f"eval={np.mean(eval_vals):.4e}")

        if on_epoch_end is not None:
            on_epoch_end(ep, policy, history)

        if killer is not None and killer.kill_now:
            print(f"[Arrêt propre] à l'epoch {ep+1}.")
            break

    return policy, history


def maml_adapt(
    policy: torch.nn.Module,
    mass: MassParams,
    x0: torch.Tensor,
    *,
    dynamics_step: Callable,
    n_steps: int,
    n_steps_adapt: int,
    lr_inner: float,
    terminal_weight: float = 50.0,
    pos_weight: float = 10.0,
    z_weight: float = 1.0,
    obs_noise_std: torch.Tensor = None,
    tau_div: float = None,
    inner_grad_clip: float = 1.0,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = True,
):
    """Few-shot adaptation on a single test task (e.g. the held-out motor).

    The defaults for ``obs_noise_std``, ``tau_div``, ``pos_weight`` and
    ``inner_grad_clip`` should match those used in ``meta_train`` so that
    adaptation sees the same distribution it was trained on.

    Args:
        x0 : (N, 12) support batch of initial states; reused across all
             adaptation steps (standard MAML).

    Returns:
        theta_adapt : dict[name -> Tensor]  adapted parameters
        losses      : list[float]            inner-loop losses per step
    """
    theta = {n: p.detach().clone().requires_grad_(True)
             for n, p in policy.named_parameters()}
    gen = torch.Generator(device=device).manual_seed(seed)
    losses = []
    mass = mass.to(device)
    x0 = x0.to(device)
    for s in range(n_steps_adapt):
        X = rollout(_make_fn(policy, theta), x0, n_steps, mass, dynamics_step,
                    tau_div=tau_div, obs_noise_std=obs_noise_std, gen=gen)
        L = trajectory_cost(X, terminal_weight,
                            pos_weight=pos_weight,
                            z_weight=z_weight).mean()
        grads = torch.autograd.grad(L, tuple(theta.values()), create_graph=False)
        if inner_grad_clip > 0:
            grads = _clip_grads(grads, max_norm=inner_grad_clip)
        theta = {n: (p - lr_inner * g).detach().requires_grad_(True)
                 for (n, p), g in zip(theta.items(), grads)}
        losses.append(float(L.item()))
        if verbose:
            print(f"  [adapt {s+1}/{n_steps_adapt}] loss={L.item():.4e}")
    return theta, losses
