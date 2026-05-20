"""MAML (Finn et al. 2017) and multi-task baseline training.

Both ``meta_train`` and ``baseline_train`` consume a
:class:`maml_lib.tasks.CompositeTaskSet` whose tasks (and per-task support
and query points) are sampled *once at startup* and reused every epoch.

A held-out :class:`CompositeTaskSet` can be passed as ``target_set`` to
log a generalisation metric every epoch *without ever feeding it to the
optimiser*:

  * MAML : few-shot adaptation on the target's support set, then loss
           on its query set (recorded in ``history["target_loss"]``).
  * Base : raw-policy loss on the target's query set.

GPU optimisations preserved across the refactor
-----------------------------------------------
The inner-loop machinery (vmapped grouped policy forward via
``_make_batched_fn``, batched per-task gradient via ``is_grads_batched``
through ``_batched_inner_grads``) is identical: it works on (N*M, ·)
batches regardless of whether the per-row masses are repeated (legacy)
or per-point (new). ``cat_mass_params`` simply concatenates per-task
batched masses along their batch dim instead of the old per-task tiling.
"""
import time
from typing import Callable, List, Optional

import torch

try:
    from torch.func import functional_call, vmap
except ImportError:
    from torch.nn.utils.stateless import functional_call
    from functorch import vmap

from .cost import trajectory_cost
from .rollout import rollout
from .mass_params import MassParams, cat_mass_params
from .tasks import CompositeTaskSet


# ---------------------------------------------------------------------------
# RNG restoration helpers (resume)
# ---------------------------------------------------------------------------

def _restore_torch_gen(device: str, state, seed: int) -> torch.Generator:
    gen = torch.Generator(device=device)
    try:
        if state is not None:
            gen.set_state(state.cpu().to(torch.uint8))
        else:
            gen.manual_seed(seed)
    except (RuntimeError, TypeError) as e:
        print(f"[Resume] could not restore torch.Generator state "
              f"({e.__class__.__name__}: {e}); reseeding with {seed}.")
        gen.manual_seed(seed)
    return gen


def _restore_global_rng(resume_state: dict, seed: int) -> None:
    cpu_state = resume_state.get("torch_rng_state", None)
    if cpu_state is not None:
        try:
            torch.set_rng_state(cpu_state.cpu().to(torch.uint8))
        except (RuntimeError, TypeError) as e:
            print(f"[Resume] could not restore torch CPU RNG state "
                  f"({e.__class__.__name__}: {e}); reseeding with {seed}.")
            torch.manual_seed(seed)
    else:
        torch.manual_seed(seed)
    if torch.cuda.is_available() and "torch_cuda_rng_state" in resume_state:
        try:
            torch.cuda.set_rng_state(
                resume_state["torch_cuda_rng_state"].cpu().to(torch.uint8))
        except (RuntimeError, TypeError) as e:
            print(f"[Resume] could not restore torch CUDA RNG state "
                  f"({e.__class__.__name__}: {e}); reseeding with {seed}.")
            torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Policy callables (functional + vmapped grouped)
# ---------------------------------------------------------------------------

def _make_fn(policy, theta):
    def _fn(state, target_rel):
        return functional_call(policy, theta, (state, target_rel))
    return _fn


def _make_batched_fn(policy, stacked_theta, n_tasks, batch):
    """Policy callable that applies a *different* theta to each task block.

    ``stacked_theta`` maps every parameter name to a tensor whose leading
    dim of size ``n_tasks`` indexes the per-task adapted weights. Inputs
    are flat ``(n_tasks*batch, .)``; row-block ``i`` (rows
    ``i*batch : (i+1)*batch``) is run with ``stacked_theta[i]``.
    """
    def _one(theta_i, state_i, target_i):
        return functional_call(policy, theta_i, (state_i, target_i))
    _vcall = vmap(_one, in_dims=(0, 0, 0))

    def _fn(state, target_rel):
        s  = state.view(n_tasks, batch, state.shape[-1])
        tr = target_rel.view(n_tasks, batch, target_rel.shape[-1])
        a  = _vcall(stacked_theta, s, tr)               # (n_tasks, batch, 4)
        return a.reshape(n_tasks * batch, a.shape[-1])
    return _fn


# ---------------------------------------------------------------------------
# Gradient utilities
# ---------------------------------------------------------------------------

def _clip_grads(grads, max_norm: float = 1.0):
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    coef  = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    return [g * coef for g in grads]


def _clip_grads_batched(batched_grads, max_norm: float = 1.0):
    """Per-task grad-norm clipping for batched grads ``(n_tasks, *param)``."""
    sq = sum((g.detach() ** 2).flatten(1).sum(dim=1) for g in batched_grads)
    total = torch.sqrt(sq)
    coef  = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    out = []
    for g in batched_grads:
        c = coef.view(coef.shape[0], *([1] * (g.dim() - 1)))
        out.append(g * c)
    return out


def _batched_inner_grads(costs, inputs, n_tasks, create_graph, chunk):
    """Per-task gradients ∂costs[i]/∂inputs for all i, in one vmapped VJP."""
    eye = torch.eye(n_tasks, device=costs.device, dtype=costs.dtype)
    if chunk is None or chunk >= n_tasks:
        return torch.autograd.grad(
            costs, inputs, grad_outputs=eye, is_grads_batched=True,
            retain_graph=create_graph, create_graph=create_graph)
    pieces = []
    for start in range(0, n_tasks, chunk):
        last = start + chunk >= n_tasks
        g = torch.autograd.grad(
            costs, inputs, grad_outputs=eye[start:start + chunk],
            is_grads_batched=True, create_graph=create_graph,
            retain_graph=create_graph or not last)
        pieces.append(g)
    return tuple(torch.cat([p[j] for p in pieces], dim=0)
                 for j in range(len(inputs)))


# ---------------------------------------------------------------------------
# Target evaluation (held-out, never feeds the optimiser)
# ---------------------------------------------------------------------------

def _eval_target_maml(policy, target_set, *,
                      dynamics_step, n_steps, n_inner_steps, lr_inner,
                      terminal_weight, pos_weight, z_weight,
                      obs_noise_std, tau_div, inner_grad_clip,
                      device, seed):
    """Per-target-task: few-shot adapt on support, query loss on eval."""
    losses = []
    for t in range(target_set.N):
        adapted_theta, _ = maml_adapt(
            policy,
            mass=target_set.masses_train[t],
            x0=target_set.x0_train[t],
            dynamics_step=dynamics_step,
            n_steps=n_steps, n_steps_adapt=n_inner_steps,
            lr_inner=lr_inner,
            terminal_weight=terminal_weight,
            pos_weight=pos_weight, z_weight=z_weight,
            obs_noise_std=obs_noise_std, tau_div=tau_div,
            inner_grad_clip=inner_grad_clip,
            device=device, seed=seed * 31 + t,
            verbose=False,
        )
        with torch.no_grad():
            X = rollout(
                _make_fn(policy, adapted_theta),
                target_set.x0_eval[t].to(device), n_steps,
                target_set.masses_eval[t].to(device), dynamics_step,
                tau_div=tau_div, obs_noise_std=None, gen=None)
            L = trajectory_cost(X, terminal_weight,
                                pos_weight=pos_weight, z_weight=z_weight).mean()
        losses.append(float(L.item()))
    return losses


def _eval_target_baseline(policy, target_set, *,
                          dynamics_step, n_steps,
                          terminal_weight, pos_weight, z_weight,
                          tau_div, device):
    """Per-target-task: raw-policy loss on the query set (no adaptation)."""
    losses = []
    for t in range(target_set.N):
        with torch.no_grad():
            X = rollout(
                policy,
                target_set.x0_eval[t].to(device), n_steps,
                target_set.masses_eval[t].to(device), dynamics_step,
                tau_div=tau_div, obs_noise_std=None, gen=None)
            L = trajectory_cost(X, terminal_weight,
                                pos_weight=pos_weight, z_weight=z_weight).mean()
        losses.append(float(L.item()))
    return losses


def _format_targets(losses: List[float]) -> str:
    return ",".join(f"{l:.3e}" for l in losses)


# ---------------------------------------------------------------------------
# MAML meta-training
# ---------------------------------------------------------------------------

def meta_train(
    policy: torch.nn.Module,
    task_set: CompositeTaskSet,
    *,
    target_set: Optional[CompositeTaskSet] = None,
    dynamics_step: Callable,
    n_steps: int,
    epochs: int,
    n_inner_steps: int,
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
    resume_state: dict = None,
    profile: bool = False,
    grad_chunk: int = None,
):
    """Meta-train ``policy`` with MAML over a CompositeTaskSet.

    ``target_set`` (optional) is held strictly out of the optimiser: its
    loss after few-shot adaptation is computed every epoch with
    ``torch.no_grad`` on the outer rollout, and logged to
    ``history["target_loss"]``.
    """
    N = task_set.N
    B_train = task_set.M_train
    B_eval  = task_set.M_eval
    create_graph = (maml_order == 2)

    opt = torch.optim.Adam(policy.parameters(), lr=lr_outer)

    if resume_state is None:
        torch.manual_seed(seed)
        torch_gen = torch.Generator(device=device).manual_seed(seed)
        history = {"epoch": [], "meta_loss": [], "inner_pre": [], "target_loss": []}
        start_epoch = 0
    else:
        _restore_global_rng(resume_state, seed)
        torch_gen = _restore_torch_gen(
            device, resume_state.get("torch_gen_state", None), seed)
        opt.load_state_dict(resume_state["optimizer_state"])
        history = resume_state["history"]
        for k, v in [("epoch", []), ("meta_loss", []), ("inner_pre", []),
                     ("target_loss", [])]:
            history.setdefault(k, v)
        start_epoch = len(history["epoch"])
        print(f"[Resume] from epoch {start_epoch}/{epochs}.")

    # ── Per-task device-resident tensors ────────────────────────────────
    masses_train_dev = [m.to(device) for m in task_set.masses_train]
    masses_eval_dev  = [m.to(device) for m in task_set.masses_eval]
    x0_train_dev = task_set.x0_train.to(device)         # (N, M_train, 12)
    x0_eval_dev  = task_set.x0_eval.to(device)          # (N, M_eval,  12)

    # Pre-built big tensors for the batched inner/outer rollouts.
    x0_inner_big  = x0_train_dev.reshape(N * B_train, 12)
    mass_inner_big = cat_mass_params(masses_train_dev, device)

    x0_outer_big  = x0_eval_dev.reshape(N * B_eval, 12)
    mass_outer_big = cat_mass_params(masses_eval_dev, device)

    _use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    def _now() -> float:
        if _use_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    for ep in range(start_epoch, epochs):
        opt.zero_grad()
        t0 = _now()

        # ── Step 1 inner loop: one big batched forward pass ─────────────
        theta = {n: p for n, p in policy.named_parameters()}

        X_inner_big = rollout(
            _make_fn(policy, theta), x0_inner_big, n_steps,
            mass_inner_big, dynamics_step,
            tau_div=tau_div, obs_noise_std=obs_noise_std, gen=torch_gen,
        )                                                # (N*B_train, n_steps, 12)

        costs_0 = (trajectory_cost(
                       X_inner_big.view(N * B_train, n_steps, 12),
                       terminal_weight, pos_weight=pos_weight, z_weight=z_weight)
                   .view(N, B_train).mean(dim=1))         # (N,)

        inner_pre = costs_0.detach().mean().item()
        t_inner = _now()

        # ── Per-task gradient + adaptation ──────────────────────────────
        theta_items  = list(theta.items())
        theta_values = tuple(p for _, p in theta_items)

        if n_inner_steps == 1:
            batched = _batched_inner_grads(
                costs_0, theta_values, N, create_graph, grad_chunk)
            if inner_grad_clip > 0:
                batched = _clip_grads_batched(batched, inner_grad_clip)
            stacked_theta = {
                name: p.unsqueeze(0) - lr_inner * g
                for (name, p), g in zip(theta_items, batched)
            }
        else:
            adapted_thetas: List[dict] = []
            for i in range(N):
                retain = create_graph or (i < N - 1)
                grads = torch.autograd.grad(
                    costs_0[i], theta_values,
                    create_graph=create_graph, retain_graph=retain,
                )
                if inner_grad_clip > 0:
                    grads = _clip_grads(grads, inner_grad_clip)
                theta_i = {n: p - lr_inner * g
                           for (n, p), g in zip(theta_items, grads)}

                for _step in range(1, n_inner_steps):
                    X_in = rollout(
                        _make_fn(policy, theta_i),
                        x0_train_dev[i], n_steps,
                        masses_train_dev[i], dynamics_step,
                        tau_div=tau_div, obs_noise_std=obs_noise_std,
                        gen=torch_gen,
                    )
                    L_in = trajectory_cost(
                        X_in, terminal_weight,
                        pos_weight=pos_weight, z_weight=z_weight).mean()
                    grads_s = torch.autograd.grad(
                        L_in, tuple(theta_i.values()),
                        create_graph=create_graph)
                    if inner_grad_clip > 0:
                        grads_s = _clip_grads(grads_s, inner_grad_clip)
                    theta_i = {n: p - lr_inner * g
                               for (n, p), g in zip(theta_i.items(), grads_s)}

                adapted_thetas.append(theta_i)
            stacked_theta = {
                name: torch.stack([t[name] for t in adapted_thetas], dim=0)
                for name in adapted_thetas[0]
            }
        t_grad = _now()

        # ── Outer loss (all N tasks batched into one rollout) ───────────
        X_out = rollout(
            _make_batched_fn(policy, stacked_theta, N, B_eval),
            x0_outer_big, n_steps, mass_outer_big, dynamics_step,
            tau_div=tau_div, obs_noise_std=obs_noise_std, gen=torch_gen,
        )                                                # (N*B_eval, n_steps, 12)
        meta_loss = trajectory_cost(
            X_out, terminal_weight,
            pos_weight=pos_weight, z_weight=z_weight).mean()
        t_outer = _now()
        meta_loss.backward()
        t_back = _now()

        has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                      for p in policy.parameters())
        if has_nan:
            print(f"[ep {ep+1}] NaN in grads — skipping optimizer step.")
        else:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()

        # ── Target eval (held out from the optimiser) ───────────────────
        target_losses = None
        if target_set is not None:
            target_losses = _eval_target_maml(
                policy, target_set,
                dynamics_step=dynamics_step,
                n_steps=n_steps, n_inner_steps=n_inner_steps,
                lr_inner=lr_inner,
                terminal_weight=terminal_weight,
                pos_weight=pos_weight, z_weight=z_weight,
                obs_noise_std=obs_noise_std, tau_div=tau_div,
                inner_grad_clip=inner_grad_clip,
                device=device, seed=seed)
        t_target = _now()

        history["epoch"].append(ep + 1)
        history["meta_loss"].append(float(meta_loss.item()))
        history["inner_pre"].append(float(inner_pre))
        if target_losses is not None:
            history["target_loss"].append(target_losses)

        if verbose_every and ((ep + 1) % verbose_every == 0):
            line = (f"  [ep {ep+1:4d}/{epochs}] "
                    f"meta={meta_loss.item():.4e}  "
                    f"inner_pre={inner_pre:.4e}")
            if target_losses is not None:
                line += f"  target=[{_format_targets(target_losses)}]"
            print(line)

        if on_epoch_end is not None:
            on_epoch_end(ep, policy, history,
                         optimizer=opt, torch_gen=torch_gen)
        t_ckpt = _now()

        if profile:
            d_inner  = t_inner  - t0
            d_grad   = t_grad   - t_inner
            d_outer  = t_outer  - t_grad
            d_back   = t_back   - t_outer
            d_target = t_target - t_back
            d_ckpt   = t_ckpt   - t_target
            d_total  = t_ckpt   - t0
            print(f"  [profile ep {ep+1}] total={d_total:.2f}s | "
                  f"inner_rollout={d_inner:.2f}s ({100*d_inner/d_total:.0f}%)  "
                  f"grad_loop={d_grad:.2f}s ({100*d_grad/d_total:.0f}%)  "
                  f"outer_loop={d_outer:.2f}s ({100*d_outer/d_total:.0f}%)  "
                  f"meta_backward={d_back:.2f}s ({100*d_back/d_total:.0f}%)  "
                  f"target_eval={d_target:.2f}s ({100*d_target/d_total:.0f}%)  "
                  f"ckpt_save={d_ckpt:.2f}s ({100*d_ckpt/d_total:.0f}%)")

        if killer is not None and killer.kill_now:
            print(f"[Arrêt propre] à l'epoch {ep+1}.")
            break

    return policy, history


# ---------------------------------------------------------------------------
# Multi-task baseline training (no adaptation)
# ---------------------------------------------------------------------------

def baseline_train(
    policy: torch.nn.Module,
    task_set: CompositeTaskSet,
    *,
    target_set: Optional[CompositeTaskSet] = None,
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
    resume_state: dict = None,
):
    """Multi-task baseline: train on all N tasks simultaneously.

    Logs ``train_loss``, ``eval_loss`` (on the CompositeTaskSet's query
    points) and, if ``target_set`` is given, ``target_loss`` — all
    measured on the same query points the MAML run uses, so the two
    histories are directly comparable.
    """
    N = task_set.N
    B_train = task_set.M_train
    B_eval  = task_set.M_eval

    opt = torch.optim.Adam(policy.parameters(), lr=lr_outer)

    if resume_state is None:
        torch.manual_seed(seed)
        torch_gen = torch.Generator(device=device).manual_seed(seed)
        history = {"epoch": [], "train_loss": [], "eval_loss": [], "target_loss": []}
        start_epoch = 0
    else:
        _restore_global_rng(resume_state, seed)
        torch_gen = _restore_torch_gen(
            device, resume_state.get("torch_gen_state", None), seed)
        opt.load_state_dict(resume_state["optimizer_state"])
        history = resume_state["history"]
        for k, v in [("epoch", []), ("train_loss", []),
                     ("eval_loss", []), ("target_loss", [])]:
            history.setdefault(k, v)
        start_epoch = len(history["epoch"])
        print(f"[Resume] from epoch {start_epoch}/{epochs}.")

    # Pre-built big tensors.
    masses_train_dev = [m.to(device) for m in task_set.masses_train]
    masses_eval_dev  = [m.to(device) for m in task_set.masses_eval]
    mass_train_big = cat_mass_params(masses_train_dev, device)
    mass_eval_big  = cat_mass_params(masses_eval_dev,  device)
    x0_train_big = task_set.x0_train.reshape(N * B_train, 12).to(device)
    x0_eval_big  = task_set.x0_eval.reshape(N * B_eval, 12).to(device)

    for ep in range(start_epoch, epochs):
        opt.zero_grad()

        X_tr = rollout(policy, x0_train_big, n_steps,
                       mass_train_big, dynamics_step,
                       tau_div=tau_div, obs_noise_std=obs_noise_std,
                       gen=torch_gen)
        L_tr = trajectory_cost(
            X_tr, terminal_weight,
            pos_weight=pos_weight, z_weight=z_weight).mean()
        L_tr.backward()

        has_nan = any(p.grad is not None and torch.isnan(p.grad).any()
                      for p in policy.parameters())
        if has_nan:
            print(f"[ep {ep+1}] NaN in grads — skipping optimizer step.")
        else:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
            opt.step()

        with torch.no_grad():
            X_ev = rollout(policy, x0_eval_big, n_steps,
                           mass_eval_big, dynamics_step,
                           tau_div=tau_div, obs_noise_std=obs_noise_std,
                           gen=torch_gen)
            L_ev = trajectory_cost(
                X_ev, terminal_weight,
                pos_weight=pos_weight, z_weight=z_weight).mean()

        target_losses = None
        if target_set is not None:
            target_losses = _eval_target_baseline(
                policy, target_set,
                dynamics_step=dynamics_step, n_steps=n_steps,
                terminal_weight=terminal_weight,
                pos_weight=pos_weight, z_weight=z_weight,
                tau_div=tau_div, device=device)

        history["epoch"].append(ep + 1)
        history["train_loss"].append(float(L_tr.item()))
        history["eval_loss"].append(float(L_ev.item()))
        if target_losses is not None:
            history["target_loss"].append(target_losses)

        if verbose_every and ((ep + 1) % verbose_every == 0):
            line = (f"  [ep {ep+1:4d}/{epochs}] "
                    f"train={L_tr.item():.4e}  eval={L_ev.item():.4e}")
            if target_losses is not None:
                line += f"  target=[{_format_targets(target_losses)}]"
            print(line)

        if on_epoch_end is not None:
            on_epoch_end(ep, policy, history,
                         optimizer=opt, torch_gen=torch_gen)

        if killer is not None and killer.kill_now:
            print(f"[Arrêt propre] à l'epoch {ep+1}.")
            break

    return policy, history


# ---------------------------------------------------------------------------
# Few-shot adaptation (test time, also reused by the target eval helper)
# ---------------------------------------------------------------------------

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
    """Few-shot adaptation on a single task (batched mass + x0).

    ``mass`` may be a single-task or batched ``MassParams``; ``x0`` should
    have the matching leading batch dim. The adapted theta is a detached
    leaf — it does **not** touch the policy's meta-parameters.
    """
    theta = {n: p.detach().clone().requires_grad_(True)
             for n, p in policy.named_parameters()}
    gen = torch.Generator(device=device).manual_seed(seed)
    losses = []
    mass = mass.to(device)
    x0   = x0.to(device)
    for s in range(n_steps_adapt):
        X = rollout(_make_fn(policy, theta), x0, n_steps, mass, dynamics_step,
                    tau_div=tau_div, obs_noise_std=obs_noise_std, gen=gen)
        L = trajectory_cost(X, terminal_weight,
                            pos_weight=pos_weight, z_weight=z_weight).mean()
        grads = torch.autograd.grad(L, tuple(theta.values()), create_graph=False)
        if inner_grad_clip > 0:
            grads = _clip_grads(grads, max_norm=inner_grad_clip)
        theta = {n: (p - lr_inner * g).detach().requires_grad_(True)
                 for (n, p), g in zip(theta.items(), grads)}
        losses.append(float(L.item()))
        if verbose:
            print(f"  [adapt {s+1}/{n_steps_adapt}] loss={L.item():.4e}")
    return theta, losses
