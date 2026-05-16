"""MAML (Finn et al. 2017) and multi-task baseline training.

Task set
--------
Both ``meta_train`` and ``baseline_train`` accept a ``FixedUniformMassSet``
whose N tasks are sampled *once at startup* and reused every epoch.  This
makes the gradient estimate stable and lets the two functions train on
exactly the same data for a fair comparison.

GPU parallelism
---------------
Inner loop (first adaptation step)
    All N tasks share the same meta-parameters θ, so their inner rollouts
    are stacked into one big batch (N*B_train, 12) — a single GPU call.
    Per-task gradients are then computed with N sequential ``autograd.grad``
    calls (retain_graph=True), each touching only its own slice of the graph.
    For n_inner_steps > 1, subsequent steps are sequential (different θ per
    task after step 1).

Outer loop / baseline
    Sequential over tasks (different adapted θ per task), but the outer
    meta_loss is accumulated *before* calling backward(), so PyTorch fuses
    the backward passes internally.

Baseline
    All N tasks share the same policy, so train and eval rollouts are fully
    batched (N*B_train, 12) — one GPU call covers the whole epoch.
"""
import time
from typing import Callable, List

import torch


def _restore_torch_gen(device: str, state, seed: int) -> torch.Generator:
    """Rebuild a torch.Generator on ``device`` from a saved state.

    Saved states are CPU ByteTensors. ``Generator.set_state`` is strict about
    device/size, so if the saved state was produced for a different device
    (e.g. checkpoint trained on CPU, resume on CUDA, or vice-versa), the call
    fails. In that case fall back to re-seeding deterministically so training
    can still resume without crashing.
    """
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
    """Restore torch CPU / CUDA RNG, falling back to manual_seed on mismatch."""
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
try:
    from torch.func import functional_call, vmap
except ImportError:
    from torch.nn.utils.stateless import functional_call
    from functorch import vmap

from .cost import trajectory_cost
from .rollout import rollout
from .mass_params import MassParams, batch_mass_params
from .tasks import FixedUniformMassSet


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_fn(policy, theta):
    def _fn(state, target_rel):
        return functional_call(policy, theta, (state, target_rel))
    return _fn


def _make_batched_fn(policy, stacked_theta, n_tasks, batch):
    """Policy callable that applies a *different* theta to each task block.

    ``stacked_theta`` maps every parameter name to a tensor whose leading
    dim of size ``n_tasks`` indexes the per-task adapted weights. The
    returned fn expects flat ``(n_tasks*batch, .)`` inputs whose row-block
    ``i`` (rows ``i*batch : (i+1)*batch``) is run with ``stacked_theta[i]``.
    This replaces N sequential outer rollouts with a single batched call.
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


def _clip_grads(grads, max_norm: float = 1.0):
    total = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
    coef  = torch.clamp(max_norm / (total + 1e-6), max=1.0)
    return [g * coef for g in grads]


# ---------------------------------------------------------------------------
# MAML meta-training
# ---------------------------------------------------------------------------

def meta_train(
    policy: torch.nn.Module,
    task_set: FixedUniformMassSet,
    x0_train: torch.Tensor,
    x0_eval: torch.Tensor,
    *,
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
):
    """Meta-train ``policy`` with MAML over a fixed uniform task set.

    The N tasks in ``task_set`` are fixed for the whole run; each epoch is a
    complete pass over all N tasks (no re-sampling).

    Inner loop GPU speedup
        Step 1 of the inner loop batches all N tasks into one rollout on
        (N*B_train, 12) states.  Per-task gradients are extracted with
        sequential ``autograd.grad`` calls that each touch only their slice.
        Subsequent inner steps (if n_inner_steps > 1) are sequential because
        each task uses its own adapted parameters after step 1.

    Returns:
        policy, history
    """
    N        = len(task_set)
    B_train  = x0_train.shape[0]
    B_eval   = x0_eval.shape[0]
    create_graph = (maml_order == 2)

    x0_train = x0_train.to(device)
    x0_eval  = x0_eval.to(device)

    opt = torch.optim.Adam(policy.parameters(), lr=lr_outer)

    if resume_state is None:
        torch.manual_seed(seed)
        torch_gen = torch.Generator(device=device).manual_seed(seed)
        history = {"epoch": [], "meta_loss": [], "inner_pre": []}
        start_epoch = 0
    else:
        _restore_global_rng(resume_state, seed)
        torch_gen = _restore_torch_gen(
            device, resume_state.get("torch_gen_state", None), seed)
        opt.load_state_dict(resume_state["optimizer_state"])
        history = resume_state["history"]
        for k, v in [("epoch", []), ("meta_loss", []), ("inner_pre", [])]:
            history.setdefault(k, v)
        start_epoch = len(history["epoch"])
        print(f"[Resume] from epoch {start_epoch}/{epochs}.")

    # Pre-build the big x0 / batched mass for the inner step-1 rollout.
    x0_inner_big = x0_train.repeat(N, 1)                        # (N*B_train, 12)
    mass_inner_big = batch_mass_params(task_set.masses, B_train, device)

    # Same for the batched outer rollout (all N tasks in one call).
    x0_outer_big   = x0_eval.repeat(N, 1)                       # (N*B_eval, 12)
    mass_outer_big = batch_mass_params(task_set.masses, B_eval, device)

    # Per-task MassParams moved to ``device`` once (reused every epoch by the
    # extra inner steps and the outer loop).
    masses_dev = [m.to(device) for m in task_set.masses]

    _use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    def _now() -> float:
        """Wall-clock time, syncing CUDA so GPU work is actually finished."""
        if _use_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()

    for ep in range(start_epoch, epochs):
        opt.zero_grad()
        t0 = _now()

        # ── Step 1 inner loop: one big batched forward pass ──────────────
        theta = {n: p for n, p in policy.named_parameters()}

        X_inner_big = rollout(
            _make_fn(policy, theta), x0_inner_big, n_steps,
            mass_inner_big, dynamics_step,
            tau_div=tau_div, obs_noise_std=obs_noise_std, gen=torch_gen,
        )   # (N*B_train, n_steps, 12)

        costs_0 = (trajectory_cost(
                       X_inner_big.view(N * B_train, n_steps, 12),
                       terminal_weight, pos_weight=pos_weight, z_weight=z_weight)
                   .view(N, B_train).mean(dim=1))   # (N,)

        inner_pre = costs_0.detach().mean().item()
        t_inner = _now()

        # ── Per-task gradient + adaptation ───────────────────────────────
        adapted_thetas: List[dict] = []
        for i in range(N):
            # Keep the big inner graph alive until the last task's grad is
            # computed.  For 2nd-order MAML it must stay alive even longer
            # (until meta_loss.backward()), so always retain when create_graph.
            retain = create_graph or (i < N - 1)
            grads = torch.autograd.grad(
                costs_0[i], tuple(theta.values()),
                create_graph=create_graph, retain_graph=retain,
            )
            if inner_grad_clip > 0:
                grads = _clip_grads(grads, inner_grad_clip)
            theta_i = {n: p - lr_inner * g
                       for (n, p), g in zip(theta.items(), grads)}

            # Additional inner steps (sequential — each task has its own θ).
            for _step in range(1, n_inner_steps):
                X_in = rollout(
                    _make_fn(policy, theta_i), x0_train, n_steps,
                    masses_dev[i], dynamics_step,
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
        t_grad = _now()

        # ── Outer loss (all N tasks batched into one rollout) ────────────
        # Stack the N adapted parameter sets along a new leading dim; the
        # batched policy fn then applies task i's weights to row-block i of
        # an (N*B_eval) rollout — one GPU call instead of N sequential ones.
        # mean() over the (N*B_eval) costs equals (1/N) Σ_i mean_B(cost_i),
        # i.e. the exact same meta-loss as the per-task sum it replaces.
        stacked_theta = {
            name: torch.stack([t[name] for t in adapted_thetas], dim=0)
            for name in adapted_thetas[0]
        }
        X_out = rollout(
            _make_batched_fn(policy, stacked_theta, N, B_eval),
            x0_outer_big, n_steps, mass_outer_big, dynamics_step,
            tau_div=tau_div, obs_noise_std=obs_noise_std, gen=torch_gen,
        )                                          # (N*B_eval, n_steps, 12)
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

        history["epoch"].append(ep + 1)
        history["meta_loss"].append(float(meta_loss.item()))
        history["inner_pre"].append(float(inner_pre))

        if verbose_every and ((ep + 1) % verbose_every == 0):
            print(f"  [ep {ep+1:4d}/{epochs}] "
                  f"meta={meta_loss.item():.4e}  "
                  f"inner_pre={inner_pre:.4e}")

        if profile:
            d_inner = t_inner - t0
            d_grad  = t_grad  - t_inner
            d_outer = t_outer - t_grad
            d_back  = t_back  - t_outer
            d_total = t_back  - t0
            print(f"  [profile ep {ep+1}] total={d_total:.2f}s | "
                  f"inner_rollout={d_inner:.2f}s ({100*d_inner/d_total:.0f}%)  "
                  f"grad_loop={d_grad:.2f}s ({100*d_grad/d_total:.0f}%)  "
                  f"outer_loop={d_outer:.2f}s ({100*d_outer/d_total:.0f}%)  "
                  f"meta_backward={d_back:.2f}s ({100*d_back/d_total:.0f}%)")

        if on_epoch_end is not None:
            on_epoch_end(ep, policy, history,
                         optimizer=opt, torch_gen=torch_gen)

        if killer is not None and killer.kill_now:
            print(f"[Arrêt propre] à l'epoch {ep+1}.")
            break

    return policy, history


# ---------------------------------------------------------------------------
# Multi-task baseline training (no adaptation)
# ---------------------------------------------------------------------------

def baseline_train(
    policy: torch.nn.Module,
    task_set: FixedUniformMassSet,
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
    resume_state: dict = None,
):
    """Multi-task baseline: train on all N fixed tasks simultaneously.

    All N tasks are batched into a single rollout of (N*B_train, 12) states,
    giving full GPU utilisation with no extra code complexity.

    history["train_loss"] : loss on x0_train (gradient batch)
    history["eval_loss"]  : loss on x0_eval  (no grad)
    """
    N       = len(task_set)
    B_train = x0_train.shape[0]
    B_eval  = x0_eval.shape[0]

    x0_train = x0_train.to(device)
    x0_eval  = x0_eval.to(device)

    opt = torch.optim.Adam(policy.parameters(), lr=lr_outer)

    if resume_state is None:
        torch.manual_seed(seed)
        torch_gen = torch.Generator(device=device).manual_seed(seed)
        history = {"epoch": [], "train_loss": [], "eval_loss": []}
        start_epoch = 0
    else:
        _restore_global_rng(resume_state, seed)
        torch_gen = _restore_torch_gen(
            device, resume_state.get("torch_gen_state", None), seed)
        opt.load_state_dict(resume_state["optimizer_state"])
        history = resume_state["history"]
        for k, v in [("epoch", []), ("train_loss", []), ("eval_loss", [])]:
            history.setdefault(k, v)
        start_epoch = len(history["epoch"])
        print(f"[Resume] from epoch {start_epoch}/{epochs}.")

    # Pre-build batched masses (same for every epoch).
    mass_train_big = batch_mass_params(task_set.masses, B_train, device)
    mass_eval_big  = batch_mass_params(task_set.masses, B_eval,  device)
    x0_train_big   = x0_train.repeat(N, 1)   # (N*B_train, 12)
    x0_eval_big    = x0_eval.repeat(N, 1)    # (N*B_eval,  12)

    for ep in range(start_epoch, epochs):
        opt.zero_grad()

        # All N tasks in one forward pass.
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

        history["epoch"].append(ep + 1)
        history["train_loss"].append(float(L_tr.item()))
        history["eval_loss"].append(float(L_ev.item()))

        if verbose_every and ((ep + 1) % verbose_every == 0):
            print(f"  [ep {ep+1:4d}/{epochs}] "
                  f"train={L_tr.item():.4e}  eval={L_ev.item():.4e}")

        if on_epoch_end is not None:
            on_epoch_end(ep, policy, history,
                         optimizer=opt, torch_gen=torch_gen)

        if killer is not None and killer.kill_now:
            print(f"[Arrêt propre] à l'epoch {ep+1}.")
            break

    return policy, history


# ---------------------------------------------------------------------------
# Few-shot adaptation (test time)
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
    """Few-shot adaptation on a single test task."""
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
