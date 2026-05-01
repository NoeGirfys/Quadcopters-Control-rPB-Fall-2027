"""Differentiable rollout. T=1 (NN queried every step).

The rollout is parametric in:
  * a ``policy_fn`` callable ``(state, target_rel) -> action``. This lets
    us pass either ``nn.Module`` instances or ``functools.partial`` /
    ``functional_call`` wrappers used by MAML's inner loop.
  * a ``mass: MassParams`` shared across the whole batch (one task).
  * a ``dynamics_step`` callable (nonlinear or linearized).
"""
import torch

from . import config as C
from .mass_params import MassParams
from .pid_chain import init_pid_state, one_nn_step, reset_pid_diverged


def compute_relative_targets(state: torch.Tensor) -> torch.Tensor:
    """Regulation-to-origin: target - current_pos, with target at origin."""
    return -state[:, [0, 2, 4]]


def rollout(policy_fn, x0: torch.Tensor, n_steps: int,
            mass: MassParams, dynamics_step,
            tau_div: float = None,
            log_actions: bool = False,
            obs_noise_std: torch.Tensor = None):
    """Run ``n_steps`` NN queries (each = PID_STEPS_PER_NN PID substeps).

    Curriculum reset: when ``tau_div`` is set, drones whose horizontal
    error exceeds it are zeroed (state + PID state). The rollout stays
    differentiable for non-diverged drones.

    Returns:
        X (B, n_steps, 12) state trajectory.
        If ``log_actions``, also returns A (B, n_steps, 4) and R (B, n_steps, 4).
    """
    B = x0.shape[0]
    dev = x0.device
    X = torch.zeros(B, n_steps, 12, device=dev)
    if log_actions:
        A = torch.zeros(B, n_steps, 4, device=dev)
        R = torch.zeros(B, n_steps, 4, device=dev)

    state = x0
    pid_state = init_pid_state(state, mass.hover_rpm)

    for k in range(n_steps):
        if obs_noise_std is None:
            state_obs = state
        else:
            state_obs = state + torch.randn_like(state) * obs_noise_std

        target_rel = compute_relative_targets(state_obs)
        action = policy_fn(state_obs, target_rel)             # (B, 4)
        state, pid_state = one_nn_step(state, action, pid_state, mass, dynamics_step)
        X[:, k, :] = state
        if log_actions:
            A[:, k, :] = action
            R[:, k, :] = pid_state[5]

        if tau_div is not None:
            pos_err = state[:, [0, 2, 4]].norm(dim=1)
            diverged = pos_err > tau_div
            if diverged.any():
                m = diverged.unsqueeze(1).expand_as(state)
                state = torch.where(m, torch.zeros_like(state), state)
                pid_state = reset_pid_diverged(pid_state, diverged, mass.hover_rpm)

    if log_actions:
        return X, A, R
    return X
