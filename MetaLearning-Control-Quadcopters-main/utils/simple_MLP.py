import os
import numpy as np
import scipy.linalg as la
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.stateless import functional_call
import matplotlib.pyplot as plt

import utils.config as cfg
import utils.quadcopter_model as qcm


# Policy: small MLP with tanh + limits
class PolicyMLP(nn.Module):
    
    def __init__(self, x_scale, u_max, hidden=16):
        super().__init__()
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))  # (12,)
        self.register_buffer("u_max",   torch.tensor(u_max,   dtype=torch.float32))  # (4,)
        self.net = nn.Sequential(
            nn.Linear(12, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        x_n = x / self.x_scale
        u   = self.net(x_n)

        # Torques: symmetric [-1,1] → [-umax, umax]
        torques = torch.tanh(u[..., 1:]) * self.u_max[1:]

        # Thrust: force ≥ 0 → [0, umax] using sigmoid
        thrust = torch.sigmoid(u[..., [0]]) * self.u_max[0] # replaced tanh with sigmoid to enforce non-negative thrust

        return torch.cat([thrust, torques], dim=-1)


# Policy: RNN-based controller with temporal memory
class PolicyRNN(nn.Module):
    
    def __init__(self, x_scale, u_max, hidden=32, num_layers=1, rnn_type='rnn'):
        """
        RNN-based policy network.
        
        Args:
            x_scale: State normalization scale (12,)
            u_max: Maximum control values (4,)
            hidden: Hidden state dimension
            num_layers: Number of RNN layers
            rnn_type: Type of RNN cell ('gru', 'lstm', or 'rnn')
        """
        super().__init__()
        self.hidden_size = hidden
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))  # (12,)
        self.register_buffer("u_max",   torch.tensor(u_max,   dtype=torch.float32))  # (4,)
        
        # RNN layer
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(12, hidden, num_layers, batch_first=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(12, hidden, num_layers, batch_first=True)
        elif self.rnn_type == 'rnn':
            self.rnn = nn.RNN(12, hidden, num_layers, batch_first=True, nonlinearity='tanh')
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}. Choose 'gru', 'lstm', or 'rnn'.")
        
        # Output layer
        self.fc_out = nn.Linear(hidden, 4)
        
    def forward(self, x, hidden=None):
        """
        Forward pass through RNN policy.
        
        Args:
            x: State tensor of shape [B, state_dim] or [B, T, state_dim]
            hidden: Hidden state (optional). If None, initialized to zeros.
                    For GRU/RNN: [num_layers, B, hidden_size]
                    For LSTM: tuple of (h, c) each [num_layers, B, hidden_size]
        
        Returns:
            u: Control output [B, 4] or [B, T, 4]
            hidden: Updated hidden state
        """
        # Normalize input
        x_n = x / self.x_scale
        
        # Handle both single timestep [B, 12] and sequence [B, T, 12]
        is_single_step = (x_n.dim() == 2)
        if is_single_step:
            x_n = x_n.unsqueeze(1)  # [B, 12] -> [B, 1, 12]
        
        # Initialize hidden state if not provided
        if hidden is None:
            batch_size = x_n.size(0)
            hidden = self.init_hidden(batch_size, x_n.device)
        
        # RNN forward pass
        rnn_out, hidden = self.rnn(x_n, hidden)  # rnn_out: [B, T, hidden]
        
        # Output layer
        u = self.fc_out(rnn_out)  # [B, T, 4]
        
        # Remove time dimension if input was single step
        if is_single_step:
            u = u.squeeze(1)  # [B, 1, 4] -> [B, 4]
        
        # Torques: symmetric [-1,1] → [-umax, umax]
        torques = torch.tanh(u[..., 1:]) * self.u_max[1:]
        
        # Thrust: force ≥ 0 → [0, umax] using sigmoid
        thrust = torch.sigmoid(u[..., [0]]) * self.u_max[0]
        
        return torch.cat([thrust, torques], dim=-1), hidden
    
    def init_hidden(self, batch_size, device):
        """Initialize hidden state to zeros."""
        if self.rnn_type == 'lstm':
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h, c)
        else:  # GRU or RNN
            return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


# Rollout policy for T_steps steps
def rollout_policy(A_d, B_tilde_d, policy, x0_batch, dist_batch, linearized=True):
    """
    If linearized=True  -> use discrete linear model
    If linearized=False -> use nonlinear dynamics + Euler integration
    """

    # Linear-model matrices initialization
    B_d = B_tilde_d[:, :4]
    B, n = x0_batch.shape
    m = B_d.shape[1]
    X = torch.zeros(B, cfg.T_STEPS, n, device=x0_batch.device)
    U = torch.zeros(B, cfg.T_STEPS, m, device=x0_batch.device)
    x_state = x0_batch.clone()

    # Disturbances input vectors
    B_dist_x = B_tilde_d[:, [4]]
    B_dist_y = B_tilde_d[:, [5]]
    B_dist_z = B_tilde_d[:, [6]]

    # Initialize hidden state for RNN (if applicable)
    hidden = None

    for k in range(cfg.T_STEPS):
        # Check if policy is RNN-based (has init_hidden method)
        if isinstance(policy, PolicyRNN):
            u, hidden = policy(x_state, hidden)
        else:
            u = policy(x_state)
        
        # Extract disturbances for this timestep
        d_x = dist_batch[:, k, [0]] # [B,1]
        d_y = dist_batch[:, k, [1]]
        d_z = dist_batch[:, k, [2]]

        if linearized:
            # ---- LINEAR DYNAMICS ----
            x_state = (
                (A_d @ x_state.T).T
                + (B_d @ u.T).T
                + (B_dist_x @ d_x.T).T
                + (B_dist_y @ d_y.T).T
                + (B_dist_z @ d_z.T).T
            )
        else:
            # ---- NONLINEAR DYNAMICS ----
            x_dot = qcm.quad_nonlinear_dynamics_torch(x_state, u)

            # Disturbance terms added to the nonlinear dynamics
            x_dot[:, 0] +=  d_x.flatten()/cfg.M    # x disturbance
            x_dot[:, 2] +=  d_y.flatten()/cfg.M    # y disturbance
            # no need to add z disturbance to vertical velocity (x_dot[:, 4]) since gravity is already included in non linear dynamics

            # Forward Euler discretization
            x_state = x_state + cfg.T_D * x_dot

        X[:, k, :] = x_state
        U[:, k, :] = u

    return X, U


# Cost: sum_k x'Qx + u'Ru (batched over trajectories)
def traj_cost(X, U):
    Q = torch.as_tensor(cfg.Q_DIAG, dtype=X.dtype, device=X.device)
    R = torch.as_tensor(cfg.R_DIAG, dtype=U.dtype, device=U.device)

    # Weighted squared terms (broadcast over [N, T, dims])
    cost_x = (X**2) * Q
    cost_u = (U**2) * R

    # Sum over state/control dims and time
    L = cost_x.sum(dim=(1, 2)) + cost_u.sum(dim=(1, 2))

    # Optional terminal position cost
    if cfg.Q_TERM > 0.0:
        pT = X[:, -1, [0, 2, 4]]  # x, y, z
        L += cfg.Q_TERM * (pT**2).sum(dim=1)

    return L


# Training loop
def train_nn_policy(A_d_nt, B_tilde_d_nt, init_points, disturbances, lr=1e-3, epochs=1000, device="cpu", verbose_every=100, shuffle_each_epoch=True, use_linearized=True):
    A_d = torch.tensor(A_d_nt, dtype=torch.float32, device=device)
    B_tilde_d = torch.tensor(B_tilde_d_nt, dtype=torch.float32, device=device)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize policy
    policy = PolicyMLP(x_scale=cfg.X_SCALE, u_max=cfg.U_MAX).to(device)
    # policy = PolicyRNN(x_scale=cfg.X_SCALE, u_max=cfg.U_MAX).to(device)

    # Optimizer + LR scheduler
    opt = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-5)                               # weight_decay: add an L2 regularization term to the loss for each parameter
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=50)    # reduce lr if no improvement for 50 epochs
    
    init_points = init_points.to(device)
    N = len(init_points)
    batch_size = 1 if N == 1 else N // 4

    train_means = []

    mean_eval = float('inf') # initialize for scheduler

    for ep in range(epochs):
        if shuffle_each_epoch:
            perm = torch.randperm(N, device=device)
            init_points = init_points[perm]
            disturbances = disturbances[perm]
            
        # ---- TRAIN LOOP ----
        batch_losses = []
        for i in range(0, N, batch_size):
            x0_batch = init_points[i : i + batch_size] 
            dist_batch = disturbances[i : i + batch_size]
            X, U = rollout_policy(A_d, B_tilde_d, policy, x0_batch, dist_batch, linearized=use_linearized)
            loss = traj_cost(X, U)
            loss = loss.mean()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            batch_losses.append(loss.item())

        # ---- LOG / EVALUATION ----
        if verbose_every and ((ep + 1) % verbose_every == 0):
            # Compute training stats only now
            train_mean = np.mean(batch_losses)
            train_means.append(train_mean)

            with torch.no_grad():
                x0_all = init_points.to(device)
                X_eval, U_eval = rollout_policy(A_d, B_tilde_d, policy, x0_all, disturbances, linearized=use_linearized)
                losses_eval = traj_cost(X_eval, U_eval)

                mean_eval = losses_eval.mean().item()

                p_end = X_eval[:, -1, [0, 2, 4]].norm(dim=1).mean().item()

            print(
                f"[ep {ep+1}/{epochs}]:   "
                f"train_loss={train_mean:.4e}   |    "
                f"eval_loss={mean_eval:.4e}   |   "
                f"|x_T|_avg={p_end:.3f}"
            )

        # scheduler.step(mean_eval)

    return policy, train_means


# MAML meta-training loop
def meta_train_nn_policy(A_d_nt, B_tilde_d_nt, init_points_list, disturbances_list, lr_outer=1e-3, lr_inner=1e-2, meta_batch_size=None, K_inner=5, K_outer=5, epochs=500, device="cpu", verbose_every=50, use_linearized=True):
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Convert dynamics
    A_d = torch.tensor(A_d_nt, dtype=torch.float32, device=device)
    B_tilde_d = torch.tensor(B_tilde_d_nt, dtype=torch.float32, device=device)

    num_tasks = len(disturbances_list)
    assert num_tasks == len(init_points_list)

    if meta_batch_size is None:
        meta_batch_size = num_tasks

    theta_policy = PolicyMLP(x_scale=cfg.X_SCALE, u_max=cfg.U_MAX).to(device)
    opt = torch.optim.Adam(theta_policy.parameters(), lr=lr_outer)

    print(f"[MAML] Meta-training on {num_tasks} tasks.")

    # ============================================================
    # STORAGE FOR PLOTTING
    # ============================================================
    logs = {
        "epochs_logged": [],
        "meta_loss": [],
        "inner_loss_tasks": [],      # shape: [num_logs][num_tasks]
        "inner_loss_mean": [],
        "inner_loss_min": [],
        "inner_loss_max": [],
        "outer_loss_mean": [],
        "grad_norms_mean": [],       # list of arrays [num_layers]
        "grad_norms_max": [],        # list of arrays [num_layers]
    }

    def tensor_norm(t):
        return t.detach().norm().item()

    for ep in range(epochs):

        task_indices = torch.randperm(num_tasks)[:meta_batch_size]

        meta_loss = 0.0
        opt.zero_grad()

        # Store inner/outer losses per task for logging
        inner_losses = np.zeros(num_tasks)
        outer_losses = np.zeros(num_tasks)

        grad_norms_this_epoch = []   # list of [num_layers]

        # ------------------------------------------------------------
        # TASK LOOP
        # ------------------------------------------------------------
        for task_counter, ti in enumerate(task_indices):

            dist_task = disturbances_list[ti].to(device)
            x0_task   = init_points_list[ti].to(device)
            N_i = x0_task.shape[0]

            # ------------------------------
            # INNER LOOP
            # ------------------------------
            perm = torch.randperm(N_i)[:K_inner]
            x0_inner = x0_task[perm]
            dist_inner = dist_task[perm]

            X_inner, U_inner = rollout_policy(A_d, B_tilde_d, theta_policy, x0_inner, dist_inner, linearized=use_linearized)
            loss_inner = traj_cost(X_inner, U_inner).mean()
            inner_losses[ti] = loss_inner.item()

            grads = torch.autograd.grad(
                loss_inner,
                tuple(theta_policy.parameters()),
                create_graph=True
            ) # if True, then 2nd order gradient, else first order only

            # Clip inner-loop gradients
            max_norm = 1.0
            total_norm = torch.sqrt(sum(g.norm()**2 for g in grads))
            clip_coef = max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                grads = [g * clip_coef for g in grads]

            grad_norms_this_epoch.append([tensor_norm(g) for g in grads])

            # Build θ'
            theta_i_prime = {
                name: p - lr_inner * g
                for (name, p), g in zip(theta_policy.named_parameters(), grads)
            }

            # ------------------------------
            # OUTER LOOP
            # ------------------------------
            perm2 = torch.randperm(N_i)[:K_outer]
            x0_outer = x0_task[perm2]
            dist_outer = dist_task[perm2]

            def forward_with_prime(x):
                return functional_call(theta_policy, theta_i_prime, (x,))

            X_outer, U_outer = rollout_policy(A_d, B_tilde_d, forward_with_prime, x0_outer, dist_outer, linearized=use_linearized)
            loss_outer = traj_cost(X_outer, U_outer).mean()
            outer_losses[ti] = loss_outer.item()

            meta_loss = meta_loss + loss_outer

        # ------------------------------------------------------------
        # META UPDATE
        # ------------------------------------------------------------
        meta_loss /= meta_batch_size
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(theta_policy.parameters(), 1.0)
        opt.step()

        # ------------------------------------------------------------
        # LOGGING (only at verbose intervals)
        # ------------------------------------------------------------
        if verbose_every and ((ep + 1) % verbose_every == 0):

            logs["epochs_logged"].append(ep + 1)
            logs["meta_loss"].append(meta_loss.item())

            logs["inner_loss_tasks"].append(inner_losses.copy())
            logs["inner_loss_mean"].append(inner_losses.mean())
            logs["inner_loss_min"].append(inner_losses.min())
            logs["inner_loss_max"].append(inner_losses.max())

            logs["outer_loss_mean"].append(outer_losses.mean())

            grad_norms_mean = np.mean(grad_norms_this_epoch, axis=0)
            grad_norms_max  = np.max(grad_norms_this_epoch, axis=0)
            logs["grad_norms_mean"].append(grad_norms_mean)
            logs["grad_norms_max"].append(grad_norms_max)

            print(f"[ep {ep+1}/{epochs}] meta_loss = {meta_loss.item():.4e}")

    return theta_policy, logs


# Weights adaptation on a new unseen task
def maml_adapt(theta_policy, A_d_nt, B_tilde_d_nt, x0_test, disturbances_test, lr_inner, nb_gradient_descent=5, K_samples=5, device="cpu", use_linearized=True):
    """
    Returns:
        theta_adapt : dict of adapted parameters
        adapt_losses : inner-loop losses (K samples)
        full_mean_losses : mean loss over ALL test samples at each step
        full_min_losses : min loss over ALL test samples
        full_max_losses : max loss over ALL test samples
    """

    # Convert dynamics
    A_d = torch.tensor(A_d_nt, dtype=torch.float32, device=device)
    B_tilde_d = torch.tensor(B_tilde_d_nt, dtype=torch.float32, device=device)

    # Prepare full test dataset
    x0_full = x0_test.to(device)
    dist_full = disturbances_test.to(device)

    # Sample K points for adaptation
    N = x0_test.shape[0]
    perm = torch.randperm(N)[:K_samples]
    x0_adapt  = x0_full[perm]
    dist_adapt = dist_full[perm]

    # Copy meta-parameters θ → θ_adapt
    theta_adapt = {name: p.clone() for name, p in theta_policy.named_parameters()}

    adapt_losses = []
    full_mean_losses = []
    full_min_losses  = []
    full_max_losses  = []

    # Helper forward function
    def forward_with_params(x):
        return functional_call(theta_policy, theta_adapt, (x,))

    # Begin adaptation loop
    for step in range(nb_gradient_descent):

        # FULL EVALUATION (NO GRAD) on ALL TESTS
        with torch.no_grad():
            Xf, Uf = rollout_policy(A_d, B_tilde_d, forward_with_params, x0_full, dist_full, linearized=use_linearized)
            # Loss per trajectory
            traj_losses = traj_cost(Xf, Uf)       # shape [N]
            full_mean_losses.append(traj_losses.mean().item())
            full_min_losses.append(traj_losses.min().item())
            full_max_losses.append(traj_losses.max().item())

        # ---- INNER-LOOP LOSS on K samples ----
        X, U = rollout_policy(A_d, B_tilde_d, forward_with_params, x0_adapt, dist_adapt, linearized=use_linearized)
        inner_loss = traj_cost(X, U).mean()
        adapt_losses.append(inner_loss.item())

        # Compute gradients
        grads = torch.autograd.grad(inner_loss,
                                    tuple(theta_adapt.values()),
                                    create_graph=False)

        # Gradient clipping
        max_norm = 1.0
        total_norm = torch.sqrt(sum(g.norm()**2 for g in grads))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            grads = [g * clip_coef for g in grads]

        # Update parameters
        theta_adapt = {
            n: p - lr_inner * g
            for (n, p), g in zip(theta_adapt.items(), grads)
        }

        print(f"[Adapt {step+1}] inner_loss={inner_loss.item():.4f}, "
              f"full_mean={full_mean_losses[-1]:.4f}")

    return theta_adapt, adapt_losses, full_mean_losses, full_min_losses, full_max_losses


@torch.no_grad()
def evaluate_policy_cost(A_d_nt, B_tilde_d_nt, policy, X0, disturbances, device="cpu", use_linearized=True):
    """
    Roll out a policy on a batch of initial states X0 and compute
    the average trajectory cost.
    mean_cost : float
        Mean cost over all trajectories.
    std_cost : float
        Standard deviation of costs.
    all_costs : torch.Tensor [N]
        Individual per-trajectory costs.
    """
    A_d = torch.tensor(A_d_nt, dtype=torch.float32, device=device)
    B_tilde_d = torch.tensor(B_tilde_d_nt, dtype=torch.float32, device=device)
    policy = policy.to(device)
    X0 = X0.to(device)

    X, U = rollout_policy(A_d, B_tilde_d, policy, X0, disturbances, linearized=use_linearized)
    costs = traj_cost(X, U)

    mean_cost = costs.mean().item()
    std_cost = costs.std(unbiased=False).item()
    return X, U, mean_cost, std_cost, costs


def plot_maml_diagnostics(logs, filename):
    epochs = np.array(logs["epochs_logged"])

    # META LOSS
    plt.figure(figsize=(7,5))
    plt.plot(epochs, logs["meta_loss"], 'blue', alpha=0.8, label="Meta-loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Meta loss")
    plt.grid(True, ls="--", lw=0.4)
    plt.tight_layout()
    plt.savefig(filename.replace(".png", "_outer.png"))
    plt.close()

    # PER-TASK INNER LOSSES
    inner_per_task = np.array(logs["inner_loss_tasks"])   # shape [num_logs, num_tasks]
    num_tasks = inner_per_task.shape[1]

    plt.figure(figsize=(7,5))
    for t in range(num_tasks):
        plt.plot(epochs, inner_per_task[:, t], label=f"Task {t}")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Inner losses")
    plt.legend()
    plt.grid(True, ls="--", lw=0.4)
    plt.tight_layout()
    plt.savefig(filename.replace(".png", "_inner.png"))
    plt.close()


def plot_adapted_losses(adapt_losses, full_mean_losses, full_min_losses, full_max_losses, filename):
    steps = np.arange(0, len(full_mean_losses))

    plt.figure(figsize=(6,4))
    plt.plot(steps, adapt_losses, marker='o', markersize=4, color="red", alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Adaptation Step")
    plt.ylabel("Inner Loss (5 samples)")
    plt.title("Inner Adaptation Loss (used for weight updates)")
    plt.grid(True)
    plt.xlim(0, 5)
    plt.tight_layout()
    plt.savefig(filename.replace(".png", "_inner.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(6,4))
    plt.plot(steps, full_mean_losses, marker='o', markersize=4, label="Mean Loss", color="red", alpha=0.8)
    plt.fill_between(
        steps,
        full_min_losses,
        full_max_losses,
        alpha=0.15,
        edgecolor="none",
        color="red",
        label="Min-Max Range"
    )

    plt.yscale("log")
    plt.xlabel("Adaptation Step")
    plt.ylabel("Full Test Loss")
    plt.title("Performance on Entire Test Distribution")
    plt.grid(True)
    plt.xlim(0, 5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename.replace(".png", "_full.png"), dpi=300)
    plt.close()


def plot_adaptation_comparison(full_mean_losses_joint, full_min_losses_joint, full_max_losses_joint, full_mean_losses_meta, full_min_losses_meta, full_max_losses_meta, filename):
    steps = np.arange(0, len(full_mean_losses_joint))
    plt.figure(figsize=(6,4))

    color_meta = "#D81B60"
    plt.plot(steps, full_mean_losses_meta, marker='o', markersize=4, label="Meta adaptation: mean loss", color=color_meta, alpha=0.8)
    plt.fill_between(
        steps,
        full_min_losses_meta,
        full_max_losses_meta,
        alpha=0.15,
        edgecolor="none",
        color=color_meta,
        label="Meta adaptation: min/max range"
    )

    color_joint = "#2E7D32" 
    plt.plot(steps, full_mean_losses_joint, marker='o', markersize=4, label="Joint adaptation: mean loss", color=color_joint, alpha=0.8)
    plt.fill_between(
        steps,
        full_min_losses_joint,
        full_max_losses_joint,
        alpha=0.15,
        edgecolor="none",
        color=color_joint,
        label="Joint adaptation: min/max range"
    )

    plt.yscale("log")
    plt.xlabel("Adaptation Step")
    plt.ylabel("Full Test Loss")
    plt.xlim(0, 5)
    plt.ylim(5e2, 3e7)
    plt.title("Performance on Entire Test Distribution")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_loss_evolution(train_means, total_epochs, log_interval, filename="training_loss_evolution.png"):
    plt.figure(figsize=(7, 5))

    epochs_range = np.arange(log_interval, total_epochs + 1, log_interval)

    plt.plot(epochs_range, train_means, 'blue', alpha=0.8)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean training loss")
    plt.grid(True, which="both", ls="--", lw=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()