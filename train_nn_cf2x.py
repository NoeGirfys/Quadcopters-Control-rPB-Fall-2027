#!/usr/bin/env python3
"""
Train a neural network controller for the Crazyflie 2.x (CF2X) quadcopter.

Reproduces Goffin's section 4.1 sanity check (regulation to origin) but with
the CF2X physical parameters from gym-pybullet-drones, so the trained weights
can later be validated inside the pybullet-drones simulator (DYN mode).

The dynamics model is the standard linearized quadcopter around hover,
discretized via ZOH.  The NN outputs absolute wrench [F, τx, τy, τz].

Usage:
    python train_nn_cf2x.py
    → saves  trained_policy_cf2x.pt  (full model)
    → saves  trained_weights_cf2x.pt (state_dict only)
"""

import numpy as np
import scipy.linalg as la
from scipy.signal import cont2discrete
import torch
import torch.nn as nn
import itertools, os

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CF2X Physical Parameters  (from cf2x.urdf in gym-pybullet-drones)
# ═══════════════════════════════════════════════════════════════════════════════
M       = 0.027                         # mass [kg]
G       = 9.8                           # gravity [m/s²]  (pybullet uses 9.8)
I_X     = 1.4e-5                        # Ixx [kg·m²]
I_Y     = 1.4e-5                        # Iyy [kg·m²]
I_Z     = 2.17e-5                       # Izz [kg·m²]
L       = 0.0397                        # arm length [m]
KF      = 3.16e-10                      # thrust coefficient [N/RPM²]
KM      = 7.94e-12                      # torque coefficient [N·m/RPM²]
T2W     = 2.25                          # thrust-to-weight ratio

# Derived limits
GRAVITY     = M * G                                         # weight [N]
HOVER_RPM   = np.sqrt(GRAVITY / (4 * KF))                  # ≈ 14468 RPM
MAX_RPM     = np.sqrt((T2W * GRAVITY) / (4 * KF))          # ≈ 21702 RPM
MAX_RPM_SQ  = MAX_RPM**2
F_MAX       = 4 * KF * MAX_RPM_SQ                          # ≈ 0.596 N
TAU_XY_MAX  = (2 * L * KF * MAX_RPM_SQ) / np.sqrt(2)      # ≈ 8.4e-3 N·m
TAU_Z_MAX   = 2 * KM * MAX_RPM_SQ                          # ≈ 7.5e-3 N·m

U_MAX = np.array([F_MAX, TAU_XY_MAX, TAU_XY_MAX, TAU_Z_MAX], dtype=np.float32)

# State scaling (Bryson-style max acceptable deviations)
X_MAX       = 1.0       # [m]
X_DMAX      = 1.0       # [m/s]
Y_MAX       = 1.0
Y_DMAX      = 1.0
Z_MAX       = 1.0
Z_DMAX      = 1.0
PHI_MAX     = np.deg2rad(30)
PHI_DMAX    = np.deg2rad(200)       # CF2X has fast angular dynamics
THETA_MAX   = np.deg2rad(30)
THETA_DMAX  = np.deg2rad(200)
PSI_MAX     = np.deg2rad(45)
PSI_DMAX    = np.deg2rad(120)

X_SCALE = np.array([X_MAX, X_DMAX, Y_MAX, Y_DMAX, Z_MAX, Z_DMAX,
                     PHI_MAX, PHI_DMAX, THETA_MAX, THETA_DMAX,
                     PSI_MAX, PSI_DMAX], dtype=np.float32)

# Simulation
CTRL_FREQ = 48                          # matches pybullet ctrl_freq=48
TS        = 1.0 / CTRL_FREQ            # ≈ 0.02083 s
T_SIM     = 4.0                         # [s]
T_STEPS   = int(T_SIM / TS)            # 192 steps

print(f"[CF2X] m={M}, g={G}, Ixx={I_X}, Iyy={I_Y}, Izz={I_Z}")
print(f"[CF2X] F_max={F_MAX:.4f} N, τxy_max={TAU_XY_MAX:.6f} N·m, τz_max={TAU_Z_MAX:.6f} N·m")
print(f"[CF2X] HOVER_RPM={HOVER_RPM:.1f}, MAX_RPM={MAX_RPM:.1f}")
print(f"[CF2X] Ts={TS:.5f} s, T_sim={T_SIM} s, T_steps={T_STEPS}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Linearized Quadcopter Model
# ═══════════════════════════════════════════════════════════════════════════════
# State: [x, xdot, y, ydot, z, zdot, phi, phidot, theta, thetadot, psi, psidot]
# Input: [F, tau_x, tau_y, tau_z]  (absolute thrust, not deviation)

def build_linear_model():
    """Build continuous-time A, B, c for the linearized hover model.

    The coupling signs match pybullet's rotation convention:
        x_ddot = +g * theta    (pitch forward → move in +x)
        y_ddot = -g * phi      (roll right → move in -y)
    The gravity enters as an affine bias c (not in A or B).
    """
    A = np.zeros((12, 12))
    # Kinematics: position ← velocity
    A[0, 1]   = 1.0    # x_dot
    A[2, 3]   = 1.0    # y_dot
    A[4, 5]   = 1.0    # z_dot
    A[6, 7]   = 1.0    # phi_dot
    A[8, 9]   = 1.0    # theta_dot
    A[10, 11] = 1.0    # psi_dot
    # Translational–rotational coupling at hover (F_hover/m = g)
    A[1, 8]   =  G     # x_ddot ← +g * theta
    A[3, 6]   = -G     # y_ddot ← -g * phi   (CRITICAL: negative sign!)

    B = np.zeros((12, 4))
    B[5, 0]  = 1.0 / M     # z_ddot ← F/m
    B[7, 1]  = 1.0 / I_X   # phi_ddot ← tau_x / Ix
    B[9, 2]  = 1.0 / I_Y   # theta_ddot ← tau_y / Iy
    B[11, 3] = 1.0 / I_Z   # psi_ddot ← tau_z / Iz

    # Affine gravity bias (only z_ddot gets -g)
    c = np.zeros((12, 1))
    c[5, 0] = -G

    C = np.eye(12)
    D = np.zeros((12, 4))
    return A, B, C, D, c


def discretize_affine(A, B, C, D, c, Ts):
    """ZOH discretization of the affine system  x_dot = Ax + Bu + c.

    Uses the standard augmented-state trick:
        [x; 1]_{k+1} = [Ad d; 0 1] [x; 1]_k + [Bd; 0] u_k
    """
    n, m = A.shape[0], B.shape[1]
    # Augment
    A_aug = np.zeros((n + 1, n + 1))
    B_aug = np.zeros((n + 1, m))
    A_aug[:n, :n] = A
    A_aug[:n, n]  = c.squeeze()
    B_aug[:n, :]  = B

    C_aug = np.zeros((n, n + 1))
    C_aug[:, :n] = C
    D_aug = D.copy()

    Ad_aug, Bd_aug, _, _, _ = cont2discrete(
        (A_aug, B_aug, C_aug, D_aug), Ts, method="zoh"
    )

    Ad = Ad_aug[:n, :n]
    Bd = Bd_aug[:n, :]
    d  = Ad_aug[:n, n]      # discrete affine bias vector (12,)
    return Ad, Bd, d


# Build model
A_ct, B_ct, C_ct, D_ct, c_ct = build_linear_model()
Ad, Bd, d_vec = discretize_affine(A_ct, B_ct, C_ct, D_ct, c_ct, TS)

print(f"\n[Model] Ad shape={Ad.shape}, Bd shape={Bd.shape}, d shape={d_vec.shape}")
print(f"[Model] d (gravity bias) = {d_vec}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LQR Baseline (for comparison)
# ═══════════════════════════════════════════════════════════════════════════════
Q_diag = np.array([
    1/X_MAX**2,     1/X_DMAX**2,
    1/Y_MAX**2,     1/Y_DMAX**2,
    1/Z_MAX**2,     1/Z_DMAX**2,
    1/PHI_MAX**2,   1/PHI_DMAX**2,
    1/THETA_MAX**2, 1/THETA_DMAX**2,
    1/PSI_MAX**2,   1/PSI_DMAX**2,
], dtype=np.float32)

R_diag = np.array([
    1/F_MAX**2,
    1/TAU_XY_MAX**2,
    1/TAU_XY_MAX**2,
    1/TAU_Z_MAX**2,
], dtype=np.float32)

Q = np.diag(Q_diag).astype(np.float64)
R = np.diag(R_diag).astype(np.float64)

P = la.solve_discrete_are(Ad, Bd, Q, R)
K_LQR = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
u_eq = -np.linalg.pinv(Bd) @ d_vec     # feedforward to cancel gravity at origin
print(f"\n[LQR] K shape={K_LQR.shape}")
print(f"[LQR] u_eq (hover feedforward) = {u_eq}")
print(f"[LQR] u_eq[0] should ≈ mg = {M*G:.4f} → got {u_eq[0]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Neural Network Policy
# ═══════════════════════════════════════════════════════════════════════════════
class PolicyMLP(nn.Module):
    """MLP controller: state (12) → wrench (4).

    Thrust output uses sigmoid → [0, F_max].
    Torque outputs use tanh → [-τ_max, +τ_max].
    """
    def __init__(self, x_scale, u_max, hidden=64):
        super().__init__()
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        self.register_buffer("u_max",   torch.tensor(u_max,   dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(12, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
        )

    def forward(self, x):
        x_n = x / self.x_scale
        raw = self.net(x_n)
        thrust  = torch.sigmoid(raw[..., [0]]) * self.u_max[0]
        torques = torch.tanh(raw[..., 1:])     * self.u_max[1:]
        return torch.cat([thrust, torques], dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Differentiable Rollout & Cost
# ═══════════════════════════════════════════════════════════════════════════════
def rollout(Ad_t, Bd_t, d_t, policy, x0, T_steps):
    """Roll out the policy through the affine linear dynamics.

    x_{k+1} = Ad x_k + Bd u_k + d
    u_k     = policy(x_k)

    Returns X (B, T, 12), U (B, T, 4).
    """
    B = x0.shape[0]
    dev = Ad_t.device
    X = torch.zeros(B, T_steps, 12, device=dev)
    U = torch.zeros(B, T_steps, 4,  device=dev)
    x = x0
    d_b = d_t.unsqueeze(0)  # (1, 12) for broadcasting

    for k in range(T_steps):
        u = policy(x)
        x = (x @ Ad_t.T) + (u @ Bd_t.T) + d_b
        X[:, k, :] = x
        U[:, k, :] = u
    return X, U


def traj_cost(X, U, Q_d, R_d, terminal_weight=0.0):
    """Quadratic trajectory cost, mean over batch, sum over time."""
    Q = torch.as_tensor(Q_d, dtype=X.dtype, device=X.device)
    R = torch.as_tensor(R_d, dtype=U.dtype, device=U.device)

    cost_x = (X**2 * Q).sum(dim=2).mean(dim=0).sum()
    cost_u = (U**2 * R).sum(dim=2).mean(dim=0).sum()
    L = cost_x + cost_u

    if terminal_weight > 0.0:
        pT = X[:, -1, [0, 2, 4]]   # terminal [x, y, z]
        L += terminal_weight * (pT**2).mean()
    return L


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Training
# ═══════════════════════════════════════════════════════════════════════════════
def make_x0_batch(xyz_list, device="cpu"):
    """Convert list of (x,y,z) to (B, 12) initial state tensor."""
    X0 = torch.zeros(len(xyz_list), 12, dtype=torch.float32, device=device)
    for b, (x, y, z) in enumerate(xyz_list):
        X0[b, 0] = x
        X0[b, 2] = y
        X0[b, 4] = z
    return X0


def generate_cube_points(half_side=0.3):
    """Generate 27 points describing a cube (vertices + face centers +
    edge centers + center), same as Goffin's section 4.1."""
    vals = [-half_side, 0.0, half_side]
    return list(itertools.product(vals, repeat=3))


def train(epochs=2000, lr=1e-3, hidden=64, terminal_weight=10.0,
          half_side=0.3, device="cpu"):
    """Train the NN policy on a regulation task (go to origin)."""
    Ad_t = torch.tensor(Ad, dtype=torch.float32, device=device)
    Bd_t = torch.tensor(Bd, dtype=torch.float32, device=device)
    d_t  = torch.tensor(d_vec, dtype=torch.float32, device=device)

    policy = PolicyMLP(x_scale=X_SCALE, u_max=U_MAX, hidden=hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # 27 cube points as training initial conditions
    cube_pts = generate_cube_points(half_side)
    x0_batch = make_x0_batch(cube_pts, device=device)
    print(f"\n[Train] {len(cube_pts)} initial conditions in cube ±{half_side} m")
    print(f"[Train] epochs={epochs}, lr={lr}, hidden={hidden}")
    print(f"[Train] terminal_weight={terminal_weight}")

    for ep in range(epochs):
        X, U = rollout(Ad_t, Bd_t, d_t, policy, x0_batch, T_STEPS)
        loss = traj_cost(X, U, Q_diag, R_diag, terminal_weight)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if (ep + 1) % 200 == 0 or ep == 0:
            with torch.no_grad():
                pos_end = X[:, -1, [0, 2, 4]].norm(dim=1)
                print(f"  [ep {ep+1:4d}/{epochs}]  loss={loss.item():.4e}  "
                      f"|x_T|_mean={pos_end.mean().item():.4f}  "
                      f"|x_T|_max={pos_end.max().item():.4f}")

    return policy


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Evaluation
# ═══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(policy, test_pts, device="cpu"):
    """Evaluate policy on test initial conditions, print results."""
    Ad_t = torch.tensor(Ad, dtype=torch.float32, device=device)
    Bd_t = torch.tensor(Bd, dtype=torch.float32, device=device)
    d_t  = torch.tensor(d_vec, dtype=torch.float32, device=device)

    x0 = make_x0_batch(test_pts, device=device)
    X, U = rollout(Ad_t, Bd_t, d_t, policy, x0, T_STEPS)

    pos_end = X[:, -1, [0, 2, 4]]
    errors = pos_end.norm(dim=1)
    print(f"\n[Eval] {len(test_pts)} test points:")
    print(f"  terminal error  mean={errors.mean().item():.5f} m  "
          f"max={errors.max().item():.5f} m")

    # Also evaluate with LQR for comparison
    K_t = torch.tensor(K_LQR, dtype=torch.float32, device=device)
    u_eq_t = torch.tensor(u_eq, dtype=torch.float32, device=device).unsqueeze(0)
    X_lqr = torch.zeros_like(X)
    x = x0.clone()
    for k in range(T_STEPS):
        u = u_eq_t - (x @ K_t.T)
        u[:, 0] = torch.clamp(u[:, 0], 0, F_MAX)
        u[:, 1] = torch.clamp(u[:, 1], -TAU_XY_MAX, TAU_XY_MAX)
        u[:, 2] = torch.clamp(u[:, 2], -TAU_XY_MAX, TAU_XY_MAX)
        u[:, 3] = torch.clamp(u[:, 3], -TAU_Z_MAX, TAU_Z_MAX)
        x = (x @ Ad_t.T) + (u @ Bd_t.T) + d_t.unsqueeze(0)
        X_lqr[:, k, :] = x

    pos_end_lqr = X_lqr[:, -1, [0, 2, 4]]
    errors_lqr = pos_end_lqr.norm(dim=1)
    print(f"  LQR baseline    mean={errors_lqr.mean().item():.5f} m  "
          f"max={errors_lqr.max().item():.5f} m")

    return X, U


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device}")

    # Train
    policy = train(epochs=600, lr=1e-3, hidden=64,
                   terminal_weight=10.0, half_side=0.3, device=device)

    # Evaluate on 20 random test points within the cube
    rng = np.random.default_rng(42)
    test_pts = [(rng.uniform(-0.3, 0.3),
                 rng.uniform(-0.3, 0.3),
                 rng.uniform(-0.3, 0.3)) for _ in range(20)]
    X_nn, U_nn = evaluate(policy, test_pts, device=device)

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    path_full  = os.path.join(out_dir, "trained_policy_cf2x.pt")
    path_dict  = os.path.join(out_dir, "trained_weights_cf2x.pt")

    # Save full model (for easy loading)
    torch.save(policy.cpu(), path_full)
    # Save state dict + metadata (for portable loading)
    torch.save({
        "state_dict": policy.state_dict(),
        "hidden": 64,
        "x_scale": X_SCALE,
        "u_max": U_MAX,
        # Also save the dynamics for reference
        "Ad": Ad,
        "Bd": Bd,
        "d": d_vec,
        "Ts": TS,
        "ctrl_freq": CTRL_FREQ,
        # CF2X allocation info (for wrench → RPM conversion)
        "KF": KF,
        "KM": KM,
        "L": L,
        "M": M,
        "G": G,
        "MAX_RPM": MAX_RPM,
    }, path_dict)

    print(f"\n[Saved] {path_full}")
    print(f"[Saved] {path_dict}")
    print("\nDone! Run  validate_nn_pybullet.py  to test in pybullet-drones.")