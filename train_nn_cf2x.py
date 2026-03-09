#!/usr/bin/env python3
"""
Train a neural network controller for the Crazyflie 2.x (CF2X) quadcopter.

Reproduces Goffin's section 4.1 sanity check (regulation to origin) but with
CF2X physical parameters from gym-pybullet-drones.

Two dynamics modes (--linear flag):
  - Nonlinear (default): EXACT replica of pybullet's DYN mode, including
    quaternion integration, gyroscopic coupling, semi-implicit Euler.
  - Linear (--linear): affine linearized model with ZOH discretization.

Usage:
    python train_nn_cf2x.py              # nonlinear dynamics (default)
    python train_nn_cf2x.py --linear     # linearized dynamics (faster)
    python train_nn_cf2x.py --epochs 3000 --lr 5e-4
"""

import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import itertools, os


# =====================================================================
# 1. CF2X Physical Parameters  (from cf2x.urdf in gym-pybullet-drones)
# =====================================================================

M       = 0.027          # mass [kg]
G       = 9.8            # gravity [m/s^2]  (pybullet uses 9.8)
I_X     = 1.4e-5         # Ixx [kg*m^2]
I_Y     = 1.4e-5         # Iyy [kg*m^2]
I_Z     = 2.17e-5        # Izz [kg*m^2]
L       = 0.0397         # arm length [m]
KF      = 3.16e-10       # thrust coeff [N/RPM^2]
KM      = 7.94e-12       # torque coeff [N*m/RPM^2]
T2W     = 2.25           # thrust-to-weight ratio

# Derived limits (same computation as BaseAviary.__init__)
GRAVITY     = M * G
HOVER_RPM   = np.sqrt(GRAVITY / (4 * KF))
MAX_RPM     = np.sqrt((T2W * GRAVITY) / (4 * KF))
F_MAX       = 4 * KF * MAX_RPM**2
TAU_XY_MAX  = (2 * L * KF * MAX_RPM**2) / np.sqrt(2)
TAU_Z_MAX   = 2 * KM * MAX_RPM**2

U_MAX = np.array([F_MAX, TAU_XY_MAX, TAU_XY_MAX, TAU_Z_MAX], dtype=np.float32)

# State scaling (hyperparameters — also used for Bryson Q weights)
X_MAX      = 1.0;           X_DMAX     = 1.0
Y_MAX      = 1.0;           Y_DMAX     = 1.0
Z_MAX      = 1.0;           Z_DMAX     = 1.0
PHI_MAX    = np.deg2rad(30); PHI_DMAX   = np.deg2rad(200)
THETA_MAX  = np.deg2rad(30); THETA_DMAX = np.deg2rad(200)
PSI_MAX    = np.deg2rad(45); PSI_DMAX   = np.deg2rad(120)

X_SCALE = np.array([X_MAX, X_DMAX, Y_MAX, Y_DMAX, Z_MAX, Z_DMAX,
                     PHI_MAX, PHI_DMAX, THETA_MAX, THETA_DMAX,
                     PSI_MAX, PSI_DMAX], dtype=np.float32)

# Timing (matches pybullet)
PYB_FREQ   = 240
CTRL_FREQ  = 48
PYB_STEPS_PER_CTRL = PYB_FREQ // CTRL_FREQ  # 5
DT_PYB     = 1.0 / PYB_FREQ                 # 1/240 s
DT_CTRL    = 1.0 / CTRL_FREQ                # 1/48 s
T_SIM      = 4.0
T_STEPS    = int(T_SIM / DT_CTRL)           # 192

print(f"[CF2X] m={M}, g={G}, Ixx={I_X}, Iyy={I_Y}, Izz={I_Z}")
print(f"[CF2X] F_max={F_MAX:.4f} N, tau_xy_max={TAU_XY_MAX:.6f} N*m, "
      f"tau_z_max={TAU_Z_MAX:.6f} N*m")
print(f"[CF2X] HOVER_RPM={HOVER_RPM:.1f}, MAX_RPM={MAX_RPM:.1f}")
print(f"[CF2X] DT_pyb={DT_PYB:.6f}s, DT_ctrl={DT_CTRL:.5f}s, "
      f"sub_steps={PYB_STEPS_PER_CTRL}, T_steps={T_STEPS}")


# =====================================================================
# 2. Nonlinear dynamics — EXACT replica of pybullet DYN mode
# =====================================================================
#
# Internal state is 13D: [x, vx, y, vy, z, vz, qx, qy, qz, qw, p, q, r]
# Quaternion convention: [x, y, z, w] (scalar-last, same as pybullet)
#
# The NN receives 12D: quaternion is converted to Euler before forward pass.
#
# Every operation below has a comment pointing to the corresponding
# line in BaseAviary._dynamics() or _integrateQ().

# ---- Quaternion utilities ----

def quat_to_rotmat(qx, qy, qz, qw):
    """Quaternion [x,y,z,w] -> rotation matrix (B, 3, 3).
    Same formula as pybullet getMatrixFromQuaternion."""
    xx = qx*qx; yy = qy*qy; zz = qz*qz
    xy = qx*qy; xz = qx*qz; yz = qy*qz
    wx = qw*qx; wy = qw*qy; wz = qw*qz

    R = torch.stack([
        torch.stack([1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)  ], dim=-1),
        torch.stack([2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)   ], dim=-1),
        torch.stack([2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy) ], dim=-1),
    ], dim=-2)
    return R


def quat_to_euler(qx, qy, qz, qw):
    """Quaternion [x,y,z,w] -> Euler angles (phi, theta, psi).
    Same formula as pybullet getEulerFromQuaternion."""
    # Roll (phi)
    phi = torch.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    # Pitch (theta)
    sinp = torch.clamp(2*(qw*qy - qz*qx), -1.0, 1.0)
    theta = torch.asin(sinp)
    # Yaw (psi)
    psi = torch.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return phi, theta, psi


def integrate_quat(qx, qy, qz, qw, p, q, r, dt):
    """Exact quaternion integration for constant angular velocity.
    Replicates BaseAviary._integrateQ() (lines 879-892).

    q(t+dt) = (cos(theta)*I + sin(theta)/||omega|| * Omega) @ q(t)
    with theta = ||omega|| * dt / 2.
    """
    omega_norm = torch.sqrt(p*p + q*q + r*r + 1e-12)
    half_angle = omega_norm * dt * 0.5
    cos_ha = torch.cos(half_angle)
    sin_ha_over_norm = torch.sin(half_angle) / omega_norm

    # Matrix-vector product expanded (same as lambda_ @ quat in pybullet)
    qx_n = cos_ha*qx + sin_ha_over_norm*( r*qy - q*qz + p*qw)
    qy_n = cos_ha*qy + sin_ha_over_norm*(-r*qx + p*qz + q*qw)
    qz_n = cos_ha*qz + sin_ha_over_norm*( q*qx - p*qy + r*qw)
    qw_n = cos_ha*qw + sin_ha_over_norm*(-p*qx - q*qy - r*qz)

    # Renormalize
    norm = torch.sqrt(qx_n**2 + qy_n**2 + qz_n**2 + qw_n**2 + 1e-12)
    return qx_n/norm, qy_n/norm, qz_n/norm, qw_n/norm


# ---- State conversion (13D internal <-> 12D for NN) ----

def state12_to_state13(s12):
    """(B,12) with rpy -> (B,13) with quaternion. For initial conditions."""
    B, dev = s12.shape[0], s12.device
    phi = s12[:,6]; theta = s12[:,8]; psi = s12[:,10]
    cy = torch.cos(psi*0.5);   sy = torch.sin(psi*0.5)
    cp = torch.cos(theta*0.5); sp = torch.sin(theta*0.5)
    cr = torch.cos(phi*0.5);   sr = torch.sin(phi*0.5)
    s13 = torch.zeros(B, 13, device=dev, dtype=s12.dtype)
    s13[:,0]=s12[:,0]; s13[:,1]=s12[:,1]  # x, vx
    s13[:,2]=s12[:,2]; s13[:,3]=s12[:,3]  # y, vy
    s13[:,4]=s12[:,4]; s13[:,5]=s12[:,5]  # z, vz
    s13[:,6] = sr*cp*cy - cr*sp*sy       # qx
    s13[:,7] = cr*sp*cy + sr*cp*sy       # qy
    s13[:,8] = cr*cp*sy - sr*sp*cy       # qz
    s13[:,9] = cr*cp*cy + sr*sp*sy       # qw
    s13[:,10]=s12[:,7]; s13[:,11]=s12[:,9]; s13[:,12]=s12[:,11]  # p, q, r
    return s13


def state13_to_state12(s13):
    """(B,13) with quaternion -> (B,12) with rpy. For NN input and cost."""
    B, dev = s13.shape[0], s13.device
    phi, theta, psi = quat_to_euler(s13[:,6], s13[:,7], s13[:,8], s13[:,9])
    s12 = torch.zeros(B, 12, device=dev, dtype=s13.dtype)
    s12[:,0]=s13[:,0]; s12[:,1]=s13[:,1]  # x, vx
    s12[:,2]=s13[:,2]; s12[:,3]=s13[:,3]  # y, vy
    s12[:,4]=s13[:,4]; s12[:,5]=s13[:,5]  # z, vz
    s12[:,6]=phi;      s12[:,7]=s13[:,10] # phi, p
    s12[:,8]=theta;    s12[:,9]=s13[:,11] # theta, q
    s12[:,10]=psi;     s12[:,11]=s13[:,12] # psi, r
    return s12


# ---- Dynamics step ----

def dynamics_substep(s13, wrench, dt):
    """One sub-step — exact replica of BaseAviary._dynamics().

    Parameters: s13 (B,13), wrench (B,4), dt float.
    Returns: s13_new (B,13).
    """
    x=s13[:,0]; vx=s13[:,1]; y=s13[:,2]; vy=s13[:,3]
    z=s13[:,4]; vz=s13[:,5]
    qx=s13[:,6]; qy=s13[:,7]; qz=s13[:,8]; qw=s13[:,9]
    p=s13[:,10]; q=s13[:,11]; r=s13[:,12]

    F=wrench[:,0]; tau_x=wrench[:,1]; tau_y=wrench[:,2]; tau_z=wrench[:,3]

    # (a) Rotation matrix from quaternion             [line 836]
    R = quat_to_rotmat(qx, qy, qz, qw)

    # (b) Thrust in world frame                       [lines 839-841, 858]
    tw = F.unsqueeze(-1) * R[:, :, 2]
    ax = tw[:,0]/M;  ay = tw[:,1]/M;  az = tw[:,2]/M - G

    # (c) Gyroscopic coupling                         [lines 856-857]
    p_dot = (tau_x - (I_Z-I_Y)*q*r) / I_X
    q_dot = (tau_y - (I_X-I_Z)*p*r) / I_Y
    r_dot = (tau_z - (I_Y-I_X)*p*q) / I_Z

    # (d) Semi-implicit Euler — velocities first      [lines 860-861]
    vx_n=vx+dt*ax; vy_n=vy+dt*ay; vz_n=vz+dt*az
    p_n=p+dt*p_dot; q_n=q+dt*q_dot; r_n=r+dt*r_dot

    # (d) Positions with NEW velocities               [line 862]
    x_n=x+dt*vx_n; y_n=y+dt*vy_n; z_n=z+dt*vz_n

    # (d) Quaternion integration with NEW omega       [line 863]
    qx_n, qy_n, qz_n, qw_n = integrate_quat(qx, qy, qz, qw, p_n, q_n, r_n, dt)

    return torch.stack([x_n,vx_n, y_n,vy_n, z_n,vz_n,
                        qx_n,qy_n,qz_n,qw_n, p_n,q_n,r_n], dim=-1)


def dynamics_one_ctrl_step(s13, wrench, n_substeps=PYB_STEPS_PER_CTRL):
    """One control step = n_substeps physics sub-steps.
    n_substeps=5 at dt=1/240 replicates pybullet exactly.
    n_substeps=1 at dt=1/48 is faster for training."""
    dt = DT_CTRL / n_substeps
    for _ in range(n_substeps):
        s13 = dynamics_substep(s13, wrench, dt)
    return s13


# =====================================================================
# 3. Linear model (for --linear mode and LQR baseline)
# =====================================================================

def _build_linear_model():
    """Linearized affine model + ZOH discretization."""
    from scipy.signal import cont2discrete
    A = np.zeros((12,12))
    A[0,1]=1; A[2,3]=1; A[4,5]=1; A[6,7]=1; A[8,9]=1; A[10,11]=1
    A[1,8]=G; A[3,6]=-G
    B = np.zeros((12,4))
    B[5,0]=1/M; B[7,1]=1/I_X; B[9,2]=1/I_Y; B[11,3]=1/I_Z
    c = np.zeros((12,1)); c[5,0]=-G

    n, m = 12, 4
    A_aug = np.zeros((n+1,n+1)); A_aug[:n,:n]=A; A_aug[:n,n]=c.squeeze()
    B_aug = np.zeros((n+1,m));   B_aug[:n,:]=B
    C_aug = np.zeros((n,n+1));   C_aug[:,:n]=np.eye(n)
    D_aug = np.zeros((n,m))
    Ad_a, Bd_a, _, _, _ = cont2discrete((A_aug,B_aug,C_aug,D_aug), DT_CTRL, method='zoh')
    Ad = Ad_a[:n,:n]; Bd = Bd_a[:n,:]; d = Ad_a[:n,n]

    Q_d = np.array([1/X_MAX**2, 1/X_DMAX**2, 1/Y_MAX**2, 1/Y_DMAX**2,
                     1/Z_MAX**2, 1/Z_DMAX**2, 1/PHI_MAX**2, 1/PHI_DMAX**2,
                     1/THETA_MAX**2, 1/THETA_DMAX**2, 1/PSI_MAX**2, 1/PSI_DMAX**2])
    R_d = np.array([1/F_MAX**2, 1/TAU_XY_MAX**2, 1/TAU_XY_MAX**2, 1/TAU_Z_MAX**2])
    P = la.solve_discrete_are(Ad, Bd, np.diag(Q_d), np.diag(R_d))
    K = np.linalg.inv(Bd.T@P@Bd + np.diag(R_d)) @ (Bd.T@P@Ad)
    u_eq = -np.linalg.pinv(Bd) @ d
    return Ad, Bd, d, K, u_eq

Ad_lin, Bd_lin, d_lin, K_LQR, U_EQ = _build_linear_model()
print(f"\n[LQR] u_eq[0] (hover thrust) = {U_EQ[0]:.6f} N  (mg = {M*G:.6f} N)")


# =====================================================================
# 4. Neural network policy
# =====================================================================

class PolicyMLP(nn.Module):
    """MLP controller: state (12) -> wrench (4).

    Initialized so that the output at x=0 is [mg, 0, 0, 0] (hover).
    This is critical for nonlinear training stability.
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
        self._init_hover()

    def _init_hover(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        hover_ratio = (M * G) / float(self.u_max[0])
        last.bias.data[0] = float(np.log(hover_ratio / (1.0 - hover_ratio)))

    def forward(self, x):
        x_n = x / self.x_scale
        raw = self.net(x_n)
        thrust  = torch.sigmoid(raw[..., [0]]) * self.u_max[0]
        torques = torch.tanh(raw[..., 1:])     * self.u_max[1:]
        return torch.cat([thrust, torques], dim=-1)


# =====================================================================
# 5. Rollout and cost
# =====================================================================

def rollout_nonlinear(policy, x0_12, T_steps, n_substeps=1):
    """Rollout with nonlinear quaternion dynamics.
    State is 13D internally, converted to 12D for NN and storage."""
    B, dev = x0_12.shape[0], x0_12.device
    X = torch.zeros(B, T_steps, 12, device=dev)
    U = torch.zeros(B, T_steps, 4,  device=dev)
    s13 = state12_to_state13(x0_12)

    for k in range(T_steps):
        s12 = state13_to_state12(s13)
        wrench = policy(s12)
        s13 = dynamics_one_ctrl_step(s13, wrench, n_substeps)
        X[:, k, :] = state13_to_state12(s13)
        U[:, k, :] = wrench
    return X, U


def rollout_linear(policy, x0_12, T_steps):
    """Rollout with linearized affine dynamics: x+ = Ad x + Bd u + d."""
    B, dev = x0_12.shape[0], x0_12.device
    Ad_t = torch.tensor(Ad_lin, dtype=torch.float32, device=dev)
    Bd_t = torch.tensor(Bd_lin, dtype=torch.float32, device=dev)
    d_t  = torch.tensor(d_lin,  dtype=torch.float32, device=dev).unsqueeze(0)

    X = torch.zeros(B, T_steps, 12, device=dev)
    U = torch.zeros(B, T_steps, 4,  device=dev)
    state = x0_12
    for k in range(T_steps):
        wrench = policy(state)
        state  = (state @ Ad_t.T) + (wrench @ Bd_t.T) + d_t
        X[:, k, :] = state
        U[:, k, :] = wrench
    return X, U


Q_DIAG = np.array([1/X_MAX**2, 1/X_DMAX**2, 1/Y_MAX**2, 1/Y_DMAX**2,
                    1/Z_MAX**2, 1/Z_DMAX**2, 1/PHI_MAX**2, 1/PHI_DMAX**2,
                    1/THETA_MAX**2, 1/THETA_DMAX**2, 1/PSI_MAX**2,
                    1/PSI_DMAX**2], dtype=np.float32)

R_DIAG = np.array([1/F_MAX**2, 1/TAU_XY_MAX**2,
                    1/TAU_XY_MAX**2, 1/TAU_Z_MAX**2], dtype=np.float32)


def traj_cost(X, U, terminal_weight=0.0):
    Q = torch.as_tensor(Q_DIAG, dtype=X.dtype, device=X.device)
    R = torch.as_tensor(R_DIAG, dtype=U.dtype, device=U.device)
    L = (X**2 * Q).sum(2).mean(0).sum() + (U**2 * R).sum(2).mean(0).sum()
    if terminal_weight > 0:
        L += terminal_weight * (X[:, -1, [0,2,4]]**2).mean()
    return L


# =====================================================================
# 6. Training
# =====================================================================

def make_x0_batch(xyz_list, device="cpu"):
    X0 = torch.zeros(len(xyz_list), 12, dtype=torch.float32, device=device)
    for b, (x, y, z) in enumerate(xyz_list):
        X0[b,0]=x; X0[b,2]=y; X0[b,4]=z
    return X0


def generate_cube_points(half_side=0.3):
    vals = [-half_side, 0.0, half_side]
    return list(itertools.product(vals, repeat=3))


def train(epochs=2000, lr=1e-3, hidden=64, terminal_weight=10.0,
          half_side=0.3, linearized=False, device="cpu"):
    policy = PolicyMLP(x_scale=X_SCALE, u_max=U_MAX, hidden=hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    x0_batch = make_x0_batch(generate_cube_points(half_side), device=device)

    with torch.no_grad():
        u0 = policy(torch.zeros(1, 12, device=device))
        print(f"\n[Init] NN(0) = F={u0[0,0]:.4f} N (mg={M*G:.4f}), tau={u0[0,1:]}")

    dyn = "LINEAR" if linearized else "NONLINEAR (quaternion, pybullet-exact)"
    print(f"[Train] {x0_batch.shape[0]} pts, {epochs} epochs, lr={lr}, dyn={dyn}")

    rollout_fn = rollout_linear if linearized else rollout_nonlinear

    for ep in range(epochs):
        X, U = rollout_fn(policy, x0_batch, T_STEPS)
        loss = traj_cost(X, U, terminal_weight)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if (ep+1) % 100 == 0 or ep == 0:
            with torch.no_grad():
                err = X[:, -1, [0,2,4]].norm(dim=1)
                print(f"  [ep {ep+1:4d}/{epochs}]  loss={loss.item():.4e}  "
                      f"|x_T|_mean={err.mean().item():.4f}  "
                      f"|x_T|_max={err.max().item():.4f}")
    return policy


# =====================================================================
# 7. Evaluation
# =====================================================================

@torch.no_grad()
def evaluate(policy, test_pts, device="cpu"):
    x0 = make_x0_batch(test_pts, device=device)

    # NN with nonlinear dynamics (5 substeps = pybullet exact)
    X_nn, _ = rollout_nonlinear(policy, x0, T_STEPS, n_substeps=PYB_STEPS_PER_CTRL)
    err_nn = X_nn[:, -1, [0,2,4]].norm(dim=1)
    print(f"\n[Eval] {len(test_pts)} points (nonlinear, {PYB_STEPS_PER_CTRL} substeps):")
    print(f"  NN   |x_T| mean={err_nn.mean().item():.5f} max={err_nn.max().item():.5f}")

    # LQR baseline (linear dynamics)
    Ad_t  = torch.tensor(Ad_lin, dtype=torch.float32, device=device)
    Bd_t  = torch.tensor(Bd_lin, dtype=torch.float32, device=device)
    d_t   = torch.tensor(d_lin,  dtype=torch.float32, device=device).unsqueeze(0)
    K_t   = torch.tensor(K_LQR,  dtype=torch.float32, device=device)
    ueq_t = torch.tensor(U_EQ,   dtype=torch.float32, device=device).unsqueeze(0)

    x = x0.clone()
    for k in range(T_STEPS):
        u = ueq_t - (x @ K_t.T)
        u[:,0] = torch.clamp(u[:,0], 0, F_MAX)
        u[:,1] = torch.clamp(u[:,1], -TAU_XY_MAX, TAU_XY_MAX)
        u[:,2] = torch.clamp(u[:,2], -TAU_XY_MAX, TAU_XY_MAX)
        u[:,3] = torch.clamp(u[:,3], -TAU_Z_MAX,  TAU_Z_MAX)
        x = (x @ Ad_t.T) + (u @ Bd_t.T) + d_t
    err_lqr = x[:, [0,2,4]].norm(dim=1)
    print(f"  LQR  |x_T| mean={err_lqr.mean().item():.5f} max={err_lqr.max().item():.5f} (linear dyn)")
    return X_nn


# =====================================================================
# 8. Main
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear", action="store_true",
                        help="Use linearized dynamics (faster)")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device}")

    policy = train(epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                   terminal_weight=10.0, half_side=0.3,
                   linearized=args.linear, device=device)

    rng = np.random.default_rng(42)
    test_pts = [(rng.uniform(-0.3,0.3), rng.uniform(-0.3,0.3),
                 rng.uniform(-0.3,0.3)) for _ in range(20)]
    evaluate(policy, test_pts, device=device)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save(policy.cpu(), os.path.join(out_dir, "trained_policy_cf2x.pt"))
    torch.save({
        "state_dict": policy.state_dict(),
        "hidden": args.hidden,
        "x_scale": X_SCALE, "u_max": U_MAX,
        "ctrl_freq": CTRL_FREQ, "Ts": DT_CTRL,
        "KF": KF, "KM": KM, "L": L, "M": M, "G": G,
        "MAX_RPM": MAX_RPM, "linearized": args.linear,
    }, os.path.join(out_dir, "trained_weights_cf2x.pt"))

    print(f"\n[Saved] trained_policy_cf2x.pt + trained_weights_cf2x.pt")
    print("Done! Run  validate_nn_pybullet.py  to test in pybullet-drones.")