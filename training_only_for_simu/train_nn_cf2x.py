#!/usr/bin/env python3
"""
Train a neural network controller for the Crazyflie 2.x (CF2X) quadcopter.

Reproduces Goffin's section 4.1 sanity check (regulation to origin) but with:
  - CF2X physical parameters from gym-pybullet-drones
  - Nonlinear dynamics matching pybullet's DYN mode exactly:
      * full rotation matrix (no small-angle approximation for translation)
      * gyroscopic coupling  tau_net = tau - omega x (J*omega)
      * semi-implicit Euler integration (velocity first, then position)
      * same sub-stepping structure (5 sub-steps at 1/240 s per ctrl step)

The NN outputs absolute wrench [F, tau_x, tau_y, tau_z].

Usage:
    cd training_only_for_simu
    python train_nn_cf2x.py                          # nonlinear dynamics (default)
    python train_nn_cf2x.py --linear                 # linearized dynamics (faster)
    python train_nn_cf2x.py --epochs 3000 --lr 5e-4
    python train_nn_cf2x.py --tag exp1
"""

import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import itertools, os


# =====================================================================
# 1. CF2X Physical Parameters  (from cf2x.urdf in gym-pybullet-drones)
# =====================================================================

# ---------- Identical to the cf2x.urdf values ----------
M       = 0.027                         # mass [kg]
G       = 9.8                           # gravity [m/s^2]  (pybullet uses 9.8)
I_X     = 1.4e-5                        # Ixx [kg*m^2]
I_Y     = 1.4e-5                        # Iyy [kg*m^2]
I_Z     = 2.17e-5                       # Izz [kg*m^2]
L       = 0.0397                        # arm length [m]
KF      = 3.16e-10                      # thrust coeff [N/RPM^2]
KM      = 7.94e-12                      # torque coeff [N*m/RPM^2]
T2W     = 2.25                          # thrust-to-weight ratio

# ---------- Derived limits (same computation as BaseAviary.__init__) ----------
GRAVITY     = M * G                                         # weight [N]
HOVER_RPM   = np.sqrt(GRAVITY / (4 * KF))
MAX_RPM     = np.sqrt((T2W * GRAVITY) / (4 * KF))
F_MAX       = 4 * KF * MAX_RPM**2                          # max thrust [N]
TAU_XY_MAX  = (2 * L * KF * MAX_RPM**2) / np.sqrt(2)      # couple roll/pitch max
TAU_Z_MAX   = 2 * KM * MAX_RPM**2                          # couple yaw max

U_MAX = np.array([F_MAX, TAU_XY_MAX, TAU_XY_MAX, TAU_Z_MAX], dtype=np.float32)

# ---------- State scaling ----------
# Used to:
#   a) normalize the NN inputs  (the network receives x / x_scale)
#   b) weight the cost matrix Q (Q_ii = 1 / x_scale_i^2)
#
# These are hyperparameters. We choose them as the maximum
# "acceptable" deviations for each state component. They will need tuning.
X_MAX       = 1.0                   # position max acceptable [m]
X_DMAX      = 1.0                   # max acceptable velocity [m/s]
Y_MAX       = 1.0
Y_DMAX      = 1.0
Z_MAX       = 1.0
Z_DMAX      = 1.0
PHI_MAX     = np.deg2rad(30)        # max acceptable tilt
PHI_DMAX    = np.deg2rad(200)       # max acceptable angular rate
THETA_MAX   = np.deg2rad(30)
THETA_DMAX  = np.deg2rad(200)
PSI_MAX     = np.deg2rad(45)
PSI_DMAX    = np.deg2rad(120)

X_SCALE = np.array([X_MAX, X_DMAX, Y_MAX, Y_DMAX, Z_MAX, Z_DMAX,
                     PHI_MAX, PHI_DMAX, THETA_MAX, THETA_DMAX,
                     PSI_MAX, PSI_DMAX], dtype=np.float32)

# ---------- Timing  (identical to pybullet) ----------
PYB_FREQ   = 240                    # pybullet physics frequency [Hz]
CTRL_FREQ  = 48                     # control frequency [Hz]
PYB_STEPS_PER_CTRL = PYB_FREQ // CTRL_FREQ   # = 5 sub-steps per control step
DT_PYB     = 1.0 / PYB_FREQ        # = 1/240 s  (time step of each sub-step)
DT_CTRL    = 1.0 / CTRL_FREQ       # = 1/48 s   (interval between two NN actions)
T_SIM      = 4.0                    # simulation duration [s]
T_STEPS    = int(T_SIM / DT_CTRL)  # = 192 control steps

print(f"[CF2X] m={M}, g={G}, Ixx={I_X}, Iyy={I_Y}, Izz={I_Z}")
print(f"[CF2X] F_max={F_MAX:.4f} N, tau_xy_max={TAU_XY_MAX:.6f} N*m, tau_z_max={TAU_Z_MAX:.6f} N*m")
print(f"[CF2X] HOVER_RPM={HOVER_RPM:.1f}, MAX_RPM={MAX_RPM:.1f}")
print(f"[CF2X] DT_pyb={DT_PYB:.6f}s, DT_ctrl={DT_CTRL:.5f}s, "
      f"sub-steps={PYB_STEPS_PER_CTRL}, T_steps={T_STEPS}")


# =====================================================================
# 2. Nonlinear dynamics (replicates pybullet DYN mode)
# =====================================================================
#
# Reference: BaseAviary._dynamics()   (lines 815-877)
#
# The state is:  [x, xdot, y, ydot, z, zdot, phi, p, theta, q, psi, r]
#
# where p,q,r are body-frame angular rates (= rpy_rates in pybullet).
# Near hover, p~=phi_dot, q~=theta_dot, r~=psi_dot.
#
# The input is:  u = [F, tau_x, tau_y, tau_z]   (wrench)
#
# Three physics blocks reproduced faithfully:
#
#   (a) Rotation of thrust into the world frame:
#         R(phi,theta,psi) @ [0, 0, F]
#       Identical to pybullet  "thrust_world_frame = np.dot(rotation, thrust)"
#       except that pybullet obtains R from the quaternion while we use Euler angles.
#       The rotation matrix R is the same (intrinsic ZYX convention).
#
#   (b) Gyroscopic coupling:
#         tau_net = [tau_x,tau_y,tau_z] - omega x (J*omega)
#       Identical to pybullet  "torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))"
#
#   (c) Semi-implicit (symplectic) Euler integration:
#         v  <- v  + dt*a          (velocity updated FIRST)
#         omega <- omega + dt*omega_dot
#         pos <- pos + dt*v        (position updated WITH the new velocity)
#         rpy <- rpy + dt*omega    (same for orientation)
#       Identical to pybullet lines 860-863.
#
#   (d) Sub-step structure:
#         For each control step, we perform PYB_STEPS_PER_CTRL sub-steps
#         at dt = 1/240 s, with the same wrench held constant.
#       Identical to the loop "for _ in range(self.PYB_STEPS_PER_CTRL)"
#
# NOTE: the only difference is the orientation integration:
#   - pybullet uses quaternions (_integrateQ)
#   - we use rpy <- rpy + dt*omega
#   These two approaches are first-order equivalent and give the
#   same results near hover. For aggressive maneuvers
#   (> 60 deg of tilt), quaternions would be more accurate.


def rotation_matrix_zyx(phi, theta, psi):
    """Intrinsic ZYX rotation matrix (= extrinsic XYZ).

    Identical to p.getMatrixFromQuaternion(p.getQuaternionFromEuler([phi,theta,psi]))
    near hover.

    Parameters: tensors of shape (B,) or scalars.
    Returns:    tensor of shape (B, 3, 3).
    """
    cphi = torch.cos(phi);   sphi = torch.sin(phi)
    cth  = torch.cos(theta); sth  = torch.sin(theta)
    cpsi = torch.cos(psi);   spsi = torch.sin(psi)

    # Row 1
    r00 = cth * cpsi
    r01 = sphi * sth * cpsi - cphi * spsi
    r02 = cphi * sth * cpsi + sphi * spsi
    # Row 2
    r10 = cth * spsi
    r11 = sphi * sth * spsi + cphi * cpsi
    r12 = cphi * sth * spsi - sphi * cpsi
    # Row 3
    r20 = -sth
    r21 = sphi * cth
    r22 = cphi * cth

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)  # (B, 3, 3)
    return R


def dynamics_substep(state, wrench, dt):
    """One semi-implicit Euler integration sub-step.

    Exactly replicates BaseAviary._dynamics() lines 838-863.

    Parameters
    ----------
    state  : (B, 12)  [x, vx, y, vy, z, vz, phi, p, theta, q, psi, r]
    wrench : (B, 4)   [F, tau_x, tau_y, tau_z]
    dt     : float    time step (= 1/240 s)

    Returns
    -------
    state_new : (B, 12)
    """
    # --- Unpack the state ---
    x     = state[:, 0];  vx = state[:, 1]
    y     = state[:, 2];  vy = state[:, 3]
    z     = state[:, 4];  vz = state[:, 5]
    phi   = state[:, 6];  p  = state[:, 7]     # p = rpy_rates[0]
    theta = state[:, 8];  q  = state[:, 9]     # q = rpy_rates[1]
    psi   = state[:, 10]; r  = state[:, 11]    # r = rpy_rates[2]

    F     = wrench[:, 0]
    tau_x = wrench[:, 1]
    tau_y = wrench[:, 2]
    tau_z = wrench[:, 3]

    # --- (a) Thrust in the world frame ---
    # pybullet : thrust = [0, 0, F]
    #            thrust_world = R @ thrust
    #            force_world = thrust_world - [0, 0, GRAVITY]
    #            acc = force_world / M
    R = rotation_matrix_zyx(phi, theta, psi)  # (B, 3, 3)

    # R @ [0, 0, F]  =  F * R[:, :, 2]   (3rd column of R)
    thrust_world = F.unsqueeze(-1) * R[:, :, 2]  # (B, 3)

    ax = (thrust_world[:, 0]) / M
    ay = (thrust_world[:, 1]) / M
    az = (thrust_world[:, 2]) / M - G      # gravity: -[0,0,GRAVITY]/M = -g

    # --- (b) Gyroscopic coupling ---
    # pybullet : torques = [tau_x, tau_y, tau_z] - cross(omega, J @ omega)
    #            omega_dot = J_inv @ torques
    #
    # cross(omega, J*omega) with diagonal J:
    #   [p]     [Ix*p]     [q*Iz*r - r*Iy*q]     [(Iz-Iy)*q*r]
    #   [q]  x  [Iy*q]  =  [r*Ix*p - p*Iz*r]  =  [(Ix-Iz)*p*r]
    #   [r]     [Iz*r]     [p*Iy*q - q*Ix*p]     [(Iy-Ix)*p*q]
    gyro_x = (I_Z - I_Y) * q * r
    gyro_y = (I_X - I_Z) * p * r
    gyro_z = (I_Y - I_X) * p * q

    p_dot = (tau_x - gyro_x) / I_X
    q_dot = (tau_y - gyro_y) / I_Y
    r_dot = (tau_z - gyro_z) / I_Z

    # --- (c) Semi-implicit Euler integration ---
    # Step 1: update the velocities
    vx_new = vx + dt * ax
    vy_new = vy + dt * ay
    vz_new = vz + dt * az
    p_new  = p  + dt * p_dot
    q_new  = q  + dt * q_dot
    r_new  = r  + dt * r_dot

    # Step 2: update the positions WITH THE NEW velocities
    x_new     = x     + dt * vx_new
    y_new     = y     + dt * vy_new
    z_new     = z     + dt * vz_new
    phi_new   = phi   + dt * p_new
    theta_new = theta + dt * q_new
    psi_new   = psi   + dt * r_new

    state_new = torch.stack([
        x_new, vx_new, y_new, vy_new, z_new, vz_new,
        phi_new, p_new, theta_new, q_new, psi_new, r_new
    ], dim=-1)
    return state_new


def dynamics_one_ctrl_step(state, wrench, n_substeps=PYB_STEPS_PER_CTRL):
    """Advance by one full control step.

    Parameters
    ----------
    n_substeps : int
        Number of Euler sub-steps.
        - n_substeps=5, dt=1/240  : exactly replicates pybullet (for validation)
        - n_substeps=1, dt=1/48   : 5x faster (for training)
        Both give similar results because dt=1/48 remains small.
        (Goffin used dt=0.1s with Euler in his nonlinear experiments!)
    """
    dt = DT_CTRL / n_substeps  # duration of each sub-step
    for _ in range(n_substeps):
        state = dynamics_substep(state, wrench, dt)
    return state


# =====================================================================
# 3. LQR baseline (for comparison - uses the linearized model)
# =====================================================================
# We linearize only to compute the LQR gain as a baseline.
# NN training uses the nonlinear dynamics above.

def _build_linear_model_for_lqr():
    """Linearized model + ZOH discretization (only for the LQR baseline)."""
    from scipy.signal import cont2discrete

    A = np.zeros((12, 12))
    A[0,1]=1; A[2,3]=1; A[4,5]=1; A[6,7]=1; A[8,9]=1; A[10,11]=1
    A[1,8] = G;  A[3,6] = -G
    B = np.zeros((12, 4))
    B[5,0]=1/M; B[7,1]=1/I_X; B[9,2]=1/I_Y; B[11,3]=1/I_Z

    c = np.zeros((12,1)); c[5,0] = -G
    n, m = 12, 4
    A_aug = np.zeros((n+1,n+1)); A_aug[:n,:n]=A; A_aug[:n,n]=c.squeeze()
    B_aug = np.zeros((n+1,m)); B_aug[:n,:]=B
    C_aug = np.zeros((n,n+1)); C_aug[:,:n]=np.eye(n)
    D_aug = np.zeros((n,m))
    Ad_aug,Bd_aug,_,_,_ = cont2discrete((A_aug,B_aug,C_aug,D_aug), DT_CTRL, method='zoh')
    Ad = Ad_aug[:n,:n]; Bd = Bd_aug[:n,:]; d = Ad_aug[:n,n]

    Q_diag = np.array([1/X_MAX**2, 1/X_DMAX**2, 1/Y_MAX**2, 1/Y_DMAX**2,
                        1/Z_MAX**2, 1/Z_DMAX**2, 1/PHI_MAX**2, 1/PHI_DMAX**2,
                        1/THETA_MAX**2, 1/THETA_DMAX**2, 1/PSI_MAX**2, 1/PSI_DMAX**2])
    R_diag = np.array([1/F_MAX**2, 1/TAU_XY_MAX**2, 1/TAU_XY_MAX**2, 1/TAU_Z_MAX**2])
    P = la.solve_discrete_are(Ad, Bd, np.diag(Q_diag), np.diag(R_diag))
    K = np.linalg.inv(Bd.T@P@Bd + np.diag(R_diag)) @ (Bd.T@P@Ad)
    u_eq = -np.linalg.pinv(Bd) @ d
    return K, u_eq, Ad, Bd, d

K_LQR, U_EQ, Ad_lqr, Bd_lqr, d_lqr = _build_linear_model_for_lqr()
print(f"\n[LQR] u_eq[0] (hover thrust) = {U_EQ[0]:.6f} N  (mg = {M*G:.6f} N)")


# =====================================================================
# 3b. Differentiable linear (affine) rollout
# =====================================================================
# x_{k+1} = Ad x_k + Bd u_k + d
# Same matrices as for LQR, but u comes from the NN instead of -Kx.

def rollout_linear(policy, x0, T_steps):
    """Rollout with affine linear dynamics (exact ZOH)."""
    B = x0.shape[0]
    dev = x0.device
    Ad_t = torch.tensor(Ad_lqr, dtype=torch.float32, device=dev)
    Bd_t = torch.tensor(Bd_lqr, dtype=torch.float32, device=dev)
    d_t  = torch.tensor(d_lqr,  dtype=torch.float32, device=dev).unsqueeze(0)

    X = torch.zeros(B, T_steps, 12, device=dev)
    U = torch.zeros(B, T_steps, 4,  device=dev)
    state = x0

    for k in range(T_steps):
        wrench = policy(state)
        state  = (state @ Ad_t.T) + (wrench @ Bd_t.T) + d_t
        X[:, k, :] = state
        U[:, k, :] = wrench
    return X, U


# =====================================================================
# 4. Neural network
# =====================================================================

class PolicyMLP(nn.Module):
    """MLP controller: state (12) -> wrench (4).

    Thrust  -> sigmoid -> [0, F_max]
    Torques -> tanh    -> [-tau_max, +tau_max]

    CRITICAL INITIALIZATION:
    The last layer is initialized with weight=0, bias=[logit(mg/Fmax), 0, 0, 0].
    Therefore, regardless of the input, the initial output is:
        thrust  = sigmoid(logit(mg/Fmax)) * Fmax = mg    (hover)
        torques = tanh(0) * tau_max = 0                   (no rotation)
    The drone therefore starts in hover, and the gradient can guide
    learning from this stable equilibrium point.
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
        """Initialize the last layer to output the hover wrench."""
        last_layer = self.net[-1]   # nn.Linear(hidden, 4)
        # Zero weights: the output does not depend on the input initially
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        # Thrust bias: sigmoid(b) * Fmax = mg  =>  b = logit(mg/Fmax)
        hover_ratio = (M * G) / float(self.u_max[0])  # mg / Fmax ≈ 0.444
        last_layer.bias.data[0] = float(np.log(hover_ratio / (1.0 - hover_ratio)))
        # Torque biases = 0: tanh(0) = 0, no torque

    def forward(self, x):
        x_n = x / self.x_scale
        raw = self.net(x_n)
        thrust  = torch.sigmoid(raw[..., [0]]) * self.u_max[0]
        torques = torch.tanh(raw[..., 1:])     * self.u_max[1:]
        return torch.cat([thrust, torques], dim=-1)


# =====================================================================
# 5. Differentiable rollout (nonlinear) and cost
# =====================================================================

def rollout(policy, x0, T_steps, n_substeps=1):
    """Roll out the policy for T_steps control steps with nonlinear dynamics.

    At each control step:
      1. The NN chooses wrench = policy(state)
      2. We integrate n_substeps nonlinear dynamics sub-steps
         while keeping the wrench constant.

    Parameters
    ----------
    policy     : PolicyMLP
    x0         : (B, 12) initial states
    T_steps    : int  number of control steps
    n_substeps : int  sub-steps per control step (1=fast, 5=exact pybullet)
    """
    B = x0.shape[0]
    dev = x0.device
    X = torch.zeros(B, T_steps, 12, device=dev)
    U = torch.zeros(B, T_steps, 4,  device=dev)
    state = x0

    for k in range(T_steps):
        wrench = policy(state)
        state  = dynamics_one_ctrl_step(state, wrench, n_substeps)
        X[:, k, :] = state
        U[:, k, :] = wrench
    return X, U


# --- Cost matrix ---
Q_DIAG = np.array([
    1/X_MAX**2,     1/X_DMAX**2,
    1/Y_MAX**2,     1/Y_DMAX**2,
    1/Z_MAX**2,     1/Z_DMAX**2,
    1/PHI_MAX**2,   1/PHI_DMAX**2,
    1/THETA_MAX**2, 1/THETA_DMAX**2,
    1/PSI_MAX**2,   1/PSI_DMAX**2,
], dtype=np.float32)

R_DIAG = np.array([
    1/F_MAX**2,
    1/TAU_XY_MAX**2,
    1/TAU_XY_MAX**2,
    1/TAU_Z_MAX**2,
], dtype=np.float32)


def traj_cost(X, U, terminal_weight=0.0):
    """Quadratic cost: sum_k (x'Qx + u'Ru), averaged over the batch."""
    Q = torch.as_tensor(Q_DIAG, dtype=X.dtype, device=X.device)
    R = torch.as_tensor(R_DIAG, dtype=U.dtype, device=U.device)

    cost_x = (X**2 * Q).sum(dim=2).mean(dim=0).sum()
    cost_u = (U**2 * R).sum(dim=2).mean(dim=0).sum()
    L = cost_x + cost_u

    if terminal_weight > 0.0:
        pT = X[:, -1, [0, 2, 4]]   # terminal [x, y, z]
        L += terminal_weight * (pT**2).mean()
    return L


# =====================================================================
# 6. Training
# =====================================================================

def make_x0_batch(xyz_list, device="cpu"):
    """Convert a list of (x,y,z) into a tensor (B, 12) of initial states.
    All velocities and angles are zero."""
    X0 = torch.zeros(len(xyz_list), 12, dtype=torch.float32, device=device)
    for b, (x, y, z) in enumerate(xyz_list):
        X0[b, 0] = x
        X0[b, 2] = y
        X0[b, 4] = z
    return X0


def generate_cube_points(half_side=0.3):
    """27 points: vertices + face centers + edge centers + center."""
    vals = [-half_side, 0.0, half_side]
    return list(itertools.product(vals, repeat=3))


def train(epochs=2000, lr=1e-3, hidden=64, terminal_weight=10.0,
          half_side=0.3, linearized=False, device="cpu"):
    """Train the NN on the regulation-to-origin task.

    Parameters
    ----------
    linearized : bool
        True  = affine linear dynamics (ZOH, fast)
        False = nonlinear dynamics (replicates pybullet DYN, 1 sub-step)
    """
    policy = PolicyMLP(x_scale=X_SCALE, u_max=U_MAX, hidden=hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    cube_pts = generate_cube_points(half_side)
    x0_batch = make_x0_batch(cube_pts, device=device)

    # Check hover initialization
    with torch.no_grad():
        u_test = policy(torch.zeros(1, 12, device=device))
        print(f"\n[Init] NN output at x=0: F={u_test[0,0]:.4f} N (mg={M*G:.4f}), "
              f"tau={u_test[0,1:]}")

    dyn_label = "LINEAIRE" if linearized else "NON-LINEAIRE"
    print(f"[Train] {len(cube_pts)} pts, epochs={epochs}, lr={lr}")
    print(f"[Train] dynamique {dyn_label}, horizon={T_STEPS} steps ({T_SIM}s)")

    # Choose the rollout function
    rollout_fn = rollout_linear if linearized else rollout

    for ep in range(epochs):
        X, U = rollout_fn(policy, x0_batch, T_STEPS)
        loss = traj_cost(X, U, terminal_weight)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        scheduler.step()

        if (ep + 1) % 100 == 0 or ep == 0:
            with torch.no_grad():
                pos_end = X[:, -1, [0, 2, 4]].norm(dim=1)
                print(f"  [ep {ep+1:4d}/{epochs}]  loss={loss.item():.4e}  "
                      f"|x_T|_mean={pos_end.mean().item():.4f}  "
                      f"|x_T|_max={pos_end.max().item():.4f}")

    return policy


# =====================================================================
# 7. Evaluation (NN vs LQR)
# =====================================================================

@torch.no_grad()
def evaluate(policy, test_pts, device="cpu"):
    """Evaluate the NN and the LQR on test positions."""

    x0 = make_x0_batch(test_pts, device=device)

    # --- NN (nonlinear dynamics, 5 sub-steps = exact pybullet) ---
    X_nn, U_nn = rollout(policy, x0, T_STEPS, n_substeps=PYB_STEPS_PER_CTRL)
    err_nn = X_nn[:, -1, [0, 2, 4]].norm(dim=1)
    print(f"\n[Eval] {len(test_pts)} test points (non-lineaire, {PYB_STEPS_PER_CTRL} sous-pas):")
    print(f"  NN   terminal error  mean={err_nn.mean().item():.5f} m  "
          f"max={err_nn.max().item():.5f} m")

    # --- LQR (linear dynamics, for baseline) ---
    Ad_t  = torch.tensor(Ad_lqr, dtype=torch.float32, device=device)
    Bd_t  = torch.tensor(Bd_lqr, dtype=torch.float32, device=device)
    d_t   = torch.tensor(d_lqr,  dtype=torch.float32, device=device).unsqueeze(0)
    K_t   = torch.tensor(K_LQR,  dtype=torch.float32, device=device)
    ueq_t = torch.tensor(U_EQ,   dtype=torch.float32, device=device).unsqueeze(0)

    x = x0.clone()
    X_lqr = torch.zeros_like(X_nn)
    for k in range(T_STEPS):
        u = ueq_t - (x @ K_t.T)
        u[:, 0] = torch.clamp(u[:, 0], 0, F_MAX)
        u[:, 1] = torch.clamp(u[:, 1], -TAU_XY_MAX, TAU_XY_MAX)
        u[:, 2] = torch.clamp(u[:, 2], -TAU_XY_MAX, TAU_XY_MAX)
        u[:, 3] = torch.clamp(u[:, 3], -TAU_Z_MAX,  TAU_Z_MAX)
        x = (x @ Ad_t.T) + (u @ Bd_t.T) + d_t
        X_lqr[:, k, :] = x
    err_lqr = X_lqr[:, -1, [0, 2, 4]].norm(dim=1)
    print(f"  LQR  terminal error  mean={err_lqr.mean().item():.5f} m  "
          f"max={err_lqr.max().item():.5f} m  (dynamique lineaire)")

    return X_nn, U_nn


def build_run_name(linearized, epochs, lr, hidden, tag=""):
    """Build a descriptive run name for saved models/checkpoints."""
    dyn = "linear" if linearized else "nonlinear"
    lr_str = f"{lr:.0e}" if lr < 1e-2 else str(lr).replace(".", "p")
    name = f"cf2x_{dyn}_h{hidden}_ep{epochs}_lr{lr_str}"
    if tag:
        safe_tag = str(tag).replace(" ", "_")
        name += f"_{safe_tag}"
    return name


# =====================================================================
# 8. Main
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear", action="store_true",
                        help="Use linearized dynamics (faster, less accurate)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix added to saved filenames")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device}")

    policy = train(epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                   terminal_weight=10.0, half_side=0.3,
                   linearized=args.linear, device=device)

    # Test on 20 random points
    rng = np.random.default_rng(42)
    test_pts = [(rng.uniform(-0.3, 0.3),
                 rng.uniform(-0.3, 0.3),
                 rng.uniform(-0.3, 0.3)) for _ in range(20)]
    evaluate(policy, test_pts, device=device)





    # Save
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # On définit le nom du nouveau dossier (ex: "checkpoints" ou "saved_models")
    out_dir = os.path.join(base_dir, "saved_policies_and_weights") 
    
    # On crée le dossier s'il n'existe pas déjà
    os.makedirs(out_dir, exist_ok=True)

    run_name = build_run_name(args.linear, args.epochs, args.lr, args.hidden, args.tag)

    path_full = os.path.join(out_dir, f"trained_policy_{run_name}.pt")
    path_dict = os.path.join(out_dir, f"trained_weights_{run_name}.pt")

    torch.save(policy.cpu(), path_full)
    torch.save({
        "state_dict": policy.state_dict(),
        "hidden": args.hidden,
        "x_scale": X_SCALE,
        "u_max": U_MAX,
        "ctrl_freq": CTRL_FREQ,
        "Ts": DT_CTRL,
        "KF": KF, "KM": KM, "L": L, "M": M, "G": G,
        "MAX_RPM": MAX_RPM,
        "linearized": args.linear,
        "run_name": run_name,
        "train_dynamics": "linear" if args.linear else "nonlinear",
        "epochs": args.epochs,
        "lr": args.lr,
        "target_definition": "error state regulated to origin",
    }, path_dict)

    print(f"\n[Saved] {path_full}")
    print(f"[Saved] {path_dict}")
    print("\nDone! Run  validate_nn_pybullet.py --weights <checkpoint_name>  to test in pybullet-drones.")
