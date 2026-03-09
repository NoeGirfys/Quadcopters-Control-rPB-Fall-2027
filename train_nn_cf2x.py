#!/usr/bin/env python3
"""
Train a neural network controller for the Crazyflie 2.x (CF2X) quadcopter.

Three dynamics modes:
  --linear              Affine linearized model, ZOH discretization
  --no-quaternions      Nonlinear with rotation matrix, rpy += dt*omega
  (default)             Nonlinear with quaternion integration (= pybullet DYN)

Usage:
    python train_nn_cf2x.py                          # nonlinear + quaternions
    python train_nn_cf2x.py --no-quaternions          # nonlinear, no quaternions
    python train_nn_cf2x.py --linear                  # linearized
    python train_nn_cf2x.py --epochs 3000 --lr 5e-4
"""

import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import itertools, os

# =====================================================================
# 1. CF2X Physical Parameters  (from cf2x.urdf)
# =====================================================================

M       = 0.027;     G    = 9.8
I_X     = 1.4e-5;    I_Y  = 1.4e-5;   I_Z = 2.17e-5
L       = 0.0397;    KF   = 3.16e-10;  KM  = 7.94e-12;  T2W = 2.25

GRAVITY    = M * G
HOVER_RPM  = np.sqrt(GRAVITY / (4*KF))
MAX_RPM    = np.sqrt((T2W*GRAVITY) / (4*KF))
F_MAX      = 4*KF*MAX_RPM**2
TAU_XY_MAX = (2*L*KF*MAX_RPM**2) / np.sqrt(2)
TAU_Z_MAX  = 2*KM*MAX_RPM**2
U_MAX      = np.array([F_MAX, TAU_XY_MAX, TAU_XY_MAX, TAU_Z_MAX], dtype=np.float32)

X_MAX=1.0; X_DMAX=1.0; Y_MAX=1.0; Y_DMAX=1.0; Z_MAX=1.0; Z_DMAX=1.0
PHI_MAX=np.deg2rad(30); PHI_DMAX=np.deg2rad(200)
THETA_MAX=np.deg2rad(30); THETA_DMAX=np.deg2rad(200)
PSI_MAX=np.deg2rad(45); PSI_DMAX=np.deg2rad(120)
X_SCALE = np.array([X_MAX,X_DMAX, Y_MAX,Y_DMAX, Z_MAX,Z_DMAX,
                     PHI_MAX,PHI_DMAX, THETA_MAX,THETA_DMAX,
                     PSI_MAX,PSI_DMAX], dtype=np.float32)

PYB_FREQ = 240;  CTRL_FREQ = 48
PYB_STEPS_PER_CTRL = PYB_FREQ // CTRL_FREQ
DT_PYB  = 1.0/PYB_FREQ;  DT_CTRL = 1.0/CTRL_FREQ
T_SIM   = 4.0;  T_STEPS = int(T_SIM/DT_CTRL)

print(f"[CF2X] m={M}, g={G}, Ixx={I_X}, Iyy={I_Y}, Izz={I_Z}")
print(f"[CF2X] F_max={F_MAX:.4f}, tau_xy={TAU_XY_MAX:.6f}, tau_z={TAU_Z_MAX:.6f}")
print(f"[CF2X] DT_ctrl={DT_CTRL:.5f}s, substeps={PYB_STEPS_PER_CTRL}, T_steps={T_STEPS}")

# =====================================================================
# 2. Quaternion utilities
# =====================================================================

def quat_to_rotmat(qx, qy, qz, qw):
    xx=qx*qx; yy=qy*qy; zz=qz*qz
    xy=qx*qy; xz=qx*qz; yz=qy*qz
    wx=qw*qx; wy=qw*qy; wz=qw*qz
    return torch.stack([
        torch.stack([1-2*(yy+zz), 2*(xy-wz),   2*(xz+wy)],  dim=-1),
        torch.stack([2*(xy+wz),   1-2*(xx+zz), 2*(yz-wx)],   dim=-1),
        torch.stack([2*(xz-wy),   2*(yz+wx),   1-2*(xx+yy)], dim=-1),
    ], dim=-2)

def quat_to_euler(qx, qy, qz, qw):
    phi   = torch.atan2(2*(qw*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    theta = torch.asin(torch.clamp(2*(qw*qy-qz*qx), -1.0, 1.0))
    psi   = torch.atan2(2*(qw*qz+qx*qy), 1-2*(qy*qy+qz*qz))
    return phi, theta, psi

def euler_to_quat(phi, theta, psi):
    cy=torch.cos(psi*0.5);   sy=torch.sin(psi*0.5)
    cp=torch.cos(theta*0.5); sp=torch.sin(theta*0.5)
    cr=torch.cos(phi*0.5);   sr=torch.sin(phi*0.5)
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    qw = cr*cp*cy + sr*sp*sy
    return qx, qy, qz, qw

def integrate_quat(qx, qy, qz, qw, p, q, r, dt):
    """Exact quaternion integration — replicates BaseAviary._integrateQ()."""
    on = torch.sqrt(p*p + q*q + r*r + 1e-12)
    ha = on * dt * 0.5
    c = torch.cos(ha);  s = torch.sin(ha) / on
    qx_n = c*qx + s*( r*qy - q*qz + p*qw)
    qy_n = c*qy + s*(-r*qx + p*qz + q*qw)
    qz_n = c*qz + s*( q*qx - p*qy + r*qw)
    qw_n = c*qw + s*(-p*qx - q*qy - r*qz)
    nm = torch.sqrt(qx_n**2 + qy_n**2 + qz_n**2 + qw_n**2 + 1e-12)
    return qx_n/nm, qy_n/nm, qz_n/nm, qw_n/nm

def rotation_matrix_euler(phi, theta, psi):
    """Rotation matrix from Euler angles (for no-quaternion mode)."""
    cphi=torch.cos(phi); sphi=torch.sin(phi)
    cth=torch.cos(theta); sth=torch.sin(theta)
    cpsi=torch.cos(psi); spsi=torch.sin(psi)
    return torch.stack([
        torch.stack([cth*cpsi, sphi*sth*cpsi-cphi*spsi, cphi*sth*cpsi+sphi*spsi], dim=-1),
        torch.stack([cth*spsi, sphi*sth*spsi+cphi*cpsi, cphi*sth*spsi-sphi*cpsi], dim=-1),
        torch.stack([-sth,     sphi*cth,                 cphi*cth],                dim=-1),
    ], dim=-2)

# =====================================================================
# 3. Nonlinear dynamics — state is always 12D externally
# =====================================================================
# State: [x, vx, y, vy, z, vz, phi, p, theta, q, psi, r]
# The quaternion conversion is hidden inside dynamics_substep_quat.

def dynamics_substep_quat(s12, wrench, dt):
    """One sub-step with quaternion integration (= pybullet DYN exact).
    Takes (B,12), returns (B,12). Quaternion conversion is internal."""
    x=s12[:,0]; vx=s12[:,1]; y=s12[:,2]; vy=s12[:,3]
    z=s12[:,4]; vz=s12[:,5]
    phi=s12[:,6]; p=s12[:,7]; theta=s12[:,8]; q=s12[:,9]; psi=s12[:,10]; r=s12[:,11]
    F=wrench[:,0]; tau_x=wrench[:,1]; tau_y=wrench[:,2]; tau_z=wrench[:,3]

    # Euler -> quaternion -> rotation matrix (same as pybullet's getMatrixFromQuaternion)
    qx, qy, qz, qw = euler_to_quat(phi, theta, psi)
    R = quat_to_rotmat(qx, qy, qz, qw)

    # Thrust world frame
    tw = F.unsqueeze(-1) * R[:,:,2]
    ax=tw[:,0]/M; ay=tw[:,1]/M; az=tw[:,2]/M - G

    # Gyroscopic coupling
    p_dot = (tau_x - (I_Z-I_Y)*q*r) / I_X
    q_dot = (tau_y - (I_X-I_Z)*p*r) / I_Y
    r_dot = (tau_z - (I_Y-I_X)*p*q) / I_Z

    # Semi-implicit Euler: velocities first
    vx_n=vx+dt*ax; vy_n=vy+dt*ay; vz_n=vz+dt*az
    p_n=p+dt*p_dot; q_n=q+dt*q_dot; r_n=r+dt*r_dot

    # Positions with NEW velocities
    x_n=x+dt*vx_n; y_n=y+dt*vy_n; z_n=z+dt*vz_n

    # Quaternion integration with NEW omega, then back to Euler
    qx_n, qy_n, qz_n, qw_n = integrate_quat(qx, qy, qz, qw, p_n, q_n, r_n, dt)
    phi_n, theta_n, psi_n = quat_to_euler(qx_n, qy_n, qz_n, qw_n)

    return torch.stack([x_n,vx_n, y_n,vy_n, z_n,vz_n,
                        phi_n,p_n, theta_n,q_n, psi_n,r_n], dim=-1)


def dynamics_substep_simple(s12, wrench, dt):
    """One sub-step WITHOUT quaternions: same physics but rpy += dt*omega.
    Takes (B,12), returns (B,12)."""
    x=s12[:,0]; vx=s12[:,1]; y=s12[:,2]; vy=s12[:,3]
    z=s12[:,4]; vz=s12[:,5]
    phi=s12[:,6]; p=s12[:,7]; theta=s12[:,8]; q=s12[:,9]; psi=s12[:,10]; r=s12[:,11]
    F=wrench[:,0]; tau_x=wrench[:,1]; tau_y=wrench[:,2]; tau_z=wrench[:,3]

    # Rotation matrix from Euler angles directly
    R = rotation_matrix_euler(phi, theta, psi)

    # Thrust world frame
    tw = F.unsqueeze(-1) * R[:,:,2]
    ax=tw[:,0]/M; ay=tw[:,1]/M; az=tw[:,2]/M - G

    # Gyroscopic coupling
    p_dot = (tau_x - (I_Z-I_Y)*q*r) / I_X
    q_dot = (tau_y - (I_X-I_Z)*p*r) / I_Y
    r_dot = (tau_z - (I_Y-I_X)*p*q) / I_Z

    # Semi-implicit Euler: velocities first
    vx_n=vx+dt*ax; vy_n=vy+dt*ay; vz_n=vz+dt*az
    p_n=p+dt*p_dot; q_n=q+dt*q_dot; r_n=r+dt*r_dot

    # Positions with NEW velocities
    x_n=x+dt*vx_n; y_n=y+dt*vy_n; z_n=z+dt*vz_n

    # Orientation: simple Euler integration (the approximation)
    phi_n=phi+dt*p_n; theta_n=theta+dt*q_n; psi_n=psi+dt*r_n

    return torch.stack([x_n,vx_n, y_n,vy_n, z_n,vz_n,
                        phi_n,p_n, theta_n,q_n, psi_n,r_n], dim=-1)


def dynamics_one_ctrl_step(s12, wrench, substep_fn, n_substeps=PYB_STEPS_PER_CTRL):
    dt = DT_CTRL / n_substeps
    for _ in range(n_substeps):
        s12 = substep_fn(s12, wrench, dt)
    return s12

# =====================================================================
# 4. Linear model (for --linear and LQR baseline)
# =====================================================================

def _build_linear_model():
    from scipy.signal import cont2discrete
    A = np.zeros((12,12))
    A[0,1]=1; A[2,3]=1; A[4,5]=1; A[6,7]=1; A[8,9]=1; A[10,11]=1
    A[1,8]=G; A[3,6]=-G
    B = np.zeros((12,4))
    B[5,0]=1/M; B[7,1]=1/I_X; B[9,2]=1/I_Y; B[11,3]=1/I_Z
    c = np.zeros((12,1)); c[5,0]=-G
    n,m = 12,4
    Aa = np.zeros((n+1,n+1)); Aa[:n,:n]=A; Aa[:n,n]=c.squeeze()
    Ba = np.zeros((n+1,m)); Ba[:n,:]=B
    Ca = np.zeros((n,n+1)); Ca[:,:n]=np.eye(n)
    Da = np.zeros((n,m))
    Ada,Bda,_,_,_ = cont2discrete((Aa,Ba,Ca,Da), DT_CTRL, method='zoh')
    Ad=Ada[:n,:n]; Bd=Bda[:n,:]; d=Ada[:n,n]
    Qd = np.array([1/X_MAX**2,1/X_DMAX**2, 1/Y_MAX**2,1/Y_DMAX**2,
                    1/Z_MAX**2,1/Z_DMAX**2, 1/PHI_MAX**2,1/PHI_DMAX**2,
                    1/THETA_MAX**2,1/THETA_DMAX**2, 1/PSI_MAX**2,1/PSI_DMAX**2])
    Rd = np.array([1/F_MAX**2, 1/TAU_XY_MAX**2, 1/TAU_XY_MAX**2, 1/TAU_Z_MAX**2])
    P = la.solve_discrete_are(Ad, Bd, np.diag(Qd), np.diag(Rd))
    K = np.linalg.inv(Bd.T@P@Bd+np.diag(Rd)) @ (Bd.T@P@Ad)
    u_eq = -np.linalg.pinv(Bd) @ d
    return Ad, Bd, d, K, u_eq

Ad_lin, Bd_lin, d_lin, K_LQR, U_EQ = _build_linear_model()
print(f"[LQR] u_eq[0]={U_EQ[0]:.6f} N (mg={M*G:.6f})")

# =====================================================================
# 5. Neural network
# =====================================================================

class PolicyMLP(nn.Module):
    def __init__(self, x_scale, u_max, hidden=64):
        super().__init__()
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        self.register_buffer("u_max",   torch.tensor(u_max,   dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(12, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 4),
        )
        self._init_hover()

    def _init_hover(self):
        last = self.net[-1]
        nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
        hr = (M*G) / float(self.u_max[0])
        last.bias.data[0] = float(np.log(hr / (1-hr)))

    def forward(self, x):
        raw = self.net(x / self.x_scale)
        thrust  = torch.sigmoid(raw[..., [0]]) * self.u_max[0]
        torques = torch.tanh(raw[..., 1:])     * self.u_max[1:]
        return torch.cat([thrust, torques], dim=-1)

# =====================================================================
# 6. Rollout and cost
# =====================================================================

def rollout_nonlinear(policy, x0, T_steps, substep_fn, n_substeps=1):
    B, dev = x0.shape[0], x0.device
    X = torch.zeros(B, T_steps, 12, device=dev)
    U = torch.zeros(B, T_steps, 4,  device=dev)
    s = x0
    for k in range(T_steps):
        w = policy(s)
        s = dynamics_one_ctrl_step(s, w, substep_fn, n_substeps)
        X[:,k,:] = s;  U[:,k,:] = w
    return X, U

def rollout_linear(policy, x0, T_steps):
    B, dev = x0.shape[0], x0.device
    Ad_t = torch.tensor(Ad_lin, dtype=torch.float32, device=dev)
    Bd_t = torch.tensor(Bd_lin, dtype=torch.float32, device=dev)
    d_t  = torch.tensor(d_lin,  dtype=torch.float32, device=dev).unsqueeze(0)
    X = torch.zeros(B, T_steps, 12, device=dev)
    U = torch.zeros(B, T_steps, 4,  device=dev)
    s = x0
    for k in range(T_steps):
        w = policy(s)
        s = (s @ Ad_t.T) + (w @ Bd_t.T) + d_t
        X[:,k,:] = s;  U[:,k,:] = w
    return X, U

Q_DIAG = np.array([1/X_MAX**2,1/X_DMAX**2, 1/Y_MAX**2,1/Y_DMAX**2,
                    1/Z_MAX**2,1/Z_DMAX**2, 1/PHI_MAX**2,1/PHI_DMAX**2,
                    1/THETA_MAX**2,1/THETA_DMAX**2, 1/PSI_MAX**2,1/PSI_DMAX**2],
                   dtype=np.float32)
R_DIAG = np.array([1/F_MAX**2, 1/TAU_XY_MAX**2, 1/TAU_XY_MAX**2,
                    1/TAU_Z_MAX**2], dtype=np.float32)

def traj_cost(X, U, terminal_weight=0.0):
    Q = torch.as_tensor(Q_DIAG, dtype=X.dtype, device=X.device)
    R = torch.as_tensor(R_DIAG, dtype=U.dtype, device=U.device)
    L = (X**2*Q).sum(2).mean(0).sum() + (U**2*R).sum(2).mean(0).sum()
    if terminal_weight > 0:
        L += terminal_weight * (X[:,-1,[0,2,4]]**2).mean()
    return L

# =====================================================================
# 7. Training
# =====================================================================

def make_x0_batch(xyz_list, device="cpu"):
    X0 = torch.zeros(len(xyz_list), 12, dtype=torch.float32, device=device)
    for b,(x,y,z) in enumerate(xyz_list):
        X0[b,0]=x; X0[b,2]=y; X0[b,4]=z
    return X0

def generate_cube_points(hs=0.3):
    return list(itertools.product([-hs, 0.0, hs], repeat=3))

def dynamics_label(linear, no_quat):
    if linear: return "linear"
    if no_quat: return "nonlinear_no_quat"
    return "nonlinear_quat"

def train(epochs=2000, lr=1e-3, hidden=64, tw=10.0, hs=0.3,
          linear=False, no_quat=False, device="cpu"):
    policy = PolicyMLP(x_scale=X_SCALE, u_max=U_MAX, hidden=hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    x0 = make_x0_batch(generate_cube_points(hs), device=device)

    with torch.no_grad():
        u0 = policy(torch.zeros(1,12,device=device))
        print(f"\n[Init] NN(0): F={u0[0,0]:.4f} (mg={M*G:.4f}), tau={u0[0,1:]}")

    dl = dynamics_label(linear, no_quat)
    print(f"[Train] {x0.shape[0]} pts, {epochs} ep, lr={lr}, dyn={dl}")

    # Choose rollout function
    if linear:
        def do_rollout(p, x, T): return rollout_linear(p, x, T)
    else:
        sfn = dynamics_substep_simple if no_quat else dynamics_substep_quat
        def do_rollout(p, x, T): return rollout_nonlinear(p, x, T, sfn)

    for ep in range(epochs):
        X, U = do_rollout(policy, x0, T_STEPS)
        loss = traj_cost(X, U, tw)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step(); sched.step()
        if (ep+1)%100==0 or ep==0:
            with torch.no_grad():
                err = X[:,-1,[0,2,4]].norm(dim=1)
                print(f"  [ep {ep+1:4d}/{epochs}] loss={loss.item():.4e} "
                      f"|x_T| mean={err.mean().item():.4f} max={err.max().item():.4f}")
    return policy

# =====================================================================
# 8. Evaluation
# =====================================================================

@torch.no_grad()
def evaluate(policy, test_pts, device="cpu"):
    x0 = make_x0_batch(test_pts, device=device)
    # NN eval with quaternion dynamics + 5 substeps (pybullet exact)
    X,_ = rollout_nonlinear(policy, x0, T_STEPS, dynamics_substep_quat,
                             n_substeps=PYB_STEPS_PER_CTRL)
    e = X[:,-1,[0,2,4]].norm(dim=1)
    print(f"\n[Eval] {len(test_pts)} pts (quat, {PYB_STEPS_PER_CTRL} substeps):")
    print(f"  NN  |x_T| mean={e.mean().item():.5f} max={e.max().item():.5f}")
    # LQR baseline
    Ad_t=torch.tensor(Ad_lin,dtype=torch.float32,device=device)
    Bd_t=torch.tensor(Bd_lin,dtype=torch.float32,device=device)
    d_t=torch.tensor(d_lin,dtype=torch.float32,device=device).unsqueeze(0)
    K_t=torch.tensor(K_LQR,dtype=torch.float32,device=device)
    ueq_t=torch.tensor(U_EQ,dtype=torch.float32,device=device).unsqueeze(0)
    x=x0.clone()
    for k in range(T_STEPS):
        u=ueq_t-(x@K_t.T)
        u[:,0]=torch.clamp(u[:,0],0,F_MAX)
        u[:,1]=torch.clamp(u[:,1],-TAU_XY_MAX,TAU_XY_MAX)
        u[:,2]=torch.clamp(u[:,2],-TAU_XY_MAX,TAU_XY_MAX)
        u[:,3]=torch.clamp(u[:,3],-TAU_Z_MAX,TAU_Z_MAX)
        x=(x@Ad_t.T)+(u@Bd_t.T)+d_t
    el=x[:,[0,2,4]].norm(dim=1)
    print(f"  LQR |x_T| mean={el.mean().item():.5f} max={el.max().item():.5f} (linear)")

# =====================================================================
# 9. Main
# =====================================================================

if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--linear", action="store_true")
    pa.add_argument("--no-quaternions", action="store_true")
    pa.add_argument("--epochs", type=int, default=2000)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--hidden", type=int, default=64)
    args = pa.parse_args()

    if args.linear and args.no_quaternions:
        print("[WARN] --linear ignores --no-quaternions (linear has no rotation)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    policy = train(epochs=args.epochs, lr=args.lr, hidden=args.hidden,
                   linear=args.linear, no_quat=args.no_quaternions, device=device)

    rng = np.random.default_rng(42)
    test_pts = [(rng.uniform(-0.3,0.3), rng.uniform(-0.3,0.3),
                 rng.uniform(-0.3,0.3)) for _ in range(20)]
    evaluate(policy, test_pts, device=device)

    # ---- Save with descriptive name ----
    dl = dynamics_label(args.linear, args.no_quaternions)
    tag = f"{dl}_h{args.hidden}"
    out = os.path.dirname(os.path.abspath(__file__))

    torch.save(policy.cpu(), os.path.join(out, f"policy_{tag}.pt"))

    ckpt_path = os.path.join(out, f"weights_{tag}.pt")
    torch.save({
        "state_dict": policy.state_dict(), "hidden": args.hidden,
        "x_scale": X_SCALE, "u_max": U_MAX,
        "ctrl_freq": CTRL_FREQ, "Ts": DT_CTRL,
        "KF": KF, "KM": KM, "L": L, "M": M, "G": G,
        "MAX_RPM": MAX_RPM,
        "dynamics": dl,
    }, ckpt_path)

    print(f"\n[Saved] policy_{tag}.pt + weights_{tag}.pt")
    print(f"Validate: python validate_nn_pybullet.py --weights weights_{tag}.pt")