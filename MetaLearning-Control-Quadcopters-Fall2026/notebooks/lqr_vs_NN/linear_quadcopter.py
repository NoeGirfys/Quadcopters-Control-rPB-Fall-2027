"""
Simulate linear quadcopter model with LQR controller in PyTorch.
Supports batched simulations: states are tensors of shape (B, T, n).
Generates 3D trajectory animation + Euler angle plots.
"""

import numpy as np
import scipy.linalg as la
from scipy.signal import cont2discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib as mpl


# Enable LaTeX rendering
mpl.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",  # or "sans-serif" or another family
    "font.serif": ["Computer Modern Roman"],  # the default LaTeX font
    "axes.unicode_minus": False  # ensure minus sign renders correctly with LaTeX
})

# Consistent printing of numpy arrays
np.set_printoptions(precision=3, suppress=True)

# --------------------------------------------------------------------SYSTEM DEFINITION-------------------------------------------------------------------
def quad_hover_linear_model(m, g, Ix, Iy, Iz):
    """
    State: [X, Xd, Y, Yd, Z, Zd, phi, phid, theta, thetad, psi, psid]^T
    Inputs: [F, tau_x, tau_y, tau_z]^T (total thrust deviation from hover, body torques)
    Source: Faraz Ahmad, Pushpendra Kumar, Anamika Bhandari, Pravin P. Patil, Simulation of the Quadcopter Dynamics with LQR based Control
    """
    Ac = np.zeros((12, 12))
    # kinematic relationships
    Ac[0,1]  = 1.0   # Xdot
    Ac[2,3]  = 1.0   # Ydot
    Ac[4,5]  = 1.0   # Zdot
    Ac[6,7]  = 1.0   # phidot
    Ac[8,9]  = 1.0   # thetadot
    Ac[10,11]= 1.0   # psidot

    # gravity-induced lateral coupling (small-angle)
    Ac[1,8]  =  g    # Xddot depends on theta
    Ac[3,6]  = -g    # Yddot depends on phi

    Bc = np.zeros((12, 4))
    # thrust -> Z acceleration
    Bc[5,0]  = 1.0/m
    # torques -> angular accelerations
    Bc[7,1]  = 1.0/Ix   # phiddot
    Bc[9,2]  = 1.0/Iy   # thetaddot
    Bc[11,3] = 1.0/Iz   # psiddot

    # Identity output by default
    Cc = np.eye(12)
    Dc = np.zeros((12, 4))
    return Ac, Bc, Cc, Dc


def discretize(Ac, Bc, Cc, Dc, Ts, method="zoh"):
    Ad, Bd, Cd, Dd, _ = cont2discrete((Ac, Bc, Cc, Dc), Ts, method=method)
    return Ad, Bd, Cd, Dd


def _euler_zyx_to_R(phi, theta, psi):
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth,  sth  = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    Rx = np.array([[1, 0, 0],
                   [0, cphi, -sphi],
                   [0, sphi,  cphi]])
    Ry = np.array([[ cth, 0, sth],
                   [   0, 1,   0],
                   [-sth, 0, cth]])
    Rz = np.array([[cpsi, -spsi, 0],
                   [spsi,  cpsi, 0],
                   [   0,     0, 1]])
    return Rz @ Ry @ Rx
# ---------------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------LQR CONTROLLER-----------------------------------------------------------------------
def design_lqr(Ad, Bd, Q, R):
    # Discrete-time LQR
    P = la.solve_discrete_are(Ad, Bd, Q, R)
    K_LQR = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    return K_LQR
# ---------------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------NN CONTROLLER------------------------------------------------------------------------
# Policy: small MLP with tanh + limits (smooth saturation)
class PolicyMLP(nn.Module):
    
    def __init__(self, x_scale, u_max, hidden=64):
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
    def forward(self, x):                        # x: (..., 12)
        x_n = x / self.x_scale                   # normalize inputs
        u   = self.net(x_n)
        return torch.tanh(u) * self.u_max        # smooth clamp to limits


# Rollout: differentiable plant x_{k+1} = Ad x_k + Bd u_k
@torch.no_grad()
def make_x0_batch(xyz_list):
    B = len(xyz_list)
    X0 = torch.zeros(B, 12, dtype=torch.float32)
    for b,(x,y,z) in enumerate(xyz_list):
        X0[b,0] = x; X0[b,2] = y; X0[b,4] = z    # [X, Xd, Y, Yd, Z, Zd, φ, φ̇, θ, θ̇, ψ, ψ̇]
    return X0


def rollout_policy(Ad, Bd, policy, x0, T_steps):
    """
    Ad, Bd: torch.FloatTensor (12x12), (12x4) on same device as policy
    x0: (B,12) initial states
    returns X (B,T,12), U (B,T,4)
    """
    B = x0.shape[0]; n = Ad.shape[0]; m = Bd.shape[1]
    X = torch.zeros(B, T_steps, n, device=Ad.device)
    U = torch.zeros(B, T_steps, m, device=Ad.device)
    x = x0
    for k in range(T_steps):
        u = policy(x)             # (B,4)
        x = (x @ Ad.T) + (u @ Bd.T)
        X[:,k,:] = x
        U[:,k,:] = u
    return X, U


# Cost: sum_k x'Qx + u'Ru (batched over trajectories)
def traj_cost(X, U, Q_diag, R_diag, terminal_pos_weight=0.0):
    """
    X: (B,T,12), U:(B,T,4)
    Q_diag: (12,), R_diag: (4,)   (torch or numpy)
    """
    Q = torch.as_tensor(Q_diag, dtype=X.dtype, device=X.device)
    R = torch.as_tensor(R_diag, dtype=U.dtype, device=U.device)
    stage_x = (X**2) * Q                      # (B,T,12)
    stage_u = (U**2) * R                      # (B,T,4)
    L = stage_x.sum(dim=2).mean(dim=0).sum()  # sum over state dims, mean over batch, sum over time
    L += stage_u.sum(dim=2).mean(dim=0).sum()
    if terminal_pos_weight > 0.0:
        pT = X[:, -1, [0,2,4]]                # terminal XYZ
        L += terminal_pos_weight * (pT**2).mean()
    return L


# Training loop
def train_nn_policy(Ad_np, Bd_np,
                    init_batches,     # list of lists of (x,y,z), one batch per iter
                    Ts, T_sim,
                    x_scale, u_max,   # normalization and control limits
                    Q_diag, R_diag,   # cost weights
                    lr=1e-3, epochs=2000,
                    terminal_pos_weight=0.0,
                    device="cpu", verbose_every=100):
    """
    init_batches: e.g. [[(0,2,3), (2,0,3), (2,2,3)], [(0,2,3)], ...]
                  Each element is a batch of start states for one optimizer step.
    """
    Ad = torch.tensor(Ad_np, dtype=torch.float32, device=device)
    Bd = torch.tensor(Bd_np, dtype=torch.float32, device=device)
    T_steps = int(T_sim / Ts)

    policy = PolicyMLP(x_scale=x_scale, u_max=u_max).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    for ep in range(epochs):
        # Iterate over provided batches of initial conditions
        for xyz_list in init_batches:
            x0 = make_x0_batch(xyz_list).to(device)

            # rollout (differentiable through policy & plant)
            X, U = rollout_policy(Ad, Bd, policy, x0, T_steps)

            # loss = sum_k x'Qx + u'Ru (+ optional terminal penalty)
            loss = traj_cost(X, U, Q_diag, R_diag, terminal_pos_weight)

            # SGD step
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        if verbose_every and ((ep+1) % verbose_every == 0):
            with torch.no_grad():
                p_end = X[:, -1, [0,2,4]].norm(dim=1).mean().item()
            print(f"[ep {ep+1}/{epochs}] loss={loss.item():.4e} |x_T|_avg={p_end:.3f}")

    return policy
# ---------------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------SIMULATION FUNCTION------------------------------------------------------------------
def simulate_torch(Ad, Bd, K, init_pos, m, g, Ts, T_sim, Fmax, tauxmax, tauymax, tauzmax, device="cpu"):
    Ad = torch.tensor(Ad, dtype=torch.float32, device=device)
    Bd = torch.tensor(Bd, dtype=torch.float32, device=device)
    K  = torch.tensor(K,  dtype=torch.float32, device=device)

    init_pos = init_pos.to(device)  # ensure same device

    B = init_pos.shape[0]       # batch size: number of different initial positions
    n_x = Ad.shape[0]           # state dimension
    N_steps = int(T_sim / Ts)   # number of simulation steps

    X = torch.zeros(B, N_steps, n_x, device=device)
    U = torch.zeros(B, N_steps, 4, device=device)

    x = torch.zeros(B, n_x, device=device)
    x[:, 0] = init_pos[:, 0]  # X
    x[:, 2] = init_pos[:, 1]  # Y
    x[:, 4] = init_pos[:, 2]  # Z

    for k in range(N_steps):
        u = -(x @ K.T)                                      
        
        u[:, 0] = torch.clamp(u[:, 0], -Fmax, Fmax)         # clamp thrust deviation and torques
        u[:, 1] = torch.clamp(u[:, 1], -tauxmax, tauxmax)
        u[:, 2] = torch.clamp(u[:, 2], -tauymax, tauymax)
        u[:, 3] = torch.clamp(u[:, 3], -tauzmax, tauzmax)

        x = (x @ Ad.T) + (u @ Bd.T)
        X[:, k, :] = x
        U[:, k, :] = u
    return X, U
# ---------------------------------------------------------------------------------------------------------------------------------------------------------



# --------------------------------------------------------------------PLOTTING & ANIMATION-----------------------------------------------------------------
def animate_trajectories(X, Ts, T_sim, filename, triad_scale=0.6, triad_lw=1.0, trail_lw=2.0, stride=1, progress=True):
    # Retrieve trajectories
    ix, iy, iz = 0, 2, 4
    iphi, itheta, ipsi = 6, 8, 10
    if hasattr(X, "detach"):  # torch tensor
        X = X.detach().cpu().numpy()
    B, T, n = X.shape
    xs, ys, zs = X[:, :, ix], X[:, :, iy], X[:, :, iz]

    # Derive/validate counts
    N_expected = int(round(T_sim / Ts))
    if T != N_expected:
        print(f"[warn] T={T} differs from T_sim/Ts={N_expected}. Using T={T} from data.")

    # Frame list & fps to keep duration = T * Ts (≈ T_sim)
    frames = list(range(0, T, max(1, int(stride))))
    fps = max(1, int(round((1.0 / Ts) / max(1, int(stride)))))  # (1/Ts)/stride

    # Figure
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    axis_lim = 5.0
    ax.set_xlim([-axis_lim, axis_lim])
    ax.set_ylim([-axis_lim, axis_lim])
    ax.set_zlim([-axis_lim, axis_lim])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(fr"Quadcopter trajectories ($T_{{\mathrm{{sim}}}}\approx {len(frames)/fps:.2f}\,\mathrm{{s}}$)")

    colors = plt.cm.viridis(np.linspace(0, 1, B))

    trails = []
    markers = []
    for b in range(B):
        # initial position for legend label
        x0 = xs[b, 0]
        y0 = ys[b, 0]
        z0 = zs[b, 0]
        label = f"Batch {b}: ({x0:.1f}, {y0:.1f}, {z0:.1f})"

        line, = ax.plot([], [], [], lw=trail_lw, color=colors[b], label=label)
        point, = ax.plot([], [], [], "o", color=colors[b])
        trails.append(line)
        markers.append(point)

    # Add legend once, after creating the lines
    ax.legend(loc="best")

    colors = plt.cm.viridis(np.linspace(0, 1, B))
    trails  = [ax.plot([], [], [], lw=trail_lw, color=colors[b])[0] for b in range(B)]
    markers = [ax.plot([], [], [], "o", color=colors[b])[0] for b in range(B)]

    # Triads: R/G/B for body x/y/z
    triads = []
    for _ in range(B):
        tx, = ax.plot([], [], [], lw=triad_lw, color="r")
        ty, = ax.plot([], [], [], lw=triad_lw, color="g")
        tz, = ax.plot([], [], [], lw=triad_lw, color="b")
        triads.append((tx, ty, tz))

    def _euler_zyx_to_R(phi, theta, psi):
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)
        Rx = np.array([[1,0,0],[0,cph,-sph],[0,sph,cph]])
        Ry = np.array([[cth,0,sth],[0,1,0],[-sth,0,cth]])
        Rz = np.array([[cps,-sps,0],[sps,cps,0],[0,0,1]])
        return Rz @ Ry @ Rx

    def init():
        return trails + markers + [l for tri in triads for l in tri]

    def update(i):
        if progress and (i % max(1, len(frames)//50) == 0):
            # progress over *rendered* frames
            print(f"Frame {frames.index(i)+1}/{len(frames)}")

        for b in range(B):
            # trail & marker at frame i
            trails[b].set_data(xs[b, :i], ys[b, :i])
            trails[b].set_3d_properties(zs[b, :i])
            markers[b].set_data([xs[b, i]], [ys[b, i]])
            markers[b].set_3d_properties([zs[b, i]])

            # attitude triad at pose i
            phi, theta, psi = X[b, i, iphi], X[b, i, itheta], X[b, i, ipsi]
            R = _euler_zyx_to_R(phi, theta, psi)
            p = np.array([xs[b, i], ys[b, i], zs[b, i]])
            ex, ey, ez = R[:,0], R[:,1], R[:,2]
            tx, ty, tz = triads[b]
            px, py, pz = p + triad_scale*ex, p + triad_scale*ey, p + triad_scale*ez
            tx.set_data([p[0], px[0]], [p[1], px[1]]); tx.set_3d_properties([p[2], px[2]])
            ty.set_data([p[0], py[0]], [p[1], py[1]]); ty.set_3d_properties([p[2], py[2]])
            tz.set_data([p[0], pz[0]], [p[1], pz[1]]); tz.set_3d_properties([p[2], pz[2]])
        return trails + markers + [l for tri in triads for l in tri]

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)

    fps = int(round(1.0 / Ts))  # 100 for Ts=0.01
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1)
    ani.save(filename, writer=writer)


def plot_angles(X, Ts, filename):
    if hasattr(X, "detach"):  # torch tensor
        X = X.detach().cpu().numpy()

    B, T, n = X.shape
    t = np.arange(T) * Ts

    fig, axs = plt.subplots(B, 1, sharex=True, figsize=(8, 2.6 * B))
    axs = np.atleast_1d(axs)

    for b in range(B):
        roll  = X[b, :, 6] 
        pitch = X[b, :, 8] 
        yaw   = X[b, :, 10]

        x0 = X[b, 0, 0] 
        y0 = X[b, 0, 2] 
        z0 = X[b, 0, 4]

        axs[b].plot(t, np.rad2deg(roll),  label="Roll $\phi$")
        axs[b].plot(t, np.rad2deg(pitch), label="Pitch $\\theta$")
        axs[b].plot(t, np.rad2deg(yaw),   label="Yaw  $\psi$")
        axs[b].grid(True)
        axs[b].set_ylabel("Angle [°]")
        axs[b].set_title(f"Batch {b}: ({x0:.1f}, {y0:.1f}, {z0:.1f})")
        axs[b].legend(loc="best")

    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    # plt.show()


def plot_xyz(X, Ts, filename):
    if hasattr(X, "detach"):  # torch tensor
        X = X.detach().cpu().numpy()

    B, T, n = X.shape
    t = np.arange(T) * Ts

    fig, axs = plt.subplots(B, 1, sharex=True, figsize=(8, 2.6 * B))
    axs = np.atleast_1d(axs)

    for b in range(B):
        x = X[b, :, 0] 
        y = X[b, :, 2]
        z = X[b, :, 4] 

        x0 = X[b, 0, 0] 
        y0 = X[b, 0, 2] 
        z0 = X[b, 0, 4]

        axs[b].plot(t, x, label="x")
        axs[b].plot(t, y, label="y")
        axs[b].plot(t, z, label="z")
        axs[b].grid(True)
        axs[b].set_ylabel("Position [m]")
        axs[b].set_title(f"Batch {b}: ({x0:.1f}, {y0:.1f}, {z0:.1f})")
        axs[b].legend(loc="best")

    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    # plt.show()


def plot_controls_grid(U, Ts, m, g, Fmax, tauxmax, tauymax, tauzmax, X_for_titles, filename):
    if hasattr(U, "detach"):  # torch tensor
        U = U.detach().cpu().numpy()
    B, T, m_in = U.shape
    assert m_in == 4, "U must be shape (B, T, 4)"

    t = np.arange(T) * Ts

    col_meta = [
        ("Thrust $F$ [N]",          Fmax),
        (r"$\tau_x$ [N$\cdot$m]",   tauxmax),
        (r"$\tau_y$ [N$\cdot$m]",   tauymax),
        (r"$\tau_z$ [N$\cdot$m]",   tauzmax),
    ]

    # optional titles per row
    row_titles = [f"Batch {b}" for b in range(B)]
    if X_for_titles is not None:
        Xn = X_for_titles.detach().cpu().numpy() if hasattr(X_for_titles, "detach") else X_for_titles
        for b in range(B):
            x0, y0, z0 = Xn[b, 0, 0], Xn[b, 0, 2], Xn[b, 0, 4]
            row_titles[b] = f"Batch {b} — $(x_0,y_0,z_0)=({x0:.1f},{y0:.1f},{z0:.1f})$"

    # figure
    fig, axs = plt.subplots(B, 4, sharex='col', figsize=(4*4, 2.6*B), squeeze=False)

    for b in range(B):
        for j, (ylabel, umax) in enumerate(col_meta):
            ax = axs[b, j]
            ax.plot(t, U[b, :, j], label="command")
            ax.axhline( umax, ls="--", c="k", lw=1, alpha=0.7, label="+limit" if (b==0 and j==0) else None)
            ax.axhline(-umax, ls="--", c="k", lw=1, alpha=0.7, label="-limit" if (b==0 and j==0) else None)
            ax.grid(True)

            # labels
            if b == 0:
                ax.set_title(ylabel)
            if b == B-1:
                ax.set_xlabel("Time [s]")
            if j == 0:
                ax.set_ylabel(row_titles[b])

    # one legend (top-left only) to avoid clutter
    axs[0,0].legend(loc="best")

    fig.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
# ---------------------------------------------------------------------------------------------------------------------------------------------------------



if __name__=="__main__":
    # ----------------------------------------------------SETUP & PARAMETERS----------------------------------------------------------------------
    # Model parameters
    m = 1.0; g = 9.81; Ix = 0.11; Iy = 0.11; Iz = 0.04; Ts = 0.01; T_sim = 8.0


    # Bryson-style max acceptable magnitudes for each state
    # Commands
    Fmax = m * g; tauxmax = 0.08; tauymax = 0.08; tauzmax = 0.05
    u_max = np.array([Fmax, tauxmax, tauymax, tauzmax], dtype=np.float32)
    # States
    Xmax, Xdmax = 5.0, 5.0
    Ymax, Ydmax = 5.0, 5.0
    Zmax, Zdmax = 5.0, 5.0
    phimax, phidmax = np.deg2rad(30), np.deg2rad(120)       # max orientation of 30 deg and max angular rate of 120 deg/s = full rotation in 3s
    thetamax, thetadmax = np.deg2rad(30), np.deg2rad(120)
    psimax, psidmax = np.deg2rad(45), np.deg2rad(90)
    x_scale = np.array([Xmax, Xdmax, Ymax, Ydmax, Zmax, Zdmax, phimax, phidmax, thetamax, thetadmax, psimax, psidmax], dtype=np.float32)


    # Continuous-time system
    Ac, Bc, Cc, Dc = quad_hover_linear_model(m, g, Ix, Iy, Iz)
    print(f"Continuous-time system:\nAc={Ac}\nBc={Bc}\nCc={Cc}\nDc={Dc}")


    # Discrete-time system
    Ad, Bd, Cd, Dd = discretize(Ac, Bc, Cc, Dc, Ts)
    print(f"Discrete-time system (Ts={Ts}):\nAd={Ad}\nBd={Bd}\nCd={Cd}\nDd={Dd}")


    # State/input weights (diagonals)
    Q_diag= np.array([1/Xmax**2,  1/Xdmax**2, 1/Ymax**2,  1/Ydmax**2, 1/Zmax**2,  1/Zdmax**2, 1/phimax**2,  1/phidmax**2, 1/thetamax**2, 1/thetadmax**2, 1/psimax**2,  1/psidmax**2], dtype=np.float32)
    Q = np.diag(Q_diag)
    R_diag= np.array([1/Fmax**2, 1/tauxmax**2, 1/tauymax**2, 1/tauzmax**2], dtype=np.float32)
    R = np.diag(R_diag)
    print(f"Q={Q}\nR={R}")
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    


    # ---------------------------------------------------------------------LQR----------------------------------------------------------------------
    # # Design LQR controller (x_(k+1) = Ad x_k + Bd u_k; u_(k) = - K x_(k))
    # K_LQR = design_lqr(Ad, Bd, Q, R)
    # print(f"LQR gain K={K_LQR}")


    # # Batch of initial positions
    # x0_s = [[0, 2, 3],
    #         [2, 0, 3],
    #         [2, 2, 3],
    #         [-2, -2, -3],
    #         [-5, -5, -5]] # Initial (x, y, z) coordiniates for four batches (trajectories) -- this is dimension (B,3)
    # init_pos = torch.tensor(x0_s, dtype=torch.float32)


    # # Simulate
    # X_LQR, U_LQR = simulate_torch(Ad, Bd, K_LQR, init_pos, m, g, Ts, T_sim, Fmax, tauxmax, tauymax, tauzmax)
    # print("Simulation done.")


    # # Plot results
    # plot_angles(X_LQR, Ts, "lqr_angles.png")
    # plot_xyz(X_LQR, Ts, "lqr_xyz.png")
    # plot_controls_grid(U_LQR, Ts, m, g, Fmax, tauxmax, tauymax, tauzmax, X_LQR, "lqr_controls.png")


    # # Create animation
    # print("Generating animation...")
    # animate_trajectories(X_LQR, Ts, T_sim, "lqr_3d_xyz.mp4")
    # print("All done.")
    # ------------------------------------------------------------------------------------------------------------------------------------------------



    # ----------------------------------------------------------------------NN------------------------------------------------------------------------
    device = "cpu"

    # Build a list of "minibatches" of initial conditions (always the same one=strong overfit to that start)
    init_batches = [[(0.0, 2.0, 3.0)]] * 1    # one initial condition per step (batch = set of ICs trained on in one optimizer step) = training
    # Or a small set per step:
    # init_batches = [[(0.0,2.0,3.0), (2.0,0.0,3.0), (2.0,2.0,3.0)]]

    print("Training NN policy...")
    policy = train_nn_policy(
        Ad_np=Ad, Bd_np=Bd,
        init_batches=init_batches,
        Ts=Ts, T_sim=T_sim,
        x_scale=x_scale, u_max=u_max,
        Q_diag=Q_diag, R_diag=R_diag,
        lr=1e-3, epochs=1500,               # learning rate = θ←θ−lr⋅∇θ​L (step in gradient descent) | epoch = a full loop over your dataset of batches
        terminal_pos_weight=10.0,           # nudges position error at T to be small (make sure it actually ends up at the target)
        device=device, verbose_every=100
    )

    # Evaluate from the training start
    X0 = make_x0_batch([(0.0, 2.0, 3.0)]).to(device) # testing on this point
    Ad_t = torch.tensor(Ad, dtype=torch.float32, device=device)
    Bd_t = torch.tensor(Bd, dtype=torch.float32, device=device)
    T_steps = int(T_sim / Ts)

    with torch.no_grad():
        X_nn, U_nn = rollout_policy(Ad_t, Bd_t, policy, X0, T_steps)

    # Reuse your plotting
    plot_xyz(X_nn, Ts, filename="nn_xyz.png")
    plot_angles(X_nn, Ts, filename="nn_angles.png")
    plot_controls_grid(U_nn, Ts, m, g, Fmax, tauxmax, tauymax, tauzmax, X_nn,filename="nn_controls.png")
    
    # Evaluate from a DIFFERENT start (to see the expected degradation)
    X1 = make_x0_batch([(2.0, 2.0, 3.0)]).to(device)
    with torch.no_grad():
        X_nn2, U_nn2 = rollout_policy(Ad_t, Bd_t, policy, X1, T_steps)
    plot_xyz(X_nn2, Ts, filename="nn_xyz_unseen.png")
    # ------------------------------------------------------------------------------------------------------------------------------------------------





