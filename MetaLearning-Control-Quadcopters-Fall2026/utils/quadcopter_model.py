import numpy as np
import scipy.linalg as la
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib as mpl
from scipy.stats import norm
import numpy as np
from numpy.linalg import matrix_rank, eigvals
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.config as cfg


def quad_hover_linear_model():
    A = np.zeros((12,12))
    A[0,1]  = 1.0
    A[2,3]  = 1.0
    A[4,5]  = 1.0
    A[6,7]  = 1.0
    A[8,9]  = 1.0
    A[10,11]= 1.0
    A[1,8] = cfg.G
    A[3,6] = cfg.G 

    B = np.zeros((12,4))
    B[5,0]  = 1.0/cfg.M
    B[7,1]  = 1.0/cfg.I_X
    B[9,2]  = 1.0/cfg.I_Y
    B[11,3] = 1.0/cfg.I_Z

    C = np.eye(A.shape[0])

    D = np.zeros((12, 4))

    return A, B, C, D


def create_discrete_model(A, B, C, D):
    n, m = A.shape[0], B.shape[1]
    p = C.shape[0]

    B_tilde = np.zeros((n, m+3))  # extra columns for disturbance inputs
    B_tilde[:, :m] = B
    B_tilde[1, 4] = 1.0/cfg.M     # for disturbance in x
    B_tilde[3, 5] = 1.0/cfg.M     # for disturbance in y
    B_tilde[5, 6] = 1.0/cfg.M     # for disturbance in z

    A_d, B_tilde_d, C_d, D_d, _ = cont2discrete((A, B_tilde, C, D), cfg.T_D, method="zoh")

    return A_d, B_tilde_d, C_d, D_d


def quad_nonlinear_dynamics_torch(x, u):
    """
    x: [B, 12]
    u: [B, 4]
    returns x_dot: [B, 12]
    """

    # Unpack states
    x_pos     = x[:, 0]
    x_dot     = x[:, 1]
    y_pos     = x[:, 2]
    y_dot     = x[:, 3]
    z_pos     = x[:, 4]
    z_dot     = x[:, 5]
    phi       = x[:, 6]
    phi_dot   = x[:, 7]
    theta     = x[:, 8]
    theta_dot = x[:, 9]
    psi       = x[:,10]
    psi_dot   = x[:,11]

    # Unpack inputs
    F         = u[:, 0]
    tau_x     = u[:, 1]
    tau_y     = u[:, 2]
    tau_z     = u[:, 3]

    # Helpers
    m  = cfg.M
    g  = cfg.G
    Ix = cfg.I_X
    Iy = cfg.I_Y
    Iz = cfg.I_Z
    cphi = torch.cos(phi)
    sphi = torch.sin(phi)
    cth  = torch.cos(theta)
    sth  = torch.sin(theta)
    cpsi = torch.cos(psi)
    spsi = torch.sin(psi)

    # Translational accelerations
    x_ddot = (F / m) * (cphi * sth * cpsi + sphi * spsi)
    y_ddot = (F / m) * (cphi * sth * spsi - sphi * cpsi)
    z_ddot = (F / m) * (cphi * cth) - g

    # Rotational accelerations
    phi_ddot = (tau_x + (Iy - Iz) * theta_dot * psi_dot) / Ix
    theta_ddot = (tau_y + (Iz - Ix) * phi_dot * psi_dot) / Iy
    psi_ddot = (tau_z + (Ix - Iy) * phi_dot * theta_dot) / Iz

    # Assemble derivative
    x_dot_nl = torch.stack([
        x_dot,
        x_ddot,
        y_dot,
        y_ddot,
        z_dot,
        z_ddot,
        phi_dot,
        phi_ddot,
        theta_dot,
        theta_ddot,
        psi_dot,
        psi_ddot
    ], dim=1)

    return x_dot_nl


def _ctrb(A, B):
    n = A.shape[0]
    mats = [B]
    AB = B
    for _ in range(1, n):
        AB = A @ AB
        mats.append(AB)
    return np.concatenate(mats, axis=1)


def _obsv(A, C):
    n = A.shape[0]
    mats = [C]
    CA = C
    for _ in range(1, n):
        CA = CA @ A
        mats.append(CA)
    return np.concatenate(mats, axis=0)


def controllability(A, B, tol=1e-9):
    n = A.shape[0]
    Co = _ctrb(A, B)
    r = matrix_rank(Co, tol)
    return r == n, r, Co


def reachability_index(A, B, tol=1e-9):
    n = A.shape[0]
    rank_prev = 0
    AB = B
    M = B
    for k in range(1, n + 1):
        r = matrix_rank(M, tol)
        if r == n:
            return k
        AB = A @ AB
        M = np.concatenate([M, AB], axis=1)
        if r == rank_prev:
            rank_prev = r
        else:
            rank_prev = r
    return None  # unreachable (shouldn't happen if full reachable)


def observability(A, C, tol=1e-9):
    n = A.shape[0]
    Ob = _obsv(A, C)
    r = matrix_rank(Ob, tol)
    return r == n, r, Ob


def pbh_stabilizable(A, B, discrete=False, tol=1e-9):
    """
    PBH stabilizability:
      - continuous: check λ with Re(λ) >= -tol
      - discrete:   check |λ| >= 1 - tol
    Need rank([λI - A, B]) == n for those λ.
    """
    n = A.shape[0]
    lam = eigvals(A)
    for l in lam:
        if (not discrete and np.real(l) >= -tol) or \
           (discrete and np.abs(l) >= 1 - tol):
            M = np.concatenate([l * np.eye(n) - A, B], axis=1)
            if matrix_rank(M, tol) < n:
                return False
    return True


def pbh_detectable(A, C, discrete=False, tol=1e-9):
    """
    PBH detectability is stabilizability of (A^T, C^T).
    Equivalent rank([λI - A^T, C^T]) == n for 'unstable' λ.
    """
    return pbh_stabilizable(A.T, C.T, discrete=discrete, tol=tol)


def asymptotically_stable(A, discrete=False, tol=1e-12):
    lam = eigvals(A)
    if discrete:
        return np.max(np.abs(lam)) < 1 - tol, lam
    return np.max(np.real(lam)) < -tol, lam


def summarize_checks(name, A, B, C, discrete=False, tol=1e-9):
    n = A.shape[0]
    ctrl_ok, ctrl_rank, _ = controllability(A, B, tol)
    obsv_ok, obsv_rank, _ = observability(A, C, tol)
    stab_ok = pbh_stabilizable(A, B, discrete=discrete, tol=tol)
    det_ok = pbh_detectable(A, C, discrete=discrete, tol=tol)
    stable_ok, lam = asymptotically_stable(A, discrete=discrete)

    r_idx = reachability_index(A, B, tol)

    print(f"\n=== {name} ({'discrete' if discrete else 'continuous'}) ===")
    print(f"n = {n}")
    print(f"Eigenvalues: {np.array2string(lam, precision=5)}")
    print(f"Asymptotically stable: {stable_ok}")
    print(f"Controllable: {ctrl_ok} (rank {ctrl_rank}/{n})")
    print(f"Reachability index: {r_idx}")
    print(f"Observable: {obsv_ok} (rank {obsv_rank}/{n})")
    print(f"Stabilizable (PBH): {stab_ok}")
    print(f"Detectable (PBH): {det_ok}")


def design_lqr(Ad, Bd, Q, R):
    # Discrete-time LQR
    P = la.solve_discrete_are(Ad, Bd, Q, R)
    K_LQR = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    return K_LQR


def gaussian_dist(mean, sigma, n_points, device="cpu"):
    """
    Generate a batch of Gaussian disturbances, constant over time
    for now (but structured [N, T, 1]).
    """
    samples = np.random.normal(loc=mean, scale=sigma, size=(n_points, 1))
    # Repeat along time dimension
    samples = np.repeat(samples, cfg.T_STEPS, axis=1)
    return torch.tensor(samples, dtype=torch.float32, device=device).unsqueeze(-1)
    # shape [N, T, 1]


def gaussian_dist_time_varying(mean, sigma, n_points, device="cpu"):
    """
    Generate a batch of Gaussian disturbances varying at each timestep.
    Shape: [N, T, 1]
    """
    samples = np.random.normal(
        loc=mean, scale=sigma, 
        size=(n_points, cfg.T_STEPS)
    )  # each timestep gets its own sample

    return torch.tensor(samples, dtype=torch.float32, device=device).unsqueeze(-1)


def gaussian_wind_direction(mean, sigma, N, direction_deg, device="cpu"):
    """
    Returns dx, dy disturbances of shape [N, T, 1]
    where the wind vector has fixed direction but random amplitude ~ N(mean, sigma²).

    direction_deg: wind angle in degrees (0° = +x, 90° = +y)
    """

    # 1) Sample scalar wind magnitude over time (your existing Gaussian time-varying)
    mag = gaussian_dist_time_varying(mean, sigma, N, device=device)   # [N, T, 1]

    # 2) Convert direction (angle in degrees → rad)
    theta = np.deg2rad(direction_deg)
    cos_t = torch.tensor(np.cos(theta), dtype=torch.float32, device=device)
    sin_t = torch.tensor(np.sin(theta), dtype=torch.float32, device=device)

    # 3) Scale by unit direction vector → (dx, dy)
    dx = mag * cos_t     # [N, T, 1]
    dy = mag * sin_t     # [N, T, 1]

    return dx, dy

    
def visualize_gaussian_samples(samples, mean, sigma, n, filename):

    if isinstance(samples, torch.Tensor):
        samples_np = samples.detach().cpu().numpy()
    else:
        samples_np = np.asarray(samples)
    
    samples = samples_np[:, 0, 0]

    # define range for true PDF
    x = np.linspace(mean - 4 * sigma, mean + 4 * sigma, 200)
    pdf = norm.pdf(x, mean, sigma)

    # compute histogram (without plotting yet)
    bins = 30
    counts, edges = np.histogram(samples, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # normalize both histogram and PDF by their joint max
    pdf_norm = pdf / pdf.max()
    counts_norm = counts / counts.max()

    # plot normalized histogram
    plt.figure(figsize=(6, 4))
    plt.bar(centers, counts_norm, width=edges[1] - edges[0],
            alpha=0.6, color='skyblue', edgecolor='black',
            label=fr'Samples from $\mathcal{{N}}({mean:.2f},\, {sigma:.2f})$ (n={n})')

    # plot normalized PDF
    plt.plot(x, pdf_norm, 'r-', lw=2, label=fr'$\mathcal{{N}}({mean:.2f},\, {sigma:.2f})$')

    plt.axvline(mean, color='k', linestyle='--', alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Normalized density")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=400)
    plt.close()


def visualize_multi_task_gaussians(tasks, means, sigmas, filename, only_one=False):
    """
    Visualize multiple Gaussian sample sets with normalized y axis in [0,1].
    
    tasks  : list of tensors, shape [N_i, T, 1]
    means  : list of means (same length as tasks)
    sigmas : list of sigmas (same length as tasks)
    """
    assert len(tasks) == len(means) == len(sigmas), \
        "tasks, means, sigmas must have identical length."

    plt.figure(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0, 1, len(tasks)))

    if only_one:
        colors = 'red'

    for i, (samples, mean, sigma) in enumerate(zip(tasks, means, sigmas)):

        # Extract samples
        samp = samples[:, 0, 0].detach().cpu().numpy()

        # Histogram (compute only, do not plot yet)
        counts, edges = np.histogram(samp, bins=25, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        # Normalize histogram
        if counts.max() > 0:
            hist_norm = counts / counts.max()
        else:
            hist_norm = counts

        # Plot histogram
        plt.bar(
            centers,
            hist_norm,
            width=edges[1] - edges[0],
            alpha=0.45,
            color=colors[i],
            edgecolor='none',
            label=fr"$\mathcal{{N}}({mean:.1f},{sigma:.1f})$"
        )

        # Theoretical PDF, normalized
        x = np.linspace(mean - 4*sigma, mean + 4*sigma, 200)
        pdf = norm.pdf(x, mean, sigma)
        pdf_norm = pdf / pdf.max()

        plt.plot(x, pdf_norm, color=colors[i], lw=2)

    plt.xlabel("Wind disturbance $d_x$")
    plt.ylabel("Normalized density")
    plt.ylim(0, 1.05)
    plt.xlim(-9, 9)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def visualize_train_vs_test_gaussians(tasks_train, train_means, train_sigmas, tasks_test,  test_means,  test_sigmas, filename):
    """
    Plot training and testing Gaussian disturbance distributions on one figure.
    
    tasks_train : list of tensors, shape [N_i, T, 1]
    tasks_test  : list of tensors, same format
    train_means / test_means : lists
    train_sigmas / test_sigmas : lists
    """

    assert len(tasks_train) == len(train_means) == len(train_sigmas)
    assert len(tasks_test)  == len(test_means)  == len(test_sigmas)

    plt.figure(figsize=(9, 4))

    # Color palettes
    train_colors = 'blue'
    test_colors  = 'red'

    # ---- TRAINING DISTRIBUTIONS ----
    for i, (samples, mean, sigma) in enumerate(zip(tasks_train, train_means, train_sigmas)):

        samp = samples[:, 0, 0].detach().cpu().numpy()

        counts, edges = np.histogram(samp, bins=25, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist_norm = counts / counts.max() if counts.max() > 0 else counts

        plt.bar(
            centers, hist_norm,
            width=edges[1] - edges[0],
            alpha=0.4,
            color=train_colors,
            edgecolor='none',
            label=fr"Training wind samples" if i == 0 else None
        )

        # PDF curve (normalized)
        x = np.linspace(mean - 4*sigma, mean + 4*sigma, 200)
        pdf = norm.pdf(x, mean, sigma)
        pdf_norm = pdf / pdf.max()
        plt.plot(x, pdf_norm, color=train_colors, alpha=0.8, lw=2)

    # ---- TESTING DISTRIBUTIONS ----
    for i, (samples, mean, sigma) in enumerate(zip(tasks_test, test_means, test_sigmas)):

        samp = samples[:, 0, 0].detach().cpu().numpy()

        counts, edges = np.histogram(samp, bins=25, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        hist_norm = counts / counts.max() if counts.max() > 0 else counts

        plt.bar(
            centers, hist_norm,
            width=edges[1] - edges[0],
            alpha=0.4,
            color=test_colors,
            edgecolor='none',
            label=fr"Testing wind samples" if i == 0 else None
        )

        # PDF curve (normalized)
        x = np.linspace(mean - 4*sigma, mean + 4*sigma, 200)
        pdf = norm.pdf(x, mean, sigma)
        pdf_norm = pdf / pdf.max()
        plt.plot(x, pdf_norm, color=test_colors, alpha=0.8, lw=2)

    # ---- Formatting ----
    plt.xlabel("Wind disturbance $d_x$")
    plt.ylabel("Normalized density")
    plt.ylim(0, 1.05)
    plt.xlim(-9, 9)
    plt.grid(True, alpha=0.25)

    # Show only unique legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = {label: h for h, label in zip(handles, labels) if label is not None}
    plt.legend(by_label.values(), by_label.keys(), fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def visualize_multi_task_gaussians_3D(tasks_all_disturbances, filename, cols=3):
    """
    Visualize normalized histograms of dx, dy, dz for each task.
    Works for ANY number of tasks (0, 1, or many).

    Parameters
    ----------
    tasks_all_disturbances : list of tensors [N, T, 3]
    filename : str
        Output image path
    cols : int
        Number of columns for multi-task grid
    """

    import matplotlib.pyplot as plt
    import numpy as np

    num_tasks = len(tasks_all_disturbances)
    if num_tasks == 0:
        raise ValueError("No tasks provided to visualize.")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # dx, dy, dz

    # ---------------------------------------------------------
    # Helper: normalized histogram
    # ---------------------------------------------------------
    def normalized_hist(a, bins=25):
        counts, edges = np.histogram(a, bins=bins, density=True)
        if counts.max() > 0:
            counts = counts / counts.max()
        centers = 0.5 * (edges[:-1] + edges[1:])
        return centers, counts, edges[1] - edges[0]

    # ---------------------------------------------------------
    # SINGLE-TASK CASE
    # ---------------------------------------------------------
    if num_tasks == 1:
        dist = tasks_all_disturbances[0].detach().cpu().numpy()

        dx = dist[:, :, 0].flatten()
        dy = dist[:, :, 1].flatten()
        dz = dist[:, :, 2].flatten()

        plt.figure(figsize=(6, 4))
        ax = plt.subplot(1, 1, 1)

        cx, hx, wx = normalized_hist(dx)
        cy, hy, wy = normalized_hist(dy)
        cz, hz, wz = normalized_hist(dz)

        ax.bar(cx, hx, width=wx, alpha=0.5, color=colors[0], label="$d_x$")
        ax.bar(cy, hy, width=wy, alpha=0.5, color=colors[1], label="$d_y$")
        ax.bar(cz, hz, width=wz, alpha=0.5, color=colors[2], label="$d_z$")

        ax.set_title("Task 0")
        ax.set_xlabel("Disturbance value")
        ax.set_ylabel("Normalized density")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-9, 9)
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        return

    # ---------------------------------------------------------
    # MULTI-TASK CASE
    # ---------------------------------------------------------
    rows = int(np.ceil(num_tasks / cols))
    plt.figure(figsize=(cols * 5, rows * 3.5))

    for i, dist_tensor in enumerate(tasks_all_disturbances):
        dist = dist_tensor.detach().cpu().numpy()

        dx = dist[:, :, 0].flatten()
        dy = dist[:, :, 1].flatten()
        dz = dist[:, :, 2].flatten()

        ax = plt.subplot(rows, cols, i + 1)

        cx, hx, wx = normalized_hist(dx)
        cy, hy, wy = normalized_hist(dy)
        cz, hz, wz = normalized_hist(dz)

        ax.bar(cx, hx, width=wx, alpha=0.5, color=colors[0], label="$d_x$")
        ax.bar(cy, hy, width=wy, alpha=0.5, color=colors[1], label="$d_y$")
        ax.bar(cz, hz, width=wz, alpha=0.5, color=colors[2], label="$d_z$")

        ax.set_title(f"Task {i}")
        ax.set_xlabel("Disturbance value")
        ax.set_ylabel("Normalized density")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-9, 9)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_disturbances_time(tasks_dx, filename, only_one=False):
    """
    Plots the evolution of d_x over time for each task and each trajectory.
    
    Accepts either:
        - A list of tensors [task_1_dx, task_2_dx, ...]
              each with shape [N, T, 1]
        - A single tensor with shape [N, T, 1]
    """
    # Convert single tensor into list
    if isinstance(tasks_dx, torch.Tensor):
        tasks_dx = [tasks_dx]

    # Determine number of tasks
    num_tasks = len(tasks_dx)

    # Create color palette
    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, num_tasks - 1)) for i in range(num_tasks)]

    if only_one:
        colors = 'red'

    plt.figure(figsize=(8, 6))

    # Loop over tasks
    for ti, dx_task in enumerate(tasks_dx):

        # Convert to numpy
        if isinstance(dx_task, torch.Tensor):
            dx_task = dx_task.detach().cpu().numpy()

        N, T, _ = dx_task.shape
        t = np.arange(T)

        # Plot all trajectories for this task
        for i in range(N):
            dx = dx_task[i, :, 0]   # shape [T]
            plt.plot(t, dx, alpha=0.5, color=colors[ti])

    plt.xlabel("Time step")
    plt.ylabel(r"$d_x$ (wind disturbance)")
    plt.ylim(-9, 9)
    plt.grid(True, alpha=0.3)
    plt.title("Time-varying wind disturbances for all training tasks")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


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


def animate_trajectories(X, filename):
    # Animation parameters
    triad_scale = 0.6
    triad_lw = 1.0
    trail_lw = 2.0
    stride = 1
    progress = True
    alpha = 0.5 # opacity parameter

    # Retrieve trajectories
    ix, iy, iz = 0, 2, 4
    iphi, itheta, ipsi = 6, 8, 10
    if hasattr(X, "detach"):  # torch tensor
        X = X.detach().cpu().numpy()
    B, T, n = X.shape
    xs, ys, zs = X[:, :, ix], X[:, :, iy], X[:, :, iz]

    # Validate T
    N_expected = int(round(cfg.T_SIM / cfg.T_D))
    if T != N_expected:
        print(f"[warn] T={T} differs from cfg.T_SIM/cfg.T_D={N_expected}. "
              f"Using T={T} from data.")

    # Frames & FPS
    frames = list(range(0, T, max(1, int(stride))))
    fps = max(1, int(round((1.0 / cfg.T_D) / max(1, int(stride)))))

    # Figure setup
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    axis_lim = 5.0
    ax.set_xlim([-axis_lim, axis_lim])
    ax.set_ylim([-axis_lim, axis_lim])
    ax.set_zlim([-axis_lim, axis_lim])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Quadcopter trajectories")

    colors = plt.cm.viridis(np.linspace(0, 1, B))

    # Trails and markers (with transparency)
    trails = [
        ax.plot([], [], [], lw=trail_lw, color=colors[b], alpha=alpha)[0]
        for b in range(B)
    ]
    markers = [
        ax.plot([], [], [], "o", color=colors[b], alpha=alpha)[0]
        for b in range(B)
    ]

    # Triads: R/G/B for body x/y/z (with transparency)
    triads = []
    for _ in range(B):
        tx, = ax.plot([], [], [], lw=triad_lw, color="r", alpha=alpha)
        ty, = ax.plot([], [], [], lw=triad_lw, color="g", alpha=alpha)
        tz, = ax.plot([], [], [], lw=triad_lw, color="b", alpha=alpha)
        triads.append((tx, ty, tz))

    def _euler_zyx_to_R(phi, theta, psi):
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)
        Rx = np.array([[1, 0, 0], [0, cph, -sph], [0, sph, cph]])
        Ry = np.array([[cth, 0, sth], [0, 1, 0], [-sth, 0, cth]])
        Rz = np.array([[cps, -sps, 0], [sps, cps, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    def init():
        return trails + markers + [l for tri in triads for l in tri]

    def update(i):
        if progress and (i % max(1, len(frames) // 50) == 0):
            print(f"Frame {frames.index(i) + 1}/{len(frames)}")

        for b in range(B):
            # trail & marker at frame i
            trails[b].set_data(xs[b, :i], ys[b, :i])
            trails[b].set_3d_properties(zs[b, :i])
            markers[b].set_data([xs[b, i]], [ys[b, i]])
            markers[b].set_3d_properties([zs[b, i]])

            # attitude triad
            phi, theta, psi = X[b, i, iphi], X[b, i, itheta], X[b, i, ipsi]
            R = _euler_zyx_to_R(phi, theta, psi)
            p = np.array([xs[b, i], ys[b, i], zs[b, i]])
            ex, ey, ez = R[:, 0], R[:, 1], R[:, 2]
            tx, ty, tz = triads[b]
            px, py, pz = (
                p + triad_scale * ex,
                p + triad_scale * ey,
                p + triad_scale * ez,
            )
            tx.set_data([p[0], px[0]], [p[1], px[1]])
            tx.set_3d_properties([p[2], px[2]])
            ty.set_data([p[0], py[0]], [p[1], py[1]])
            ty.set_3d_properties([p[2], py[2]])
            tz.set_data([p[0], pz[0]], [p[1], pz[1]])
            tz.set_3d_properties([p[2], pz[2]])
        return trails + markers + [l for tri in triads for l in tri]

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False)

    fps = int(round(1.0 / cfg.T_D))
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=-1)
    ani.save(filename, writer=writer)
    plt.close(fig)


def plot_angles(X, filename):
    if hasattr(X, "detach"):  # torch tensor
        X = X.detach().cpu().numpy()

    B, T, n = X.shape
    t = np.arange(T) * cfg.T_D

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
    plt.savefig(filename, dpi=300)
    plt.close(fig)

    
def plot_xyz(X, filename):
    if hasattr(X, "detach"):  # torch tensor
        X = X.detach().cpu().numpy()

    B, T, n = X.shape
    t = np.arange(T) * cfg.T_D

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
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def plot_controls_grid(U, X, filename):
    if hasattr(U, "detach"):  # torch tensor
        U = U.detach().cpu().numpy()
    B, T, m_in = U.shape
    assert m_in == 4, "U must be shape (B, T, 4)"

    t = np.arange(T) * cfg.T_D

    col_meta = [
        ("Thrust $F$ [N]",          cfg.F_MAX),
        (r"$\tau_x$ [N$\cdot$m]",   cfg.TAU_X_MAX),
        (r"$\tau_y$ [N$\cdot$m]",   cfg.TAU_Y_MAX),
        (r"$\tau_z$ [N$\cdot$m]",   cfg.TAU_Z_MAX),
    ]

    # optional titles per row
    row_titles = [f"Batch {b}" for b in range(B)]
    if X is not None:
        Xn = X.detach().cpu().numpy() if hasattr(X, "detach") else X
        for b in range(B):
            x0, y0, z0 = Xn[b, 0, 0], Xn[b, 0, 2], Xn[b, 0, 4]
            row_titles[b] = f"Batch {b} — $(x_0,y_0,z_0)=({x0:.1f},{y0:.1f},{z0:.1f})$"

    # figure
    fig, axs = plt.subplots(B, 4, sharex='col', figsize=(4*4, 2.6*B), squeeze=False)

    for b in range(B):
        for j, (ylabel, umax) in enumerate(col_meta):
            ax = axs[b, j]
            ax.plot(t, U[b, :, j], label="command")

            if j == 0:  # thrust: [0, cfg.F_MAX]
                ax.axhline(0.0, ls="--", c="k", lw=1, alpha=0.7,
                        label="min" if (b==0 and j==0) else None)
                ax.axhline(umax, ls="--", c="k", lw=1, alpha=0.7,
                        label="max" if (b==0 and j==0) else None)
            else:       # torques: [-umax, +umax]
                ax.axhline( umax, ls="--", c="k", lw=1, alpha=0.7,
                        label="+limit" if (b==0 and j==1) else None)
                ax.axhline(-umax, ls="--", c="k", lw=1, alpha=0.7,
                        label="-limit" if (b==0 and j==1) else None)

            ax.grid(True)
            if b == 0:
                ax.set_title(ylabel)
            if b == B-1:
                ax.set_xlabel("Time [s]")
            if j == 0:
                ax.set_ylabel(row_titles[b])

    # one legend (top-left only) to avoid clutter
    axs[0,0].legend(loc="best")

    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


def plot_angles_mean(X, filename):
    color_map = plt.get_cmap("cividis")

    if hasattr(X, "detach"):  # torch tensor
        X = X.detach().cpu().numpy()

    B, T, n = X.shape
    t = np.arange(T) * cfg.T_D

    # Extract roll, pitch, yaw (indices 6, 8, 10)
    roll  = np.rad2deg(X[:, :, 6])
    pitch = np.rad2deg(X[:, :, 8])
    yaw   = np.rad2deg(X[:, :, 10])

    # Compute mean, min, and max across batch dimension
    roll_mean, pitch_mean, yaw_mean = (
        roll.mean(axis=0),
        pitch.mean(axis=0),
        yaw.mean(axis=0),
    )
    roll_min, pitch_min, yaw_min = (
        roll.min(axis=0),
        pitch.min(axis=0),
        yaw.min(axis=0),
    )
    roll_max, pitch_max, yaw_max = (
        roll.max(axis=0),
        pitch.max(axis=0),
        yaw.max(axis=0),
    )

    plt.figure(figsize=(8, 4))

    for mean, vmin, vmax, label, symbol, color in zip(
        [roll_mean, pitch_mean, yaw_mean],
        [roll_min, pitch_min, yaw_min],
        [roll_max, pitch_max, yaw_max],
        ["roll", "pitch", "yaw"],
        [r"\phi", r"\theta", r"\psi"],
        [color_map(0.2), color_map(0.6), color_map(0.9)]
    ):
        # Mean curve
        plt.plot(t, mean, color=color, label=fr"$\bar{{{symbol}}}$ (mean)")
        # Min–max shading
        plt.fill_between(
            t, vmin, vmax, color=color, alpha=0.2,
            label=fr"Range of ${symbol}$"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Angle (°)")
    plt.title("Mean Orientation Angles (shaded: min–max range across batch)")
    plt.legend(loc="best", fontsize=10, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_xyz_mean(X, filename):
    color_map = plt.get_cmap("plasma")

    if hasattr(X, "detach"):  # torch tensor
        X = X.detach().cpu().numpy()

    B, T, n = X.shape
    t = np.arange(T) * cfg.T_D

    # Extract x, y, z (assuming indices 0, 2, 4)
    x = X[:, :, 0]
    y = X[:, :, 2]
    z = X[:, :, 4]

    # Compute mean, min, max across batch
    x_mean, y_mean, z_mean = x.mean(axis=0), y.mean(axis=0), z.mean(axis=0)
    x_min,  y_min,  z_min  = x.min(axis=0),  y.min(axis=0),  z.min(axis=0)
    x_max,  y_max,  z_max  = x.max(axis=0),  y.max(axis=0),  z.max(axis=0)

    plt.figure(figsize=(8, 4))

    for mean, vmin, vmax, label, color in zip(
        [x_mean, y_mean, z_mean],
        [x_min,  y_min,  z_min],
        [x_max,  y_max,  z_max],
        ["x", "y", "z"],
        [color_map(0.2), color_map(0.4), color_map(0.7)]
    ):
        # Mean curve
        plt.plot(t, mean, color=color, label=fr"$\bar{{{label}}}$ (mean)")
        # Min–max shading
        plt.fill_between(
            t, vmin, vmax, color=color, alpha=0.2,
            label=fr"Range of ${label}$"
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.title("Mean Position Trajectories (shaded: min–max range across batch)")
    plt.legend(loc="best", fontsize=10, frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_controls_grid_mean(U, filename):
    color_map = plt.get_cmap("inferno")

    if hasattr(U, "detach"):  # torch tensor
        U = U.detach().cpu().numpy()

    B, T, m_in = U.shape
    assert m_in == 4, "U must have shape (B, T, 4)"
    t = np.arange(T) * cfg.T_D

    # Metadata for each control channel
    col_meta = [
        (r"Thrust $F$ (N)",        r"F",      cfg.F_MAX),
        (r"$\tau_x$ (N$\cdot$m)",  r"\tau_x", cfg.TAU_X_MAX),
        (r"$\tau_y$ (N$\cdot$m)",  r"\tau_y", cfg.TAU_Y_MAX),
        (r"$\tau_z$ (N$\cdot$m)",  r"\tau_z", cfg.TAU_Z_MAX),
    ]

    fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    axs = np.atleast_1d(axs)

    for j, (ylabel, symbol, umax) in enumerate(col_meta):
        u = U[:, :, j]
        u_mean = u.mean(axis=0)
        u_min  = u.min(axis=0)
        u_max  = u.max(axis=0)
        color = color_map(0.3 + 0.15 * j)

        # Mean curve
        axs[j].plot(t, u_mean, color=color, label=fr"$\bar{{{symbol}}}$ (mean)")
        # Min–max shading
        axs[j].fill_between(
            t, u_min, u_max, color=color, alpha=0.2,
            label=fr"Range of ${symbol}$"
        )

        # Control limits
        if j == 0:  # thrust (0 to F_MAX)
            axs[j].axhline(0.0, ls="--", c="k", lw=1, alpha=0.7, label="limit")
            axs[j].axhline(umax, ls="--", c="k", lw=1, alpha=0.7)
        else:  # torques (-τmax to +τmax)
            axs[j].axhline( umax, ls="--", c="k", lw=1, alpha=0.7)
            axs[j].axhline(-umax, ls="--", c="k", lw=1, alpha=0.7)

        axs[j].set_title(ylabel)
        axs[j].grid(True)
        axs[j].legend(loc="best", fontsize=9, frameon=True)
        axs[j].set_xlabel("Time (s)")

    axs[0].set_ylabel("Command Value")
    fig.suptitle("Mean Control Inputs (shaded: min–max range across batch)")
    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()