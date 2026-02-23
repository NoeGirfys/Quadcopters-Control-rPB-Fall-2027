"""
LQR controller for a Crazyflie 2.x in gym-pybullet-drones.

v2 — fixes from initial testing:
  • angular velocity: world→body frame rotation
  • anti-flip safety cutoff when angles exceed linear regime
  • softer LQR gains (higher R on torques)
  • incremental integral term on Z for steady-state offset

Usage:
    python lqr_pybullet_drone_v2.py --mode single --gui
    python lqr_pybullet_drone_v2.py --mode batch --no-gui
"""

import numpy as np
import scipy.linalg as la
from scipy.signal import cont2discrete
import matplotlib.pyplot as plt
import argparse, time


# gym-pybullet-drones
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary


# ══════════════════════════════════════════════════════════════════════
#  1.  CRAZYFLIE PHYSICAL PARAMETERS
# ══════════════════════════════════════════════════════════════════════
CF_MASS     = 0.027
CF_G        = 9.81
CF_IX       = 1.4e-5
CF_IY       = 1.4e-5
CF_IZ       = 2.17e-5
CF_ARM      = 0.0397
CF_KF       = 3.16e-10
CF_KM       = 7.94e-12
CF_MAX_RPM  = 21702.63
CF_HOVER_RPM = np.sqrt(CF_MASS * CF_G / (4 * CF_KF))  # ≈ 14577


# ══════════════════════════════════════════════════════════════════════
#  2.  LINEARIZED MODEL + LQR
# ══════════════════════════════════════════════════════════════════════
def quad_hover_linear_model(m, g, Ix, Iy, Iz):
    """
    12-state linearized hover model.
    State:  [X, Ẋ, Y, Ẏ, Z, Ż, φ, φ̇, θ, θ̇, ψ, ψ̇]
    Input:  [ΔF, τx, τy, τz]
    """
    Ac = np.zeros((12, 12))
    Ac[0,1] = 1.0;  Ac[2,3] = 1.0;  Ac[4,5] = 1.0
    Ac[6,7] = 1.0;  Ac[8,9] = 1.0;  Ac[10,11] = 1.0
    Ac[1,8] =  g    # Ẍ ← θ
    Ac[3,6] = -g    # Ÿ ← φ

    Bc = np.zeros((12, 4))
    Bc[5,0]  = 1.0 / m
    Bc[7,1]  = 1.0 / Ix
    Bc[9,2]  = 1.0 / Iy
    Bc[11,3] = 1.0 / Iz

    return Ac, Bc, np.eye(12), np.zeros((12, 4))


def discretize(Ac, Bc, Cc, Dc, Ts):
    Ad, Bd, Cd, Dd, _ = cont2discrete((Ac, Bc, Cc, Dc), Ts, method="zoh")
    return Ad, Bd


def design_lqr(Ad, Bd, Q, R):
    P = la.solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    return K


def build_lqr_gain(Ts=0.01):
    Ac, Bc, Cc, Dc = quad_hover_linear_model(CF_MASS, CF_G, CF_IX, CF_IY, CF_IZ)
    Ad, Bd = discretize(Ac, Bc, Cc, Dc, Ts)

    # ── Q weights (Bryson's rule) ──
    pos_max      = 0.5                 # tighter position tracking
    vel_max      = 1.0
    ang_max      = np.deg2rad(15)      # keep angles small!
    rate_max     = np.deg2rad(60)
    psi_max      = np.deg2rad(30)
    psi_rate_max = np.deg2rad(45)

    q = np.array([
        1/pos_max**2,  1/vel_max**2,       # X, Ẋ
        1/pos_max**2,  1/vel_max**2,       # Y, Ẏ
        1/pos_max**2,  1/vel_max**2,       # Z, Ż
        1/ang_max**2,  1/rate_max**2,      # φ, φ̇
        1/ang_max**2,  1/rate_max**2,      # θ, θ̇
        1/psi_max**2,  1/psi_rate_max**2,  # ψ, ψ̇
    ])
    Q = np.diag(q)

    # ── R weights — Bryson's rule on realistic Crazyflie actuator limits ──
    Fmax     = CF_MASS * CF_G         # max thrust deviation ≈ full weight
    tau_max  = 0.005                   # roll/pitch torque budget [N·m]
    tauz_max = 0.002                   # yaw torque budget [N·m]
    r = np.array([1/Fmax**2, 1/tau_max**2, 1/tau_max**2, 1/tauz_max**2])
    R = np.diag(r)

    K = design_lqr(Ad, Bd, Q, R)
    return K, Ad, Bd


# ══════════════════════════════════════════════════════════════════════
#  3.  WRENCH → RPM (allocation matrix)
# ══════════════════════════════════════════════════════════════════════
def build_allocation_matrix():
    """
    X-config Crazyflie 2.x:
        M0 (FL, CW)   M1 (FR, CCW)
        M3 (BL, CCW)  M2 (BR, CW)

    wrench = A · [ω0² ω1² ω2² ω3²]^T
    """
    d = CF_ARM / np.sqrt(2)
    A = np.array([
        [ CF_KF,      CF_KF,      CF_KF,      CF_KF     ],
        [+d*CF_KF,   -d*CF_KF,   -d*CF_KF,   +d*CF_KF  ],
        [+d*CF_KF,   +d*CF_KF,   -d*CF_KF,   -d*CF_KF  ],
        [-CF_KM,     +CF_KM,     -CF_KM,     +CF_KM    ],
    ])
    A_inv = np.linalg.inv(A)
    return A, A_inv

ALLOC_A, ALLOC_A_INV = build_allocation_matrix()


def wrench_to_rpm(F_total, tau_x, tau_y, tau_z):
    wrench = np.array([F_total, tau_x, tau_y, tau_z])
    rpm_sq = ALLOC_A_INV @ wrench
    rpm_sq = np.clip(rpm_sq, 0, CF_MAX_RPM**2)
    rpms = np.sqrt(rpm_sq)
    return np.clip(rpms, 0, CF_MAX_RPM)


# ══════════════════════════════════════════════════════════════════════
#  4.  STATE EXTRACTION  (with world→body ang. vel. fix)
# ══════════════════════════════════════════════════════════════════════
def rotation_matrix_ZYX(phi, theta, psi):
    """R_world_from_body  (ZYX intrinsic = extrinsic XYZ)."""
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([
        [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
        [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
        [  -st,             ct*sp,             ct*cp ],
    ])


def get_drone_state(env, drone_id=0):
    """
    Pack PyBullet state → 12-vector [X,Ẋ,Y,Ẏ,Z,Ż,φ,φ̇,θ,θ̇,ψ,ψ̇].

    CRITICAL: PyBullet's getBaseVelocity returns angular velocity in WORLD
    frame. The linearized model expects BODY-frame rates. We rotate:
        ω_body = R^T · ω_world
    """
    pos   = env.pos[drone_id]       # (3,)
    vel   = env.vel[drone_id]       # (3,)
    rpy   = env.rpy[drone_id]       # (3,)  roll, pitch, yaw
    ang_w = env.ang_v[drone_id]     # (3,)  angular vel in WORLD frame

    # Rotate angular velocity to body frame
    R = rotation_matrix_ZYX(rpy[0], rpy[1], rpy[2])
    ang_b = R.T @ ang_w             # body-frame angular velocity

    return np.array([
        pos[0],  vel[0],      # X, Ẋ
        pos[1],  vel[1],      # Y, Ẏ
        pos[2],  vel[2],      # Z, Ż
        rpy[0],  ang_b[0],    # φ, φ̇  (body)
        rpy[1],  ang_b[1],    # θ, θ̇  (body)
        rpy[2],  ang_b[2],    # ψ, ψ̇  (body)
    ])


# ══════════════════════════════════════════════════════════════════════
#  5.  ANGLE WRAPPING  (keep errors in [-π, π])
# ══════════════════════════════════════════════════════════════════════
def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_error(state, xref):
    """State error with proper angle wrapping on φ, θ, ψ."""
    err = state - xref
    err[6]  = wrap_angle(err[6])    # φ
    err[8]  = wrap_angle(err[8])    # θ
    err[10] = wrap_angle(err[10])   # ψ
    return err


# ══════════════════════════════════════════════════════════════════════
#  6.  MAIN SIMULATION
# ══════════════════════════════════════════════════════════════════════
def make_target_state(target_pos):
    xref = np.zeros(12)
    xref[0] = target_pos[0]
    xref[2] = target_pos[1]
    xref[4] = target_pos[2]
    return xref


def run_lqr_simulation(
    init_pos=np.array([0.0, 0.0, 0.5]),
    target_pos=np.array([0.0, 0.0, 1.0]),
    T_sim=10.0,
    ctrl_freq=50,
    gui=True,
    record_video=False,
):
    Ts = 1.0 / ctrl_freq
    K_lqr, Ad, Bd = build_lqr_gain(Ts)
    xref = make_target_state(target_pos)
    F_hover = CF_MASS * CF_G

    print(f"LQR gain K (first row): {K_lqr[0,:4]}...")
    print(f"Hover RPM: {CF_HOVER_RPM:.1f},  Hover thrust: {F_hover*1e3:.1f} mN")
    print(f"Init: {init_pos}  →  Target: {target_pos}")

    # ── Create environment ──
    # pyb_freq must be divisible by ctrl_freq. Default pyb_freq=240.
    pyb_freq = 240
    assert pyb_freq % ctrl_freq == 0, \
        f"pyb_freq={pyb_freq} must be divisible by ctrl_freq={ctrl_freq}. " \
        f"Try ctrl_freq in {{24, 30, 40, 48, 60, 80, 120, 240}}."

    env = CtrlAviary(
        drone_model=DroneModel.CF2X,
        num_drones=1,
        initial_xyzs=init_pos.reshape(1, 3),
        physics=Physics.PYB,
        pyb_freq=pyb_freq,
        ctrl_freq=ctrl_freq,
        gui=gui,
        record=record_video,
    )

    N_steps = int(T_sim * ctrl_freq)
    state_log  = np.zeros((N_steps, 12))
    wrench_log = np.zeros((N_steps, 4))
    rpm_log    = np.zeros((N_steps, 4))
    time_log   = np.arange(N_steps) * Ts

    # ── Integral term on Z for steady-state accuracy ──
    z_integral = 0.0
    KI_Z = 0.8          # integral gain on altitude error

    # ── Torque limits (physical, for Crazyflie) ──
    TAU_XY_MAX = 0.005   # N·m — roll/pitch torque clamp
    TAU_Z_MAX  = 0.002   # N·m — yaw torque clamp

    # ── Anti-flip threshold ──
    ANGLE_SAFE = np.deg2rad(40)

    obs, info = env.reset()
    t_start = time.time()

    for k in range(N_steps):
        # 1) Read state
        state = get_drone_state(env, drone_id=0)

        # 2) Error with angle wrapping
        error = compute_error(state, xref)

        # 3) Check if angles are in the linear regime
        phi_abs   = abs(state[6])
        theta_abs = abs(state[8])

        if phi_abs > ANGLE_SAFE or theta_abs > ANGLE_SAFE:
            # ── SAFETY MODE: reduce thrust, don't apply torques ──
            # Let gravity + reduced lift bring it back down / dampen
            rpms = np.full(4, CF_HOVER_RPM * 0.5)
            wrench = np.array([F_hover * 0.25, 0.0, 0.0, 0.0])
            z_integral = 0.0  # reset integrator — we've lost tracking
        else:
            # ── NORMAL LQR ──
            u_lqr = -K_lqr @ error      # [ΔF, τx, τy, τz]

            # Altitude integral action
            z_err = error[4]             # Z - Z_target
            z_integral += z_err * Ts
            z_integral = np.clip(z_integral, -0.5, 0.5)  # anti-windup

            delta_F = u_lqr[0] - KI_Z * z_integral
            F_total = F_hover + delta_F
            F_total = np.clip(F_total, 0.0, 4 * CF_KF * CF_MAX_RPM**2)

            tau_x = np.clip(u_lqr[1], -TAU_XY_MAX, TAU_XY_MAX)
            tau_y = np.clip(u_lqr[2], -TAU_XY_MAX, TAU_XY_MAX)
            tau_z = np.clip(u_lqr[3], -TAU_Z_MAX,  TAU_Z_MAX)

            rpms = wrench_to_rpm(F_total, tau_x, tau_y, tau_z)
            wrench = np.array([F_total, tau_x, tau_y, tau_z])

        # 4) Step environment
        action = rpms.reshape(1, 4)
        obs, reward, terminated, truncated, info = env.step(action)

        # 5) Log
        state_log[k]  = state
        wrench_log[k] = wrench
        rpm_log[k]    = rpms

        # Real-time pacing in GUI
        if gui:
            elapsed = time.time() - t_start
            expected = (k + 1) * Ts
            if elapsed < expected:
                time.sleep(expected - elapsed)

    wall = time.time() - t_start
    print(f"Done: {wall:.1f}s wall, {T_sim/wall:.2f}x real-time")

    # ── Final state ──
    final = state_log[-1]
    print(f"Final position:  ({final[0]:.3f}, {final[2]:.3f}, {final[4]:.3f})")
    print(f"Final angles:    φ={np.rad2deg(final[6]):.1f}°  θ={np.rad2deg(final[8]):.1f}°  ψ={np.rad2deg(final[10]):.1f}°")
    pos_err = np.linalg.norm([final[0]-target_pos[0], final[2]-target_pos[1], final[4]-target_pos[2]])
    print(f"Position error:  {pos_err*100:.1f} cm")

    env.close()
    return time_log, state_log, wrench_log, rpm_log, target_pos


# ══════════════════════════════════════════════════════════════════════
#  7.  PLOTTING
# ══════════════════════════════════════════════════════════════════════
def plot_results(time_log, state_log, wrench_log, rpm_log, target_pos,
                 save_prefix="lqr_pybullet_v2"):
    t = time_log
    F_hover = CF_MASS * CF_G

    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)
    fig.suptitle("LQR Hover Control  --  Crazyflie 2.x in PyBullet (v2)", fontsize=13)

    # ── Position ──
    ax = axs[0]
    ax.plot(t, state_log[:, 0], label='x')
    ax.plot(t, state_log[:, 2], label='y')
    ax.plot(t, state_log[:, 4], label='z')
    ax.axhline(target_pos[0], ls='--', color='C0', alpha=0.4)
    ax.axhline(target_pos[1], ls='--', color='C1', alpha=0.4)
    ax.axhline(target_pos[2], ls='--', color='C2', alpha=0.4, label=f'z target={target_pos[2]}')
    ax.set_ylabel('Position [m]')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # ── Angles ──
    ax = axs[1]
    ax.plot(t, np.rad2deg(state_log[:, 6]),  label='roll')
    ax.plot(t, np.rad2deg(state_log[:, 8]),  label='pitch')
    ax.plot(t, np.rad2deg(state_log[:, 10]), label='yaw')
    ax.axhline(+np.rad2deg(np.deg2rad(40)), ls=':', color='k', alpha=0.3, label='safety limit')
    ax.axhline(-np.rad2deg(np.deg2rad(40)), ls=':', color='k', alpha=0.3)
    ax.set_ylabel('Angle [deg]')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # ── RPMs ──
    ax = axs[2]
    for i in range(4):
        ax.plot(t, rpm_log[:, i], label=f'M{i}')
    ax.axhline(CF_HOVER_RPM, ls='-.', color='gray', alpha=0.5, label='Hover')
    ax.axhline(CF_MAX_RPM,   ls='--', color='gray', alpha=0.5, label='MAX')
    ax.set_ylabel('RPM')
    ax.legend(loc='best', ncol=3)
    ax.grid(True, alpha=0.3)

    # ── Thrust & Torques (dual y-axis) ──
    ax = axs[3]
    ax.plot(t, wrench_log[:, 0], 'C0', label='F total [N]')
    ax.axhline(F_hover, ls='--', color='C0', alpha=0.4, label=f'mg={F_hover:.4f}N')
    ax.set_ylabel('Thrust [N]')
    ax.set_xlabel('Time [s]')

    ax2 = ax.twinx()
    ax2.plot(t, wrench_log[:, 1] * 1e6, 'C1', alpha=0.7, label=r'$\tau_x$ [µNm]')
    ax2.plot(t, wrench_log[:, 2] * 1e6, 'C2', alpha=0.7, label=r'$\tau_y$ [µNm]')
    ax2.plot(t, wrench_log[:, 3] * 1e6, 'C3', alpha=0.7, label=r'$\tau_z$ [µNm]')
    ax2.set_ylabel(r'Torques [$\mu$N$\cdot$m]')

    # Merge legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, loc='best', ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(f"{save_prefix}_results.png", dpi=150)
    plt.show()

    # ── 3D trajectory ──
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(state_log[:, 0], state_log[:, 2], state_log[:, 4],
            'b-', lw=1.5, label='Trajectory')
    ax.scatter(*target_pos, color='r', s=100, marker='*', label='Target')
    ax.scatter(state_log[0, 0], state_log[0, 2], state_log[0, 4],
               color='g', s=80, marker='o', label='Start')
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title('3D trajectory')
    ax.legend()
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_3d.png", dpi=150)
    plt.show()


# ══════════════════════════════════════════════════════════════════════
#  8.  BATCH SIMULATION
# ══════════════════════════════════════════════════════════════════════
def run_batch_simulation(
    init_positions,
    target_pos=np.array([0.0, 0.0, 1.0]),
    T_sim=10.0,
    ctrl_freq=50,
    gui=False,
):
    all_states = []
    all_times = None

    for i, ip in enumerate(init_positions):
        print(f"\n{'='*50}")
        print(f"Trajectory {i+1}/{len(init_positions)}: init={ip}")
        t, states, wrench, rpms, _ = run_lqr_simulation(
            init_pos=np.array(ip),
            target_pos=target_pos,
            T_sim=T_sim,
            ctrl_freq=ctrl_freq,
            gui=gui,
        )
        all_states.append(states)
        if all_times is None:
            all_times = t

    # ── Comparison plot ──
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(init_positions)))
    labels_xyz = ['X', 'Y', 'Z']
    indices = [0, 2, 4]

    for ax, lbl, idx, tgt in zip(axs, labels_xyz, indices, target_pos):
        for j, (states, ip) in enumerate(zip(all_states, init_positions)):
            ax.plot(all_times, states[:, idx], color=colors[j],
                    label=f'({ip[0]:.1f},{ip[1]:.1f},{ip[2]:.1f})')
        ax.axhline(tgt, ls='--', color='r', alpha=0.5)
        ax.set_ylabel(f'{lbl} [m]')
        ax.grid(True, alpha=0.3)

    axs[0].legend(loc='best', fontsize=8, ncol=2)
    axs[0].set_title(f'Batch: multiple starts → target {target_pos}')
    axs[-1].set_xlabel('Time [s]')
    fig.tight_layout()
    plt.savefig("lqr_pybullet_v2_batch.png", dpi=150)
    plt.show()

    return all_states


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LQR drone control in PyBullet (v2)")
    parser.add_argument("--mode", choices=["single", "batch"], default="single")
    parser.add_argument("--gui", action="store_true", default=True)
    parser.add_argument("--no-gui", dest="gui", action="store_false")
    parser.add_argument("--T", type=float, default=10.0)
    parser.add_argument("--freq", type=int, default=48, help="Control frequency (Hz). Must divide 240.")
    parser.add_argument("--target", nargs=3, type=float, default=[0.0, 0.0, 1.0])
    args = parser.parse_args()

    target = np.array(args.target)

    if args.mode == "single":
        t, states, wrench, rpms, tgt = run_lqr_simulation(
            init_pos=np.array([0.0, 0.0, 0.8]),  # close to target for small-angle validity
            target_pos=target,
            T_sim=args.T,
            ctrl_freq=args.freq,
            gui=args.gui,
        )
        plot_results(t, states, wrench, rpms, tgt)

    elif args.mode == "batch":
        init_positions = [
            [0.0, 0.0, 0.8],     # directly below target
            [0.1, 0.0, 0.7],     # small lateral offset
            [0.0, 0.1, 0.9],     # small lateral, close in Z
            [-0.1, -0.1, 0.6],   # diagonal offset
            [0.2, 0.2, 0.5],     # larger offset — tests limits
        ]
        run_batch_simulation(
            init_positions=init_positions,
            target_pos=target,
            T_sim=args.T,
            ctrl_freq=args.freq,
            gui=False,
        )