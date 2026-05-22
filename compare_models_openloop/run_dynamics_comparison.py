"""
Dynamics Comparison: gym-pybullet-drones vs Goffin Model
========================================================

This script:
1. Runs the pid.py circular trajectory in gym-pybullet-drones
2. Logs RPMs and full state at every physics substep
3. Replays the same RPMs through Goffin's model (nonlinear + linearized)
4. Compares trajectories (open-loop: same initial state, diverge freely)

Supports both CF2P (+) and CF2X (x) configurations,
and multiple Physics modes (PYB, DYN, PYB_GND_DRAG_DW, etc.)

Usage:
    cd compare_models_openloop
    python run_dynamics_comparison.py

Requirements:
    - gym-pybullet-drones installed/in path
    - numpy, matplotlib
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Import gym-pybullet-drones ──────────────────────────────────────
# Adjust this path if needed to point to your gym-pybullet-drones repo
# sys.path.insert(0, '/path/to/your/repo')

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# ── Import Goffin dynamics ──────────────────────────────────────────
from goffin_dynamics import GoffinDynamics, gym_state_to_goffin, goffin_state_labels


# =====================================================================
#  STEP 1: Run gym-pybullet-drones and log at every physics substep
# =====================================================================

def run_and_log_substeps(drone_model=DroneModel.CF2X,
                          physics=Physics.PYB,
                          ctrl_freq=48,
                          pyb_freq=240,
                          duration_sec=3.0):
    """
    Reproduce the pid.py circular trajectory and log state at EVERY physics
    substep for accurate comparison with Goffin's model.

    Trajectory setup is copied from gym_pybullet_drones/examples/pid.py:
    circular path in X-Y plane, radius R=0.3, period=10s.
    """
    import pybullet as p

    # ── pid.py trajectory parameters ──
    H = 1.0
    R = 0.3
    PERIOD = 10
    NUM_WP = ctrl_freq * PERIOD

    INIT_XYZS = np.array([[
        R * np.cos(np.pi / 2),
        R * np.sin(np.pi / 2) - R,
        H
    ]])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])

    # Circular waypoints (same formula as pid.py)
    TARGET_POS = np.zeros((NUM_WP, 3))
    for i in range(NUM_WP):
        TARGET_POS[i, :] = [
            R * np.cos((i / NUM_WP) * 2 * np.pi + np.pi / 2) + INIT_XYZS[0, 0],
            R * np.sin((i / NUM_WP) * 2 * np.pi + np.pi / 2) - R + INIT_XYZS[0, 1],
            0
        ]
    wp_counter = 0

    print(f"[SIM] drone={drone_model.name}, physics={physics.name}, "
          f"pyb_freq={pyb_freq}, ctrl_freq={ctrl_freq}, duration={duration_sec}s")
    print(f"[SIM] Trajectory: circular, R={R}, period={PERIOD}s, "
          f"init_xyz={INIT_XYZS[0]}")

    env = CtrlAviary(
        drone_model=drone_model,
        num_drones=1,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        pyb_freq=pyb_freq,
        ctrl_freq=ctrl_freq,
        gui=False,
    )

    params = {
        'm':  env.M,
        'g':  env.G,
        'Ix': env.J[0, 0],
        'Iy': env.J[1, 1],
        'Iz': env.J[2, 2],
        'l':  env.L,
        'kf': env.KF,
        'km': env.KM,
    }

    pid = DSLPIDControl(drone_model=drone_model)

    dt = 1.0 / pyb_freq
    steps_per_ctrl = pyb_freq // ctrl_freq
    total_ctrl_steps = int(duration_sec * ctrl_freq)

    all_states = []
    all_rpms = []

    obs, info = env.reset()

    # Log initial state
    env._updateAndStoreKinematicInformation()
    s0 = gym_state_to_goffin(env.pos[0], env.rpy[0], env.vel[0], env.ang_v[0])
    all_states.append(s0.copy())

    print(f"[SIM-SUBSTEP] Running {total_ctrl_steps} ctrl steps x {steps_per_ctrl} substeps...")

    for ctrl_step in range(total_ctrl_steps):
        # Get current state for PID
        state_20 = env._getDroneStateVector(0) # 0 for first (and only) drone

        # Target = circular waypoint at current altitude (same as pid.py)
        target = np.hstack([TARGET_POS[wp_counter, 0:2], INIT_XYZS[0, 2]])

        rpm_cmd, _, _ = pid.computeControlFromState(
            control_timestep=1.0 / ctrl_freq,
            state=state_20,
            target_pos=target,
            target_rpy=INIT_RPYS[0, :],
        )

        # Advance waypoint (same as pid.py)
        wp_counter = wp_counter + 1 if wp_counter < (NUM_WP - 1) else 0

        # Clip RPMs (mimic _preprocessAction)
        clipped_rpm = np.clip(rpm_cmd, 0, env.MAX_RPM)

        # Manually do the physics substeps to log intermediate states
        for sub in range(steps_per_ctrl):
            all_rpms.append(clipped_rpm.copy())

            # Apply forces (depends on physics mode)
            if physics in [Physics.PYB, Physics.PYB_GND, Physics.PYB_DRAG,
                           Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                env._physics(clipped_rpm, 0)

                if physics in [Physics.PYB_GND, Physics.PYB_GND_DRAG_DW]:
                    env._groundEffect(clipped_rpm, 0)
                if physics in [Physics.PYB_DRAG, Physics.PYB_GND_DRAG_DW]:
                    env._drag(env.last_clipped_action[0], 0)
                if physics in [Physics.PYB_DW, Physics.PYB_GND_DRAG_DW]:
                    env._downwash(0)

                p.stepSimulation(physicsClientId=env.CLIENT)

            elif physics == Physics.DYN:
                env._dynamics(clipped_rpm, 0)

            env.last_clipped_action[0] = clipped_rpm

            # Read state after this substep
            env._updateAndStoreKinematicInformation()
            s = gym_state_to_goffin(env.pos[0], env.rpy[0], env.vel[0], env.ang_v[0])
            all_states.append(s.copy())

        # Update step counter (for consistency with env internals)
        env.step_counter += steps_per_ctrl

    env.close()

    all_states = np.array(all_states)
    all_rpms = np.array(all_rpms)

    print(f"[SIM-SUBSTEP] Done. {all_states.shape[0]} states, {all_rpms.shape[0]} RPMs logged.")

    return {
        'states': all_states,
        'rpms': all_rpms,
        'dt': dt,
        'ctrl_freq': ctrl_freq,
        'pyb_freq': pyb_freq,
        'steps_per_ctrl': steps_per_ctrl,
        'drone_model': drone_model,
        'physics': physics,
        'params': params,
    }


# =====================================================================
#  STEP 2: Replay through Goffin model
# =====================================================================

def replay_open_loop(goffin: GoffinDynamics, initial_state, rpms, model='nonlinear'):
    """
    Open-loop replay: start from initial_state, apply RPMs sequence.
    Errors accumulate freely.

    Parameters
    ----------
    goffin : GoffinDynamics instance
    initial_state : ndarray (12,)
    rpms : ndarray (N, 4)
    model : 'nonlinear' or 'linearized'

    Returns
    -------
    states : ndarray (N+1, 12)
    """
    N = rpms.shape[0]
    states = np.zeros((N + 1, 12))
    states[0] = initial_state.copy()

    for k in range(N):
        states[k + 1] = goffin.step(states[k], rpms[k], model=model)

    return states



# =====================================================================
#  STEP 3: Error metrics
# =====================================================================

# State groups (indices into the 12-dim Goffin state)
STATE_GROUPS = {
    'Position': [0, 2, 4],
    'Velocity': [1, 3, 5],
    'Attitude': [6, 8, 10],
    'Body rate': [7, 9, 11],
}
GROUP_UNITS = {
    'Position': 'm',
    'Velocity': 'm/s',
    'Attitude': 'rad',
    'Body rate': 'rad/s',
}
ANGLE_INDICES = [6, 8, 10]  # phi, theta, psi -- need wrap-around handling


def _wrap_to_pi(a):
    """Wrap angle(s) to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def compute_single_step_error(goffin: GoffinDynamics, ref_states, rpms, model='nonlinear'):
    """
    Single-step (one-step-ahead) prediction error.

    At every step k the Goffin model is re-initialised to the *reference*
    state x_k (i.e. the gym-pybullet state), advanced by exactly one step with
    the logged RPMs, and compared to the reference state x_{k+1}. This isolates
    the per-step discrepancy of the dynamics from the free accumulation of
    error that occurs in an open-loop replay -- and it is the quantity that
    actually matters when training with an autoregressive (horizon-1) rollout.

    Parameters
    ----------
    goffin : GoffinDynamics
    ref_states : ndarray (N+1, 12)   reference (gym-pybullet) trajectory
    rpms : ndarray (N, 4)
    model : 'nonlinear' or 'linearized'

    Returns
    -------
    err : ndarray (N, 12)            signed one-step error (Goffin - reference),
                                     with attitude components wrapped to [-pi, pi]
    """
    N = rpms.shape[0]
    err = np.zeros((N, 12))
    for k in range(N):
        pred = goffin.step(ref_states[k], rpms[k], model=model)
        err[k] = pred - ref_states[k + 1]
    err[:, ANGLE_INDICES] = _wrap_to_pi(err[:, ANGLE_INDICES])
    return err


def group_rms(err):
    """Per-group RMS of an (N, 12) error array. Returns {group_name: rms}."""
    return {
        g: float(np.sqrt(np.mean(err[:, idx] ** 2)))
        for g, idx in STATE_GROUPS.items()
    }


# =====================================================================
#  STEP 4: Report-quality plotting
# =====================================================================

# Consistent styling across all report figures
_REF_STYLE = dict(color='black',     linestyle='-',  linewidth=1.8)
_NL_STYLE  = dict(color='#1f77b4',   linestyle='--', linewidth=1.4)
_LIN_STYLE = dict(color='#d62728',   linestyle=':',  linewidth=1.4)
_REF_LABEL = 'gym-pybullet-drones'
_NL_LABEL  = 'Goffin nonlinear'
_LIN_LABEL = 'Goffin linearized'


def _ref_limits(values, pad=0.15):
    """Axis limits tracking the reference data with a relative margin."""
    lo, hi = float(np.min(values)), float(np.max(values))
    m = pad * (hi - lo) + 1e-3
    return lo - m, hi + m


def plot_trajectory_report(ref_states, nl_states, lin_states, time_vec,
                           tag="", save_path=None):
    """
    Clean, report-oriented open-loop trajectory comparison:
    top view (x-y), altitude z(t), roll phi(t), pitch theta(t).
    Axis limits track the reference so the comparison stays readable even
    when the linearized model spirals away.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle(f'{tag} — open-loop trajectory', fontsize=13, fontweight='bold')

    # (0,0) top view x-y
    ax = axes[0, 0]
    ax.plot(ref_states[:, 0], ref_states[:, 2], label=_REF_LABEL, **_REF_STYLE)
    ax.plot(nl_states[:, 0],  nl_states[:, 2],  label=_NL_LABEL,  **_NL_STYLE)
    ax.plot(lin_states[:, 0], lin_states[:, 2], label=_LIN_LABEL, **_LIN_STYLE)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('Top view (x–y plane)')
    ax.set_xlim(_ref_limits(ref_states[:, 0]))
    ax.set_ylim(_ref_limits(ref_states[:, 2]))
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    # remaining panels: z(t), phi(t), theta(t)
    panels = [((0, 1), 4, 'Altitude $z$ (m)'),
              ((1, 0), 6, r'Roll $\phi$ (rad)'),
              ((1, 1), 8, r'Pitch $\theta$ (rad)')]
    for (r, c), idx, ylabel in panels:
        ax = axes[r, c]
        ax.plot(time_vec, ref_states[:, idx], label=_REF_LABEL, **_REF_STYLE)
        ax.plot(time_vec, nl_states[:, idx],  label=_NL_LABEL,  **_NL_STYLE)
        ax.plot(time_vec, lin_states[:, idx], label=_LIN_LABEL, **_LIN_STYLE)
        ax.set_xlabel('Time (s)'); ax.set_ylabel(ylabel)
        ax.set_ylim(_ref_limits(ref_states[:, idx], pad=0.6))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


def plot_single_step_error(err_nl, err_lin, time_vec, tag="", save_path=None):
    """
    Single-step prediction error grouped by state type (position, velocity,
    attitude, body rate). For each group the per-axis errors are aggregated
    into the Euclidean norm over the three axes and shown on a log scale.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    fig.suptitle(f'{tag} — single-step prediction error', fontsize=13, fontweight='bold')

    EPS = 1e-18
    for ax, (group, idx) in zip(axes.flat, STATE_GROUPS.items()):
        norm_nl = np.linalg.norm(err_nl[:, idx], axis=1)
        norm_lin = np.linalg.norm(err_lin[:, idx], axis=1)
        ax.plot(time_vec, np.maximum(norm_nl, EPS), label=_NL_LABEL, **_NL_STYLE)
        ax.plot(time_vec, np.maximum(norm_lin, EPS), label=_LIN_LABEL, **_LIN_STYLE)
        ax.set_title(f'{group} ({GROUP_UNITS[group]} / step)')
        ax.set_ylabel('one-step error (log)')
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


def plot_global_rms_summary(rms_table, save_path=None):
    """
    Bar chart of the single-step RMS error across all experiments, one panel
    per state group, comparing nonlinear vs linearized Goffin models.

    rms_table : list of dicts with keys
        {'tag', 'model' ('nl'|'lin'), 'Position', 'Velocity', 'Attitude', 'Body rate'}
    """
    tags = sorted({row['tag'] for row in rms_table})
    x = np.arange(len(tags))
    width = 0.38

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Single-step prediction error (RMS over the trajectory)',
                 fontsize=13, fontweight='bold')

    for ax, (group, _) in zip(axes.flat, STATE_GROUPS.items()):
        nl_vals = [next(r[group] for r in rms_table if r['tag'] == t and r['model'] == 'nl') for t in tags]
        lin_vals = [next(r[group] for r in rms_table if r['tag'] == t and r['model'] == 'lin') for t in tags]
        ax.bar(x - width / 2, nl_vals, width, label=_NL_LABEL, color='#1f77b4')
        ax.bar(x + width / 2, lin_vals, width, label=_LIN_LABEL, color='#d62728')
        ax.set_yscale('log')
        ax.set_title(f'{group} ({GROUP_UNITS[group]} / step)')
        ax.set_xticks(x)
        ax.set_xticklabels(tags, rotation=30, ha='right', fontsize=8)
        ax.grid(True, axis='y', which='both', alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


# =====================================================================
#  STEP 5: Full comparison pipeline
# =====================================================================

def run_full_comparison(drone_model, physics, config_name,
                        duration_sec=10.0, output_dir='results_comparison'):
    """
    Full pipeline for one (drone_model, physics) combination:
    simulate + log, replay through Goffin (open loop), compute single-step
    prediction error, and produce the report figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{config_name}_{physics.name}"

    # ── 1. Run simulation and log ──
    print(f"\n{'='*60}\n  {tag}: Running simulation...\n{'='*60}")
    log = run_and_log_substeps(
        drone_model=drone_model, physics=physics,
        ctrl_freq=48, pyb_freq=240, duration_sec=duration_sec,
    )
    ref_states = log['states']  # (N+1, 12)
    rpms = log['rpms']          # (N, 4)
    dt = log['dt']
    p = log['params']

    N = rpms.shape[0]
    time_full = np.arange(N + 1) * dt   # for trajectories (N+1 samples)
    time_step = np.arange(N) * dt       # for single-step error (N samples)

    # ── 2. Create Goffin model ──
    goffin = GoffinDynamics(
        m=p['m'], g=p['g'], Ix=p['Ix'], Iy=p['Iy'], Iz=p['Iz'],
        l=p['l'], kf=p['kf'], km=p['km'], config=config_name, dt=dt,
    )

    # ── 3. Open-loop replay (for the trajectory figure) ──
    print(f"[{tag}] Open-loop replay (nonlinear / linearized)...")
    ol_nl = replay_open_loop(goffin, ref_states[0], rpms, model='nonlinear')
    ol_lin = replay_open_loop(goffin, ref_states[0], rpms, model='linearized')

    # ── 4. Single-step prediction error (the quantitative validation) ──
    print(f"[{tag}] Single-step prediction error...")
    ss_nl = compute_single_step_error(goffin, ref_states, rpms, model='nonlinear')
    ss_lin = compute_single_step_error(goffin, ref_states, rpms, model='linearized')
    rms_nl = group_rms(ss_nl)
    rms_lin = group_rms(ss_lin)

    # ── 5. Figures ──
    plot_trajectory_report(
        ref_states, ol_nl, ol_lin, time_full, tag=tag,
        save_path=os.path.join(output_dir, f'{tag}_traj.png'),
    )
    plot_single_step_error(
        np.abs(ss_nl), np.abs(ss_lin), time_step, tag=tag,
        save_path=os.path.join(output_dir, f'{tag}_singlestep_err.png'),
    )

    # ── 6. Console summary ──
    print(f"\n[{tag}] === single-step RMS error ===")
    for g in STATE_GROUPS:
        print(f"  {g:<10} ({GROUP_UNITS[g]:>5}/step):  "
              f"NL={rms_nl[g]:.3e}   Lin={rms_lin[g]:.3e}")

    return {
        'tag': tag,
        'ref_states': ref_states, 'rpms': rpms,
        'ol_nl': ol_nl, 'ol_lin': ol_lin,
        'ss_nl': ss_nl, 'ss_lin': ss_lin,
        'rms_nl': rms_nl, 'rms_lin': rms_lin,
    }


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == '__main__':

    OUTPUT_DIR = 'results_comparison'
    DURATION = 10.0  # seconds (one full circle period from pid.py)

    # (drone_model, physics, config_name). Cross (X) is the real-Crazyflie
    # configuration and is the focus of the report; Plus (+) reproduces
    # Goffin's original layout and serves as a cross-check.
    experiments = [
        (DroneModel.CF2X, Physics.PYB,             'cross'),
        (DroneModel.CF2X, Physics.DYN,             'cross'),
        (DroneModel.CF2X, Physics.PYB_GND_DRAG_DW, 'cross'),
        (DroneModel.CF2P, Physics.PYB,             'plus'),
        (DroneModel.CF2P, Physics.DYN,             'plus'),
        (DroneModel.CF2P, Physics.PYB_GND_DRAG_DW, 'plus'),
    ]

    all_results = {}
    rms_table = []
    for drone_model, physics, config_name in experiments:
        try:
            res = run_full_comparison(
                drone_model=drone_model, physics=physics, config_name=config_name,
                duration_sec=DURATION, output_dir=OUTPUT_DIR,
            )
            all_results[res['tag']] = res
            rms_table.append({'tag': res['tag'], 'model': 'nl', **res['rms_nl']})
            rms_table.append({'tag': res['tag'], 'model': 'lin', **res['rms_lin']})
        except Exception as e:
            print(f"[ERROR] {config_name}_{physics.name}: {e}")
            import traceback
            traceback.print_exc()

    # ── Global single-step RMS summary (figure + table) ──
    if rms_table:
        plot_global_rms_summary(
            rms_table, save_path=os.path.join(OUTPUT_DIR, 'global_singlestep_rms.png'))

        # CSV for direct tabulation in the report
        csv_path = os.path.join(OUTPUT_DIR, 'singlestep_rms.csv')
        with open(csv_path, 'w') as f:
            f.write('tag,model,' + ','.join(STATE_GROUPS.keys()) + '\n')
            for row in rms_table:
                f.write(f"{row['tag']},{row['model']},"
                        + ','.join(f"{row[g]:.6e}" for g in STATE_GROUPS) + '\n')
        print(f"\n[CSV] Saved single-step RMS table: {csv_path}")

        print(f"\n{'='*78}\n  GLOBAL SINGLE-STEP RMS SUMMARY\n{'='*78}")
        header = f"{'Experiment':<26}{'model':<6}" + ''.join(f"{g:>14}" for g in STATE_GROUPS)
        print(header)
        print('-' * len(header))
        for row in rms_table:
            print(f"{row['tag']:<26}{row['model']:<6}"
                  + ''.join(f"{row[g]:>14.4e}" for g in STATE_GROUPS))

    print(f"\nPlots saved in '{OUTPUT_DIR}/'")
    plt.show()