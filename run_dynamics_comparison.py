"""
Dynamics Comparison: gym-pybullet-drones vs Goffin Model
========================================================

This script:
1. Runs a gym-pybullet-drones simulation (with PID control)
2. Logs RPMs and full state at every physics substep
3. Replays the same RPMs through Goffin's model (nonlinear + linearized)
4. Compares trajectories with two approaches:
   - Open-loop: same initial state, diverge freely
   - Windowed reinitialization: reset Goffin state every N steps

Supports both CF2P (+) and CF2X (x) configurations,
and multiple Physics modes (PYB, DYN, PYB_GND_DRAG_DW, etc.)

Usage:
    python run_dynamics_comparison.py

Requirements:
    - gym-pybullet-drones installed/in path
    - numpy, matplotlib
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Import gym-pybullet-drones ──────────────────────────────────────
# Adjust this path if needed to point to your gym-pybullet-drones repo
# sys.path.insert(0, '/path/to/your/repo')

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

# ── Import Goffin dynamics ──────────────────────────────────────────
from goffin_dynamics import GoffinDynamics, gym_state_to_goffin, goffin_state_labels


# =====================================================================
#  STEP 1: Run gym-pybullet-drones and log everything
# =====================================================================

def run_and_log_simulation(drone_model=DroneModel.CF2X,
                           physics=Physics.PYB,
                           ctrl_freq=48,
                           pyb_freq=240,
                           duration_sec=3.0,
                           target_pos=np.array([0.0, 0.0, 1.0])):
    """
    Run a PID-controlled flight and log state + RPMs at every PHYSICS substep.

    Returns
    -------
    log : dict with keys:
        'states_20'  : (N_steps+1, 20) full gym state vectors
        'rpms'       : (N_steps, 4) RPMs applied at each physics step
        'dt'         : physics timestep
        'ctrl_freq'  : control frequency
        'pyb_freq'   : physics frequency
        'drone_model': DroneModel enum
        'physics'    : Physics enum
        'params'     : dict of drone parameters (m, l, kf, km, Ix, Iy, Iz, g)
    """
    print(f"[SIM] drone={drone_model.name}, physics={physics.name}, "
          f"pyb_freq={pyb_freq}, ctrl_freq={ctrl_freq}, duration={duration_sec}s")

    # Create environment
    env = CtrlAviary(
        drone_model=drone_model,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
        physics=physics,
        pyb_freq=pyb_freq,
        ctrl_freq=ctrl_freq,
        gui=False,
    )

    # Extract physical parameters from the environment
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
    print(f"[SIM] Drone params: m={params['m']:.4f}, l={params['l']:.4f}, "
          f"kf={params['kf']:.2e}, km={params['km']:.2e}")

    # PID controller
    pid = DSLPIDControl(drone_model=drone_model)

    dt = 1.0 / pyb_freq
    steps_per_ctrl = pyb_freq // ctrl_freq
    total_ctrl_steps = int(duration_sec * ctrl_freq)

    # Storage: we log at PHYSICS substep level
    total_phys_steps = total_ctrl_steps * steps_per_ctrl
    all_states_20 = []
    all_rpms = []

    # Reset
    obs, info = env.reset()

    # Log initial state
    state_20_init = env._getDroneStateVector(0)
    all_states_20.append(state_20_init.copy())

    print(f"[SIM] Running {total_ctrl_steps} control steps "
          f"({total_phys_steps} physics steps)...")

    for ctrl_step in range(total_ctrl_steps):
        # Get current state for PID
        state_20 = env._getDroneStateVector(0)

        # Compute PID action (RPMs)
        rpm_cmd, _, _ = pid.computeControlFromState(
            control_timestep=1.0 / ctrl_freq,
            state=state_20,
            target_pos=target_pos,
        )

        # The same RPMs are applied for all physics substeps within one control step.
        # We log them for each substep.
        for sub in range(steps_per_ctrl):
            all_rpms.append(rpm_cmd.copy())

        # Step the environment (internally does steps_per_ctrl physics steps)
        obs, reward, terminated, truncated, info = env.step(rpm_cmd.reshape(1, 4))

        # Log state AFTER step (this is the state after all substeps)
        state_20_after = env._getDroneStateVector(0)
        all_states_20.append(state_20_after.copy())

    env.close()

    # Note: we have total_ctrl_steps+1 states and total_ctrl_steps*steps_per_ctrl RPMs.
    # But the states are logged per control step, not per physics substep.
    # For the substep-level comparison, we need to also log intermediate states.
    # Let's redo with a finer approach.

    print(f"[SIM] Done. Logged {len(all_states_20)} states, {len(all_rpms)} RPM entries.")

    log = {
        'states_20': np.array(all_states_20),       # (N_ctrl+1, 20)
        'rpms': np.array(all_rpms),                  # (N_ctrl*steps_per_ctrl, 4)
        'dt': dt,
        'ctrl_freq': ctrl_freq,
        'pyb_freq': pyb_freq,
        'steps_per_ctrl': steps_per_ctrl,
        'drone_model': drone_model,
        'physics': physics,
        'params': params,
    }
    return log


def run_and_log_substeps(drone_model=DroneModel.CF2X,
                          physics=Physics.PYB,
                          ctrl_freq=48,
                          pyb_freq=240,
                          duration_sec=3.0,
                          target_pos=np.array([0.0, 0.0, 1.0])):
    """
    Same as above but logs state at EVERY physics substep for accurate comparison.

    This requires direct access to the environment internals to call
    _updateAndStoreKinematicInformation() between substeps.
    """
    import pybullet as p

    print(f"[SIM] drone={drone_model.name}, physics={physics.name}, "
          f"pyb_freq={pyb_freq}, ctrl_freq={ctrl_freq}, duration={duration_sec}s")

    env = CtrlAviary(
        drone_model=drone_model,
        num_drones=1,
        initial_xyzs=np.array([[0.0, 0.0, 0.1]]),
        initial_rpys=np.array([[0.0, 0.0, 0.0]]),
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

    all_states = []  # Goffin 12-dim states at each physics step
    all_rpms = []    # RPMs at each physics step

    obs, info = env.reset()

    # Log initial state
    env._updateAndStoreKinematicInformation()
    s0 = gym_state_to_goffin(env.pos[0], env.rpy[0], env.vel[0], env.ang_v[0])
    all_states.append(s0.copy())

    print(f"[SIM-SUBSTEP] Running {total_ctrl_steps} ctrl steps x {steps_per_ctrl} substeps...")

    for ctrl_step in range(total_ctrl_steps):
        # Get current state for PID
        state_20 = env._getDroneStateVector(0)
        rpm_cmd, _, _ = pid.computeControlFromState(
            control_timestep=1.0 / ctrl_freq,
            state=state_20,
            target_pos=target_pos,
        )

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

    all_states = np.array(all_states)  # (N_phys+1, 12)
    all_rpms = np.array(all_rpms)      # (N_phys, 4)

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


def replay_windowed(goffin: GoffinDynamics, ref_states, rpms,
                    model='nonlinear', window_size=1):
    """
    Windowed reinitialization: every `window_size` steps, reset Goffin state
    to the reference (gym) state, apply RPMs for `window_size` steps,
    measure the error.

    Parameters
    ----------
    goffin : GoffinDynamics
    ref_states : ndarray (N+1, 12) — reference states from gym
    rpms : ndarray (N, 4)
    model : 'nonlinear' or 'linearized'
    window_size : int — reinitialize every this many physics steps

    Returns
    -------
    goffin_states : ndarray (N+1, 12) — Goffin prediction at each step
    errors : ndarray (N+1, 12) — absolute error at each step
    """
    N = rpms.shape[0]
    goffin_states = np.zeros((N + 1, 12))
    goffin_states[0] = ref_states[0].copy()

    current_state = ref_states[0].copy()

    for k in range(N):
        # Reinitialize from reference at window boundaries
        if k % window_size == 0:
            current_state = ref_states[k].copy()

        next_state = goffin.step(current_state, rpms[k], model=model)
        goffin_states[k + 1] = next_state.copy()
        current_state = next_state

    errors = np.abs(goffin_states - ref_states)
    return goffin_states, errors


# =====================================================================
#  STEP 3: Plotting
# =====================================================================

def plot_trajectory_comparison(time_vec, ref_states, goffin_nl, goffin_lin,
                                title_prefix="", save_path=None):
    """
    Plot gym reference vs Goffin nonlinear vs Goffin linearized for all 12 states.
    """
    labels = goffin_state_labels()

    fig, axes = plt.subplots(4, 3, figsize=(18, 14), sharex=True)
    fig.suptitle(f'{title_prefix} — Trajectory Comparison', fontsize=14, fontweight='bold')

    for i in range(12):
        ax = axes[i // 3, i % 3]
        ax.plot(time_vec, ref_states[:, i], 'k-', linewidth=1.5, label='gym-pybullet-drones')
        ax.plot(time_vec, goffin_nl[:, i], 'b--', linewidth=1.0, label='Goffin nonlinear')
        ax.plot(time_vec, goffin_lin[:, i], 'r:', linewidth=1.0, label='Goffin linearized')
        ax.set_ylabel(labels[i], fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


def plot_error_comparison(time_vec, errors_nl, errors_lin,
                           title_prefix="", save_path=None):
    """
    Plot absolute errors for nonlinear and linearized Goffin models.
    """
    labels = goffin_state_labels()

    # Group: positions (0,2,4), velocities (1,3,5), angles (6,8,10), rates (7,9,11)
    groups = {
        'Positions': [0, 2, 4],
        'Velocities': [1, 3, 5],
        'Angles': [6, 8, 10],
        'Angular rates': [7, 9, 11],
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{title_prefix} — Absolute Errors', fontsize=14, fontweight='bold')

    for ax, (group_name, indices) in zip(axes.flat, groups.items()):
        for idx in indices:
            ax.plot(time_vec, errors_nl[:, idx], '-', linewidth=1.0,
                    label=f'NL: {labels[idx]}')
            ax.plot(time_vec, errors_lin[:, idx], ':', linewidth=1.0,
                    label=f'Lin: {labels[idx]}')
        ax.set_title(group_name, fontsize=11)
        ax.set_ylabel('Absolute error')
        ax.set_yscale('log')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel('Time (s)')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


def plot_windowed_errors_summary(time_vec, errors_nl_w1, errors_lin_w1,
                                  errors_nl_w5, errors_lin_w5,
                                  title_prefix="", save_path=None):
    """
    Compare windowed errors for different window sizes.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'{title_prefix} — Windowed Reinitialization Errors', fontsize=14)

    # Compute total state error norm at each step
    norm_nl_w1 = np.linalg.norm(errors_nl_w1, axis=1)
    norm_lin_w1 = np.linalg.norm(errors_lin_w1, axis=1)
    norm_nl_w5 = np.linalg.norm(errors_nl_w5, axis=1)
    norm_lin_w5 = np.linalg.norm(errors_lin_w5, axis=1)

    axes[0].set_title('Window = 1 step (single-step prediction error)')
    axes[0].plot(time_vec, norm_nl_w1, 'b-', linewidth=0.8, label='Goffin nonlinear')
    axes[0].plot(time_vec, norm_lin_w1, 'r-', linewidth=0.8, label='Goffin linearized')
    axes[0].set_ylabel('||error||')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Window = 5 steps')
    axes[1].plot(time_vec, norm_nl_w5, 'b-', linewidth=0.8, label='Goffin nonlinear')
    axes[1].plot(time_vec, norm_lin_w5, 'r-', linewidth=0.8, label='Goffin linearized')
    axes[1].set_ylabel('||error||')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[PLOT] Saved: {save_path}")
    return fig


# =====================================================================
#  STEP 4: Full comparison pipeline
# =====================================================================

def run_full_comparison(drone_model, physics, config_name,
                         duration_sec=3.0, output_dir='results_comparison'):
    """
    Full pipeline for one (drone_model, physics) combination.
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{config_name}_{physics.name}"

    # ── 1. Run simulation and log ──
    print(f"\n{'='*60}")
    print(f"  {tag}: Running simulation...")
    print(f"{'='*60}")

    log = run_and_log_substeps(
        drone_model=drone_model,
        physics=physics,
        ctrl_freq=48,
        pyb_freq=240,
        duration_sec=duration_sec,
        target_pos=np.array([0.0, 0.0, 1.0]),
    )

    ref_states = log['states']  # (N+1, 12)
    rpms = log['rpms']          # (N, 4)
    dt = log['dt']
    p = log['params']

    N = rpms.shape[0]
    time_vec = np.arange(N + 1) * dt

    # ── 2. Create Goffin model ──
    goffin = GoffinDynamics(
        m=p['m'], g=p['g'],
        Ix=p['Ix'], Iy=p['Iy'], Iz=p['Iz'],
        l=p['l'], kf=p['kf'], km=p['km'],
        config=config_name,
        dt=dt,
    )

    # ── 3. Open-loop replay ──
    print(f"[{tag}] Open-loop replay (nonlinear)...")
    ol_nl = replay_open_loop(goffin, ref_states[0], rpms, model='nonlinear')

    print(f"[{tag}] Open-loop replay (linearized)...")
    ol_lin = replay_open_loop(goffin, ref_states[0], rpms, model='linearized')

    # Plot open-loop trajectories
    plot_trajectory_comparison(
        time_vec, ref_states, ol_nl, ol_lin,
        title_prefix=f'{tag} — Open Loop',
        save_path=os.path.join(output_dir, f'{tag}_open_loop_traj.png'),
    )

    # Plot open-loop errors
    err_ol_nl = np.abs(ol_nl - ref_states)
    err_ol_lin = np.abs(ol_lin - ref_states)
    plot_error_comparison(
        time_vec, err_ol_nl, err_ol_lin,
        title_prefix=f'{tag} — Open Loop',
        save_path=os.path.join(output_dir, f'{tag}_open_loop_err.png'),
    )

    # ── 4. Windowed replay ──
    print(f"[{tag}] Windowed replay (window=1)...")
    _, err_w1_nl  = replay_windowed(goffin, ref_states, rpms, model='nonlinear',  window_size=1)
    _, err_w1_lin = replay_windowed(goffin, ref_states, rpms, model='linearized', window_size=1)

    print(f"[{tag}] Windowed replay (window=5)...")
    _, err_w5_nl  = replay_windowed(goffin, ref_states, rpms, model='nonlinear',  window_size=5)
    _, err_w5_lin = replay_windowed(goffin, ref_states, rpms, model='linearized', window_size=5)

    plot_windowed_errors_summary(
        time_vec, err_w1_nl, err_w1_lin, err_w5_nl, err_w5_lin,
        title_prefix=tag,
        save_path=os.path.join(output_dir, f'{tag}_windowed_err.png'),
    )

    # ── 5. Print summary statistics ──
    print(f"\n[{tag}] === SUMMARY ===")
    print(f"  Open-loop final position error (nonlinear):   "
          f"x={err_ol_nl[-1,0]:.4f} y={err_ol_nl[-1,2]:.4f} z={err_ol_nl[-1,4]:.4f}")
    print(f"  Open-loop final position error (linearized):  "
          f"x={err_ol_lin[-1,0]:.4f} y={err_ol_lin[-1,2]:.4f} z={err_ol_lin[-1,4]:.4f}")
    print(f"  Single-step mean ||error|| (nonlinear):   {np.mean(np.linalg.norm(err_w1_nl, axis=1)):.6e}")
    print(f"  Single-step mean ||error|| (linearized):  {np.mean(np.linalg.norm(err_w1_lin, axis=1)):.6e}")

    return {
        'tag': tag,
        'ref_states': ref_states,
        'rpms': rpms,
        'ol_nl': ol_nl,
        'ol_lin': ol_lin,
        'err_w1_nl': err_w1_nl,
        'err_w1_lin': err_w1_lin,
    }


# =====================================================================
#  MAIN
# =====================================================================

if __name__ == '__main__':

    OUTPUT_DIR = 'results_comparison'
    DURATION = 3.0  # seconds

    # ─────────────────────────────────────────────────────────────────
    # Define all (drone_model, physics, config_name) combos to test
    # ─────────────────────────────────────────────────────────────────
    experiments = [
        # Cross (x) configuration
        (DroneModel.CF2X, Physics.PYB,              'cross'),
        (DroneModel.CF2X, Physics.DYN,              'cross'),
        (DroneModel.CF2X, Physics.PYB_GND_DRAG_DW,  'cross'),

        # Plus (+) configuration
        (DroneModel.CF2P, Physics.PYB,              'plus'),
        (DroneModel.CF2P, Physics.DYN,              'plus'),
        (DroneModel.CF2P, Physics.PYB_GND_DRAG_DW,  'plus'),
    ]

    all_results = {}

    for drone_model, physics, config_name in experiments:
        try:
            result = run_full_comparison(
                drone_model=drone_model,
                physics=physics,
                config_name=config_name,
                duration_sec=DURATION,
                output_dir=OUTPUT_DIR,
            )
            all_results[result['tag']] = result
        except Exception as e:
            print(f"[ERROR] {config_name}_{physics.name}: {e}")
            import traceback
            traceback.print_exc()

    # ─────────────────────────────────────────────────────────────────
    # Summary comparison across all experiments
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  GLOBAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<30} {'NL 1-step err':>15} {'Lin 1-step err':>15}")
    print(f"{'-'*60}")

    for tag, res in all_results.items():
        nl_err = np.mean(np.linalg.norm(res['err_w1_nl'], axis=1))
        lin_err = np.mean(np.linalg.norm(res['err_w1_lin'], axis=1))
        print(f"{tag:<30} {nl_err:>15.6e} {lin_err:>15.6e}")

    print(f"\nPlots saved in '{OUTPUT_DIR}/'")
    plt.show()