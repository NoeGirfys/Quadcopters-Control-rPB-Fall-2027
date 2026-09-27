# Meta-Learning NN Control on a Crazyflie Quadcopter

Semester project, EPFL — DECODE lab, Spring 2026.
**Student:** Niels Rigaud (SCIPER 341284) · **Supervisors:** Simone Baratto, Nicolas Kirsch, Daniele Martinelli · **Professor:** G. Ferrari Trecate.

This project carries the meta-learning quadcopter controller of Goffin (previous
semester, simulation-only, see [`MetaLearning-Control-Quadcopters-main/`](MetaLearning-Control-Quadcopters-main/)) to a **real Crazyflie**, with an explicit sim-to-real
validation. The main additions are: a simulator made faithful to the Crazyflie by
porting its **cascade PID firmware** from C to Python, a **hybrid NN–PID**
controller (the network replaces only the outer two PID loops), and **MAML**
applied to an **offset-mass** disturbance, trained in simulation and flown on
hardware.

> **Read the report first.** The full write-up is
> [`Report_Latex_Overleaf/main.pdf`](Report_Latex_Overleaf/main.pdf). Throughout
> this README, "§N" refers to a section of that report. The folder map below tells
> you which directory produced which part of the report.

---

## 1. Setup

### 1.1 Local environment

Everything (simulation, training, real-drone flights, plotting) runs in a single
conda environment on Python 3.13:

```bash
conda create -n venv_cf_313 python=3.13
conda activate venv_cf_313
pip install -r requirements.txt
```

`requirements.txt` pins every library the code uses (versions this project was
validated against):

- **`-e ./gym-pybullet-drones`** — the vendored simulator, installed editable; its
  own install pulls `pybullet`, `gymnasium`, `scipy`, `Pillow`.
- **`torch`, `numpy`, `matplotlib`, `PyYAML`, `pybullet`** — training, differentiable
  dynamics, plotting, and the YAML task configs.
- **`cflib`, `cfclient`** — real-drone radio comms and the config client; only needed
  to fly the physical Crazyflie (`fly_*_cf_pid.py --mode attitude`).

### 1.2 Running scripts — import paths

`maml_lib/` and `crazyflie_firmware/` are **local packages** imported by name
(e.g. `from maml_lib import ...`, `from crazyflie_firmware.firmware import ...`).
**Run scripts from the repository root** so these resolve, e.g.:

```bash
python training_MAML/train_maml.py ...
python circle_comparison_simu_and_real/cf_firmware_pid_sim.py ...
```

### 1.3 Cluster (SCITAS / Izar)

Meta-training was run on the Izar GPU cluster. The `run_*.sh` files in
`training_MAML/` are SLURM batch scripts; on the cluster they use a separate venv
`venv_MAML_SCITAS` and are submitted with `sbatch`:

```bash
module load gcc python
source venv_MAML_SCITAS/bin/activate
pip install --no-cache-dir <lib>      # always --no-cache-dir on Izar
sbatch training_MAML/run_maml_izar_offdirmag.sh
```

SCITAS docs: <https://scitas-doc.epfl.ch/>.

---

## 2. Repository map

| Directory | Report § | Role |
|-----------|----------|------|
| [`gym-pybullet-drones/`](gym-pybullet-drones) | §3.1 | PyBullet drone simulator (vendored, editable). DYN physics mode + offset-mass support. |
| [`compare_models_openloop/`](compare_models_openloop) | §3.2 | Open-loop comparison of gym-pybullet dynamics vs Goffin's model. |
| [`crazyflie_firmware/`](crazyflie_firmware) | §2.2, §3.3 | Crazyflie cascade PID firmware ported C→Python (used in sim and as the inner loops). |
| [`circle_comparison_simu_and_real/`](circle_comparison_simu_and_real) | §4 | Sim-to-real validation: circle trajectory, Flowdeck vs Lighthouse. |
| [`training_regulation_simu_and_real/`](training_regulation_simu_and_real) | §5 | Classic-regulation NN–PID controller: training, sim, real flights. |
| [`maml_lib/`](maml_lib) | §2.5, §6 | Differentiable MAML library (dynamics, PID chain, policy, cost, tasks). |
| [`training_MAML/`](training_MAML) | §6 | MAML for offset-mass compensation: training, PyBullet eval, real flights. |
| [`MetaLearning-Control-Quadcopters-main/`](MetaLearning-Control-Quadcopters-main) | §1.2 | Goffin's prior work (theory + simulation-only MAML). Reference, not run here. |
| [`Report_Latex_Overleaf/`](Report_Latex_Overleaf) | — | The report itself (`main.tex`, `main.pdf`, `images/`). |
| [`ANGLE_CONVENTIONS.md`](ANGLE_CONVENTIONS.md) | §2.1 | Reference note on body rates vs Euler-angle derivatives and sign conventions. |

**Pipeline order (mirrors the report):** simulator → model comparison → firmware
port → sim-to-real circle → regulation NN → MAML.

---

## 3. Folder-by-folder guide

### 3.1 `gym-pybullet-drones/` — the simulator (§3.1)

Vendored copy of [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones),
installed editable. We use the **DYN** physics mode (explicit rigid-body dynamics
with quaternion orientation integration) rather than the default controller.

Touch-points added/used by this project (in `gym_pybullet_drones/envs/BaseAviary.py`):
- `_integrateQ` — quaternion orientation integration (avoids gimbal lock).
- `set_offset_mass` — Steiner-corrected inertia + shifted CoM for the
  `Physics.DYN_OFFSET` mode used during MAML offset-mass training/eval.

You normally don't edit this folder; the project drives it from the outside.

### 3.2 `compare_models_openloop/` — dynamics comparison (§3.2)

Shows that gym-pybullet's dynamics and Goffin's model agree in open loop.

- `goffin_dynamics.py` — Goffin's nonlinear + linearized model re-implemented, and
  `gym_state_to_goffin`, the **only** explicit body-rate → Euler-rate converter in
  the repo (see `ANGLE_CONVENTIONS.md` §3.1).
- `run_dynamics_comparison.py` — runs the same open-loop input through PYB/DYN and
  Goffin's nonlinear/linearized models and plots trajectories + single-step error.
- `results_comparison/` — output figures (the report uses `cross_*` plots).

```bash
python compare_models_openloop/run_dynamics_comparison.py
```

### 3.3 `crazyflie_firmware/` — cascade PID port (§2.2, §3.3)

The real Crazyflie's 4-loop cascade PID (position → velocity → attitude → rate →
motors), ported from the official C firmware to Python. This is what makes the
closed-loop simulation faithful to the hardware, and it provides the **inner loops**
(attitude + rate) that the NN feeds into.

- `constants.py` — gains, rates, saturations, timing (`ATTITUDE_RATE`, etc.).
- `firmware.py` — the PID controllers and power distribution.
- Imported by the regulation and MAML flight/sim scripts; not run directly.

### 3.4 `circle_comparison_simu_and_real/` — sim-to-real validation (§4)

Commands a circle trajectory in simulation and on the real drone, and compares the
two. Motivates the switch from Flowdeck (optical-flow jitter) to Lighthouse.

- `cf_firmware_pid_sim.py` — the driver: runs the sim (and processes real logs),
  produces the comparison / top-view / jitter / metrics figures.
- `flights/` — recorded logs: `flight_{flowdeck,lighthouse}_<timestamp>.npz`
  (the `20260523` pair backs the report figures).
- `metrics/` — per-flight numeric metrics and discrepancies (`*.json`), the numbers
  behind the §4.3 error discussion.

Figures were removed to keep the repo clean (they live in the report's `images/`);
regenerate them from the kept logs with `cf_firmware_pid_sim.py`.

### 3.5 `training_regulation_simu_and_real/` — classic regulation (§5)

The hybrid NN–PID controller that regulates the drone to a setpoint. The network
replaces the outer position/velocity loops (outputs roll/pitch/thrust + yaw-rate
setpoints), the cascade PID handles attitude/rate/motors.

- `train_nn_cf_pid.py` — trains the policy against a differentiable near-hover model
  (BPTT, single-step horizon `T=1`, curriculum reset, observation-noise domain
  randomisation). Saves `trained_cf_pid_T1_ch200_h64_ep<E>_<tag>.pt`.
- `fly_nn_cf_pid.py` — deploys the four-phase pipeline (PID takeoff → hover →
  **NN regulation** → PID landing) in `--mode sim` (PyBullet) or `--mode attitude`
  (real drone); logs to `flights/flight_{sim,real}_<timestamp>.npz`.
- `plot_report_figures.py` — evaluates a checkpoint on the 27-point init cube →
  the §5.3 regulation figures.
- `plot_sim_real_overlay.py` — overlays a sim vs a real flight → the §5.4 figures
  (defaults to the latest pair in `flights/`).
- **Deployed model:** `trained_cf_pid_T1_ch200_h64_ep50_noisy.pt` (the report's §5
  results; 50 epochs, terminal weight 50, noise scale 2.0).

```bash
# train (defaults ≈ report hyper-params)
python training_regulation_simu_and_real/train_nn_cf_pid.py --tag noisy
# deploy in sim (--mode sim), then on the real drone (--mode attitude)
python training_regulation_simu_and_real/fly_nn_cf_pid.py --mode sim      --weights training_regulation_simu_and_real/trained_cf_pid_T1_ch200_h64_ep50_noisy.pt
python training_regulation_simu_and_real/fly_nn_cf_pid.py --mode attitude --weights training_regulation_simu_and_real/trained_cf_pid_T1_ch200_h64_ep50_noisy.pt
# report figures
python training_regulation_simu_and_real/plot_report_figures.py --weights trained_cf_pid_T1_ch200_h64_ep50_noisy.pt
python training_regulation_simu_and_real/plot_sim_real_overlay.py
```

### 3.6 `maml_lib/` — the MAML library (§2.5, §6)

Reusable, fully differentiable pieces that `training_MAML/` composes. Everything is
batched over tasks and drones.

| Module | Role |
|--------|------|
| `config.py` | Physical constants / timing for the CF2X (mass-baseline). |
| `mass_params.py` | Per-task mass parameters: total mass, CoM offset, Steiner inertia. |
| `dynamics.py` | Differentiable nonlinear + linearized dynamics (offset-mass aware). |
| `pid_chain.py` | Differentiable firmware-like attitude+rate PID (inner loops). |
| `policy.py` | Single-step (`T=1`) policy MLP. |
| `rollout.py` | Differentiable closed-loop rollout (NN queried every step). |
| `cost.py` | Quadratic per-trajectory trajectory cost (batched). |
| `maml.py` | MAML (Finn et al. 2017) inner/outer loop + multi-task baseline. |
| `tasks/` | Task distributions: `uniform_mass`, `composite` (mixtures). |

### 3.7 `training_MAML/` — MAML offset-mass experiments (§6)

Meta-training and evaluation for the offset-mass task, built on `maml_lib` and the
`crazyflie_firmware` inner loops.

- `train_maml.py` / `train_baseline.py` — meta-training (MAML) and the joint-training
  baseline. Task distribution comes entirely from the YAML files (single source of
  truth): `tasks_offdirmag.yaml`, `target_offdirmag.yaml`.
- `run_maml_izar_offdirmag.sh` / `run_baseline_izar_offdirmag.sh` — the exact SLURM
  invocations that produced the report models.
- `test_maml_pybullet.py` — few-shot adaptation + closed-loop PyBullet evaluation.
- `compare_pybullet.py` — the four-controller PyBullet comparison (firmware PID /
  baseline / baseline-adapted / MAML-adapted) → §6.3 figure + metrics CSV.
- `eval_gap.py` — held-out adaptation-gap curves.
- `plot_report_curves.py` — convergence / adaptation figures for §6.
- `fly_maml_cf_pid.py` — real-drone deployment of the four controllers with a taped
  payload → `flights/maml_{sim,real}_<ctrl>_<mass>g_<timestamp>.npz`.
- `plot_real_compare.py` — overlays the four real flights → §6.4 figure.
- **Models:** `maml_nonlinear_h64_o1_n6_izar_offdirmag_ep500.pt` (MAML meta-init) and
  `baseline_nonlinear_h64_n6_izar_offdirmag_ep500.pt` (joint baseline). The
  `offdirmag` task (crossed offset **direction × magnitude**) is the configuration
  where MAML beats the joint baseline.

```bash
# meta-train (locally; on Izar use the run_*.sh)
python training_MAML/train_maml.py     --tasks-config training_MAML/tasks_offdirmag.yaml --target-config training_MAML/target_offdirmag.yaml ...
python training_MAML/train_baseline.py --tasks-config training_MAML/tasks_offdirmag.yaml --target-config training_MAML/target_offdirmag.yaml ...
# evaluate & compare in PyBullet
python training_MAML/compare_pybullet.py
# flights: --maml-ckpt is required; --mode sim (PyBullet) or --mode attitude (real drone);
# one --controller per run (pid | base | base-adapt | maml)
python training_MAML/fly_maml_cf_pid.py --maml-ckpt training_MAML/maml_nonlinear_h64_o1_n6_izar_offdirmag_ep500.pt --controller maml --mass 8.21 --mode attitude
# overlay the four real flights
python training_MAML/plot_real_compare.py --pid ... --base ... --base-adapt ... --maml ... --mass 8.21
```

`flights/` naming: `maml_real_<controller>_<mass>g_<timestamp>.npz`, controllers
`pid` / `base` / `base-adapt` / `maml`. **Only the `20260611` real flights are
post-yaw-fix** (see §4 below) — earlier real MAML logs predate that fix.

### 3.8 `MetaLearning-Control-Quadcopters-main/` — Goffin's prior work (§1.2)

Goffin's repository (theory, LQR-vs-NN and meta-learning notebooks, all in
simulation). Kept for reference and to reproduce the prior-work baseline; not part
of this project's runtime.

### 3.9 `Report_Latex_Overleaf/` — the report

`main.tex` (source), `main.bib` (IEEE bibliography), `images/` (all figures, as
self-contained copies), `main.pdf` (compiled). Build with `latexmk -pdf main.tex`.

---

## 4. Conventions & gotchas

Read [`ANGLE_CONVENTIONS.md`](ANGLE_CONVENTIONS.md) before touching the dynamics or
the sim↔real bridge. The essentials:

- **State vector (code order):** `[x, vx, y, vy, z, vz, φ, p, θ, q, ψ, r]` — slots
  7/9/11 hold **body rates** `(p,q,r)`, not Euler-angle derivatives. They coincide
  only near hover; the training simulators make that small-angle approximation.
- **Pitch sign:** PyBullet uses pitch-positive = nose **down**; the firmware uses
  nose **up**. At every PyBullet↔firmware/real boundary, **θ and q are negated**;
  roll and yaw are not.
- **Yaw-rate sign (important):** the Crazyflie `send_setpoint` yawrate argument has
  the **opposite** sign to the sim/training convention. Real-flight scripts negate
  the network's yaw-rate command (`send_setpoint(roll, -pitch, -yaw_rate, thrust)`).
  A missing negation here was the bug that initially sent the payload-laden drone to
  the ceiling; it was fixed for the `2026-06-11` MAML flights.

---

## 5. Reproducing the report figures

Generated figures were deleted from the working folders to keep the repo lean; the
report's copies live in `Report_Latex_Overleaf/images/` and every figure is
regenerable from the kept models, logs, and scripts:

| Report figures | Regenerate with |
|----------------|-----------------|
| §3.2 model comparison | `compare_models_openloop/run_dynamics_comparison.py` |
| §4 circle sim-vs-real | `circle_comparison_simu_and_real/cf_firmware_pid_sim.py` |
| §5.3 / §5.4 regulation | `plot_report_figures.py` / `plot_sim_real_overlay.py` |
| §6.1–6.3 MAML | `plot_report_curves.py`, `compare_pybullet.py`, `eval_gap.py` |
| §6.4 real MAML | `plot_real_compare.py` (on the `20260611` flights) |

---

## 6. Continuing the project

The report's conclusion (§7) lists the open directions. The most actionable, with
their code entry points:

1. **Harder task distribution** — a centred payload is nearly rejected by the inner
   integral loops; strongly offset / asymmetric / time-varying masses give MAML a
   regime where it matters (`training_MAML/tasks_*.yaml`, `maml_lib/tasks/`).
2. **Closed-loop thrust on the real drone** — the network emits an open-loop
   feed-forward hover thrust; adding an integral/estimated hover-thrust channel would
   close the last steady-state gap to the firmware cascade
   (`training_regulation_simu_and_real/`, `training_MAML/fly_maml_cf_pid.py`).
3. **Model the communication latency in training** — the off-board NN acts one radio
   round-trip after the state it saw; delaying observations/actions in the simulator
   would let the policy learn to anticipate it (`maml_lib/rollout.py`,
   `training_regulation_simu_and_real/train_nn_cf_pid.py`).
4. **Exact orientation kinematics** — the training dynamics conflate `(p,q,r)` with
   `(φ̇,θ̇,ψ̇)`; differentiating through the exact quaternion dynamics would tighten
   the model (`maml_lib/dynamics.py`, `train_nn_cf_pid.py`; see `ANGLE_CONVENTIONS.md`).
