# Project Context: Differentiable Quadrotor Control via Analytic Policy Gradient (APG)

## 1. Scientific Background
This project implements the methodology from the paper: **"Training Efficient Controllers via Analytic Policy Gradient" (Wiedemann et al., ICRA 2023)**.
The objective is to train a Neural Network (NN) to control a Crazyflie 2.1 quadrotor offline. Instead of standard Reinforcement Learning (RL), we use **Analytic Policy Gradients (APG)**: the loss gradients are backpropagated directly through time (BPTT) across a fully differentiable PyTorch simulator containing both the quadrotor's nonlinear physics and its onboard firmware PID cascade.

### Key Concepts from the Paper:
1. **Concurrent Architecture:** Instead of an autoregressive loop (where the NN is queried at every time step), the NN is queried *once* per chunk. It takes the current state and a future trajectory horizon, and outputs **all $T$ future actions simultaneously**. This drastically shortens the computational graph for the NN weights and stabilizes BPTT.
2. **Relative Target Inputs:** To ensure translation invariance, the NN receives future targets as *relative vectors* (Target Position - Current Drone Position), not absolute world coordinates.
3. **Curriculum Learning (Reset Mechanism):** If the drone deviates too far from the reference trajectory during the unrolled simulation (error > $\tau_{div}$), its state is detached from the gradient graph and "teleported" back to the reference path. The tolerance $\tau_{div}$ starts tight (e.g., $0.1$m) and is relaxed over epochs.

## 2. Technical Architecture & Frequencies
The control pipeline mimics the real Crazyflie firmware (`cf_firmware_pid_sim.py`):
* **Neural Network (100 Hz):** Acts as the high-level planner. 
  * *Inputs:* Current 12D State + $T$ Relative 3D Targets.
  * *Outputs:* $T$ sets of `(roll, pitch, yaw_rate, thrust)` depending on the chosen mode.
* **PID Controller (500 Hz):** Low-level deterministic control. Runs 5 times for every NN action.
* **Physics (500 Hz):** Exact nonlinear Newton-Euler dynamics matching `gym-pybullet-drones`. 

## 3. Curriculum Reset Implementation
When `norm(pos - target) > tau_div`:
1. Clone the state to avoid in-place operation errors in PyTorch.
2. Overwrite `[x, y, z]` with the target `[x, y, z]` and `.detach()` it to break the gradient chain for the failed trajectory.
3. Zero out velocities and angular rates to stabilize the reset state.

## 4. Provided Reference Files
* `cf_firmware_pid_sim.py`: The ground truth. Contains the exact C-to-Python translation of the Crazyflie firmware (Attitude PID, Rate PID, LPF filters, Motor mixing, Battery compensation). Use this to verify physical constants and logic.
* `train_nn_cf2x.py`: Contains the dynamics of the drone that replicates "DYN" mode from gym-pybullet-drones.