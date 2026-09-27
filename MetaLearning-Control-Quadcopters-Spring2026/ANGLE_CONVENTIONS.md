# Angular conventions: body rates vs Euler-angle derivatives

Reference note for the semester project. It clarifies the distinction between
**Euler-angle derivatives** $\dot\Phi=(\dot\phi,\dot\theta,\dot\psi)$ and
**body rates** $\omega_B=(p,q,r)$, catalogues the conventions used across the
project, and locates in each codebase where each quantity is used.

---

## 1. Theoretical background

### 1.1 Is there a notion of "body angles" (order 0)?

**No.** The body rates $(p,q,r)$ are not the derivative of any vector of angles.
The angular velocity $\omega_B$ is an **anholonomic** (non-integrable) quantity:
there is no parametrisation $\Theta$ of the orientation such that
$\dot\Theta=\omega_B$. Integrating $p$ over time does not produce a well-defined
angle — the result depends on the path taken.

| Order | "Euler" representation | "Body" representation | Difference? |
|-------|------------------------|-----------------------|-------------|
| 0 — orientation | $(\phi,\theta,\psi)$, chart on $SO(3)$ | *does not exist* | — the orientation is unique, frame-independent |
| 1 — angular velocity | $\dot\Phi$ | $\omega_B=(p,q,r)$ | **yes** |
| 2 — angular acceleration | $\ddot\Phi$ | $\dot\omega_B$ | **yes** (also involves $\dot W$) |

At order 0 there is only **the orientation**, an element of $SO(3)$ on which all
representations agree; the Euler angles are one chart of it. The body/Euler
distinction only appears from order 1 onward.

### 1.2 Kinematic relation

Body rates and Euler-angle derivatives are related by the kinematic
transformation matrix:

$$\omega_B = W(\phi,\theta)\,\dot\Phi,\qquad
W(\phi,\theta)=\begin{bmatrix}
1 & 0 & -s_\theta\\
0 & c_\phi & s_\phi c_\theta\\
0 & -s_\phi & c_\phi c_\theta
\end{bmatrix}$$

Since $W(0,0)=I_3$, we have $\omega_B\simeq\dot\Phi$ near hover. The distinction
becomes non-negligible away from the hover regime, and $W^{-1}$ degenerates at
$\theta=\pm\pi/2$ (gimbal lock).

### 1.3 Consequence for the state vector

The 12-D state vector of the code stacks `[φ, p, θ, q, ψ, r]`: an Euler angle
**paired with a body rate that is not its derivative**. The two coincide only
near hover. The report (Section 2.1) writes this vector with
$\dot\phi,\dot\theta,\dot\psi$; strictly speaking the angular slots contain
$p,q,r$.

---

## 2. Project conventions

### 2.1 Frames

- **World frame** $\mathcal{W}$: inertial, $z$ pointing up.
- **Body frame** $\mathcal{B}$: attached to the CoM, $x_B$ forward, $y_B$ left,
  $z_B$ aligned with the thrust.
- Orientation parametrised by the ZYX Euler angles $(\phi,\theta,\psi)$.

### 2.2 State vector (code order)

```
[x, vx, y, vy, z, vz, phi, p, theta, q, psi, r]
 0   1   2   3   4   5    6   7    8   9   10  11
```

Slots 7, 9, 11 hold the **body rates** $(p,q,r)$, not $\dot\Phi$.

### 2.3 Pitch sign convention

- **pybullet** (used in simulation): positive pitch = nose **down**.
- **Crazyflie firmware** (aeronautical): positive pitch = nose **up**.
- Roll and yaw: identical conventions between the two.

→ When converting pybullet ↔ firmware/real, we **negate the pitch** $\theta$
**and the pitch rate** $q$. Roll and yaw are left untouched.

### 2.4 Angular velocity by source

| Source | Native form | Conversion applied |
|--------|-------------|--------------------|
| `obs[13:16]` from gym-pybullet | $\omega$ in **world** frame | `R.T @ ang_v` → body |
| Real Crazyflie gyroscope (`gyro.x/y/z`) | body rates (deg/s) | none (already body) |
| Goffin's model | Euler-angle derivatives $\dot\Phi$ | — |

---

## 3. Codebase-by-codebase audit

**Summary:** a single codebase uses Euler-angle derivatives — Goffin's model in
`compare_models_openloop`, and that is intentional (the goal is to reproduce it
faithfully). Everywhere else the project works in body rates.

### 3.1 `compare_models_openloop/` — Euler-angle derivatives

`goffin_dynamics.py` implements Goffin's model **entirely in Euler rates**.

State (L24-37) — Euler angles + Euler-angle derivatives:

```python
# x = [x, xd, y, yd, z, zd, phi, phid, theta, thetad, psi, psid]^T
IPHI, IPHID   = 6, 7
ITHETA, ITHETAD = 8, 9
IPSI, IPSID   = 10, 11
```

`nonlinear_continuous` (L178-180) — gyroscopic coupling written in Euler-angle
derivatives (Goffin eq. 2.10-2.12, approximate form):

```python
phi_dd   = (tau_x + (self.Iy - self.Iz) * thetad * psid) / self.Ix   # (2.10)
theta_dd = (tau_y + (self.Iz - self.Ix) * phid   * psid) / self.Iy   # (2.11)
psi_dd   = (tau_z + (self.Ix - self.Iy) * phid   * thetad) / self.Iz # (2.12)
```

Orientation integration (L189, L191, L193): `dxdt[IPHI] = phid` etc.
`linearized_continuous` (L201-245): same structure.

**`gym_state_to_goffin` (L268-310) — the only explicit converter in the
project.** It takes pybullet's world angular velocity and maps it back to
Euler-angle derivatives:

```python
# World -> body angular velocity
omega_body = R.T @ ang_v  # [p, q, r]                              (L292)

# Body rates -> Euler rates
if np.abs(c_theta) > 1e-8:
    t_theta = s_theta / c_theta
    phi_dot   = omega_body[0] + s_phi*t_theta*omega_body[1] + c_phi*t_theta*omega_body[2]
    theta_dot = c_phi*omega_body[1] - s_phi*omega_body[2]
    psi_dot   = (s_phi/c_theta)*omega_body[1] + (c_phi/c_theta)*omega_body[2]
```

This is the single place in the project where the conversion $\omega_B\to\dot\Phi$
(matrix $W^{-1}$) is performed.

`run_dynamics_comparison.py` (L121, L171): gym-pybullet's `env.ang_v` is in the
world frame; everything entering Goffin's model goes through
`gym_state_to_goffin`. The comparison is therefore honest — pybullet (body rates
internally) is converted to Euler rates to be compared against Goffin.

### 3.2 `circle_comparison_simu_and_real/` — body rates

`cf_firmware_pid_sim.py`, `obs_to_firmware_state` (L329-376):

```python
ang_v  = obs[13:16]                      # [rad/s] world           (L343)
...
rpy[1] = -rpy[1]                         # pitch: pybullet -> firmware (L354)
...
gyro_body = R.T @ ang_v          # [rad/s] body frame              (L370)
gyro_deg  = np.degrees(gyro_body)
gyro_deg[1] = -gyro_deg[1]       # pitch rate: same reason         (L374)
```

- `rpy` = Euler angles (order 0), `gyro_deg` = **body rates**.
- The Rate PID loop consumes `gyro_deg` = body rates (L473, L491).
- Real-drone logs (L1312-1314): `gyro.x/y/z` = raw gyroscope = native body rates.

### 3.3 `training_regulation_simu_and_real/` — body rates

`train_nn_cf_pid.py`, `dynamics_substep` (L153-204). State `[...phi,p,theta,q,
psi,r]`. Gyroscopic coupling in body rates (L188-190):

```python
gyro_x = (I_Z - I_Y) * q * r
gyro_y = (I_X - I_Z) * p * r
gyro_z = (I_Y - I_X) * p * q
```

⚠️ **Caveat** — orientation integration (L201):

```python
phi_n = phi + dt * p_n;   theta_n = theta + dt * q_n;   psi_n = psi + dt * r_n
```

The rotational *dynamics* are correct in body rates, but the orientation
*kinematics* use the small-angle approximation $\omega_B\simeq\dot\Phi$ (the
matrix $W^{-1}$ is omitted).

`fly_nn_cf_pid.py`:
- `obs_to_nn_state` (L126-157): `gyro_body = R.T @ ang_v` (L150) → body rates.
- `drone_state_to_nn_state` (L160-178): `gyro_x/y/z` from the real IMU → body
  rates; pitch and `q` negated (L168, L173).

### 3.4 `training_MAML/` — body rates

`maml_lib/dynamics.py`:

`nonlinear_step` (L53-98) — body rates, full gyroscopic term:

```python
omega = torch.stack([p, q, r], dim=-1)                             # (L85)
Iomega = (mass.I @ omega.unsqueeze(-1)).squeeze(-1)
gyro = torch.cross(omega, Iomega, dim=-1)                          # (L87)
omega_dot = (mass.I_inv @ (tau - gyro).unsqueeze(-1)).squeeze(-1)
```

Orientation integration (L95): `phi_n = phi + dt*p_n` — same small-angle
approximation as in 3.3.

`linearized_step` (L101-140): body rates, without gyroscopic coupling,
orientation L137 identical.

`test_maml_pybullet.py`, `obs_to_nn_state` (L66-89): `gyro_body = R.T @ ang_v`
(L82) → body rates. `rollout.py` / `policy.py` merely pass the state through.

### 3.5 Goffin vs training: the same model

A natural question: are Goffin's model and the dynamics of
`train_nn_cf_pid.py` / `maml_lib/dynamics.py` the same model, up to a variable
renaming (`phi_dot` ↔ `p`)? **Yes.** Term-by-term comparison of the nonlinear
model.

**Translation** — `train_nn` computes `thrust_world = F·R[:,:,2]` (3rd column of
the ZYX rotation matrix). This column gives exactly Goffin's expressions:

```
ax = F/m·(cφ·sθ·cψ + sφ·sψ)   ≡ Goffin (2.7)
ay = F/m·(cφ·sθ·sψ − sφ·cψ)   ≡ Goffin (2.8)
az = F/m·cφ·cθ − g            ≡ Goffin (2.9)
```

**Rotation** — Euler's equation. `train_nn`:

```
p_dot = (tau_x − (I_Z−I_Y)·q·r) / I_X
      = (tau_x + (I_Y−I_Z)·q·r) / I_X
```

Goffin (2.10): `phi_dd = (tau_x + (Iy−Iz)·θ̇·ψ̇) / Ix`. **Identical formula**
under the renaming `p↔φ̇, q↔θ̇, r↔ψ̇`.

**Motor mixing** (X configuration) — identical.

**Orientation integration** — both write
`angle_{k+1} = angle_k + dt · (angular-velocity variable)`. Goffin calls this
variable `φ̇`, `train_nn` calls it `p`, but the computation performed is the
same. In other words **both codebases make the same, single approximation**
$\dot\Phi\equiv\omega_B$: they use the variable both (a) as a body rate in
Euler's equation and (b) as an Euler-angle derivative to integrate the
orientation. This is only consistent near hover.

**Conclusion: physically it is the same model.** The variable names (`phi_dot`
vs `p`) change nothing in the computations; neither is "more correct" than the
other — it is the same approximation described from two opposite viewpoints.

The only real differences are **not** in the dynamics model:

1. **Integration scheme.** Goffin uses an explicit (forward) Euler scheme
   (`state + dt·dxdt`, all derivatives evaluated at the old state). `train_nn`
   and `maml_lib` use a **semi-implicit (symplectic) Euler** scheme:
   velocities/rates first, then positions/angles from the *new* velocities. A
   numerical, not physical, difference; negligible at small `dt`, more stable
   over a long horizon for the symplectic one.
2. **Offset mass.** `maml_lib/dynamics.py` generalises to the off-centre point
   mass (`M_total`, Steiner-corrected inertia, CoM offset `r_com`). For the
   baseline drone (centred mass, diagonal inertia) it reduces exactly to the
   dynamics of `train_nn_cf_pid.py`.

---

## 4. Summary table

There are in fact only **two distinct dynamics models** in the project:

| Model | Codebases | Dynamics (transl. + rotation) | Orientation kinematics |
|-------|-----------|-------------------------------|------------------------|
| Approximate (hover) | `goffin_dynamics.py`, `train_nn_cf_pid.py`, `maml_lib/dynamics.py` | same Euler equations | $\omega_B\equiv\dot\Phi$, `angle + dt·ω` |
| Exact | gym-pybullet DYN (`_integrateQ`) | same Euler equations | exact quaternion |

Goffin and the training simulators implement **the same** approximate model
(cf. 3.5), up to an explicit/semi-implicit integration variant. The only truly
distinct model is gym-pybullet's: identical dynamics equations, but orientation
integrated via an exact quaternion instead of `φ + dt·p`. The body-rates /
Euler-derivatives distinction therefore only concerns the **name** of the
variables and the **orientation kinematics** — never the dynamics equations
themselves.

---

## 5. Points to watch

1. **Training ↔ deployment consistency.** The neural network always sees body
   rates: in simulation because we apply `R.T @ ang_v`, on the real drone
   because the gyroscope measures them natively. A mismatch here would be a pure
   sim-to-real gap.
2. **The pitch.** At every pybullet ↔ firmware/real boundary, $\theta$ and $q$
   are negated. Roll and yaw never.
3. **The misleading name in gym-pybullet.** `BaseAviary`'s `rpy_rates` variable
   actually holds $\omega_B=(p,q,r)$, not $\dot\Phi$ — see the report,
   Section 2.1.
4. **Report wording.** The remark in Section 2.3 ("propagate $(p,q,r)$ through
   eq. 2.10-2.12") only holds for the *dynamics* part; the integration of $\Phi$
   in the training simulators is approximate (small-angle). To be clarified in
   Sections 5.2 / 6.2.
