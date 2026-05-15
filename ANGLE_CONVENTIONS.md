# Conventions angulaires : body rates vs dérivées d'Euler

Document de référence pour le projet de semestre. Il clarifie la distinction
entre **dérivées des angles d'Euler** $\dot\Phi=(\dot\phi,\dot\theta,\dot\psi)$
et **body rates** $\omega_B=(p,q,r)$, recense les conventions utilisées dans le
projet, et localise dans chaque base de code où chaque grandeur est employée.

---

## 1. Rappel théorique

### 1.1 Y a-t-il une notion de « body angles » (ordre 0) ?

**Non.** Les body rates $(p,q,r)$ ne sont la dérivée d'aucun vecteur d'angles.
La vitesse angulaire $\omega_B$ est une grandeur **anholonome** (non
intégrable) : il n'existe aucune paramétrisation $\Theta$ de l'orientation
telle que $\dot\Theta=\omega_B$. Intégrer $p$ dans le temps ne produit pas un
angle bien défini — le résultat dépend du chemin parcouru.

| Ordre | Représentation « Euler » | Représentation « body » | Différence ? |
|-------|--------------------------|-------------------------|--------------|
| 0 — orientation | $(\phi,\theta,\psi)$, carte sur $SO(3)$ | *n'existe pas* | — l'orientation est unique, indépendante du repère |
| 1 — vitesse angulaire | $\dot\Phi$ | $\omega_B=(p,q,r)$ | **oui** |
| 2 — accélération angulaire | $\ddot\Phi$ | $\dot\omega_B$ | **oui** (implique aussi $\dot W$) |

À l'ordre 0 il y a juste **l'orientation**, un élément de $SO(3)$ sur lequel
toutes les représentations sont d'accord ; les angles d'Euler en sont une
carte. La distinction body/Euler n'apparaît qu'à partir de l'ordre 1.

### 1.2 Relation cinématique

Body rates et dérivées d'Euler sont reliés par la matrice de transformation
cinématique :

$$\omega_B = W(\phi,\theta)\,\dot\Phi,\qquad
W(\phi,\theta)=\begin{bmatrix}
1 & 0 & -s_\theta\\
0 & c_\phi & s_\phi c_\theta\\
0 & -s_\phi & c_\phi c_\theta
\end{bmatrix}$$

Comme $W(0,0)=I_3$, on a $\omega_B\simeq\dot\Phi$ près du hover. La
distinction devient non négligeable hors du régime hover, et $W^{-1}$
dégénère à $\theta=\pm\pi/2$ (gimbal lock).

### 1.3 Conséquence sur le vecteur d'état

Le vecteur d'état 12-D du code empile `[φ, p, θ, q, ψ, r]` : un angle d'Euler
**couplé à un body rate qui n'en est pas la dérivée**. Les deux coïncident
seulement près du hover. Le rapport (section 2.1) écrit ce vecteur avec
$\dot\phi,\dot\theta,\dot\psi$ ; rigoureusement les emplacements angulaires
contiennent $p,q,r$.

---

## 2. Conventions du projet

### 2.1 Repères

- **Repère monde** $\mathcal{W}$ : inertiel, $z$ vers le haut.
- **Repère corps** $\mathcal{B}$ : attaché au CoM, $x_B$ avant, $y_B$ gauche,
  $z_B$ aligné avec la poussée.
- Orientation paramétrée par les angles d'Euler ZYX $(\phi,\theta,\psi)$.

### 2.2 Vecteur d'état (ordre du code)

```
[x, vx, y, vy, z, vz, phi, p, theta, q, psi, r]
 0   1   2   3   4   5    6   7    8   9   10  11
```

Les emplacements 7, 9, 11 contiennent les **body rates** $(p,q,r)$, pas
$\dot\Phi$.

### 2.3 Convention de signe du pitch

- **pybullet** (utilisé en simulation) : pitch positif = nez **vers le bas**.
- **Firmware Crazyflie** (aéronautique) : pitch positif = nez **vers le haut**.
- Roll et yaw : conventions identiques entre les deux.

→ Lors de la conversion pybullet ↔ firmware/réel, on **nie le pitch** $\theta$
**et le pitch rate** $q$. Le roll et le yaw ne sont pas touchés.

### 2.4 Vitesse angulaire selon la source

| Source | Forme native | Conversion appliquée |
|--------|---------------|----------------------|
| `obs[13:16]` de gym-pybullet | $\omega$ repère **monde** | `R.T @ ang_v` → body |
| Gyroscope du vrai Crazyflie (`gyro.x/y/z`) | body rates (deg/s) | aucune (déjà body) |
| Modèle de Goffin | dérivées d'Euler $\dot\Phi$ | — |

---

## 3. Audit par base de code

**Résumé :** un seul code utilise les dérivées d'Euler — le modèle de Goffin
dans `compare_models_openloop`, et c'est voulu (le but est de le reproduire
fidèlement). Partout ailleurs on travaille en body rates.

### 3.1 `compare_models_openloop/` — dérivées d'Euler

`goffin_dynamics.py` implémente le modèle de Goffin **entièrement en Euler
rates**.

État (L24-37) — angles d'Euler + dérivées d'Euler :

```python
# x = [x, xd, y, yd, z, zd, phi, phid, theta, thetad, psi, psid]^T
IPHI, IPHID   = 6, 7
ITHETA, ITHETAD = 8, 9
IPSI, IPSID   = 10, 11
```

`nonlinear_continuous` (L178-180) — couplage gyroscopique écrit en dérivées
d'Euler (eq. 2.10-2.12 de Goffin, forme approchée) :

```python
phi_dd   = (tau_x + (self.Iy - self.Iz) * thetad * psid) / self.Ix   # (2.10)
theta_dd = (tau_y + (self.Iz - self.Ix) * phid   * psid) / self.Iy   # (2.11)
psi_dd   = (tau_z + (self.Ix - self.Iy) * phid   * thetad) / self.Iz # (2.12)
```

Intégration de l'orientation (L189, L191, L193) : `dxdt[IPHI] = phid` etc.
`linearized_continuous` (L201-245) : même structure.

**`gym_state_to_goffin` (L268-310) — le seul convertisseur explicite du
projet.** Il prend la vitesse angulaire monde de pybullet et la ramène en
dérivées d'Euler :

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

C'est l'unique endroit du projet où la conversion $\omega_B\to\dot\Phi$
(matrice $W^{-1}$) est faite.

`run_dynamics_comparison.py` (L121, L171) : `env.ang_v` de gym-pybullet est en
repère monde ; tout ce qui entre dans le modèle Goffin passe par
`gym_state_to_goffin`. La comparaison est donc honnête — pybullet (body rates
en interne) est converti en Euler rates pour être comparé à Goffin.

### 3.2 `circle_comparison_simu_and_real/` — body rates

`cf_firmware_pid_sim.py`, `obs_to_firmware_state` (L329-376) :

```python
ang_v  = obs[13:16]                      # [rad/s] global          (L343)
...
rpy[1] = -rpy[1]                         # pitch : pybullet -> firmware (L354)
...
gyro_body = R.T @ ang_v          # [rad/s] body frame              (L370)
gyro_deg  = np.degrees(gyro_body)
gyro_deg[1] = -gyro_deg[1]       # pitch rate : même raison        (L374)
```

- `rpy` = angles d'Euler (ordre 0), `gyro_deg` = **body rates**.
- La boucle Rate PID consomme `gyro_deg` = body rates (L473, L491).
- Logs du vrai drone (L1312-1314) : `gyro.x/y/z` = gyroscope brut = body rates
  natifs.

### 3.3 `training_regulation_simu_and_real/` — body rates

`train_nn_cf_pid.py`, `dynamics_substep` (L153-204). État `[...phi,p,theta,q,
psi,r]`. Couplage gyroscopique en body rates (L188-190) :

```python
gyro_x = (I_Z - I_Y) * q * r
gyro_y = (I_X - I_Z) * p * r
gyro_z = (I_Y - I_X) * p * q
```

⚠️ **Nuance** — intégration de l'orientation (L201) :

```python
phi_n = phi + dt * p_n;   theta_n = theta + dt * q_n;   psi_n = psi + dt * r_n
```

La *dynamique* rotationnelle est correcte en body rates, mais la *cinématique*
d'orientation utilise l'approximation petit-angle $\omega_B\simeq\dot\Phi$
(la matrice $W^{-1}$ est omise).

`fly_nn_cf_pid.py` :
- `obs_to_nn_state` (L126-157) : `gyro_body = R.T @ ang_v` (L150) → body rates.
- `drone_state_to_nn_state` (L160-178) : `gyro_x/y/z` du vrai IMU → body rates ;
  pitch et `q` niés (L168, L173).

### 3.4 `training_MAML/` — body rates

`maml_lib/dynamics.py` :

`nonlinear_step` (L53-98) — body rates, gyroscopique complet :

```python
omega = torch.stack([p, q, r], dim=-1)                             # (L85)
Iomega = (mass.I @ omega.unsqueeze(-1)).squeeze(-1)
gyro = torch.cross(omega, Iomega, dim=-1)                          # (L87)
omega_dot = (mass.I_inv @ (tau - gyro).unsqueeze(-1)).squeeze(-1)
```

Intégration de l'orientation (L95) : `phi_n = phi + dt*p_n` — même
approximation petit-angle qu'en 3.3.

`linearized_step` (L101-140) : body rates, sans couplage gyroscopique,
orientation L137 identique.

`test_maml_pybullet.py`, `obs_to_nn_state` (L66-89) : `gyro_body = R.T @ ang_v`
(L82) → body rates. `rollout.py` / `policy.py` ne font que transmettre l'état.

### 3.5 Goffin vs entraînement : le même modèle

Question naturelle : le modèle de Goffin et la dynamique de
`train_nn_cf_pid.py` / `maml_lib/dynamics.py` sont-ils le même modèle, à un
renommage de variables près (`phi_dot` ↔ `p`) ? **Oui.** Comparaison terme à
terme du modèle nonlinéaire.

**Translation** — `train_nn` calcule `thrust_world = F·R[:,:,2]` (3ᵉ colonne de
la matrice de rotation ZYX). Cette colonne donne exactement les expressions de
Goffin :

```
ax = F/m·(cφ·sθ·cψ + sφ·sψ)   ≡ Goffin (2.7)
ay = F/m·(cφ·sθ·sψ − sφ·cψ)   ≡ Goffin (2.8)
az = F/m·cφ·cθ − g            ≡ Goffin (2.9)
```

**Rotation** — équation d'Euler. `train_nn` :

```
p_dot = (tau_x − (I_Z−I_Y)·q·r) / I_X
      = (tau_x + (I_Y−I_Z)·q·r) / I_X
```

Goffin (2.10) : `phi_dd = (tau_x + (Iy−Iz)·θ̇·ψ̇) / Ix`. **Formule identique**
sous le renommage `p↔φ̇, q↔θ̇, r↔ψ̇`.

**Mixage moteurs** (config X) — identique.

**Intégration de l'orientation** — les deux écrivent
`angle_{k+1} = angle_k + dt · (variable de vitesse angulaire)`. Goffin appelle
cette variable `φ̇`, `train_nn` l'appelle `p`, mais le calcul exécuté est le
même. Autrement dit **les deux codes font la même et unique approximation**
$\dot\Phi\equiv\omega_B$ : ils utilisent la variable à la fois (a) comme body
rate dans l'équation d'Euler et (b) comme dérivée d'Euler pour intégrer
l'orientation. Ce n'est cohérent que près du hover.

**Conclusion : physiquement c'est le même modèle.** Le nom des variables
(`phi_dot` vs `p`) ne change rien aux calculs ; aucun des deux n'est « plus
correct » que l'autre — c'est la même approximation décrite depuis deux points
de vue opposés.

Les seules vraies différences ne sont **pas** dans le modèle de dynamique :

1. **Schéma d'intégration.** Goffin utilise un Euler explicite
   (`state + dt·dxdt`, toutes les dérivées évaluées sur l'état ancien).
   `train_nn` et `maml_lib` utilisent un Euler **semi-implicite (symplectique)**:
   vitesses/rates d'abord, puis positions/angles à partir des *nouvelles*
   vitesses. Différence numérique, pas physique ; négligeable à petit `dt`,
   plus stable sur long horizon pour le symplectique.
2. **Masse décentrée.** `maml_lib/dynamics.py` généralise au point-masse
   décentré (`M_total`, inertie corrigée de Steiner, offset du CoM `r_com`).
   Pour le drone de base (masse centrée, inertie diagonale) il se réduit
   exactement à la dynamique de `train_nn_cf_pid.py`.

---

## 4. Tableau de synthèse

Il n'existe en réalité que **deux modèles de dynamique distincts** dans le
projet :

| Modèle | Codes | Dynamique (transl. + rotation) | Cinématique d'orientation |
|--------|-------|-------------------------------|---------------------------|
| Approché (hover) | `goffin_dynamics.py`, `train_nn_cf_pid.py`, `maml_lib/dynamics.py` | mêmes équations d'Euler | $\omega_B\equiv\dot\Phi$, `angle + dt·ω` |
| Exact | gym-pybullet DYN (`_integrateQ`) | mêmes équations d'Euler | quaternion exact |

Goffin et les simulateurs d'entraînement implémentent **le même** modèle
approché (cf. 3.5), à une variante d'intégration explicite/semi-implicite près.
Le seul modèle réellement distinct est celui de gym-pybullet : équations de
dynamique identiques, mais intégration de l'orientation par quaternion exact au
lieu de `φ + dt·p`. La distinction body rates / dérivées d'Euler ne porte donc
que sur le **nom** des variables et la **cinématique d'orientation** — jamais
sur les équations de dynamique elles-mêmes.

---

## 5. Points de vigilance

1. **Cohérence entraînement ↔ déploiement.** Le réseau de neurones voit
   toujours des body rates : en simulation parce qu'on applique `R.T @ ang_v`,
   sur le réel parce que le gyroscope les mesure nativement. Un désaccord ici
   serait un pur écart sim-to-real.
2. **Le pitch.** À chaque frontière pybullet ↔ firmware/réel, $\theta$ et $q$
   sont niés. Roll et yaw jamais.
3. **Le nom trompeur dans gym-pybullet.** La variable `rpy_rates` de
   `BaseAviary` contient en réalité $\omega_B=(p,q,r)$, pas $\dot\Phi$ — voir
   le rapport, section 2.1.
4. **Formulation du rapport.** La remarque de la section 2.3
   (« propagate $(p,q,r)$ through eq. 2.10-2.12 ») ne vaut que pour la partie
   *dynamique* ; l'intégration de $\Phi$ dans les simulateurs d'entraînement
   est approchée (petit-angle). À préciser dans les sections 5.2 / 6.2.
