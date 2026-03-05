#!/usr/bin/env python3
"""
Train a neural network controller for the Crazyflie 2.x (CF2X) quadcopter.

Reproduces Goffin's section 4.1 sanity check (regulation to origin) but with:
  - CF2X physical parameters from gym-pybullet-drones
  - Nonlinear dynamics matching pybullet's DYN mode exactly:
      * full rotation matrix (no small-angle approximation for translation)
      * gyroscopic coupling  tau_net = tau - omega x (J*omega)
      * semi-implicit Euler integration (velocity first, then position)
      * same sub-stepping structure (5 sub-steps at 1/240 s per ctrl step)

The NN outputs absolute wrench [F, tau_x, tau_y, tau_z].

Usage:
    python train_nn_cf2x.py
    -> saves  trained_policy_cf2x.pt  (full model)
    -> saves  trained_weights_cf2x.pt (state_dict only)
"""

import numpy as np
import scipy.linalg as la
import torch
import torch.nn as nn
import itertools, os


# =====================================================================
# 1. CF2X Physical Parameters  (from cf2x.urdf in gym-pybullet-drones)
# =====================================================================

# ---------- Identiques au URDF cf2x.urdf ----------
M       = 0.027                         # mass [kg]
G       = 9.8                           # gravity [m/s^2]  (pybullet uses 9.8)
I_X     = 1.4e-5                        # Ixx [kg*m^2]
I_Y     = 1.4e-5                        # Iyy [kg*m^2]
I_Z     = 2.17e-5                       # Izz [kg*m^2]
L       = 0.0397                        # arm length [m]
KF      = 3.16e-10                      # thrust coeff [N/RPM^2]
KM      = 7.94e-12                      # torque coeff [N*m/RPM^2]
T2W     = 2.25                          # thrust-to-weight ratio

# ---------- Limites derivees (meme calcul que BaseAviary.__init__) ----------
GRAVITY     = M * G                                         # poids [N]
HOVER_RPM   = np.sqrt(GRAVITY / (4 * KF))
MAX_RPM     = np.sqrt((T2W * GRAVITY) / (4 * KF))
F_MAX       = 4 * KF * MAX_RPM**2                          # poussee max [N]
TAU_XY_MAX  = (2 * L * KF * MAX_RPM**2) / np.sqrt(2)      # couple roll/pitch max
TAU_Z_MAX   = 2 * KM * MAX_RPM**2                          # couple yaw max

U_MAX = np.array([F_MAX, TAU_XY_MAX, TAU_XY_MAX, TAU_Z_MAX], dtype=np.float32)

# ---------- State scaling ----------
# Sert a :
#   a) normaliser les entrees du NN  (le reseau recoit x / x_scale)
#   b) ponderer la matrice Q du cout (Q_ii = 1 / x_scale_i^2)
#
# Ce sont des hyperparametres. On les choisit comme les deviations maximales
# "acceptables" de chaque composante d'etat.  Il faudra les tuner.
X_MAX       = 1.0                   # position max acceptable [m]
X_DMAX      = 1.0                   # vitesse max acceptable [m/s]
Y_MAX       = 1.0
Y_DMAX      = 1.0
Z_MAX       = 1.0
Z_DMAX      = 1.0
PHI_MAX     = np.deg2rad(30)        # inclinaison max acceptable
PHI_DMAX    = np.deg2rad(200)       # taux angulaire max acceptable
THETA_MAX   = np.deg2rad(30)
THETA_DMAX  = np.deg2rad(200)
PSI_MAX     = np.deg2rad(45)
PSI_DMAX    = np.deg2rad(120)

X_SCALE = np.array([X_MAX, X_DMAX, Y_MAX, Y_DMAX, Z_MAX, Z_DMAX,
                     PHI_MAX, PHI_DMAX, THETA_MAX, THETA_DMAX,
                     PSI_MAX, PSI_DMAX], dtype=np.float32)

# ---------- Timing  (identique a pybullet) ----------
PYB_FREQ   = 240                    # frequence physique pybullet [Hz]
CTRL_FREQ  = 48                     # frequence de controle [Hz]
PYB_STEPS_PER_CTRL = PYB_FREQ // CTRL_FREQ   # = 5 sous-pas par pas de controle
DT_PYB     = 1.0 / PYB_FREQ        # = 1/240 s  (pas de temps de chaque sous-pas)
DT_CTRL    = 1.0 / CTRL_FREQ       # = 1/48 s   (intervalle entre deux actions du NN)
T_SIM      = 4.0                    # duree de simulation [s]
T_STEPS    = int(T_SIM / DT_CTRL)  # = 192 pas de controle

print(f"[CF2X] m={M}, g={G}, Ixx={I_X}, Iyy={I_Y}, Izz={I_Z}")
print(f"[CF2X] F_max={F_MAX:.4f} N, tau_xy_max={TAU_XY_MAX:.6f} N*m, tau_z_max={TAU_Z_MAX:.6f} N*m")
print(f"[CF2X] HOVER_RPM={HOVER_RPM:.1f}, MAX_RPM={MAX_RPM:.1f}")
print(f"[CF2X] DT_pyb={DT_PYB:.6f}s, DT_ctrl={DT_CTRL:.5f}s, "
      f"sub-steps={PYB_STEPS_PER_CTRL}, T_steps={T_STEPS}")


# =====================================================================
# 2. Dynamique non-lineaire (replique le mode DYN de pybullet)
# =====================================================================
#
# Reference : BaseAviary._dynamics()   (lignes 815-877)
#
# L'etat est :  [x, xdot, y, ydot, z, zdot, phi, p, theta, q, psi, r]
#
# ou p,q,r sont les vitesses angulaires body-frame (= rpy_rates dans pybullet).
# Au voisinage du hover, p~=phi_dot, q~=theta_dot, r~=psi_dot.
#
# L'entree est :  u = [F, tau_x, tau_y, tau_z]   (wrench)
#
# Trois blocs de physique reproduits fidelement :
#
#   (a) Rotation du thrust en world-frame :
#         R(phi,theta,psi) @ [0, 0, F]
#       Identique a pybullet  "thrust_world_frame = np.dot(rotation, thrust)"
#       sauf que pybullet obtient R via le quaternion et nous via les angles d'Euler.
#       La matrice R est la meme (convention ZYX intrinsic).
#
#   (b) Couplage gyroscopique :
#         tau_net = [tau_x,tau_y,tau_z] - omega x (J*omega)
#       Identique a pybullet  "torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))"
#
#   (c) Integration Euler semi-implicite (symplectique) :
#         v  <- v  + dt*a          (vitesse mise a jour EN PREMIER)
#         omega <- omega + dt*omega_dot
#         pos <- pos + dt*v        (position mise a jour AVEC la nouvelle vitesse)
#         rpy <- rpy + dt*omega    (idem pour l'orientation)
#       Identique a pybullet lignes 860-863.
#
#   (d) Structure de sous-pas :
#         Pour chaque pas de controle, on fait PYB_STEPS_PER_CTRL sous-pas
#         a dt = 1/240 s, avec le meme wrench tenu constant.
#       Identique a la boucle "for _ in range(self.PYB_STEPS_PER_CTRL)"
#
# NOTE : la seule difference est l'integration de l'orientation :
#   - pybullet utilise des quaternions (_integrateQ)
#   - nous utilisons rpy <- rpy + dt*omega
#   Ces deux approches sont equivalentes au premier ordre et donnent les
#   memes resultats au voisinage du hover.  Pour des manoeuvres agressives
#   (> 60 deg de tilt), les quaternions seraient plus precis.


def rotation_matrix_zyx(phi, theta, psi):
    """Matrice de rotation ZYX intrinsic (= XYZ extrinsic).

    Identique a p.getMatrixFromQuaternion(p.getQuaternionFromEuler([phi,theta,psi]))
    au voisinage du hover.

    Parametres : tenseurs de shape (B,) ou scalaires.
    Retourne :   tenseur de shape (B, 3, 3).
    """
    cphi = torch.cos(phi);   sphi = torch.sin(phi)
    cth  = torch.cos(theta); sth  = torch.sin(theta)
    cpsi = torch.cos(psi);   spsi = torch.sin(psi)

    # Ligne 1
    r00 = cth * cpsi
    r01 = sphi * sth * cpsi - cphi * spsi
    r02 = cphi * sth * cpsi + sphi * spsi
    # Ligne 2
    r10 = cth * spsi
    r11 = sphi * sth * spsi + cphi * cpsi
    r12 = cphi * sth * spsi - sphi * cpsi
    # Ligne 3
    r20 = -sth
    r21 = sphi * cth
    r22 = cphi * cth

    R = torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)  # (B, 3, 3)
    return R


def dynamics_substep(state, wrench, dt):
    """Un sous-pas d'integration Euler semi-implicite.

    Replique exactement BaseAviary._dynamics() lignes 838-863.

    Parametres
    ----------
    state  : (B, 12)  [x, vx, y, vy, z, vz, phi, p, theta, q, psi, r]
    wrench : (B, 4)   [F, tau_x, tau_y, tau_z]
    dt     : float    pas de temps (= 1/240 s)

    Retourne
    --------
    state_new : (B, 12)
    """
    # --- Unpacker l'etat ---
    x     = state[:, 0];  vx = state[:, 1]
    y     = state[:, 2];  vy = state[:, 3]
    z     = state[:, 4];  vz = state[:, 5]
    phi   = state[:, 6];  p  = state[:, 7]     # p = rpy_rates[0]
    theta = state[:, 8];  q  = state[:, 9]     # q = rpy_rates[1]
    psi   = state[:, 10]; r  = state[:, 11]    # r = rpy_rates[2]

    F     = wrench[:, 0]
    tau_x = wrench[:, 1]
    tau_y = wrench[:, 2]
    tau_z = wrench[:, 3]

    # --- (a) Thrust en world-frame ---
    # pybullet : thrust = [0, 0, F]
    #            thrust_world = R @ thrust
    #            force_world = thrust_world - [0, 0, GRAVITY]
    #            acc = force_world / M
    R = rotation_matrix_zyx(phi, theta, psi)  # (B, 3, 3)

    # R @ [0, 0, F]  =  F * R[:, :, 2]   (3eme colonne de R)
    thrust_world = F.unsqueeze(-1) * R[:, :, 2]  # (B, 3)

    ax = (thrust_world[:, 0]) / M
    ay = (thrust_world[:, 1]) / M
    az = (thrust_world[:, 2]) / M - G      # gravity: -[0,0,GRAVITY]/M = -g

    # --- (b) Couplage gyroscopique ---
    # pybullet : torques = [tau_x, tau_y, tau_z] - cross(omega, J @ omega)
    #            omega_dot = J_inv @ torques
    #
    # cross(omega, J*omega) avec J diagonal :
    #   [p]     [Ix*p]     [q*Iz*r - r*Iy*q]     [(Iz-Iy)*q*r]
    #   [q]  x  [Iy*q]  =  [r*Ix*p - p*Iz*r]  =  [(Ix-Iz)*p*r]
    #   [r]     [Iz*r]     [p*Iy*q - q*Ix*p]     [(Iy-Ix)*p*q]
    gyro_x = (I_Z - I_Y) * q * r
    gyro_y = (I_X - I_Z) * p * r
    gyro_z = (I_Y - I_X) * p * q

    p_dot = (tau_x - gyro_x) / I_X
    q_dot = (tau_y - gyro_y) / I_Y
    r_dot = (tau_z - gyro_z) / I_Z

    # --- (c) Integration Euler semi-implicite ---
    # Etape 1 : mettre a jour les vitesses
    vx_new = vx + dt * ax
    vy_new = vy + dt * ay
    vz_new = vz + dt * az
    p_new  = p  + dt * p_dot
    q_new  = q  + dt * q_dot
    r_new  = r  + dt * r_dot

    # Etape 2 : mettre a jour les positions AVEC LES NOUVELLES vitesses
    x_new     = x     + dt * vx_new
    y_new     = y     + dt * vy_new
    z_new     = z     + dt * vz_new
    phi_new   = phi   + dt * p_new
    theta_new = theta + dt * q_new
    psi_new   = psi   + dt * r_new

    state_new = torch.stack([
        x_new, vx_new, y_new, vy_new, z_new, vz_new,
        phi_new, p_new, theta_new, q_new, psi_new, r_new
    ], dim=-1)
    return state_new


def dynamics_one_ctrl_step(state, wrench, n_substeps=PYB_STEPS_PER_CTRL):
    """Avance d'un pas de controle complet.

    Parametres
    ----------
    n_substeps : int
        Nombre de sous-pas d'Euler.
        - n_substeps=5, dt=1/240  : replique exactement pybullet (pour validation)
        - n_substeps=1, dt=1/48   : 5x plus rapide (pour entrainement)
        Les deux donnent des resultats proches car dt=1/48 reste petit.
        (Goffin utilisait dt=0.1s avec Euler dans ses exp. non-lineaires !)
    """
    dt = DT_CTRL / n_substeps  # duree de chaque sous-pas
    for _ in range(n_substeps):
        state = dynamics_substep(state, wrench, dt)
    return state


# =====================================================================
# 3. LQR baseline (pour comparaison - utilise le modele linearise)
# =====================================================================
# On linearise uniquement pour calculer le gain LQR comme baseline.
# L'entrainement du NN utilise la dynamique non-lineaire ci-dessus.

def _build_linear_model_for_lqr():
    """Modele linearise + discretisation ZOH (seulement pour le LQR baseline)."""
    from scipy.signal import cont2discrete

    A = np.zeros((12, 12))
    A[0,1]=1; A[2,3]=1; A[4,5]=1; A[6,7]=1; A[8,9]=1; A[10,11]=1
    A[1,8] = G;  A[3,6] = -G
    B = np.zeros((12, 4))
    B[5,0]=1/M; B[7,1]=1/I_X; B[9,2]=1/I_Y; B[11,3]=1/I_Z

    c = np.zeros((12,1)); c[5,0] = -G
    n, m = 12, 4
    A_aug = np.zeros((n+1,n+1)); A_aug[:n,:n]=A; A_aug[:n,n]=c.squeeze()
    B_aug = np.zeros((n+1,m)); B_aug[:n,:]=B
    C_aug = np.zeros((n,n+1)); C_aug[:,:n]=np.eye(n)
    D_aug = np.zeros((n,m))
    Ad_aug,Bd_aug,_,_,_ = cont2discrete((A_aug,B_aug,C_aug,D_aug), DT_CTRL, method='zoh')
    Ad = Ad_aug[:n,:n]; Bd = Bd_aug[:n,:]; d = Ad_aug[:n,n]

    Q_diag = np.array([1/X_MAX**2, 1/X_DMAX**2, 1/Y_MAX**2, 1/Y_DMAX**2,
                        1/Z_MAX**2, 1/Z_DMAX**2, 1/PHI_MAX**2, 1/PHI_DMAX**2,
                        1/THETA_MAX**2, 1/THETA_DMAX**2, 1/PSI_MAX**2, 1/PSI_DMAX**2])
    R_diag = np.array([1/F_MAX**2, 1/TAU_XY_MAX**2, 1/TAU_XY_MAX**2, 1/TAU_Z_MAX**2])
    P = la.solve_discrete_are(Ad, Bd, np.diag(Q_diag), np.diag(R_diag))
    K = np.linalg.inv(Bd.T@P@Bd + np.diag(R_diag)) @ (Bd.T@P@Ad)
    u_eq = -np.linalg.pinv(Bd) @ d
    return K, u_eq, Ad, Bd, d

K_LQR, U_EQ, Ad_lqr, Bd_lqr, d_lqr = _build_linear_model_for_lqr()
print(f"\n[LQR] u_eq[0] (hover thrust) = {U_EQ[0]:.6f} N  (mg = {M*G:.6f} N)")


# =====================================================================
# 4. Reseau de neurones
# =====================================================================

class PolicyMLP(nn.Module):
    """MLP controller: state (12) -> wrench (4).

    Thrust  -> sigmoid -> [0, F_max]
    Torques -> tanh    -> [-tau_max, +tau_max]

    INITIALISATION CRITIQUE :
    La derniere couche est initialisee a poids=0, biais=[logit(mg/Fmax), 0, 0, 0].
    Ainsi, quelle que soit l'entree, la sortie initiale est :
        thrust  = sigmoid(logit(mg/Fmax)) * Fmax = mg    (hover)
        torques = tanh(0) * tau_max = 0                   (pas de rotation)
    Le drone commence donc en vol stationnaire, et le gradient peut guider
    l'apprentissage depuis ce point d'equilibre stable.
    """
    def __init__(self, x_scale, u_max, hidden=64):
        super().__init__()
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32))
        self.register_buffer("u_max",   torch.tensor(u_max,   dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(12, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 4),
        )
        self._init_hover()

    def _init_hover(self):
        """Initialise la derniere couche pour sortir le wrench de hover."""
        last_layer = self.net[-1]   # nn.Linear(hidden, 4)
        # Poids a zero : la sortie ne depend pas de l'entree au depart
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        # Biais du thrust : sigmoid(b) * Fmax = mg  =>  b = logit(mg/Fmax)
        hover_ratio = (M * G) / float(self.u_max[0])  # mg / Fmax ≈ 0.444
        last_layer.bias.data[0] = float(np.log(hover_ratio / (1.0 - hover_ratio)))
        # Biais des torques = 0 : tanh(0) = 0, pas de couple

    def forward(self, x):
        x_n = x / self.x_scale
        raw = self.net(x_n)
        thrust  = torch.sigmoid(raw[..., [0]]) * self.u_max[0]
        torques = torch.tanh(raw[..., 1:])     * self.u_max[1:]
        return torch.cat([thrust, torques], dim=-1)


# =====================================================================
# 5. Rollout differentiable (non-lineaire) et cout
# =====================================================================

def rollout(policy, x0, T_steps, n_substeps=1):
    """Deroule la politique sur T_steps pas de controle avec dynamique non-lineaire.

    A chaque pas de controle :
      1. Le NN choisit wrench = policy(state)
      2. On integre n_substeps sous-pas de dynamique non-lineaire
         avec le wrench tenu constant.

    Parametres
    ----------
    policy     : PolicyMLP
    x0         : (B, 12) etats initiaux
    T_steps    : int  nombre de pas de controle
    n_substeps : int  sous-pas par pas de controle (1=rapide, 5=pybullet exact)
    """
    B = x0.shape[0]
    dev = x0.device
    X = torch.zeros(B, T_steps, 12, device=dev)
    U = torch.zeros(B, T_steps, 4,  device=dev)
    state = x0

    for k in range(T_steps):
        wrench = policy(state)
        state  = dynamics_one_ctrl_step(state, wrench, n_substeps)
        X[:, k, :] = state
        U[:, k, :] = wrench
    return X, U


# --- Matrice de cout ---
Q_DIAG = np.array([
    1/X_MAX**2,     1/X_DMAX**2,
    1/Y_MAX**2,     1/Y_DMAX**2,
    1/Z_MAX**2,     1/Z_DMAX**2,
    1/PHI_MAX**2,   1/PHI_DMAX**2,
    1/THETA_MAX**2, 1/THETA_DMAX**2,
    1/PSI_MAX**2,   1/PSI_DMAX**2,
], dtype=np.float32)

R_DIAG = np.array([
    1/F_MAX**2,
    1/TAU_XY_MAX**2,
    1/TAU_XY_MAX**2,
    1/TAU_Z_MAX**2,
], dtype=np.float32)


def traj_cost(X, U, terminal_weight=0.0):
    """Cout quadratique : sum_k (x'Qx + u'Ru), moyenne sur le batch."""
    Q = torch.as_tensor(Q_DIAG, dtype=X.dtype, device=X.device)
    R = torch.as_tensor(R_DIAG, dtype=U.dtype, device=U.device)

    cost_x = (X**2 * Q).sum(dim=2).mean(dim=0).sum()
    cost_u = (U**2 * R).sum(dim=2).mean(dim=0).sum()
    L = cost_x + cost_u

    if terminal_weight > 0.0:
        pT = X[:, -1, [0, 2, 4]]   # terminal [x, y, z]
        L += terminal_weight * (pT**2).mean()
    return L


# =====================================================================
# 6. Entrainement
# =====================================================================

def make_x0_batch(xyz_list, device="cpu"):
    """Convertit une liste de (x,y,z) en tenseur (B, 12) d'etats initiaux.
    Toutes les vitesses et angles sont a zero."""
    X0 = torch.zeros(len(xyz_list), 12, dtype=torch.float32, device=device)
    for b, (x, y, z) in enumerate(xyz_list):
        X0[b, 0] = x
        X0[b, 2] = y
        X0[b, 4] = z
    return X0


def generate_cube_points(half_side=0.3):
    """27 points : sommets + centres des faces + centres des aretes + centre."""
    vals = [-half_side, 0.0, half_side]
    return list(itertools.product(vals, repeat=3))


def train(epochs=2000, lr=1e-3, hidden=64, terminal_weight=10.0,
          half_side=0.3, device="cpu"):
    """Entraine le NN avec dynamique non-lineaire + curriculum sur l'horizon.

    Curriculum : on commence avec un horizon court (le drone n'a qu'a rester
    stable 0.5s) puis on allonge progressivement jusqu'a T_SIM.
    Cela permet au NN d'apprendre d'abord a stabiliser, puis a reguler.

    On utilise 1 sous-pas par pas de controle (dt=1/48s) pendant l'entrainement
    pour la vitesse.  La dynamique reste la meme, juste avec un dt plus grand.
    C'est equivalent a Goffin qui utilisait dt=0.1s en Euler non-lineaire.
    """
    policy = PolicyMLP(x_scale=X_SCALE, u_max=U_MAX, hidden=hidden).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)

    cube_pts = generate_cube_points(half_side)
    x0_batch = make_x0_batch(cube_pts, device=device)

    # Verifier l'init hover
    with torch.no_grad():
        u_test = policy(torch.zeros(1, 12, device=device))
        print(f"\n[Init] NN output at x=0: F={u_test[0,0]:.4f} N (mg={M*G:.4f}), "
              f"tau={u_test[0,1:]}")

    # --- Curriculum : horizons croissants ---
    # Phase 1 : 24 pas  = 0.5 s    (stabiliser)
    # Phase 2 : 48 pas  = 1.0 s    (commencer a reguler)
    # Phase 3 : 96 pas  = 2.0 s    (reguler)
    # Phase 4 : 192 pas = 4.0 s    (horizon complet)
    horizons = [24, 48, 96, T_STEPS]
    epochs_per_phase = epochs // len(horizons)

    print(f"[Train] {len(cube_pts)} pts, epochs={epochs}, lr={lr}")
    print(f"[Train] Curriculum: {len(horizons)} phases, "
          f"{epochs_per_phase} epochs/phase")
    print(f"[Train] Horizons (ctrl steps): {horizons}")
    print(f"[Train] 1 sous-pas/ctrl (dt={DT_CTRL:.4f}s) pour la vitesse")

    global_ep = 0
    for phase_idx, horizon in enumerate(horizons):
        t_horizon = horizon * DT_CTRL
        print(f"\n--- Phase {phase_idx+1}/{len(horizons)}: "
              f"horizon={horizon} steps ({t_horizon:.1f}s) ---")

        # Reset le scheduler pour chaque phase
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs_per_phase, eta_min=lr*0.1)

        for ep in range(epochs_per_phase):
            X, U = rollout(policy, x0_batch, horizon, n_substeps=1)
            loss = traj_cost(X, U, terminal_weight)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            scheduler.step()
            global_ep += 1

            if (ep + 1) % 100 == 0 or ep == 0:
                with torch.no_grad():
                    pos_end = X[:, -1, [0, 2, 4]].norm(dim=1)
                    print(f"  [ep {global_ep:4d}]  loss={loss.item():.4e}  "
                          f"|x_T|_mean={pos_end.mean().item():.4f}  "
                          f"|x_T|_max={pos_end.max().item():.4f}")

    return policy


# =====================================================================
# 7. Evaluation (NN vs LQR)
# =====================================================================

@torch.no_grad()
def evaluate(policy, test_pts, device="cpu"):
    """Evalue le NN et le LQR sur des positions de test."""

    x0 = make_x0_batch(test_pts, device=device)

    # --- NN (dynamique non-lineaire, 5 sous-pas = pybullet exact) ---
    X_nn, U_nn = rollout(policy, x0, T_STEPS, n_substeps=PYB_STEPS_PER_CTRL)
    err_nn = X_nn[:, -1, [0, 2, 4]].norm(dim=1)
    print(f"\n[Eval] {len(test_pts)} test points (non-lineaire, {PYB_STEPS_PER_CTRL} sous-pas):")
    print(f"  NN   terminal error  mean={err_nn.mean().item():.5f} m  "
          f"max={err_nn.max().item():.5f} m")

    # --- LQR (dynamique lineaire, pour baseline) ---
    Ad_t  = torch.tensor(Ad_lqr, dtype=torch.float32, device=device)
    Bd_t  = torch.tensor(Bd_lqr, dtype=torch.float32, device=device)
    d_t   = torch.tensor(d_lqr,  dtype=torch.float32, device=device).unsqueeze(0)
    K_t   = torch.tensor(K_LQR,  dtype=torch.float32, device=device)
    ueq_t = torch.tensor(U_EQ,   dtype=torch.float32, device=device).unsqueeze(0)

    x = x0.clone()
    X_lqr = torch.zeros_like(X_nn)
    for k in range(T_STEPS):
        u = ueq_t - (x @ K_t.T)
        u[:, 0] = torch.clamp(u[:, 0], 0, F_MAX)
        u[:, 1] = torch.clamp(u[:, 1], -TAU_XY_MAX, TAU_XY_MAX)
        u[:, 2] = torch.clamp(u[:, 2], -TAU_XY_MAX, TAU_XY_MAX)
        u[:, 3] = torch.clamp(u[:, 3], -TAU_Z_MAX,  TAU_Z_MAX)
        x = (x @ Ad_t.T) + (u @ Bd_t.T) + d_t
        X_lqr[:, k, :] = x
    err_lqr = X_lqr[:, -1, [0, 2, 4]].norm(dim=1)
    print(f"  LQR  terminal error  mean={err_lqr.mean().item():.5f} m  "
          f"max={err_lqr.max().item():.5f} m  (dynamique lineaire)")

    return X_nn, U_nn


# =====================================================================
# 8. Main
# =====================================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Device] {device}")

    policy = train(epochs=2000, lr=1e-3, hidden=64,
                   terminal_weight=10.0, half_side=0.3, device=device)

    # Test sur 20 points aleatoires
    rng = np.random.default_rng(42)
    test_pts = [(rng.uniform(-0.3, 0.3),
                 rng.uniform(-0.3, 0.3),
                 rng.uniform(-0.3, 0.3)) for _ in range(20)]
    evaluate(policy, test_pts, device=device)

    # Sauvegarde
    out_dir = os.path.dirname(os.path.abspath(__file__))
    path_full = os.path.join(out_dir, "trained_policy_cf2x.pt")
    path_dict = os.path.join(out_dir, "trained_weights_cf2x.pt")

    torch.save(policy.cpu(), path_full)
    torch.save({
        "state_dict": policy.state_dict(),
        "hidden": 64,
        "x_scale": X_SCALE,
        "u_max": U_MAX,
        "ctrl_freq": CTRL_FREQ,
        "Ts": DT_CTRL,
        "KF": KF, "KM": KM, "L": L, "M": M, "G": G,
        "MAX_RPM": MAX_RPM,
    }, path_dict)

    print(f"\n[Saved] {path_full}")
    print(f"[Saved] {path_dict}")
    print("\nDone! Run  validate_nn_pybullet.py  to test in pybullet-drones.")