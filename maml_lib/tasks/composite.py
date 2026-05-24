"""Composite task set: tasks defined as mixtures of atomic distributions.

A *task* is a row in a YAML/JSON configuration file. Each task selects, on
three independent axes:

  * **mass-position Gaussians** (``mass_gaussians``): isotropic 3-D Gaussians;
    indices 1..4 map to motors M1..M4, index 5 to the drone centre (0,0,0).
    Shared std ``mass_pos_sigma``.
  * **drone-start octants** (``octants``) of a cube ``[-half_side, half_side]^3``
    (indices 1..8, see ``OCTANT_SIGNS``),
  * **mass-magnitude Gaussians** (``mass_values``): 1-D Gaussians selected from
    a top-level ``mass_value_gaussians`` pool (each ``{mean, std}`` in kg),
    clamped to >= 0. This is the per-task axis that makes MAML meaningful: the
    collective thrust the NN must output (to hold a given weight) has no
    integrator in the firmware cascade — the NN replaced it — so different
    per-task weights cannot all be held by one fixed policy without steady
    state error, whereas the inner attitude/rate PIDs already integrate away a
    *static* CoM-offset torque (hence offset *direction* is a poor task axis).

All distribution parameters (``mass_pos_sigma``, ``half_side``, the magnitude
pool) live in the YAML — it is the single source of truth for the task
distribution, so MAML and the standalone baseline only need to point at the
same file. Sampling counts (``M_train``/``M_eval``) and the seed stay on the
CLI.

At startup we draw ``M_train + M_eval`` points per task — each point is
``(mass_x, mass_y, mass_z, m_extra, drone_x, drone_y, drone_z)`` — by picking
uniformly among the selected Gaussians/octants on a per-point basis. The
first ``M_train`` points are the support (inner-loop) set, the last
``M_eval`` the query (outer-loop) set. These points are *fixed* for the
lifetime of the run.

The class also serves as the *target* task set (held out from training) when
loaded from a separate config file — same schema, typically one task.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .. import config as C
from ..mass_params import MassParams, compute_mass_params_batched


# Octant index -> (sign_x, sign_y, sign_z), 1-indexed externally:
#   1: (+,+,+)   2: (-,+,+)   3: (+,-,+)   4: (-,-,+)
#   5: (+,+,-)   6: (-,+,-)   7: (+,-,-)   8: (-,-,-)
OCTANT_SIGNS = np.array([
    [+1, +1, +1],
    [-1, +1, +1],
    [+1, -1, +1],
    [-1, -1, +1],
    [+1, +1, -1],
    [-1, +1, -1],
    [+1, -1, -1],
    [-1, -1, -1],
], dtype=np.float32)


def _mass_centers() -> np.ndarray:
    """Centres of the mass-position Gaussians: 4 motors + drone centre.

    Index (0-based) 0..3 -> motors M1..M4 (``C.MOTOR_POS``), 4 -> (0,0,0).
    Externally these are 1-based (1..5, with 5 = centre).
    """
    return np.vstack([np.asarray(C.MOTOR_POS, dtype=np.float32),
                      np.zeros((1, 3), dtype=np.float32)])


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_task_file(path: str) -> dict:
    """Load a task config file and return the full top-level mapping.

    Accepts ``.yaml``/``.yml`` (requires ``pyyaml``) and ``.json`` (built-in).
    The returned dict holds the ``tasks`` list plus the distribution
    parameters (``mass_pos_sigma``, ``half_side``, ``mass_value_gaussians``).
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Reading a YAML task config requires PyYAML. "
                    "Install it with `pip install pyyaml`, or convert "
                    "the file to .json.") from e
            data = yaml.safe_load(f)
        elif ext == ".json":
            data = json.load(f)
        else:
            raise ValueError(
                f"Unsupported task-config format: {ext!r}. "
                "Use .yaml, .yml or .json.")

    if not isinstance(data, dict) or "tasks" not in data:
        raise ValueError(
            f"{path}: top-level must be a mapping with a 'tasks' key.")
    tasks = data["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path}: 'tasks' must be a non-empty list.")
    return data


def _validate_mass_pool(data: dict) -> np.ndarray:
    """Validate the top-level ``mass_value_gaussians`` pool.

    Returns an ``(P, 2)`` float32 array of ``(mean, std)`` per 1-D Gaussian.
    """
    pool = data.get("mass_value_gaussians")
    if not isinstance(pool, list) or not pool:
        raise ValueError(
            "config: top-level 'mass_value_gaussians' must be a non-empty "
            "list of {mean, std} mappings (extra-mass magnitude pool, kg).")
    out = np.zeros((len(pool), 2), dtype=np.float32)
    for j, g in enumerate(pool):
        if not isinstance(g, dict) or "mean" not in g:
            raise ValueError(
                f"mass_value_gaussians[{j}]: must be a mapping with at least "
                f"a 'mean' (kg); 'std' optional (got {g!r}).")
        mean = float(g["mean"])
        std = float(g.get("std", 0.0))
        if mean < 0 or std < 0:
            raise ValueError(
                f"mass_value_gaussians[{j}]: 'mean' and 'std' must be >= 0 "
                f"(got mean={mean}, std={std}).")
        out[j] = (mean, std)
    return out


def _validate_task(entry: dict, idx: int, n_mass_pool: int) -> tuple:
    """Validate one task entry; return (name, gauss_0idx, oct_0idx, mval_0idx)."""
    if not isinstance(entry, dict):
        raise ValueError(f"task #{idx+1}: must be a mapping, got {type(entry).__name__}")
    name = entry.get("name", f"task_{idx+1}")

    gauss = entry.get("mass_gaussians")
    if (not isinstance(gauss, list) or not gauss
            or not all(isinstance(x, int) and 1 <= x <= 5 for x in gauss)):
        raise ValueError(
            f"task '{name}': 'mass_gaussians' must be a non-empty list of "
            f"ints in 1..5 (1..4 = motors M1..M4, 5 = centre; got {gauss!r}).")

    oct_ = entry.get("octants")
    if (not isinstance(oct_, list) or not oct_
            or not all(isinstance(x, int) and 1 <= x <= 8 for x in oct_)):
        raise ValueError(
            f"task '{name}': 'octants' must be a non-empty list of ints in "
            f"1..8 (got {oct_!r}).")

    mvals = entry.get("mass_values")
    if (not isinstance(mvals, list) or not mvals
            or not all(isinstance(x, int) and 1 <= x <= n_mass_pool for x in mvals)):
        raise ValueError(
            f"task '{name}': 'mass_values' must be a non-empty list of ints "
            f"in 1..{n_mass_pool} (indices into 'mass_value_gaussians'); "
            f"got {mvals!r}.")

    # Deduplicate while preserving order.
    gauss = list(dict.fromkeys(gauss))
    oct_ = list(dict.fromkeys(oct_))
    mvals = list(dict.fromkeys(mvals))

    return (name, [g - 1 for g in gauss], [o - 1 for o in oct_],
            [m - 1 for m in mvals])


# ---------------------------------------------------------------------------
# Per-task point sampler
# ---------------------------------------------------------------------------

def _sample_points(rng: np.random.Generator, n: int,
                   gauss_indices: List[int],
                   octant_indices: List[int],
                   mval_indices: List[int],
                   mass_centers: np.ndarray,
                   mass_pos_sigma: float,
                   mass_pool: np.ndarray,
                   half_side: float):
    """Sample ``n`` points for one task.

    Returns three arrays: ``mass_positions (n, 3)``, ``m_extras (n,)``,
    ``drone_positions (n, 3)``, all float32.
    """
    # Mass attachment position: pick uniformly among the selected position
    # Gaussians (motors / centre), then add isotropic 3-D Gaussian noise.
    gauss_pick = rng.integers(0, len(gauss_indices), n)
    chosen_centers = np.asarray(gauss_indices, dtype=np.int64)[gauss_pick]
    centers = mass_centers[chosen_centers]                            # (n, 3)
    noise = rng.normal(0.0, mass_pos_sigma, (n, 3)).astype(np.float32)
    mass_positions = (centers + noise).astype(np.float32)

    # Extra mass magnitude: pick uniformly among the selected 1-D Gaussians,
    # draw from it, and clamp to >= 0 (no negative payload).
    mv_pick = rng.integers(0, len(mval_indices), n)
    chosen_mv = np.asarray(mval_indices, dtype=np.int64)[mv_pick]
    means = mass_pool[chosen_mv, 0]
    stds = mass_pool[chosen_mv, 1]
    m_extras = rng.normal(means, stds).astype(np.float32)
    m_extras = np.clip(m_extras, 0.0, None).astype(np.float32)

    # Drone start: pick uniformly among selected octants, then a uniform
    # point inside that octant's slice of the cube.
    oct_pick = rng.integers(0, len(octant_indices), n)
    chosen_octs = np.asarray(octant_indices, dtype=np.int64)[oct_pick]
    signs = OCTANT_SIGNS[chosen_octs]                                  # (n, 3)
    mag = rng.uniform(0.0, half_side, (n, 3)).astype(np.float32)
    drone_positions = (signs * mag).astype(np.float32)

    return mass_positions, m_extras, drone_positions


def _positions_to_x0(drone_positions: np.ndarray) -> torch.Tensor:
    """Wrap (M, 3) drone positions into the canonical 12-D hover state."""
    M = drone_positions.shape[0]
    X0 = torch.zeros(M, 12, dtype=torch.float32)
    X0[:, 0] = torch.from_numpy(drone_positions[:, 0])
    X0[:, 2] = torch.from_numpy(drone_positions[:, 1])
    X0[:, 4] = torch.from_numpy(drone_positions[:, 2])
    return X0


# ---------------------------------------------------------------------------
# CompositeTaskSet
# ---------------------------------------------------------------------------

@dataclass
class CompositeTaskSet:
    """Fixed composite task set sampled once at startup.

    Attributes
    ----------
    names                 : list of N task names (from the YAML, or ``task_<i>``)
    spec                  : (N, 12) bool — 4 columns for Gaussians + 8 for octants
    mass_positions_train  : (N, M_train, 3) float32 [m]
    m_extras_train        : (N, M_train)    float32 [kg]
    mass_positions_eval   : (N, M_eval,  3) float32 [m]
    m_extras_eval         : (N, M_eval)     float32 [kg]
    masses_train          : list of N MassParams (each batched over M_train)
    masses_eval           : list of N MassParams (each batched over M_eval)
    x0_train              : (N, M_train, 12) float32
    x0_eval               : (N, M_eval,  12) float32
    """
    names: List[str]
    spec: np.ndarray

    mass_positions_train: np.ndarray
    m_extras_train:       np.ndarray
    mass_positions_eval:  np.ndarray
    m_extras_eval:        np.ndarray

    masses_train: List[MassParams] = field(repr=False)
    masses_eval:  List[MassParams] = field(repr=False)

    x0_train: torch.Tensor = field(repr=False)
    x0_eval:  torch.Tensor = field(repr=False)

    M_train: int
    M_eval:  int

    # Atom parameters (kept for reproducibility and re-plotting).
    mass_pos_sigma: float
    half_side: float
    mass_min: float
    mass_max: float

    # --------------------------------------------------------------------
    # Convenience accessors
    # --------------------------------------------------------------------

    @property
    def N(self) -> int:
        return len(self.names)

    def __len__(self) -> int:
        return self.N

    @property
    def m_extra(self) -> float:
        """Mean extra mass across all sampled training points.

        Used as the hover-thrust bias of the policy at construction time.
        """
        return float(
            np.concatenate([self.m_extras_train.reshape(-1),
                            self.m_extras_eval.reshape(-1)]).mean())

    # --------------------------------------------------------------------
    # Construction
    # --------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str, *,
                    M_train: int, M_eval: int,
                    rng: np.random.Generator) -> "CompositeTaskSet":
        """Sample a fresh task set from a YAML/JSON config.

        The distribution parameters (``mass_pos_sigma``, ``half_side`` and the
        ``mass_value_gaussians`` magnitude pool) are read from the config file
        itself — it is the single source of truth for the task distribution.
        Only the sampling counts and ``rng`` come from the caller.
        """
        if M_train < 1 or M_eval < 1:
            raise ValueError(
                f"M_train and M_eval must be >= 1 (got {M_train}, {M_eval}).")

        data = _load_task_file(config_path)
        entries = data["tasks"]
        mass_pos_sigma = float(data.get("mass_pos_sigma", 0.01))
        half_side = float(data.get("half_side", 0.3))
        if mass_pos_sigma < 0 or half_side < 0:
            raise ValueError(
                f"{config_path}: 'mass_pos_sigma' and 'half_side' must be "
                f">= 0 (got {mass_pos_sigma}, {half_side}).")
        mass_pool = _validate_mass_pool(data)                        # (P, 2)
        P = mass_pool.shape[0]

        N = len(entries)
        M_total = M_train + M_eval
        mass_centers = _mass_centers()                               # (5, 3)

        names: List[str] = []
        spec = np.zeros((N, 5 + 8 + P), dtype=bool)   # pos(5) | octants(8) | mag(P)
        mp_all = np.zeros((N, M_total, 3), dtype=np.float32)
        me_all = np.zeros((N, M_total),    dtype=np.float32)
        dp_all = np.zeros((N, M_total, 3), dtype=np.float32)

        for i, entry in enumerate(entries):
            name, gauss_idx, oct_idx, mval_idx = _validate_task(entry, i, P)
            names.append(name)
            for g in gauss_idx:
                spec[i, g] = True
            for o in oct_idx:
                spec[i, 5 + o] = True
            for m in mval_idx:
                spec[i, 13 + m] = True
            mp, me, dp = _sample_points(
                rng, M_total, gauss_idx, oct_idx, mval_idx,
                mass_centers, mass_pos_sigma, mass_pool, half_side)
            mp_all[i] = mp
            me_all[i] = me
            dp_all[i] = dp

        return cls._assemble(
            names=names, spec=spec,
            mass_positions=mp_all, m_extras=me_all, drone_positions=dp_all,
            M_train=M_train, M_eval=M_eval,
            mass_pos_sigma=mass_pos_sigma, half_side=half_side)

    @classmethod
    def _assemble(cls, *, names, spec,
                  mass_positions, m_extras, drone_positions,
                  M_train, M_eval,
                  mass_pos_sigma, half_side
                  ) -> "CompositeTaskSet":
        """Internal: build masses/x0 from raw arrays and split adapt/eval.

        ``mass_min``/``mass_max`` are *derived* as the min/max sampled extra
        mass across all points (used only for plotting / provenance now that
        the magnitude is a per-task Gaussian pool, not a global uniform).
        """
        N = len(names)
        masses_train, masses_eval = [], []
        x0_train = torch.zeros(N, M_train, 12, dtype=torch.float32)
        x0_eval  = torch.zeros(N, M_eval,  12, dtype=torch.float32)
        for i in range(N):
            mp_t, me_t = mass_positions[i, :M_train], m_extras[i, :M_train]
            mp_e, me_e = mass_positions[i, M_train:], m_extras[i, M_train:]
            masses_train.append(compute_mass_params_batched(me_t, mp_t))
            masses_eval .append(compute_mass_params_batched(me_e, mp_e))
            x0_train[i] = _positions_to_x0(drone_positions[i, :M_train])
            x0_eval[i]  = _positions_to_x0(drone_positions[i, M_train:])

        mass_min = float(m_extras.min())
        mass_max = float(m_extras.max())

        return cls(
            names=names, spec=spec,
            mass_positions_train=mass_positions[:, :M_train].copy(),
            m_extras_train      =m_extras      [:, :M_train].copy(),
            mass_positions_eval =mass_positions[:, M_train:].copy(),
            m_extras_eval       =m_extras      [:, M_train:].copy(),
            masses_train=masses_train, masses_eval=masses_eval,
            x0_train=x0_train, x0_eval=x0_eval,
            M_train=M_train, M_eval=M_eval,
            mass_pos_sigma=float(mass_pos_sigma), half_side=float(half_side),
            mass_min=mass_min, mass_max=mass_max,
        )

    # --------------------------------------------------------------------
    # Checkpoint serialisation
    # --------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Pure-data dict snapshot — round-trips via :meth:`from_dict`."""
        return {
            "names": list(self.names),
            "spec":  self.spec.copy(),
            "mass_positions_train": self.mass_positions_train.copy(),
            "m_extras_train":       self.m_extras_train.copy(),
            "mass_positions_eval":  self.mass_positions_eval.copy(),
            "m_extras_eval":        self.m_extras_eval.copy(),
            "x0_train": self.x0_train.detach().cpu(),
            "x0_eval":  self.x0_eval.detach().cpu(),
            "M_train":  int(self.M_train),
            "M_eval":   int(self.M_eval),
            "mass_pos_sigma": float(self.mass_pos_sigma),
            "half_side":      float(self.half_side),
            "mass_min":       float(self.mass_min),
            "mass_max":       float(self.mass_max),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CompositeTaskSet":
        """Reconstruct from a dict produced by :meth:`to_dict`. No re-sampling."""
        names = list(d["names"])
        spec  = np.asarray(d["spec"], dtype=bool)
        mp_t  = np.asarray(d["mass_positions_train"], dtype=np.float32)
        me_t  = np.asarray(d["m_extras_train"],       dtype=np.float32)
        mp_e  = np.asarray(d["mass_positions_eval"],  dtype=np.float32)
        me_e  = np.asarray(d["m_extras_eval"],        dtype=np.float32)
        x0_t  = torch.as_tensor(d["x0_train"], dtype=torch.float32)
        x0_e  = torch.as_tensor(d["x0_eval"],  dtype=torch.float32)
        M_train = int(d["M_train"]); M_eval = int(d["M_eval"])

        N = len(names)
        masses_train = [compute_mass_params_batched(me_t[i], mp_t[i])
                        for i in range(N)]
        masses_eval  = [compute_mass_params_batched(me_e[i], mp_e[i])
                        for i in range(N)]

        return cls(
            names=names, spec=spec,
            mass_positions_train=mp_t, m_extras_train=me_t,
            mass_positions_eval =mp_e, m_extras_eval =me_e,
            masses_train=masses_train, masses_eval=masses_eval,
            x0_train=x0_t, x0_eval=x0_e,
            M_train=M_train, M_eval=M_eval,
            mass_pos_sigma=float(d.get("mass_pos_sigma", 0.01)),
            half_side     =float(d.get("half_side",      0.3)),
            mass_min      =float(d.get("mass_min",       0.002)),
            mass_max      =float(d.get("mass_max",       0.014)),
        )
