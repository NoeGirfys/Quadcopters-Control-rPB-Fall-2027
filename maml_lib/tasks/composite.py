"""Composite task set: tasks defined as mixtures of atomic distributions.

A *task* is a row in a YAML/JSON configuration file. Each task selects:

  * one or more **mass-position Gaussians** (one isotropic 3-D Gaussian per
    motor of the CF2X; indices 1..4 in the config map to motors M1..M4),
  * one or more **drone-start octants** of a cube ``[-half_side, half_side]^3``
    (indices 1..8, see ``OCTANT_SIGNS``),
  * the shared 1-D uniform distribution on the extra mass magnitude
    ``[mass_min, mass_max]``.

At startup we draw ``M_train + M_eval`` 7-D points per task — each point is
``(mass_x, mass_y, mass_z, m_extra, drone_x, drone_y, drone_z)`` — by
picking uniformly among the selected Gaussians and octants on a per-point
basis. The first ``M_train`` points are the support (inner-loop) set, the
last ``M_eval`` the query (outer-loop) set. These points are *fixed* for
the lifetime of the run: every epoch sees the same support and query.

The class also serves as the *target* task set (held out from training)
when loaded from a separate config file — same shape, typically one task.
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


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_task_file(path: str) -> List[dict]:
    """Load a task config file. Accepts ``.yaml``/``.yml`` (requires
    ``pyyaml``) and ``.json`` (built-in). Returns the ``tasks:`` list."""
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
    return tasks


def _validate_task(entry: dict, idx: int) -> tuple:
    """Validate one task entry; return (name, gauss_0idx, oct_0idx)."""
    if not isinstance(entry, dict):
        raise ValueError(f"task #{idx+1}: must be a mapping, got {type(entry).__name__}")
    name = entry.get("name", f"task_{idx+1}")

    gauss = entry.get("mass_gaussians")
    if (not isinstance(gauss, list) or not gauss
            or not all(isinstance(x, int) and 1 <= x <= 4 for x in gauss)):
        raise ValueError(
            f"task '{name}': 'mass_gaussians' must be a non-empty list of "
            "ints in 1..4 (got {gauss!r}).")

    oct_ = entry.get("octants")
    if (not isinstance(oct_, list) or not oct_
            or not all(isinstance(x, int) and 1 <= x <= 8 for x in oct_)):
        raise ValueError(
            f"task '{name}': 'octants' must be a non-empty list of ints in "
            f"1..8 (got {oct_!r}).")

    # Deduplicate while preserving order.
    gauss = list(dict.fromkeys(gauss))
    oct_ = list(dict.fromkeys(oct_))

    return name, [g - 1 for g in gauss], [o - 1 for o in oct_]


# ---------------------------------------------------------------------------
# Per-task point sampler
# ---------------------------------------------------------------------------

def _sample_points(rng: np.random.Generator, n: int,
                   gauss_indices: List[int],
                   octant_indices: List[int],
                   motor_pos: np.ndarray,
                   mass_pos_sigma: float,
                   mass_min: float, mass_max: float,
                   half_side: float):
    """Sample ``n`` points for one task.

    Returns three arrays: ``mass_positions (n, 3)``, ``m_extras (n,)``,
    ``drone_positions (n, 3)``, all float32.
    """
    # Mass attachment position: pick uniformly among selected motors, then
    # add isotropic 3-D Gaussian noise around that motor's position.
    gauss_pick = rng.integers(0, len(gauss_indices), n)
    chosen_motors = np.asarray(gauss_indices, dtype=np.int64)[gauss_pick]
    centers = motor_pos[chosen_motors]                                # (n, 3)
    noise = rng.normal(0.0, mass_pos_sigma, (n, 3)).astype(np.float32)
    mass_positions = (centers + noise).astype(np.float32)

    # Extra mass magnitude: shared 1-D uniform.
    m_extras = rng.uniform(mass_min, mass_max, n).astype(np.float32)

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
                    mass_pos_sigma: float,
                    half_side: float,
                    mass_min: float, mass_max: float,
                    rng: np.random.Generator) -> "CompositeTaskSet":
        """Sample a fresh task set from a YAML/JSON config."""
        if M_train < 1 or M_eval < 1:
            raise ValueError(
                f"M_train and M_eval must be >= 1 (got {M_train}, {M_eval}).")
        if mass_min > mass_max:
            raise ValueError(
                f"mass_min ({mass_min}) must be <= mass_max ({mass_max}).")

        entries = _load_task_file(config_path)
        N = len(entries)
        M_total = M_train + M_eval
        motor_pos = np.asarray(C.MOTOR_POS, dtype=np.float32)        # (4, 3)

        names: List[str] = []
        spec = np.zeros((N, 12), dtype=bool)
        mp_all = np.zeros((N, M_total, 3), dtype=np.float32)
        me_all = np.zeros((N, M_total),    dtype=np.float32)
        dp_all = np.zeros((N, M_total, 3), dtype=np.float32)

        for i, entry in enumerate(entries):
            name, gauss_idx, oct_idx = _validate_task(entry, i)
            names.append(name)
            for g in gauss_idx:
                spec[i, g] = True
            for o in oct_idx:
                spec[i, 4 + o] = True
            mp, me, dp = _sample_points(
                rng, M_total, gauss_idx, oct_idx,
                motor_pos, mass_pos_sigma, mass_min, mass_max, half_side)
            mp_all[i] = mp
            me_all[i] = me
            dp_all[i] = dp

        return cls._assemble(
            names=names, spec=spec,
            mass_positions=mp_all, m_extras=me_all, drone_positions=dp_all,
            M_train=M_train, M_eval=M_eval,
            mass_pos_sigma=mass_pos_sigma, half_side=half_side,
            mass_min=mass_min, mass_max=mass_max)

    @classmethod
    def _assemble(cls, *, names, spec,
                  mass_positions, m_extras, drone_positions,
                  M_train, M_eval,
                  mass_pos_sigma, half_side, mass_min, mass_max
                  ) -> "CompositeTaskSet":
        """Internal: build masses/x0 from raw arrays and split adapt/eval."""
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
            mass_min=float(mass_min), mass_max=float(mass_max),
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
