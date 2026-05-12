"""Abstract Task interface: a Task knows how to sample MassParams.

Adding a new task type (e.g. wind disturbance, motor mismatch, …) is a
matter of subclassing ``Task`` and producing the appropriate
``MassParams`` (or extending ``MassParams`` with new fields if a new task
needs to perturb something the dynamics doesn't currently expose).
"""
from abc import ABC, abstractmethod
from typing import List

from ..mass_params import MassParams


class Task(ABC):
    """Generator of MAML tasks. ``sample`` is stochastic, ``deterministic_set``
    returns a fixed list useful for evaluation."""

    @abstractmethod
    def sample(self, rng) -> MassParams:
        """Return one task. ``rng`` is a ``numpy.random.Generator``."""

    @abstractmethod
    def deterministic_set(self, n: int) -> List[MassParams]:
        """Return a fixed list of ``n`` tasks (seeded internally)."""

    @abstractmethod
    def describe(self, mass: MassParams) -> str:
        """Human-readable label for one of this task's MassParams."""
