"""maml_lib — modular MAML training for the Crazyflie 2.x.

Public surface:

    from maml_lib import (
        config,                 # constants
        MassParams, compute_mass_params,
        nonlinear_step, linearized_step, DYNAMICS,
        PolicyMLP,
        rollout, compute_relative_targets, trajectory_cost,
        Task, GaussianMotorTask,
        meta_train, maml_adapt,
        sample_hover_x0, plot_training_map,
        make_x0_batch, generate_cube_points, GracefulKiller,
    )
"""
from . import config

from .mass_params import MassParams, compute_mass_params
from .dynamics    import nonlinear_step, linearized_step, DYNAMICS
from .policy      import PolicyMLP
from .rollout     import rollout, compute_relative_targets
from .cost        import trajectory_cost
from .tasks       import Task, GaussianMotorTask
from .maml        import meta_train, maml_adapt
from .utils       import (
    sample_hover_x0, plot_training_map,
    make_x0_batch, generate_cube_points, GracefulKiller,
)

__all__ = [
    "config",
    "MassParams", "compute_mass_params",
    "nonlinear_step", "linearized_step", "DYNAMICS",
    "PolicyMLP",
    "rollout", "compute_relative_targets", "trajectory_cost",
    "Task", "GaussianMotorTask",
    "meta_train", "maml_adapt",
    "sample_hover_x0", "plot_training_map",
    "make_x0_batch", "generate_cube_points", "GracefulKiller",
]
