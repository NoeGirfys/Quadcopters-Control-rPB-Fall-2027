"""maml_lib — modular MAML training for the Crazyflie 2.x.

Public surface:

    from maml_lib import (
        config,
        MassParams, compute_mass_params, batch_mass_params,
        nonlinear_step, linearized_step, DYNAMICS,
        PolicyMLP,
        rollout, compute_relative_targets, trajectory_cost,
        FixedUniformMassSet,
        meta_train, maml_adapt, baseline_train,
        sample_hover_x0, plot_fixed_task_map,
        make_x0_batch, generate_cube_points, GracefulKiller,
    )
"""
from . import config

from .mass_params import MassParams, compute_mass_params, batch_mass_params
from .dynamics    import nonlinear_step, linearized_step, DYNAMICS
from .policy      import PolicyMLP
from .rollout     import rollout, compute_relative_targets
from .cost        import trajectory_cost
from .tasks       import FixedUniformMassSet
from .maml        import meta_train, maml_adapt, baseline_train
from .utils       import (
    sample_hover_x0, plot_fixed_task_map,
    make_x0_batch, generate_cube_points, GracefulKiller,
)

__all__ = [
    "config",
    "MassParams", "compute_mass_params", "batch_mass_params",
    "nonlinear_step", "linearized_step", "DYNAMICS",
    "PolicyMLP",
    "rollout", "compute_relative_targets", "trajectory_cost",
    "FixedUniformMassSet",
    "meta_train", "maml_adapt", "baseline_train",
    "sample_hover_x0", "plot_fixed_task_map",
    "make_x0_batch", "generate_cube_points", "GracefulKiller",
]
