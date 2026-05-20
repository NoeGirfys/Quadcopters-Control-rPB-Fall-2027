"""Removed — replaced by :class:`maml_lib.tasks.composite.CompositeTaskSet`.

The old "task = one uniformly-sampled (offset, mass) configuration"
abstraction no longer fits the project. Use ``CompositeTaskSet``: each
task is a mixture of selected motor-centered Gaussians and start-position
octants, sampled from a YAML/JSON config.
"""
