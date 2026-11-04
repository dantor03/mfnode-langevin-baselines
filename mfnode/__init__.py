"""
mfnode: particle implementation of the entropic mean-field Neural ODE.

Application of arXiv:2507.08486 (Daudin & Delarue, 2025) to make_moons,
make_circles and California Housing.

The names below are imported on first use, so that ``mfnode.plotting`` (which
only needs numpy, scipy and matplotlib) can be used without torch.
"""

import importlib

_EXPORTS = {
    'SEED': 'config', 'DEVICE': 'config', 'OUTPUT_DIR': 'paths', 'DATA_DIR': 'paths',
    'get_moons': 'data', 'get_circles': 'data', 'get_california_regression': 'data',
    'MeanFieldVelocity': 'model', 'MeanFieldResNet': 'model',
    'train': 'train', 'mu_pl_estimate': 'train',
    'decision_grid': 'plots', 'velocity_grid': 'plots',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f'.{_EXPORTS[name]}', __name__), name)
    raise AttributeError(f"module 'mfnode' has no attribute {name!r}")
