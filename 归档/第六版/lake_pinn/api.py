"""Bootstrap and public API for the modular run9 LakePINN code.

The first split is intentionally mechanical: functions live in topic modules, and
this bootstrap shares their symbols across module globals so behavior matches the
former single-file script while we continue cleaning dependencies incrementally.
"""

from importlib import import_module

_MODULE_NAMES = [
    'model',
    'cli_config',
    'lake_metadata',
    'data_io',
    'forcing',
    'physics',
    'losses',
    'validation',
    'ppo',
    'checkpoint',
    'online_control',
    'train',
    'predict',
    'kalman',
    'export',
    'plotting',
    'scorecard_integration',
    'pipeline',
]

_modules = [import_module(f'{__package__}.{name}') for name in _MODULE_NAMES]
_registry = {}
for _module in _modules:
    for _name, _value in vars(_module).items():
        if _name.startswith('_'):
            continue
        _registry[_name] = _value

for _module in _modules:
    _module.__dict__.update(_registry)

globals().update(_registry)

__all__ = sorted(_registry)
