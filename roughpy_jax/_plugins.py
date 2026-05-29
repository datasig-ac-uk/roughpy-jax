from __future__ import annotations

from importlib import import_module
from roughpy_jax.ops import registration_lock as _registration_lock

CUDA_PLUGIN_MODULES = (
    "roughpy_jax_cuda13_plugin",
    "roughpy_jax_cuda12_plugin",
)

_plugins_loaded = False
_cuda_module = None

def _load_cuda_plugin(module_name):
    module = import_module(module_name)
    register = getattr(module, "register", None)
    if register is not None:
        register()
    return module


def load_plugins():
    global _plugins_loaded
    global _cuda_module

    with _registration_lock:
        if _plugins_loaded:
            return

        for module_name in CUDA_PLUGIN_MODULES:
            try:
                _cuda_module = _load_cuda_plugin(module_name)
            except ImportError:
                _cuda_module = None
            else:
                break

        _plugins_loaded = True
