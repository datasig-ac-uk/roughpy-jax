from __future__ import annotations

import platform
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from roughpy_jax.ops import registration_lock as _registration_lock

CUDA_PLUGIN_MODULES = (
    "roughpy_jax_cuda13_plugin",
    "roughpy_jax_cuda12_plugin",
)

_plugins_loaded = False
_cuda_module = None


def _check_plugin_version(module) -> None:
    plugin_version = getattr(module, "PLUGIN_VERSION", None)
    if plugin_version is None:
        distribution_name = module.__name__.replace("_", "-")
        try:
            plugin_version = version(distribution_name)
        except PackageNotFoundError:
            # Permit plugins loaded directly from a development source tree.
            return

    try:
        host_version = version("roughpy-jax")
    except PackageNotFoundError:
        # The source tree can be imported without installed distribution
        # metadata. Packaging pins still enforce compatibility for installs.
        return

    if plugin_version != host_version:
        raise ImportError(
            f"CUDA plugin {module.__name__!r} has version {plugin_version}, "
            f"but roughpy-jax has version {host_version}"
        )


def _load_cuda_plugin(module_name):
    module = import_module(module_name)
    _check_plugin_version(module)
    register = getattr(module, "register", None)
    if register is not None:
        register()
    return module


def load_plugins():
    if platform.system() != "Linux":
        return

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
