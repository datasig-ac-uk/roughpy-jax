from __future__ import annotations

import importlib.util
from pathlib import Path

_PLUGINS_PATH = Path(__file__).resolve().parents[1] / "_plugins.py"
_SPEC = importlib.util.spec_from_file_location("roughpy_jax._plugins", _PLUGINS_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_plugins = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_plugins)


def test_load_plugins_is_idempotent(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(_plugins, "_plugins_loaded", False)
    monkeypatch.setattr(
        _plugins,
        "_load_cuda_plugin",
        lambda name: calls.append(name) or True,
    )

    _plugins.load_plugins()
    _plugins.load_plugins()

    assert calls == ["roughpy_jax_cuda13_plugin"]


def test_load_plugins_falls_back_to_older_cuda(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(_plugins, "_plugins_loaded", False)

    def fake_load(name: str) -> bool:
        calls.append(name)
        if name == "roughpy_jax_cuda13_plugin":
            raise ImportError("not installed")
        return True

    monkeypatch.setattr(_plugins, "_load_cuda_plugin", fake_load)

    _plugins.load_plugins()

    assert calls == [
        "roughpy_jax_cuda13_plugin",
        "roughpy_jax_cuda12_plugin",
    ]
