from __future__ import annotations

import platform

import pytest

import roughpy_jax._plugins as _plugins


LINUX_ONLY = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="plugin loading is only supported on Linux",
)


@LINUX_ONLY
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


@LINUX_ONLY
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
