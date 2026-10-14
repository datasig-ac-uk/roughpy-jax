from __future__ import annotations

import platform
from types import SimpleNamespace

import pytest

import roughpy_jax._plugins as _plugins

LINUX_ONLY = pytest.mark.skipif(
    platform.system() != "Linux",
    reason="plugin loading is only supported on Linux",
)


def test_plugin_version_matches_host(monkeypatch):
    monkeypatch.setattr(_plugins, "version", lambda distribution: "1.1.1")

    _plugins._check_plugin_version(
        SimpleNamespace(__name__="roughpy_jax_cuda12_plugin", PLUGIN_VERSION="1.1.1")
    )


def test_plugin_version_mismatch_is_rejected(monkeypatch):
    versions = {"roughpy-jax-cuda12-plugin": "1.1.0", "roughpy-jax": "1.1.1"}
    monkeypatch.setattr(_plugins, "version", versions.__getitem__)

    with pytest.raises(ImportError, match=r"has version 1.1.0.*has version 1.1.1"):
        _plugins._check_plugin_version(
            SimpleNamespace(
                __name__="roughpy_jax_cuda12_plugin", PLUGIN_VERSION="1.1.0"
            )
        )


def test_installed_plugin_version_uses_wheel_metadata(monkeypatch):
    versions = {
        "roughpy-jax-cuda12-plugin": "1.2.1.dev4",
        "roughpy-jax": "1.2.1.dev4",
    }
    monkeypatch.setattr(_plugins, "version", versions.__getitem__)

    _plugins._check_plugin_version(
        SimpleNamespace(__name__="roughpy_jax_cuda12_plugin", PLUGIN_VERSION="1.2.1")
    )


def test_legacy_plugin_uses_distribution_version(monkeypatch):
    versions = {
        "roughpy-jax-cuda12-plugin": "1.1.0",
        "roughpy-jax": "1.1.1",
    }
    monkeypatch.setattr(_plugins, "version", versions.__getitem__)

    with pytest.raises(ImportError, match=r"has version 1.1.0.*has version 1.1.1"):
        _plugins._check_plugin_version(
            SimpleNamespace(__name__="roughpy_jax_cuda12_plugin")
        )


def test_source_plugin_without_distribution_metadata_is_accepted(monkeypatch):
    def missing_distribution(distribution):
        raise _plugins.PackageNotFoundError(distribution)

    monkeypatch.setattr(_plugins, "version", missing_distribution)

    _plugins._check_plugin_version(
        SimpleNamespace(__name__="roughpy_jax_cuda12_plugin")
    )


def test_source_plugin_uses_embedded_version(monkeypatch):
    def installed_version(distribution):
        if distribution == "roughpy-jax":
            return "1.2.1.dev4"
        raise _plugins.PackageNotFoundError(distribution)

    monkeypatch.setattr(_plugins, "version", installed_version)

    with pytest.raises(ImportError, match=r"has version 1.2.1.*1.2.1.dev4"):
        _plugins._check_plugin_version(
            SimpleNamespace(
                __name__="roughpy_jax_cuda12_plugin", PLUGIN_VERSION="1.2.1"
            )
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


@LINUX_ONLY
def test_load_plugins_reports_broken_installed_plugin(monkeypatch):
    monkeypatch.setattr(_plugins, "_plugins_loaded", False)

    def fake_load(name: str):
        if name == "roughpy_jax_cuda13_plugin":
            raise ImportError("libcudart.so.13: cannot open shared object file")
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(_plugins, "_load_cuda_plugin", fake_load)

    with pytest.raises(ImportError, match="CUDA plugin failed to load") as exc_info:
        _plugins.load_plugins()

    assert "libcudart.so.13" in str(exc_info.value.__cause__)
    assert not _plugins._plugins_loaded


@LINUX_ONLY
def test_load_plugins_allows_no_installed_cuda_plugin(monkeypatch):
    monkeypatch.setattr(_plugins, "_plugins_loaded", False)

    def fake_load(name: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    monkeypatch.setattr(_plugins, "_load_cuda_plugin", fake_load)
    _plugins.load_plugins()

    assert _plugins._plugins_loaded
