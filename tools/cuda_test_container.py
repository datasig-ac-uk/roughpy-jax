"""Container entry point for tools/run_cuda_tests.py; source is mounted read-only."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(*command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def copy_sources(source, destination):
    # Include uncommitted source changes, but never reuse local extensions,
    # environments, CMake caches, or generated build trees.
    ignore = shutil.ignore_patterns(
        ".git",
        ".venv",
        "__pycache__",
        ".pytest_cache",
        "build",
        "build-*",
        "cmake-build-*",
        "CMakeFiles",
        "CMakeCache.txt",
        "*.so",
        "*.so.*",
        "*.o",
        "*.a",
        "*.pyc",
        "*.egg-info",
    )
    destination.mkdir()
    for name in ("CMakeLists.txt", "pyproject.toml", "README.md", "LICENSE.txt"):
        shutil.copy2(source / name, destination / name)
    for name in (
        "common",
        "roughpy_jax",
        "platforms/cuda",
        "third_party/rough_path_primitives",
        "third_party/eigen",
    ):
        shutil.copytree(source / name, destination / name, ignore=ignore)


def repair_wheel(python, raw_dir, destination, *, cuda=False, env=None):
    wheels = list(raw_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one wheel in {raw_dir}, found {wheels}")
    command = [python, "-m", "auditwheel", "repair", "--plat", "manylinux_2_28_x86_64"]
    if cuda:
        command += ["--exclude", "libcudart.so.12", "--exclude", "libcudart.so.13"]
    run(*command, "-w", destination, wheels[0], env=env)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["12", "13"], required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--version", default="0.0.0")
    parser.add_argument("--cuda-architectures", default="native")
    parser.add_argument("--jax-version")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    pytest_args = (
        args.pytest_args[1:] if args.pytest_args[:1] == ["--"] else args.pytest_args
    )
    artifacts = Path("/artifacts")
    toolkit_version = subprocess.check_output(["nvcc", "--version"], text=True)
    print(toolkit_version, flush=True)
    if not re.search(rf"release {args.variant}\.", toolkit_version):
        raise RuntimeError("Container toolkit does not match requested CUDA variant")
    # Use pip's CUDA runtime libraries with JAX, as in the published extras.
    # Keep the container's toolkit libraries from overriding those libraries.
    subprocess_env = os.environ.copy()
    subprocess_env.pop("LD_LIBRARY_PATH", None)
    subprocess_env.pop("PYTHONPATH", None)
    build_env = {
        **subprocess_env,
        "CMAKE_BUILD_PARALLEL_LEVEL": str(args.jobs),
        "SETUPTOOLS_SCM_PRETEND_VERSION": args.version,
    }
    with tempfile.TemporaryDirectory(prefix="rpj-cuda-tests-") as temporary:
        work = Path(temporary)
        source = work / "source"
        copy_sources(Path("/source"), source)
        run(sys.executable, "-m", "venv", work / "build-env", env=subprocess_env)
        build_python = work / "build-env/bin/python"
        run(
            build_python,
            "-m",
            "pip",
            "install",
            "build",
            "auditwheel",
            "scikit-build-core[pyproject]>=1.0",
            "setuptools-scm>=9.0",
            "cmake>=3.31",
            "ninja>=1.11",
            "roughpy>=0.3.0",
            f"jax{('==' + args.jax_version) if args.jax_version else ''}",
            env=subprocess_env,
        )
        raw = work / "raw"
        run(
            build_python,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            raw / "main",
            "-Ccmake.define.RPJ_INPLACE_BUILD=OFF",
            source,
            env=build_env,
        )
        plugin = work / "plugin"
        run(
            build_python,
            "/source/tools/prepare_cuda_plugin_build.py",
            "--variant",
            args.variant,
            "--root",
            source,
            "--out",
            plugin,
            env=subprocess_env,
        )
        run(
            build_python,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            raw / "plugin",
            f"-Ccmake.define.CMAKE_CUDA_ARCHITECTURES={args.cuda_architectures}",
            plugin,
            env=build_env,
        )
        wheels = artifacts / "wheelhouse"
        wheels.mkdir(exist_ok=True)
        # Keep raw/repaired wheels separate from previous invocations.
        wheelhouse = work / "wheelhouse"
        wheelhouse.mkdir()
        repair_wheel(build_python, raw / "main", wheelhouse, env=subprocess_env)
        repair_wheel(
            build_python, raw / "plugin", wheelhouse, cuda=True, env=subprocess_env
        )
        for wheel in wheelhouse.glob("*.whl"):
            shutil.copy2(wheel, wheels / wheel.name)
        if args.build_only:
            return
        jax_version = subprocess.check_output(
            [
                str(build_python),
                "-c",
                "from importlib.metadata import version; print(version('jax'))",
            ],
            text=True,
            env=subprocess_env,
        ).strip()
        run(sys.executable, "-m", "venv", work / "test-env", env=subprocess_env)
        test_python = work / "test-env/bin/python"
        run(
            test_python,
            "-m",
            "pip",
            "install",
            f"jax[cuda{args.variant}]=={jax_version}",
            "pytest",
            "pytest-benchmark",
            "hypothesis",
            *sorted(wheelhouse.glob("*.whl")),
            env=subprocess_env,
        )
        tests = work / "test-run/tests"
        shutil.copytree(source / "roughpy_jax/tests", tests)
        shutil.copy2(source / "pyproject.toml", tests.parent / "pyproject.toml")
        test_env = {
            **subprocess_env,
            "JAX_PLATFORMS": "cuda,cpu",
            "RPJ_TEST_PLATFORMS": "cuda",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
        with (artifacts / "pip-freeze.txt").open("w") as output:
            run(test_python, "-m", "pip", "freeze", stdout=output, env=subprocess_env)
        run(
            test_python,
            "-m",
            "pytest",
            "-o",
            "testpaths=tests",
            "--junitxml",
            artifacts / "tests.xml",
            *pytest_args,
            cwd=tests.parent,
            env=test_env,
        )


if __name__ == "__main__":
    try:
        main()
    finally:
        # Docker runs as root so repair ownership of the output mount only.
        if "RPJ_ARTIFACT_UID" in os.environ:
            uid, gid = (
                int(os.environ["RPJ_ARTIFACT_UID"]),
                int(os.environ["RPJ_ARTIFACT_GID"]),
            )
            for path in [Path("/artifacts"), *Path("/artifacts").rglob("*")]:
                os.chown(path, uid, gid)
