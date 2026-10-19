"""Build and test installed CPU/CUDA wheels in the release manylinux images."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from pathlib import Path

IMAGES = {
    "12": "quay.io/manylinux_cuda/manylinux_2_28_x86_64_cuda12_9:latest",
    "13": "quay.io/manylinux_cuda/manylinux_2_28_x86_64_cuda13_1:latest",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["12", "13", "all"], default="all")
    parser.add_argument(
        "--python", choices=["3.11", "3.12", "3.13", "3.14"], default="3.11"
    )
    parser.add_argument(
        "--image", help="Override the image for a single variant (a tag or digest)."
    )
    parser.add_argument(
        "--gpus", default="all", help="Docker GPU selector; defaults to all."
    )
    parser.add_argument("--jobs", type=int, default=4, help="Parallel build jobs.")
    parser.add_argument(
        "--cuda-architectures",
        help="CMake architecture list; defaults to native, or 80 for build-only.",
    )
    parser.add_argument(
        "--jax-version", help="Pin JAX for both build and runtime environments."
    )
    parser.add_argument(
        "--version",
        default="0.0.0",
        help="Version assigned to both development wheels.",
    )
    parser.add_argument("--output", type=Path, default=Path("build/cuda-tests"))
    parser.add_argument(
        "--pip-cache",
        default="roughpy-jax-pip-cache",
        help="Docker volume for the shared persistent pip cache.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build and repair wheels without requiring a GPU.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print Docker commands without running them.",
    )
    parser.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="Pytest arguments after --; default: run the full suite.",
    )
    args = parser.parse_args(argv)
    if args.image and args.variant == "all":
        parser.error("--image requires a single --variant")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.pytest_args[:1] == ["--"]:
        args.pytest_args = args.pytest_args[1:]
    return args


def docker_command(args, variant, repo_root):
    output = args.output.resolve() / f"cuda{variant}"
    python_tag = "cp" + args.python.replace(".", "")
    command = ["docker", "run", "--rm", "--init"]
    if not args.build_only:
        command += [
            "--gpus",
            args.gpus,
            "--env",
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        ]
    command += [
        "--mount",
        f"type=bind,src={repo_root},dst=/source,readonly",
        "--mount",
        f"type=bind,src={output},dst=/artifacts",
        "--mount",
        f"type=volume,src={args.pip_cache},dst=/pip-cache",
        "--env",
        "PIP_CACHE_DIR=/pip-cache",
        "--env",
        f"RPJ_ARTIFACT_UID={os.getuid()}",
        "--env",
        f"RPJ_ARTIFACT_GID={os.getgid()}",
        "--workdir",
        "/tmp",
        "--entrypoint",
        f"/opt/python/{python_tag}-{python_tag}/bin/python",
        args.image or IMAGES[variant],
        "/source/tools/cuda_test_container.py",
        "--variant",
        variant,
        "--jobs",
        str(args.jobs),
        "--version",
        args.version,
        "--cuda-architectures",
        args.cuda_architectures or ("80" if args.build_only else "native"),
    ]
    if args.jax_version:
        command += ["--jax-version", args.jax_version]
    if args.build_only:
        command += ["--build-only"]
    return [*command, "--", *args.pytest_args]


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    for name, path in (
        ("RPP", "third_party/rough_path_primitives"),
        ("Eigen", "third_party/eigen"),
    ):
        if not (root / path / "CMakeLists.txt").is_file():
            raise SystemExit(
                f"{name} sources are missing; run git submodule update --init "
                "--recursive before running tests."
            )
    variants = IMAGES if args.variant == "all" else [args.variant]
    failures = []
    for variant in variants:
        command = docker_command(args, variant, root)
        print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        output = args.output.resolve() / f"cuda{variant}"
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "build-tests.log"
        with (
            log_path.open("w") as log,
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
            ) as process,
        ):
            log.write(shlex.join(command) + "\n")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                failures.append(variant)
    if failures:
        raise SystemExit(f"CUDA build/test failed for: {', '.join(failures)}")


if __name__ == "__main__":
    main()
