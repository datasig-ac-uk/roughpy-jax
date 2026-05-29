from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from textwrap import dedent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a variant-specific CUDA plugin build directory."
    )
    parser.add_argument(
        "--variant",
        required=True,
        help="CUDA plugin family, for example 12 or 13. Minor variants like 12.8 are normalised to 12.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Defaults to build/cuda-plugin/<variant> under the repo root.",
    )
    return parser.parse_args()


def normalise_variant(variant: str) -> tuple[str, str, str, str]:
    value = variant.strip().removeprefix("cuda")
    if not value:
        raise ValueError("CUDA variant must not be empty")

    major = value.split(".", 1)[0]
    if not major.isdigit():
        raise ValueError(f"Invalid CUDA variant {variant!r}")

    if major not in {"12", "13"}:
        raise ValueError("Only CUDA 12 and CUDA 13 plugin families are supported")

    family = f"cuda{major}"
    package_module = f"roughpy_jax_{family}_plugin"
    return major, family, family, package_module


def render_pyproject(major: str, family: str, package_module: str) -> str:
    return dedent(
        f"""
        [build-system]
        requires = [
            "scikit-build-core[pyproject]>=0.10",
            "roughpy>=0.3.0",
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ]
        build-backend = "scikit_build_core.build"

        [project]
        name = "roughpy-jax-{family}"
        version = "1.0.0"
        description = "CUDA {major} plugin package for roughpy-jax custom calls and kernels"
        readme = "README.md"
        authors = [{{ name = "The RoughPy Authors", email = "info@datasig.ac.uk" }}]
        license = {{ text = "BSD-3-Clause" }}
        requires-python = ">=3.11"
        keywords = ["roughpy", "jax", "cuda", "xla"]
        classifiers = [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Mathematics",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Programming Language :: Python :: 3.13",
        ]
        dependencies = [
            "roughpy-jax==1.0.0",
        ]

        [project.entry-points."roughpy_jax.plugins"]
        {family} = "{package_module}"

        [tool.scikit-build]
        cmake.version = ">=3.30"
        ninja.version = ">=1.11"
        cmake.build-type = "Release"
        logging.level = "INFO"
        experimental = true
        wheel.packages = []
        """
    ).strip() + "\n"


def render_readme(major: str, jax_extra: str) -> str:
    return dedent(
        f"""
        # roughpy-jax-{jax_extra}

        Binary CUDA plugin for `roughpy-jax`, built for the CUDA {major} family.

        Typical runtime installation:

        ```bash
        pip install "roughpy-jax[{jax_extra}]"
        ```
        """
    ).strip() + "\n"


def main() -> None:
    args = parse_args()
    repo_root = args.root.resolve()
    major, family, jax_extra, package_module = normalise_variant(args.variant)
    out_dir = (args.out or repo_root / "build" / "cuda-plugin" / family).resolve()

    if out_dir.exists():
        shutil.rmtree(out_dir)

    shutil.copytree(repo_root / "platforms" / "cuda" / "src", out_dir / "src")
    shutil.copytree(
        repo_root / "platforms" / "cuda" / "package",
        out_dir / "package",
    )
    shutil.copy2(repo_root / "platforms" / "cuda" / "CMakeLists.txt", out_dir / "CMakeLists.txt")

    (out_dir / "pyproject.toml").write_text(
        render_pyproject(major, family, package_module),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(render_readme(major, jax_extra), encoding="utf-8")

    build_cmd = (
        f'python -m build --wheel '
        f'-Ccmake.define.RPJ_CUDA_TOOLKIT_MAJOR={major} '
        f'-Ccmake.define.RPJ_CUDA_PACKAGE_DIR={package_module} '
        f'-Ccmake.define.RPJ_CUDA_VARIANT={family} '
        f'"{out_dir}"'
    )

    print(f"Prepared {out_dir}")
    print(build_cmd)


if __name__ == "__main__":
    main()
