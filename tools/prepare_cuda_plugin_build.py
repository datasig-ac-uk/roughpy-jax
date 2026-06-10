from __future__ import annotations

import argparse
import re
import shutil
import tomllib
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


def normalise_variant(variant: str) -> tuple[str, str]:
    value = variant.strip().removeprefix("cuda")
    if not value:
        raise ValueError("CUDA variant must not be empty")

    major = value.split(".", 1)[0]
    if not major.isdigit():
        raise ValueError(f"Invalid CUDA variant {variant!r}")

    if major not in {"12", "13"}:
        raise ValueError("Only CUDA 12 and CUDA 13 plugin families are supported")

    family = f"cuda{major}"
    return major, family


def load_root_pyproject(repo_root: Path) -> dict:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))


def format_toml_list(values: list[str]) -> str:
    lines = ",\n".join(f'    "{value}"' for value in values)
    return f"[\n{lines},\n]"


def wheel_python_api(root_pyproject: dict) -> str:
    value = (
        root_pyproject.get("tool", {})
        .get("scikit-build", {})
        .get("wheel", {})
        .get("py-api")
    )
    if not value:
        raise ValueError("Root pyproject.toml must define tool.scikit-build.wheel.py-api")
    return value


def plugin_distribution_name(root_pyproject: dict, family: str) -> str:
    extra_deps = root_pyproject["project"].get("optional-dependencies", {}).get(family, [])
    pattern = re.compile(r"^(roughpy-jax-[A-Za-z0-9_.-]+)")

    for dep in extra_deps:
        match = pattern.match(dep)
        if match:
            return match.group(1)

    raise ValueError(
        f"Root pyproject.toml optional dependency group {family!r} must declare the plugin package name"
    )


def plugin_runtime_dependencies(root_pyproject: dict, family: str) -> list[str]:
    version = root_pyproject["project"]["version"]
    return [
        f'    "roughpy-jax=={version}",',
        f'    "jax[{family}]>=0.4.0; platform_system == \'Linux\'",',
    ]


def supported_cuda_architectures(major: str) -> str:
    if major == "12":
        architectures = ["80", "86", "87", "89", "90", "100", "101", "103", "120", "121"]
    elif major == "13":
        architectures = ["80", "86", "87", "88", "89", "90", "100", "103", "110", "120", "121"]
    else:
        raise ValueError(f"Unsupported CUDA major version {major!r}")

    return ";".join(architectures)


def plugin_classifiers(root_pyproject: dict) -> list[str]:
    classifiers = root_pyproject["project"].get("classifiers", [])
    filtered = [
        classifier
        for classifier in classifiers
        if not classifier.startswith("Operating System :: ")
    ]
    filtered.extend(
        [
            "Operating System :: POSIX",
            "Operating System :: POSIX :: Linux",
        ]
    )
    return filtered


def render_pyproject(
    major: str,
    family: str,
    root_pyproject: dict,
) -> str:
    root_project = root_pyproject["project"]
    root_cibw = root_pyproject.get("tool", {}).get("cibuildwheel", {})
    authors = root_project.get("authors", [])
    classifiers = plugin_classifiers(root_pyproject)
    version = root_project["version"]
    requires_python = root_project["requires-python"]
    keywords = root_project.get("keywords", [])
    license_text = root_project.get("license", {}).get("text", "BSD-3-Clause")
    skip = root_cibw.get("skip", [])
    py_api = wheel_python_api(root_pyproject)
    project_name = plugin_distribution_name(root_pyproject, family)
    package_module = f"roughpy_jax_{family}_plugin"
    runtime_dependencies = plugin_runtime_dependencies(root_pyproject, family)
    cuda_architectures = supported_cuda_architectures(major)
    authors_block = format_inline_toml_list(authors)
    classifiers_block = format_toml_list(classifiers)
    keywords_block = format_inline_toml_list(keywords)
    skip_block = format_toml_list(skip)

    return "\n".join(
        [
            "[build-system]",
            "requires = [",
            '    "scikit-build-core[pyproject]>=0.10",',
            '    "roughpy>=0.3.0",',
            '    "jax>=0.4.0",',
            '    "jaxlib>=0.4.0",',
            "]",
            'build-backend = "scikit_build_core.build"',
            "",
            "[project]",
            f'name = "{project_name}"',
            f'version = "{version}"',
            f'description = "CUDA {major} plugin package for roughpy-jax custom calls and kernels"',
            'readme = "README.md"',
            f"authors = {authors_block}",
            f'license = {{ text = "{license_text}" }}',
            f'requires-python = "{requires_python}"',
            f"keywords = {keywords_block}",
            f"classifiers = {classifiers_block}",
            "dependencies = [",
            *runtime_dependencies,
            "]",
            "",
            "[tool.scikit-build]",
            'cmake.version = ">=3.30"',
            'ninja.version = ">=1.11"',
            'cmake.build-type = "Release"',
            'logging.level = "INFO"',
            "experimental = true",
            "wheel.packages = []",
            f'wheel.py-api = "{py_api}"',
            "",
            "[tool.scikit-build.cmake.define]",
            f'RPJ_CUDA_TOOLKIT_MAJOR = "{major}"',
            f'RPJ_CUDA_PACKAGE_DIR = "{package_module}"',
            f'RPJ_CUDA_VARIANT = "{family}"',
            f'CMAKE_CUDA_ARCHITECTURES = "{cuda_architectures}"',
            "",
            "[tool.cibuildwheel]",
            f'build = "{py_api}-manylinux_x86_64"',
            'build-frontend = { name = "pip", args = ["--no-build-isolation"] }',
            f"skip = {skip_block}",
            "",
        ]
    )


def format_inline_toml_list(values: list[object]) -> str:
    rendered = []
    for value in values:
        if isinstance(value, str):
            rendered.append(f'"{value}"')
        elif isinstance(value, dict):
            items = ", ".join(f'{key} = "{item}"' for key, item in value.items())
            rendered.append(f"{{ {items} }}")
        else:
            raise TypeError(f"Unsupported TOML inline list value: {value!r}")
    return f"[{', '.join(rendered)}]"


def render_readme(major: str, family: str, project_name: str) -> str:
    return dedent(
        f"""
        # {project_name}

        Binary CUDA plugin for `roughpy-jax`, built for the CUDA {major} family.

        Typical runtime installation:

        ```bash
        pip install "roughpy-jax[{family}]"
        ```
        """
    ).strip() + "\n"


def main() -> None:
    args = parse_args()
    repo_root = args.root.resolve()
    root_pyproject = load_root_pyproject(repo_root)
    major, family = normalise_variant(args.variant)
    project_name = plugin_distribution_name(root_pyproject, family)
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
        render_pyproject(major, family, root_pyproject),
        encoding="utf-8",
    )
    (out_dir / "README.md").write_text(
        render_readme(major, family, project_name),
        encoding="utf-8",
    )

    print(f"Prepared {out_dir}")
    print(f"Metadata written to {out_dir / 'pyproject.toml'}")


if __name__ == "__main__":
    main()
