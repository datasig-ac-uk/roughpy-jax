# roughpy-jax

`roughpy-jax` provides JAX bindings and operations for RoughPy. It provides
stream classes and dense algebraic objects (such as free tensors, shuffle 
tensors, and elements of the free Lie algebra) for computational rough path 
theory, and supports JAX JIT-compilation and differentiation.

`roughpy-jax` publishes stable releases and is under active development. APIs
may evolve on a faster timeline than projects with longer compatibility
cycles. Breaking changes and deprecations are documented in the
[GitHub release notes](https://github.com/datasig-ac-uk/roughpy-jax/releases).

## What This Package Provides

`roughpy-jax` builds on top of `roughpy` and `jax` and currently
includes:

- dense tensor, shuffle tensor, and Lie algebra wrappers
- algebraic operations such as multiplication, exponentials, logarithms, CBH,
  pairings, and adjoint operations
- JAX-compatible derivative and adjoint-derivative rules for core operations
- interval and partition types for stream queries
- stream types including piecewise Abelian streams and Lie increment streams


## Installation

`roughpy-jax` can be installed from PyPI with:

```bash
pip install roughpy-jax
```

CUDA plugin builds are published separately and are intended to be pulled in
through extras on the main package:

```bash
pip install "roughpy-jax[cuda12]"
pip install "roughpy-jax[cuda13]"
```

The core package auto-discovers installed backend plugins through Python entry
points, so users do not need a separate `import` for the CUDA extension.

The package requires RoughPy 0.3.0 or newer and Python 3.11 or newer.

Release notes and artifacts are available from the
[GitHub Releases page](https://github.com/datasig-ac-uk/roughpy-jax/releases).

CUDA plugin wheels are intended for Linux only and follow JAX's CUDA package
families (`cuda12` and `cuda13`). A `manylinux_2_28` baseline is used for Linux
wheel compatibility. This is compatible with Ubuntu 22.04, which ships a newer
glibc than that baseline. 

## Installing From Source

Installing from source is useful when working on the package itself or testing
changes before a release. A working C/C++ toolchain and CMake-compatible build
environment are required.

Clone the repository and install it into a virtual environment:

```bash
git clone --recurse-submodules https://github.com/datasig-ac-uk/roughpy-jax.git
cd roughpy-jax
python -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install .
```

If you are using `uv`, the equivalent workflow is:

```bash
uv venv
. .venv/bin/activate
uv pip install .
```

## Streams and Intervals

Streams are the central object of RoughPy, and so too in `roughpy-jax`.
Like RoughPy itself, `roughpy-jax` works carefully with intervals and stream
queries.

Current stream-facing functionality includes:

- `PiecewiseAbelianStream` for streams built from piecewise log-signature data
- `LieIncrementStream` for dyadic-cache-backed querying of log-signatures and
  signatures over intervals

These pieces are intended to make it practical to move between algebraic
objects and stream queries within JAX-oriented workflows.

## API Differences From RoughPy

There are some deliberate API differences compared to `roughpy`.

Context objects are not used in `roughpy-jax`. Instead, explicit basis objects
and conversion functions handle translation between algebraic objects with
different configurations. At present, only depth changes are supported
explicitly.

Streams may have several associated bases, depending on the stream type. These
can include:

- the basis of the underlying data
- the basis used for stored or cached data
- the basis used for answering queries

These bases do not need to be identical, but they do need to be compatible.
Exactly which bases exist, and whether they are user-facing, is stream-type
dependent.

The package provides real and dyadic intervals, partitions, batched interval
endpoints, and left-closed/right-open and left-open/right-closed endpoint
conventions. These types are JAX pytrees and can be used directly in stream
queries. Some operations remain incomplete; in particular, intersection of two
dyadic intervals is not yet implemented.

Direct conversion from RoughPy objects to `roughpy-jax` equivalents is not
currently provided.

## JAX Integration

Dense algebra objects are JAX pytrees. Their coefficient arrays are dynamic
leaves, while basis information is static metadata. Core algebra operations
support JIT compilation and reverse-mode transformations through custom VJP
rules, whose adjoint-derivatives backpropagate cotangents between operations.
Explicit derivative and adjoint-derivative functions are also available when
these operations need to be used directly.

`LieIncrementStream`, `PiecewiseAbelianStream`, intervals, and partitions are
also registered as pytrees. Stream queries can be JIT-compiled, and reverse-mode
transformations can propagate cotangents through the stored algebra data or
dyadic cache. Timestamps, query endpoints, partitions, bases, resolutions, and
interval conventions are treated as non-differentiable configuration.

The dyadic resolution used by `LieIncrementStream.from_increments` determines
the shape of its cache and must be static during JIT compilation. Passing
`resolution=None` is deprecated. Use `compute_separating_resolution` outside
the compiled function and pass the selected resolution explicitly.

JAX requires cotangents returned by a custom VJP to have the same pytree
structure as the corresponding primal value. This differs from the mathematical
duality between free tensors and shuffle tensors, so a cotangent produced by a
JAX reverse-mode transformation may use the primal algebra wrapper even when
the mathematical cotangent belongs to its dual algebra. The explicit
adjoint-derivative functions expose the mathematically appropriate algebra
types.

## Testing

The test suite exercises both the pure Python layer and the compiled CPU
backend. Locally, the main test command is:

```bash
pytest -m "not extra" roughpy_jax/tests
```

Main-package wheel builds are tested through `cibuildwheel` in CI, and release
artifacts are validated before publishing. CUDA plugin wheels are built but not
executed in CI because GitHub-hosted runners do not provide suitable GPUs.

Run `pytest` without the marker expression to include the longer tests marked
as `extra`.

CUDA plugin wheel builds are prepared with:

```bash
python tools/prepare_cuda_plugin_build.py --variant 12
```

That generates a variant-specific source and metadata tree under
`build/cuda-plugin/`. The release workflow passes this directory to
`cibuildwheel`; compiling it locally additionally requires the matching CUDA
toolkit and the native RoughPathPrimitives dependencies.

### Testing installed CUDA wheels in Docker

Run both CUDA families with the same manylinux CUDA 12.9 and 13.1 images used
by the release pipeline:

```bash
python tools/run_cuda_tests.py
```

The host needs Docker, NVIDIA Container Toolkit GPU passthrough, and a driver
compatible with the selected CUDA family. No host CUDA toolkit or Python build
environment is needed; the runner itself uses only Python's standard library
(Python 3.11 or newer). These images target Linux x86_64.

Each container mounts the checkout read-only and copies the current sources,
including uncommitted changes, into a temporary build tree. It builds the main
wheel and a toolkit-specific plugin using `prepare_cuda_plugin_build.py`, then
repairs both wheels with `auditwheel`, excluding the CUDA runtime library just
as the release pipeline does. The RPP checkout is never modified.

A separate virtual environment installs those wheels and the matching JAX CUDA
extra. The JAX version is resolved once for the build and reused for testing.
Both environments and both CUDA variants share a persistent pip cache in the
Docker volume `roughpy-jax-pip-cache`. Docker creates the volume on first use;
it survives container removal and is reused on later runs, including for the
large CUDA dependency wheels. Use `--pip-cache VOLUME_NAME` to select another
cache volume. Pip can still check package-index metadata, but reuses cached
downloads when available.
Tests are copied outside the source package so imports use the installed
wheels. The runner invokes the test environment's `python -m pytest` directly;
operation execution and validation belong to the test suite.
Test device fixtures select CUDA explicitly via `RPJ_TEST_PLATFORMS=cuda`;
CPU remains available for operations' host fallbacks. A missing GPU or plugin
is an error rather than a successful CPU-only run.

The default runs the complete suite, including `extra` tests, with four build
jobs, Python 3.11, and native GPU architectures. Both development wheels get
version `0.0.0` unless `--version` is supplied. Useful variations:

```bash
# One toolkit and just the half-shuffle tests, including their longer cases.
python tools/run_cuda_tests.py --variant 12 -- tests/test_left_half_shuffle.py
python tools/run_cuda_tests.py --variant 13 -- tests/test_left_half_shuffle.py

# Optionally exclude longer tests, matching the release wheel checks.
python tools/run_cuda_tests.py -- -m "not extra"

# Pin the JAX version and target Python; select a GPU and control build jobs.
python tools/run_cuda_tests.py --variant 13 --python 3.12 --jax-version 0.10.2 --gpus device=0 --jobs 2

# Build/repair wheels without a GPU. Defaults to architecture 80 in this mode.
python tools/run_cuda_tests.py --variant 12 --build-only

# Override architectures explicitly, or inspect both Docker commands.
python tools/run_cuda_tests.py --cuda-architectures "80;90"
python tools/run_cuda_tests.py --dry-run
```

Pytest arguments follow `--`; paths are relative to the isolated test working
directory, where the copied suite is named `tests/`. Normal pytest output
includes a device summary; verbose output (`-- -v`) shows CUDA fixture IDs.
CUDA cases replace the corresponding CPU cases. The native dispatch tests
check the compiled custom-call targets and result devices; fallback-only cases
are skipped on CUDA. The default images match the release
workflow; `--image` can override a single variant with a specific image tag or
digest for reproducible runs.

Wheels, build/test logs (including the toolkit version), installed dependency
versions, and the pytest JUnit report are saved under
`build/cuda-tests/cuda12/` and `build/cuda-tests/cuda13/`. Use `--output` to choose
a different artifact directory. The two variants run sequentially and both are
attempted even if one fails. The runner exits with a failure if either fails.

## Example

For examples of how to use the higher-level stream objects, see the `examples/`
folder. The `words` example from the RoughPy documentation has been converted
to use the `roughpy-jax` stream objects.

## Support

If you hit a bug or need a feature, open an issue on GitHub. Bug reports with a
minimal reproducer are the most useful.

## Contributing

Contributions are welcome, especially:

- bug fixes
- tests
- documentation improvements
- examples and API polish

If you plan to make a larger change, open an issue first so the design can be
discussed before implementation.

## License

`roughpy-jax` is licensed under the BSD 3-Clause License. See `LICENSE.txt`.
Third party libraries have their own licenses, see `THIRD_PART_NOTICES.txt` for 
more information.
