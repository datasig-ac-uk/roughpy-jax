# roughpy-jax

`roughpy-jax` provides JAX bindings and operations for RoughPy. It is aimed at
working with dense algebraic objects such as free tensors, shuffle tensors, and
elements of the free Lie algebra, while integrating with JAX arrays,
transformations, and differentiation.

This library is currently in an alpha stage. The API is still evolving, and
some features are incomplete or subject to change as the package matures.

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

Once published, `roughpy-jax` can be installed from PyPI with:

```bash
pip install roughpy-jax
```

The package requires the latest version of roughpy (0.3.0) and Python 3.11 or newer.

Release artifacts can also be downloaded from the GitHub Releases page for this
repository.

## Installing From Source

Installing from source is useful when working on the package itself or testing
changes before a release. A working C/C++ toolchain and CMake-compatible build
environment are required.

Clone the repository and install it into a virtual environment:

```bash
git clone https://github.com/datasig-ac-uk/roughpy-jax.git
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

Only very basic interval support is currently implemented. This area still
needs to be expanded.

## JAX Notes

All algebra objects and algebraic operations are intended to support JIT and
are fully differentiable. In particular, the package provides explicit
derivative and adjoint-derivative functions alongside the corresponding primal
operations, and these are the functions whose type information should be relied
upon.

Stream objects are more limited. Some stream types may support JIT in some
contexts, but stream support is not yet uniform. In particular,
`LieIncrementStream` is not currently registered as a pytree because of
technical limitations that have not yet been resolved.

There is also an important JAX-specific subtlety in reverse-mode code. Because
JAX tree handling does not preserve the intended algebraic type information in
all backward-pass cotangents, cotangents may be represented using the wrong
algebra wrapper. For example, a value that should be treated as a shuffle tensor
may arrive as a free tensor, or vice versa. To handle this, internal JAX-facing
code applies corrective conversions on incoming and outgoing cotangents. The
public derivative and adjoint-derivative APIs expose the correct algebraic
types.

## Testing

The test suite exercises both the pure Python layer and the compiled CPU
backend. Locally, the main test command is:

```bash
pytest -m "not extra" roughpy_jax/tests
```

Wheel builds are tested through `cibuildwheel` in CI, and release artifacts are
validated before publishing.

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
