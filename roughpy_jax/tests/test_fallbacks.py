import math

import jax
import jax.numpy as jnp
import pytest

import roughpy_jax as rpj
from roughpy_jax.ops import Operation


@pytest.mark.parametrize("compile", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("input_depth", [1, 2])
@pytest.mark.parametrize("out_depth", [0, 1, 3])
@pytest.mark.parametrize("operation", ["ft_exp", "ft_log", "ft_fmexp"])
def test_fallback_series_output_depth(
    monkeypatch, operation, out_depth, input_depth, compile
):
    monkeypatch.setattr(Operation, "no_acceleration", True)
    input_basis = rpj.TensorBasis(2, input_depth)
    out_basis = rpj.TensorBasis(2, out_depth)
    argument = rpj.FreeTensor(jnp.zeros(input_basis.size()).at[1].set(1), input_basis)

    def evaluate(x):
        if operation == "ft_fmexp":
            return rpj.ft_fmexp(
                rpj.FreeTensor.identity(input_basis), x, out_basis=out_basis
            )
        return getattr(rpj, operation)(x, out_basis=out_basis)

    result = (jax.jit(evaluate) if compile else evaluate)(argument)
    expected = jnp.zeros(out_basis.size(), dtype=result.dtype)
    for degree in range(out_depth + 1):
        if operation == "ft_log":
            coefficient = (-1) ** (degree + 1) / degree if degree else 0
        else:
            coefficient = 1 / math.factorial(degree)
        expected = expected.at[out_basis.degree_begin[degree]].set(coefficient)

    assert result.basis == out_basis
    assert jnp.allclose(result.data, expected)


@pytest.mark.parametrize("compile", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(("op_depth", "arg_depth"), [(1, 2), (2, 1)])
def test_fallback_right_adjoint_mixed_depth(monkeypatch, op_depth, arg_depth, compile):
    op_basis = rpj.TensorBasis(2, op_depth)
    arg_basis = rpj.TensorBasis(2, arg_depth)
    op = rpj.FreeTensor(jnp.arange(1.0, op_basis.size() + 1), op_basis)
    arg = rpj.ShuffleTensor(jnp.arange(1.0, arg_basis.size() + 1), arg_basis)
    monkeypatch.setattr(Operation, "no_acceleration", False)
    expected = rpj.ft_adjoint_right_mul(op, arg)
    monkeypatch.setattr(Operation, "no_acceleration", True)
    evaluate = rpj.ft_adjoint_right_mul
    result = (jax.jit(evaluate) if compile else evaluate)(op, arg)

    assert result.basis == expected.basis
    assert jnp.allclose(result.data, expected.data)


@pytest.mark.parametrize("compile", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float64])
@pytest.mark.parametrize(
    "force_fallback", [False, True], ids=["no-native-dispatch", "forced"]
)
@pytest.mark.parametrize("operation", ["st_mul", "st_fma"])
def test_shuffle_cpu_fallback(monkeypatch, operation, force_fallback, dtype, compile):
    basis = rpj.TensorBasis(2, 2)
    argument = rpj.ShuffleTensor(jnp.ones(basis.size(), dtype=dtype), basis)
    monkeypatch.setattr(Operation, "no_acceleration", False)
    native_argument = rpj.ShuffleTensor(argument.data.astype(jnp.float64), basis)
    arg_count = 2 if operation == "st_mul" else 3
    expected = getattr(rpj, operation)(*((native_argument,) * arg_count))
    monkeypatch.setattr(Operation, "no_acceleration", force_fallback)
    if not force_fallback:
        # Exercise automatic fallback even once compiled fp16 kernels exist.
        # Keep CPU registrations intact for fallback kernel/dtype selection.
        op_cls = Operation.get_operation(operation)
        monkeypatch.setattr(op_cls, "get_implementations", lambda self: {})

    def evaluate(*args):
        return getattr(rpj, operation)(*args)

    result = (jax.jit(evaluate) if compile else evaluate)(*((argument,) * arg_count))

    assert result.dtype == dtype
    assert jnp.allclose(result.data, expected.data.astype(dtype))
