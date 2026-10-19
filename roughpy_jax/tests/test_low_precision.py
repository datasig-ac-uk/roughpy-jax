"""CPU scalar storage and accumulation contracts shared by the operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roughpy_jax as rpj
from roughpy_jax import TensorBasis
from roughpy_jax.ops import DenseAntipode


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("factor", [0.5, 1.0, 1.5, 2.0])
def test_scalar_conversion_boundaries(dtype, factor):
    # Every encoding passes through an Eigen load, float32 arithmetic, and
    # Eigen store, including infinities, NaNs, and normal/subnormal boundaries.
    encodings = np.arange(65536, dtype=np.uint16).view(np.dtype(dtype))
    values = jax.device_put(
        jnp.asarray(encodings).reshape(-1, 1), jax.devices("cpu")[0]
    )
    factors = jnp.full_like(values, factor)
    basis = TensorBasis(1, 0)
    result = jax.jit(rpj.ft_mul)(
        rpj.FreeTensor(values, basis), rpj.FreeTensor(factors, basis)
    ).data
    expected = (values.astype(jnp.float32) * factors.astype(jnp.float32)).astype(dtype)
    np.testing.assert_array_equal(
        np.asarray(result).astype(np.float32), np.asarray(expected).astype(np.float32)
    )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_fma_accumulates_in_float32(dtype):
    cpu = jax.devices("cpu")[0]
    epsilon = float(jnp.finfo(dtype).eps)
    value = 1 + epsilon
    rounded_product = float(jnp.asarray(value * value, dtype=dtype))
    args = tuple(
        jax.device_put(jnp.array([v], dtype=dtype), cpu)
        for v in (-rounded_product, value, value)
    )
    basis = TensorBasis(1, 0)
    result = jax.jit(rpj.ft_fma)(*(rpj.FreeTensor(arg, basis) for arg in args)).data
    # Rounding the product to coefficient precision before adding would give 0.
    np.testing.assert_array_equal(result.astype(jnp.float32), [epsilon**2])


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_scalar_storage_preserves_bits(dtype):
    # Unsigned antipode copies coefficients, so even signed zero and NaN
    # payloads must survive the raw-buffer/Eigen boundary without conversion.
    encodings = np.arange(65536, dtype=np.uint16)
    values = jax.device_put(
        jnp.asarray(encodings.view(np.dtype(dtype))).reshape(-1, 1),
        jax.devices("cpu")[0],
    )
    operation = DenseAntipode(
        (TensorBasis(1, 0),),
        jnp.dtype(dtype),
        (65536,),
        arg_max_deg=0,
        no_sign=True,
    )
    (result,) = jax.jit(operation)(values)
    np.testing.assert_array_equal(np.asarray(result).view(np.uint16).ravel(), encodings)
