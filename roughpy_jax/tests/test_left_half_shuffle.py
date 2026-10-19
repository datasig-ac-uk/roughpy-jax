from functools import lru_cache
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from roughpy_jax import TensorBasis
from roughpy_jax.ops import Operation


@lru_cache(None)
def _shuffles(lhs, rhs):
    if not lhs:
        return (rhs,)
    if not rhs:
        return (lhs,)
    return tuple((lhs[0], *word) for word in _shuffles(lhs[1:], rhs)) + tuple(
        (rhs[0], *word) for word in _shuffles(lhs, rhs[1:])
    )


@lru_cache(None)
def _half_shuffle_coefficients(width, depth):
    words = tuple(
        word
        for degree in range(depth + 1)
        for word in product(range(width), repeat=degree)
    )
    index = {word: i for i, word in enumerate(words)}
    coefficients = np.zeros((len(words),) * 3)
    for i, lhs in enumerate(words):
        if not lhs:
            continue
        for j, rhs in enumerate(words):
            if len(lhs) + len(rhs) > depth:
                continue
            for trailing in _shuffles(lhs[1:], rhs):
                coefficients[index[(lhs[0], *trailing)], i, j] += 1
    return words, coefficients


@pytest.mark.parametrize(
    "name", ["left_hs_mul", "left_hs_adj_lmul", "left_hs_adj_rmul"]
)
@pytest.mark.parametrize("input_depths", [(3, 3), (1, 2), (2, 1)])
@pytest.mark.parametrize("min_degree", [0, 1])
def test_half_shuffle_ffi_reference(
    name, rpj_tensor_dtype, input_depths, min_degree, rpj_batch, rpj_tensor_tolerances
):
    dtype = jnp.dtype(rpj_tensor_dtype)
    basis = TensorBasis(2, 3)
    bases = tuple(TensorBasis(2, depth) for depth in input_depths)
    op_cls = Operation.get_operation(name)
    operation = op_cls(
        bases,
        dtype,
        rpj_batch.shape,
        specific_basis=basis,
        integrator_max_deg=input_depths[0],
        integrand_max_deg=input_depths[1],
        integrator_min_deg=min_degree,
        integrand_min_deg=min_degree,
    )
    inputs = tuple(rpj_batch.rng_uniform(-1, 1, b.size(), dtype) for b in bases)
    words, coefficients = _half_shuffle_coefficients(2, 3)
    padded = []
    for data, depth in zip(inputs, input_depths, strict=True):
        accum_dtype = np.float64 if dtype == jnp.dtype("float64") else np.float32
        values = np.zeros((*rpj_batch.shape, basis.size()), dtype=accum_dtype)
        values[..., : data.shape[-1]] = np.asarray(data)
        mask = np.array([min_degree <= len(word) <= depth for word in words])
        padded.append(values * mask)

    if name == "left_hs_mul":
        expected = np.einsum("kij,...i,...j->...k", coefficients, *padded)
    elif name == "left_hs_adj_lmul":
        expected = np.einsum("kij,...i,...k->...j", coefficients, *padded)
    else:
        # RPP's right adjoint takes the cotangent first, then the fixed integrand.
        expected = np.einsum("kij,...k,...j->...i", coefficients, *padded)
    actual = jax.jit(lambda a, b: operation(a, b)[0])(*inputs)
    assert actual.shape == (*rpj_batch.shape, basis.size())
    assert actual.dtype == dtype
    np.testing.assert_allclose(
        actual.astype(jnp.float32) if dtype.itemsize == 2 else actual,
        expected,
        **rpj_tensor_tolerances(atol=1e-6, rtol=1e-6),
    )


@pytest.mark.parametrize(
    "name", ["left_hs_mul", "left_hs_adj_lmul", "left_hs_adj_rmul"]
)
def test_half_shuffle_ffi_rejects_mismatched_batches(name):
    basis = TensorBasis(2, 2)
    operation = Operation.get_operation(name)(
        (basis, basis),
        jnp.dtype("float32"),
        (2,),
        integrator_max_deg=2,
        integrand_max_deg=2,
        integrator_min_deg=0,
        integrand_min_deg=0,
    )
    with pytest.raises(Exception, match="mismatched batch dimensions"):
        operation(jnp.zeros((2, basis.size())), jnp.zeros((3, basis.size())))[
            0
        ].block_until_ready()


@pytest.mark.parametrize("dtype", [jnp.dtype("float32"), jnp.dtype("float64")])
@pytest.mark.parametrize("input_depths", [(3, 3), (1, 2), (2, 1)])
@pytest.mark.parametrize("out_depth", [1, 3, 4])
def test_half_shuffle_algebra_derivatives(dtype, input_depths, out_depth, rpj_batch):
    import roughpy_jax as rpj

    bases = tuple(TensorBasis(2, depth) for depth in input_depths)
    out_basis = TensorBasis(2, out_depth)
    primals = tuple(
        rpj.ShuffleTensor(rpj_batch.rng_uniform(-1, 1, b.size(), dtype), b)
        for b in bases
    )
    tangents = tuple(
        rpj.ShuffleTensor(rpj_batch.rng_uniform(-1, 1, b.size(), dtype), b)
        for b in bases
    )
    cotangent = rpj.FreeTensor(
        rpj_batch.rng_uniform(-1, 1, out_basis.size(), dtype), out_basis
    )
    # Independent word enumeration preserves shuffle multiplicities. Operand
    # order and empty-word behaviour follow the integrator/integrand convention
    # of https://arxiv.org/abs/2212.00134 (with RPP's word orientation).
    _, coefficients = _half_shuffle_coefficients(2, max(*input_depths, out_depth))
    coefficients = coefficients[
        : out_basis.size(), : bases[0].size(), : bases[1].size()
    ]
    a, b = (np.asarray(p.data) for p in primals)
    da, db = (np.asarray(t.data) for t in tangents)
    ct = np.asarray(cotangent.data)
    expected = np.einsum("kij,...i,...j->...k", coefficients, a, b)
    expected_derivative = np.einsum("kij,...i,...j->...k", coefficients, da, b)
    expected_derivative += np.einsum("kij,...i,...j->...k", coefficients, a, db)
    expected_cotangents = (
        np.einsum("kij,...k,...j->...i", coefficients, ct, b),
        np.einsum("kij,...i,...k->...j", coefficients, a, ct),
    )

    def primal(a, b):
        return rpj.left_hs_mul(a, b, out_basis=out_basis)

    actual = jax.jit(primal)(*primals)
    derivative = jax.jit(
        lambda a, b, da, db: rpj.left_hs_mul_derivative(
            a, b, da, db, out_basis=out_basis
        )
    )(*primals, *tangents)
    adjoints = jax.jit(rpj.left_hs_mul_adjoint_derivative)(*primals, cotangent)
    _, pullback = jax.vjp(primal, *primals)
    jax_adjoints = jax.jit(pullback)(rpj.ShuffleTensor(cotangent.data, out_basis))
    np.testing.assert_allclose(actual.data, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        derivative.data, expected_derivative, atol=2e-6, rtol=2e-6
    )
    for basis, adjoint, jax_adjoint, expected_ct in zip(
        bases, adjoints, jax_adjoints, expected_cotangents, strict=True
    ):
        assert isinstance(adjoint, rpj.DenseFreeTensor)
        assert adjoint.basis == basis
        assert jax_adjoint.basis == basis
        np.testing.assert_allclose(adjoint.data, expected_ct, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(jax_adjoint.data, expected_ct, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        np.sum(derivative.data * cotangent.data, axis=-1),
        sum(
            np.sum(t.data * c.data, axis=-1)
            for t, c in zip(tangents, adjoints, strict=True)
        ),
        atol=4e-6,
        rtol=4e-6,
    )


@pytest.mark.parametrize("dtype", [jnp.dtype("float32"), jnp.dtype("float64")])
def test_half_shuffle_units_and_zinbiel(dtype, rpj_batch):
    import roughpy_jax as rpj

    basis = TensorBasis(2, 3)
    a, b, c = (
        rpj.ShuffleTensor(
            rpj_batch.rng_uniform(-1, 1, basis.size(), dtype)
            .at[..., 0]
            .set(0)
            .at[..., basis.degree_begin[2] :]
            .set(0),
            basis,
        )
        for _ in range(3)
    )
    unit = rpj.ShuffleTensor.identity(basis, dtype=dtype, batch_dims=rpj_batch.shape)
    np.testing.assert_allclose(rpj.left_hs_mul(a, unit).data, a.data)
    np.testing.assert_array_equal(rpj.left_hs_mul(unit, a).data, jnp.zeros_like(a.data))
    np.testing.assert_allclose(
        (rpj.left_hs_mul(a, b) + rpj.left_hs_mul(b, a)).data,
        rpj.st_mul(a, b).data,
        atol=2e-6,
        rtol=2e-6,
    )
    np.testing.assert_allclose(
        rpj.left_hs_mul(rpj.left_hs_mul(a, b), c).data,
        rpj.left_hs_mul(a, rpj.st_mul(b, c)).data,
        atol=2e-6,
        rtol=2e-6,
    )


def test_half_shuffle_algebra_rejects_incompatible_bases():
    import roughpy_jax as rpj

    a = rpj.ShuffleTensor(jnp.ones(7), TensorBasis(2, 2))
    b = rpj.ShuffleTensor(jnp.ones(13), TensorBasis(3, 2))
    ct = rpj.FreeTensor(a.data, a.basis)
    for fn in (
        rpj.left_hs_mul,
        rpj.left_hs_adjoint_left_mul,
        rpj.left_hs_adjoint_right_mul,
    ):
        with pytest.raises(ValueError, match="Incompatible width"):
            fn(b, ct if fn != rpj.left_hs_mul else a)
    with pytest.raises(ValueError, match="Incompatible width"):
        rpj.left_hs_mul_derivative(a, a, a, b)
    with pytest.raises(ValueError, match="Incompatible width"):
        rpj.left_hs_mul_adjoint_derivative(a, b, ct)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("dtype", [jnp.dtype("float32"), jnp.dtype("float64")])
@pytest.mark.parametrize(
    "input_depths", [(0, 0), (0, 3), (3, 0), (1, 2), (2, 1), (3, 3)]
)
def test_half_shuffle_adjoint_derivatives(side, dtype, input_depths, rpj_batch):
    import roughpy_jax as rpj

    op_basis, arg_basis = (TensorBasis(2, depth) for depth in input_depths)
    out_basis = TensorBasis(2, max(input_depths))
    op, t_op = (
        rpj.ShuffleTensor(
            rpj_batch.rng_uniform(-1, 1, op_basis.size(), dtype), op_basis
        )
        for _ in range(2)
    )
    arg, t_arg = (
        rpj.FreeTensor(rpj_batch.rng_uniform(-1, 1, arg_basis.size(), dtype), arg_basis)
        for _ in range(2)
    )
    ct = rpj.ShuffleTensor(
        rpj_batch.rng_uniform(-1, 1, out_basis.size(), dtype), out_basis
    )
    _, coefficients = _half_shuffle_coefficients(2, out_basis.depth)
    if side == "left":
        # Output j, fixed integrator i, free-tensor argument k.
        coefficients = coefficients[: arg_basis.size(), : op_basis.size(), :]

        def reference(a, b):
            return np.einsum("kij,...i,...k->...j", coefficients, a, b)

        expected_ct_op = np.einsum(
            "kij,...j,...k->...i", coefficients, ct.data, arg.data
        )
        expected_ct_arg = np.einsum(
            "kij,...i,...j->...k", coefficients, op.data, ct.data
        )
    else:
        # Output i, fixed integrand j, free-tensor argument k.
        coefficients = coefficients[: arg_basis.size(), :, : op_basis.size()]

        def reference(a, b):
            return np.einsum("kij,...j,...k->...i", coefficients, a, b)

        expected_ct_op = np.einsum(
            "kij,...i,...k->...j", coefficients, ct.data, arg.data
        )
        expected_ct_arg = np.einsum(
            "kij,...i,...j->...k", coefficients, ct.data, op.data
        )
    primal = getattr(rpj, f"left_hs_adjoint_{side}_mul")
    derivative = getattr(rpj, f"left_hs_adjoint_{side}_mul_derivative")
    adjoint_derivative = getattr(rpj, f"left_hs_adjoint_{side}_mul_adjoint_derivative")
    result = jax.jit(primal)(op, arg)
    tangent = jax.jit(derivative)(op, arg, t_op, t_arg)
    cotangents = jax.jit(adjoint_derivative)(op, arg, ct)
    _, pullback = jax.vjp(primal, op, arg)
    jax_cotangents = jax.jit(pullback)(rpj.FreeTensor(ct.data, ct.basis))
    np.testing.assert_allclose(
        result.data, reference(op.data, arg.data), atol=2e-6, rtol=2e-6
    )
    np.testing.assert_allclose(
        tangent.data,
        reference(t_op.data, arg.data) + reference(op.data, t_arg.data),
        atol=2e-6,
        rtol=2e-6,
    )
    assert isinstance(cotangents[0], rpj.DenseFreeTensor)
    assert isinstance(cotangents[1], rpj.DenseShuffleTensor)
    for c, jax_c, basis, expected in zip(
        cotangents,
        jax_cotangents,
        (op_basis, arg_basis),
        (expected_ct_op, expected_ct_arg),
        strict=True,
    ):
        assert c.basis == basis
        assert jax_c.basis == basis
        np.testing.assert_allclose(c.data, expected, atol=2e-6, rtol=2e-6)
        np.testing.assert_allclose(jax_c.data, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(
        np.sum(tangent.data * ct.data, axis=-1),
        np.sum(t_op.data * cotangents[0].data, axis=-1)
        + np.sum(t_arg.data * cotangents[1].data, axis=-1),
        atol=4e-6,
        rtol=4e-6,
    )


@pytest.mark.parametrize("name", ["mul", "adjoint_left_mul", "adjoint_right_mul"])
def test_half_shuffle_second_reverse_derivative(name, rpj_device):
    import roughpy_jax as rpj

    basis = TensorBasis(2, 2)
    _, coefficients = _half_shuffle_coefficients(2, 2)
    coefficients = jnp.asarray(coefficients)
    first = jax.device_put(
        jnp.linspace(-0.7, 0.9, basis.size(), dtype=jnp.float64), rpj_device
    )
    second = jax.device_put(
        jnp.linspace(0.8, -0.6, basis.size(), dtype=jnp.float64), rpj_device
    )
    primal = getattr(rpj, f"left_hs_{name}")
    arg_cls = rpj.ShuffleTensor if name == "mul" else rpj.FreeTensor
    indices = {
        "mul": "kij,i,j->k",
        "adjoint_left_mul": "kij,i,k->j",
        "adjoint_right_mul": "kij,j,k->i",
    }[name]

    def loss(a, b):
        result = primal(rpj.ShuffleTensor(a, basis), arg_cls(b, basis))
        return jnp.sum(result.data**2)

    def reference_loss(a, b):
        return jnp.sum(jnp.einsum(indices, coefficients, a, b) ** 2)

    # Reverse-over-reverse exercises differentiation through the new adjoint
    # VJPs, including the cross derivative between the two operands.
    second_derivative = jax.jit(jax.jacrev(jax.grad(loss, argnums=0), argnums=(0, 1)))
    reference_derivative = jax.jacrev(
        jax.grad(reference_loss, argnums=0), argnums=(0, 1)
    )
    for actual, expected in zip(
        second_derivative(first, second),
        reference_derivative(first, second),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("name", ["mul", "adjoint_left_mul", "adjoint_right_mul"])
@pytest.mark.parametrize("dtype", [jnp.dtype("float32"), jnp.dtype("float64")])
@pytest.mark.parametrize("depths", [(1, 3), (3, 1)])
def test_half_shuffle_numerical_derivatives_and_vjp(
    name, dtype, depths, rpj_small_batch, rpj_no_acceleration
):
    import roughpy_jax as rpj

    # Width three supplements the width-two combinatorial oracle. Distinct
    # random operands make swapping integrator and integrand observable.
    bases = tuple(TensorBasis(3, depth) for depth in depths)
    classes = (
        rpj.ShuffleTensor,
        rpj.ShuffleTensor if name == "mul" else rpj.FreeTensor,
    )

    def random_operands():
        return tuple(
            cls(rpj_small_batch.rng_uniform(-0.5, 0.5, basis.size(), dtype), basis)
            for cls, basis in zip(classes, bases, strict=True)
        )

    primals = random_operands()
    tangents = random_operands()
    other_tangents = random_operands()
    primal = getattr(rpj, f"left_hs_{name}")
    derivative = getattr(rpj, f"left_hs_{name}_derivative")
    adjoint_derivative = getattr(rpj, f"left_hs_{name}_adjoint_derivative")
    result = primal(*primals)
    compiled_primal = jax.jit(primal)
    math_ct_cls = rpj.FreeTensor if name == "mul" else rpj.ShuffleTensor
    cotangents = tuple(
        math_ct_cls(
            rpj_small_batch.rng_uniform(-0.5, 0.5, result.basis.size(), dtype),
            result.basis,
        )
        for _ in range(2)
    )
    step = 0.05 if dtype == jnp.dtype("float32") else 0.001
    tolerance = 5e-5 if dtype == jnp.dtype("float32") else 1e-10
    zero_tangents = tuple(
        cls(jnp.zeros_like(p.data), p.basis)
        for cls, p in zip(classes, primals, strict=True)
    )
    for direction in (
        (tangents[0], zero_tangents[1]),
        (zero_tangents[0], tangents[1]),
        tangents,
    ):
        plus = tuple(p + step * t for p, t in zip(primals, direction, strict=True))
        minus = tuple(p - step * t for p, t in zip(primals, direction, strict=True))
        finite_difference = (
            compiled_primal(*plus).data - compiled_primal(*minus).data
        ) / (2 * step)
        tangent = derivative(*primals, *direction)
        np.testing.assert_allclose(
            tangent.data, finite_difference, atol=tolerance, rtol=tolerance
        )
        adjoints = adjoint_derivative(*primals, cotangents[0])
        # Check the derivative/adjoint relation against numerical changes in
        # the primal rather than only against another derivative formula.
        np.testing.assert_allclose(
            np.sum(finite_difference * cotangents[0].data, axis=-1),
            sum(
                np.sum(t.data * c.data, axis=-1)
                for t, c in zip(direction, adjoints, strict=True)
            ),
            atol=tolerance,
            rtol=tolerance,
        )

    alpha, beta = 0.7, -1.3
    combined_tangents = tuple(
        alpha * t + beta * u for t, u in zip(tangents, other_tangents, strict=True)
    )
    np.testing.assert_allclose(
        derivative(*primals, *combined_tangents).data,
        alpha * derivative(*primals, *tangents).data
        + beta * derivative(*primals, *other_tangents).data,
        atol=tolerance,
        rtol=tolerance,
    )
    np.testing.assert_array_equal(
        derivative(*primals, *zero_tangents).data, jnp.zeros_like(result.data)
    )

    # Compile both the forward pass and backward rule together. Passing the
    # cotangent as a primal-shaped pytree exercises JAX's representation of
    # the mathematical dual, for each of the three operation signatures.
    def vjp(a, b, ct):
        value, pullback = jax.vjp(primal, a, b)
        return value, pullback(ct)

    compiled_vjp = jax.jit(vjp)
    expected_types = (
        rpj.DenseFreeTensor,
        rpj.DenseFreeTensor if name == "mul" else rpj.DenseShuffleTensor,
    )
    for ct in (*cotangents, alpha * cotangents[0] + beta * cotangents[1]):
        jax_ct = type(result)(ct.data, ct.basis)
        eager_result, eager_cts = vjp(*primals, jax_ct)
        compiled_result, compiled_cts = compiled_vjp(*primals, jax_ct)
        math_cts = adjoint_derivative(*primals, ct)
        np.testing.assert_allclose(
            compiled_result.data, eager_result.data, atol=tolerance, rtol=tolerance
        )
        for p, expected_type, math_ct, eager_ct, compiled_ct in zip(
            primals, expected_types, math_cts, eager_cts, compiled_cts, strict=True
        ):
            assert isinstance(math_ct, expected_type)
            assert type(eager_ct) is type(p)
            assert type(compiled_ct) is type(p)
            assert eager_ct.basis == compiled_ct.basis == math_ct.basis == p.basis
            np.testing.assert_allclose(
                eager_ct.data, math_ct.data, atol=tolerance, rtol=tolerance
            )
            np.testing.assert_allclose(
                compiled_ct.data, math_ct.data, atol=tolerance, rtol=tolerance
            )
    ct0, ct1 = (type(result)(ct.data, ct.basis) for ct in cotangents)
    _, pb0 = compiled_vjp(*primals, ct0)
    _, pb1 = compiled_vjp(*primals, ct1)
    _, pb_combined = compiled_vjp(*primals, alpha * ct0 + beta * ct1)
    _, pb_zero = compiled_vjp(
        *primals, type(result)(jnp.zeros_like(result.data), result.basis)
    )
    for c0, c1, combined, zero in zip(pb0, pb1, pb_combined, pb_zero, strict=True):
        np.testing.assert_allclose(
            combined.data,
            alpha * c0.data + beta * c1.data,
            atol=tolerance,
            rtol=tolerance,
        )
        np.testing.assert_array_equal(zero.data, jnp.zeros_like(zero.data))
