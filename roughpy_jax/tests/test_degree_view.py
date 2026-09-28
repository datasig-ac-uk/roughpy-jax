import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roughpy_jax as rpj
from roughpy_jax.algebra import _degree_bounds


@pytest.fixture(params=[rpj.DenseFreeTensor, rpj.DenseShuffleTensor])
def tensor(request):
    basis = rpj.TensorBasis(2, 3)
    data = jnp.arange(basis.size(), dtype=jnp.float32)
    return request.param(data, basis)


def test_degree_view_defaults_to_the_full_degree_range(tensor):
    view = rpj.degree_view(tensor)

    assert view.algebra is tensor
    assert view.min_degree == 0
    assert view.max_degree == tensor.basis.depth


def test_degree_view_accepts_an_explicit_degree_range(tensor):
    view = rpj.degree_view(tensor, min_degree=1, max_degree=2)

    assert view.min_degree == 1
    assert view.max_degree == 2


def test_degree_bounds_support_algebras_and_views(tensor):
    view = rpj.degree_view(tensor, min_degree=1, max_degree=2)

    assert _degree_bounds(tensor) == (0, tensor.basis.depth)
    assert _degree_bounds(view) == (1, 2)


def test_degree_view_replaces_the_range_of_an_existing_view(tensor):
    original = rpj.degree_view(tensor, min_degree=1, max_degree=2)

    view = rpj.degree_view(original, min_degree=2, max_degree=3)

    assert view.algebra is tensor
    assert view.min_degree == 2
    assert view.max_degree == 3


def test_degree_view_preserves_the_ambient_basis_and_data(tensor):
    view = rpj.degree_view(tensor, min_degree=1, max_degree=2)

    assert view.basis is tensor.basis
    assert view.data is tensor.data
    assert view.dtype == tensor.dtype
    assert view.batch_shape == tensor.batch_shape


def test_degree_view_supports_non_tensor_algebras():
    basis = rpj.LieBasis(2, 3)
    algebra = rpj.DenseLie(jnp.arange(basis.size()), basis)

    view = rpj.degree_view(algebra, min_degree=2)

    assert view.algebra is algebra
    assert view.basis is basis
    assert view.min_degree == 2
    assert view.max_degree == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_degree": -1}, "min_degree must be non-negative"),
        (
            {"min_degree": 2, "max_degree": 1},
            "max_degree must be greater than or equal to min_degree",
        ),
        ({"max_degree": 4}, "max_degree must not exceed the basis depth"),
    ],
)
def test_degree_view_rejects_invalid_ranges(tensor, kwargs, message):
    with pytest.raises(ValueError, match=message):
        rpj.degree_view(tensor, **kwargs)


def test_degree_bounds_are_static_pytree_metadata(tensor):
    view = rpj.degree_view(tensor, min_degree=1, max_degree=2)

    leaves = jax.tree.leaves(view)

    assert len(leaves) == 1
    assert leaves[0] is tensor.data


def test_degree_view_round_trips_through_jit(tensor):
    view = rpj.degree_view(tensor, min_degree=1, max_degree=2)

    result = jax.jit(lambda value: value)(view)

    assert isinstance(result, rpj.DegreeView)
    assert result.basis == tensor.basis
    assert result.min_degree == 1
    assert result.max_degree == 2
    np.testing.assert_array_equal(result.data, tensor.data)
