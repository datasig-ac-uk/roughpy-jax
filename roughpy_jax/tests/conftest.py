import jax
import jax.numpy as jnp
import numpy as np
import pytest

import roughpy_jax as rpj
from roughpy_jax.ops import Operation, get_supported_platforms

# Running both f32 and f64 tests requires enabling JAX 64 bit mode
jax.config.update("jax_enable_x64", True)


def _available_test_platforms():
    platforms = []
    for platform in sorted(get_supported_platforms()):
        try:
            if jax.devices(platform):
                platforms.append(platform)
        except RuntimeError:
            continue
    return platforms


@pytest.fixture(params=_available_test_platforms())
def rpj_platform(request):
    return request.param


@pytest.fixture
def rpj_device(rpj_platform):
    devices = jax.devices(rpj_platform)
    if not devices:
        pytest.skip(f"no {rpj_platform} device available")
    return devices[0]


@pytest.fixture
def rpj_test_fixture_type_mismatch(rpj_device):
    """
    Unit test fixture for generating dummy arrays of given type. Reduces code
    duplication when writing type mismatch tests.
    """

    class BasisFixture:
        def ft_f32(self, basis_width: int = 2, basis_depth: int = 2, **kwargs):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            kwargs.setdefault("device", rpj_device)
            return rpj.FreeTensor.zero(basis, dtype=jnp.float32, **kwargs)

        def ft_f64(self, basis_width: int = 2, basis_depth: int = 2, **kwargs):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            kwargs.setdefault("device", rpj_device)
            return rpj.FreeTensor.zero(basis, dtype=jnp.float64, **kwargs)

        def ft_i32(self, basis_width: int = 2, basis_depth: int = 2, **kwargs):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            kwargs.setdefault("device", rpj_device)
            return rpj.FreeTensor.zero(basis, dtype=jnp.int32, **kwargs)

        def st_f32(self, basis_width: int = 2, basis_depth: int = 2, **kwargs):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            kwargs.setdefault("device", rpj_device)
            return rpj.ShuffleTensor.zero(basis, dtype=jnp.float32, **kwargs)

        def st_f64(self, basis_width: int = 2, basis_depth: int = 2, **kwargs):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            kwargs.setdefault("device", rpj_device)
            return rpj.ShuffleTensor.zero(basis, dtype=jnp.float64, **kwargs)

        def st_i32(self, basis_width: int = 2, basis_depth: int = 2, **kwargs):
            basis = rpj.TensorBasis(basis_width, basis_depth)
            kwargs.setdefault("device", rpj_device)
            return rpj.ShuffleTensor.zero(basis, dtype=jnp.int32, **kwargs)

    return BasisFixture()


class BatchFixtureHelper:
    """
    This is parameterised by rpj_batch to provide helper methods to tests working
    with batched tensors. batch_shape is the prefix to the actual tensor shape; a
    null batch_shape () represents a single tensor. For example, given an
    rpj.TensorBasis with width 2, depth 2 (resulting in a data size of 7) and a
    null batch_shape of () then tensor_batch_shape would be (7,), but given a 2D
    batch_shape of (3,2) then tensor_batch_shape would be (3,2,7).

    Example usage:

        def test_xs(rpj_batch):
            data = jnp.zeros(20)
            batched_data = rpj_batch.repeat(data)
            assert batched_data.shape[:-1] == rpj_batch.shape
    """

    def __init__(self, batch_shape, device):
        self.rng = np.random.default_rng(1234)
        self.shape = batch_shape
        self.device = device

    def __repr__(self):
        return str(self.shape)

    def __str__(self):
        return str(self.shape)

    def tensor_batch_shape(self, basis):
        return (*self.shape, basis.size())

    def zeros(self, num, dtype, **kwargs):
        device = kwargs.pop("device", self.device)
        data = jnp.zeros(num, dtype=dtype, device=device, **kwargs)
        return self.repeat(data, device=device)

    def repeat(self, xs, **kwargs):
        device = kwargs.pop("device", self.device)
        data = jax.device_put(jnp.asarray(xs), device)
        return jnp.tile(data, (*self.shape, 1))

    def rng_uniform(self, min, max, num, dtype, **kwargs):
        device = kwargs.pop("device", self.device)
        data = self.rng.uniform(min, max, (*self.shape, num)).astype(dtype)
        return jax.device_put(jnp.asarray(data, dtype=dtype, **kwargs), device)

    def rng_nonzero_free_tensor(self, basis, dtype, **kwargs):
        """
        Several tests need to operate on non-zero values otherwise exp/log will
        end up always being zero and tests are not doing anything useful. Only the
        vector part (directly after scalar part, hence [1:width+1]) needs to be set
        to ensure the overall value does not collapse to zero.
        """
        # Built using np not jnp for easy mutability
        data = np.zeros(self.tensor_batch_shape(basis), dtype)
        data[..., 1: basis.width + 1] = self.rng.normal(
            size=(*self.shape, basis.width)
        )
        device = kwargs.pop("device", self.device)
        jax_data = jax.device_put(jnp.asarray(data, dtype=dtype, **kwargs), device)
        return rpj.FreeTensor(jax_data, basis)

    def rng_shuffle_tensor(self, basis, dtype, **kwargs):
        device = kwargs.pop("device", self.device)
        data = self.rng_uniform(-1.0, 1.0, basis.size(), dtype, device=device, **kwargs)
        return rpj.ShuffleTensor(data, basis)

    def identity_zero_data(self, basis, dtype, **kwargs):
        kwargs.setdefault("device", self.device)
        return rpj.FreeTensor.identity(
            basis, dtype=dtype, batch_dims=self.shape, **kwargs
        ).data


@pytest.fixture()
def rpj_nobatch(rpj_device):
    """
    Fixture for tests that do not require batching. Provides helper methods for
    creating non-batched tensors.
    """
    return BatchFixtureHelper((), rpj_device)


# Batching test fixture returns helper class for common operations
@pytest.fixture(params=[(), (2,), (3, 2), (2, 2, 2)], ids=str)
def rpj_batch(request, rpj_device):
    return BatchFixtureHelper(request.param, rpj_device)


# Minimised batch sizes for slower tests; test non-batched and small 2D batch
@pytest.fixture(params=[(), (2, 2)])
def rpj_small_batch(request, rpj_device):
    return BatchFixtureHelper(request.param, rpj_device)


# Data type test fixture
@pytest.fixture(
    params=[pytest.param(jnp.float32, marks=pytest.mark.extra), jnp.float64]
)
def rpj_dtype(request):
    return request.param


# Toggle operation between running with and without acceleration for testing CPU vs fallback code
@pytest.fixture(
    params=[
        pytest.param(False, id="default accel"),
        pytest.param(True, marks=pytest.mark.extra, id="no accel"),
    ]
)
def rpj_no_acceleration(request, rpj_platform):
    if request.param and rpj_platform != "cpu":
        pytest.skip("fallback-only mode is only exercised on cpu")
    old_setting = Operation.no_acceleration
    Operation.no_acceleration = request.param
    yield request.param
    Operation.no_acceleration = old_setting
