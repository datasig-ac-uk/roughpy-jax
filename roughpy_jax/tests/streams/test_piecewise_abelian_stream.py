import jax
import jax.numpy as jnp
import pytest
import roughpy_jax as rpj
from roughpy_jax.algebra import FreeTensor, ft_fmexp, lie_to_tensor, to_log_signature
from roughpy_jax.intervals import IntervalType, Partition, RealInterval
from roughpy_jax.streams import PiecewiseAbelianStream
from roughpy_jax.streams.piecewise_abelian_stream import to_piecewise_abelian_stream


class PASHelper:
    def __init__(self, rpj_batch, rpj_dtype):
        # Create a simple piecewise abelian stream with two intervals
        # [0, 1] and [1, 2] with corresponding Lie elements L1 and L2
        self.interval = RealInterval(0.0, 2.0, IntervalType.ClOpen)
        self.partition = Partition([0.0, 1.0, 2.0], IntervalType.ClOpen)

        # Make some Lie elements for the stream (we can just use random data for this test)
        self.lie_basis = rpj.LieBasis(2, 2)
        self.tensor_basis = rpj.to_tensor_basis(self.lie_basis)

        self.l1_data = rpj_batch.rng_uniform(-1, 1, self.lie_basis.size(), rpj_dtype)
        self.l1 = rpj.Lie(self.l1_data, self.lie_basis)
        self.l2_data = rpj_batch.rng_uniform(-1, 1, self.lie_basis.size(), rpj_dtype)
        self.l2 = rpj.Lie(self.l2_data, self.lie_basis)

        # Create the piecewise abelian stream
        self.stream = PiecewiseAbelianStream(
            jnp.stack((self.l1_data, self.l2_data)),
            self.partition,
            self.lie_basis,
            rpj.to_tensor_basis(self.lie_basis),
        )

    @property
    def dtype(self):
        return self.l1_data.dtype

    def batch_shape(self):
        return self.l1_data.shape[:-1]


@pytest.fixture
def pas_data(rpj_batch, rpj_dtype):
    return PASHelper(rpj_batch, rpj_dtype)


class TestPiecewiseAbelianStream:
    def test_construction(self, pas_data):
        """Test that the PiecewiseAbelianStream can be constructed without errors."""
        with pytest.raises(ValueError, match="Data length"):
            PiecewiseAbelianStream(
                pas_data.l1_data[None, ...],  # Incorrect piece dimension
                pas_data.partition,
                pas_data.lie_basis,
                rpj.to_tensor_basis(pas_data.lie_basis),
            )

    def test_construction_validates_normalized_data_shape(self):
        lie_basis = rpj.LieBasis(2, 2)
        group_basis = rpj.to_tensor_basis(lie_basis)
        partition = Partition([0.0, 1.0, 2.0], IntervalType.ClOpen)

        with pytest.raises(ValueError, match="must have shape"):
            PiecewiseAbelianStream(
                jnp.zeros((lie_basis.size(),)),
                partition,
                lie_basis,
                group_basis,
            )

        with pytest.raises(ValueError, match="Data Lie dimension"):
            PiecewiseAbelianStream(
                jnp.zeros((len(partition), lie_basis.size() + 1)),
                partition,
                lie_basis,
                group_basis,
            )

    def test_construction_rejects_batched_partition(self):
        lie_basis = rpj.LieBasis(2, 2)
        partition = Partition(
            [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]],
            IntervalType.ClOpen,
        )

        with pytest.raises(ValueError, match="batched partition"):
            PiecewiseAbelianStream(
                jnp.zeros((len(partition), lie_basis.size())),
                partition,
                lie_basis,
                rpj.to_tensor_basis(lie_basis),
            )

    @pytest.mark.parametrize(
        "group_basis",
        [rpj.TensorBasis(3, 2), rpj.TensorBasis(2, 3)],
    )
    def test_construction_rejects_incompatible_bases(self, group_basis):
        lie_basis = rpj.LieBasis(2, 2)
        partition = Partition([0.0, 1.0, 2.0], IntervalType.ClOpen)

        with pytest.raises(ValueError, match="Incompatible width"):
            PiecewiseAbelianStream(
                jnp.zeros((len(partition), lie_basis.size())),
                partition,
                lie_basis,
                group_basis,
            )

    def test_log_signature(self, pas_data):
        """Test the PiecewiseAbelianStream class."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)

        # Compute log signature over [0, 1]
        log_sig = pas_data.stream.log_signature(query_interval)

        # Check that it equals L1 (which is l1 in this case)
        assert jnp.allclose(log_sig.data, pas_data.l1.data, atol=1e-6)

    @pytest.mark.parametrize(
        "query_interval",
        [
            RealInterval(0.0, 1.0, IntervalType.ClOpen),
            RealInterval(1.0, 2.0, IntervalType.ClOpen),
            RealInterval(0.5, 1.5, IntervalType.ClOpen),
            RealInterval(0.0, 2.0, IntervalType.ClOpen),
            RealInterval(-0.5, 2.5, IntervalType.ClOpen),
            RealInterval(2.0, 3.0, IntervalType.ClOpen),
            RealInterval(-2.0, -1.0, IntervalType.ClOpen),
        ],
    )
    def test_log_signature_various_intervals(self, pas_data, query_interval):
        """Test log signature over various query intervals."""
        pas_data.stream.log_signature(query_interval)

    def test_signature(self, pas_data):
        """Test that the signature of the stream over [0, 1]."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)
        pas_data.stream.signature(query_interval)

    def test_log_signature_accepts_singleton_array_interval_endpoints(self, pas_data):
        query_interval = RealInterval(
            jnp.array([0.5], dtype=pas_data.dtype),
            jnp.array([1.5], dtype=pas_data.dtype),
            IntervalType.ClOpen,
        )

        log_sig = pas_data.stream.log_signature(query_interval)
        expected_log_sig = pas_data.stream.log_signature(
            RealInterval(0.5, 1.5, IntervalType.ClOpen)
        )

        assert jnp.allclose(log_sig.data, expected_log_sig.data, atol=1e-6)

    def test_query_endpoints_broadcast_before_stream_batch_dimensions(self, pas_data):
        query_interval = RealInterval(
            jnp.array([[0.0], [0.5]], dtype=pas_data.dtype),
            jnp.array([[0.5, 1.5, 2.0]], dtype=pas_data.dtype),
            IntervalType.ClOpen,
        )

        actual = pas_data.stream.log_signature(query_interval)
        expected_shape = (2, 3, *pas_data.batch_shape(), pas_data.lie_basis.size())

        assert actual.data.shape == expected_shape
        for i in range(2):
            for j in range(3):
                scalar_interval = RealInterval(
                    query_interval.inf[i, 0],
                    query_interval.sup[0, j],
                    query_interval.interval_type,
                )
                expected = pas_data.stream.log_signature(scalar_interval)
                assert jnp.allclose(actual.data[i, j], expected.data, atol=1e-6)

    def test_log_signature_cbh(self, pas_data):
        """Test that the log signature of the stream over [0.5, 1.5] is CBH(0.5*L1, 0.5*L2)."""

        query_interval = RealInterval(0.5, 1.5, IntervalType.ClOpen)
        log_sig = pas_data.stream.log_signature(query_interval)

        expected_log_sig = rpj.cbh(0.5 * pas_data.l1, 0.5 * pas_data.l2)

        assert jnp.allclose(log_sig.data, expected_log_sig.data, atol=1e-6)

    def test_log_signature_multi_piece_stream(self):
        lie_basis = rpj.LieBasis(26, 3)
        tensor_basis = rpj.to_tensor_basis(lie_basis)
        indices = (0, 1, 4, 19)

        def make_lie(index):
            data = jnp.zeros((lie_basis.size(),), dtype=jnp.float32)
            data = data.at[index].set(1.0)
            return rpj.Lie(data, lie_basis)

        lies = tuple(make_lie(index) for index in indices)
        stream = PiecewiseAbelianStream(
            jnp.stack([lie.data for lie in lies]),
            Partition([0.0, 1.0, 2.0, 3.0, 4.0], IntervalType.ClOpen),
            lie_basis,
            tensor_basis,
        )
        query_interval = RealInterval(0.0, 4.0, IntervalType.ClOpen)

        actual_log_sig = stream.log_signature(query_interval)

        expected_signature = FreeTensor.identity(tensor_basis, dtype=jnp.float32)
        for lie in lies:
            expected_signature = ft_fmexp(
                expected_signature,
                lie_to_tensor(lie),
                tensor_basis,
            )
        expected_log_sig = to_log_signature(expected_signature)

        assert jnp.allclose(actual_log_sig.data, expected_log_sig.data, atol=1e-6)

    def test_stream_metadata(self, pas_data):
        """Test that the stream exposes dtype and batch metadata."""
        assert pas_data.stream.dtype == pas_data.dtype
        assert pas_data.stream.batch_dims == pas_data.batch_shape()

    def test_support(self, pas_data):
        """Test that the support interval is correct."""
        support = pas_data.stream.support

        assert support.inf == pas_data.partition.inf
        assert support.sup == pas_data.partition.sup
        assert support.interval_type == pas_data.partition.interval_type

    def test_to_piecewise_abelian_stream(self, pas_data):
        """Conversion queries all partition pieces as one batched interval."""
        partition = Partition(
            [0.0, 0.5, 1.0, 1.5, 2.0],
            IntervalType.ClOpen,
        )

        converted = to_piecewise_abelian_stream(pas_data.stream, partition)
        query_intervals = partition.to_intervals()

        expected = pas_data.stream.log_signature(query_intervals)
        actual = converted.log_signature(query_intervals)

        assert converted._data.shape == expected.data.shape
        assert actual.data.shape == expected.data.shape
        assert jnp.allclose(actual.data, expected.data, atol=1e-6)

    @pytest.mark.extra
    @pytest.mark.parametrize("static", [True, False])
    def test_jitness_log_signature(self, pas_data, static):
        """Test that the log signature can be JIT compiled."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)

        # JIT compile the log signature method, keeping the query interval as a
        # static argument
        jit_log_sig = jax.jit(
            pas_data.stream.log_signature, static_argnums=(0,) if static else ()
        )
        log_sig = jit_log_sig(query_interval)

        # Check that we're producing the same result as the non-JIT version
        non_jit_log_sig = pas_data.stream.log_signature(query_interval)
        assert jnp.allclose(log_sig.data, non_jit_log_sig.data, atol=1e-6)
        # Check that the log signature equals L1
        assert jnp.allclose(log_sig.data, pas_data.l1.data, atol=1e-6)


@pytest.mark.extra
class TestPiecewiseAbelianStreamBench:
    def test_log_signature_bench(self, benchmark, pas_data):
        """Benchmark the log signature computation."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)

        benchmark(pas_data.stream.log_signature, query_interval)

    def test_log_signature_jit_static_bench(self, benchmark, pas_data):
        """Benchmark the JIT-compiled log signature computation."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)

        jit_log_sig = jax.jit(pas_data.stream.log_signature, static_argnums=(0,))

        benchmark(jit_log_sig, query_interval)

    def test_log_signature_jit_dynamic_bench(self, benchmark, pas_data):
        """Benchmark the JIT-compiled log signature computation with dynamic query interval."""
        query_interval = RealInterval(0.0, 1.0, IntervalType.ClOpen)

        jit_log_sig = jax.jit(pas_data.stream.log_signature, static_argnums=())

        benchmark(jit_log_sig, query_interval)
