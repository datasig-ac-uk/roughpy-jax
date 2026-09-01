import jax.numpy as jnp
import pytest

import roughpy_jax as rpj
from roughpy_jax.algebra import LieBasis, TensorBasis
from roughpy_jax.intervals import IntervalType, Partition, RealInterval
from roughpy_jax.streams import PiecewiseAbelianStream
from roughpy_jax.streams.lie_increment_stream import LieIncrementStream


def test_constructor_rejects_opencl_cache():
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.zeros((4, lie_basis.size()), dtype=jnp.float32)

    with pytest.raises(ValueError, match="only supports ClOpen dyadic caches"):
        LieIncrementStream(
            cache,
            lie_basis,
            resolution=1,
            interval_type=IntervalType.OpenCl,
        )


def test_from_stream_rejects_nonpositive_resolution():
    class DummyStream:
        lie_basis = LieBasis(width=1, depth=1)
        group_basis = TensorBasis(width=1, depth=1)
        support = RealInterval(0.0, 1.0, IntervalType.ClOpen)

        def log_signature(self, interval):
            pass

        def signature(self, interval):
            pass

    with pytest.raises(ValueError, match="resolution must be positive"):
        LieIncrementStream.from_stream(DummyStream(), resolution=0)


def test_from_stream_uses_stream_dyadic_cache_provider():
    class DummyStream:
        def __init__(self):
            self.lie_basis = LieBasis(width=1, depth=1)
            self.group_basis = TensorBasis(width=1, depth=1)
            self.support = RealInterval(0.0, 1.0, IntervalType.ClOpen)

        def log_signature(self, interval):
            pass

        def signature(self, interval):
            pass

        def __dyadic_cache__(self, resolution: int):
            return jnp.zeros(
                (1 << (resolution + 1), self.lie_basis.size()), dtype=jnp.float32
            )

    src = DummyStream()
    result = LieIncrementStream.from_stream(src, resolution=4)

    assert isinstance(result, LieIncrementStream)
    assert result.resolution == 4
    assert result.support == src.support
    assert result.lie_basis == src.lie_basis
    assert result.group_basis == src.group_basis
    assert result.__base_stream__ is src

    log_sig = result.log_signature(src.support)
    sig = result.signature(src.support)

    assert jnp.allclose(log_sig.data, rpj.Lie.zero(src.lie_basis).data)
    assert jnp.allclose(sig.data, rpj.FreeTensor.identity(src.group_basis).data)


def test_from_stream_falls_back_to_stream_to_cache(monkeypatch):
    class DummyStream:
        def __init__(self):
            self.lie_basis = LieBasis(width=1, depth=1)
            self.group_basis = TensorBasis(width=1, depth=1)
            self.support = RealInterval(0.0, 1.0, IntervalType.ClOpen)

        def log_signature(self, interval):
            pass

        def signature(self, interval):
            pass

    src = DummyStream()
    captured = {}

    def fake_stream_to_cache(stream, resolution, interval_type=IntervalType.ClOpen):
        captured["stream"] = stream
        captured["resolution"] = resolution
        captured["interval_type"] = interval_type
        return jnp.zeros(
            (1 << (resolution + 1), src.lie_basis.size()), dtype=jnp.float32
        )

    monkeypatch.setattr(
        LieIncrementStream, "_stream_to_cache", staticmethod(fake_stream_to_cache)
    )

    result = LieIncrementStream.from_stream(src, resolution=3)
    assert isinstance(result, LieIncrementStream)
    assert result.resolution == 3
    assert captured["stream"] is src
    assert captured["resolution"] == 3
    assert captured["interval_type"] == IntervalType.ClOpen

    log_sig = result.log_signature(src.support)
    sig = result.signature(src.support)

    assert jnp.allclose(log_sig.data, rpj.Lie.zero(src.lie_basis).data)
    assert jnp.allclose(sig.data, rpj.FreeTensor.identity(src.group_basis).data)


def test_from_stream_constructs_equivalent_lie_increment_stream():
    lie_basis = LieBasis(width=2, depth=2)
    group_basis = rpj.to_tensor_basis(lie_basis)
    partition = Partition([0.0, 1.0, 2.0], IntervalType.ClOpen)

    l1 = rpj.Lie(jnp.array([0.0, 0.3, -0.2], dtype=jnp.float64), lie_basis)
    l2 = rpj.Lie(jnp.array([0.0, -0.1, 0.4], dtype=jnp.float64), lie_basis)
    src = PiecewiseAbelianStream(
        _data=(l1, l2),
        _partition=partition,
        _lie_basis=lie_basis,
        _group_basis=group_basis,
    )

    result = LieIncrementStream.from_stream(src, resolution=3)

    assert isinstance(result, LieIncrementStream)
    assert result.resolution == 3
    assert result.support == src.support
    assert result.lie_basis == src.lie_basis
    assert result.group_basis == src.group_basis
    assert result.__base_stream__ is src

    intervals = [
        RealInterval(0.0, 1.0, IntervalType.ClOpen),
        RealInterval(1.0, 2.0, IntervalType.ClOpen),
        RealInterval(0.5, 1.5, IntervalType.ClOpen),
        RealInterval(0.0, 2.0, IntervalType.ClOpen),
    ]
    for interval in intervals:
        expected = src.log_signature(interval)
        actual = result.log_signature(interval)
        assert jnp.allclose(actual.data, expected.data, atol=1e-6)

        expected_sig = src.signature(interval)
        actual_sig = result.signature(interval)
        assert jnp.allclose(actual_sig.data, expected_sig.data, atol=1e-6)


def test_log_signature_accepts_singleton_array_interval_endpoints():
    lie_basis = LieBasis(width=2, depth=2)
    group_basis = rpj.to_tensor_basis(lie_basis)
    partition = Partition([0.0, 1.0, 2.0], IntervalType.ClOpen)

    l1 = rpj.Lie(jnp.array([0.0, 0.3, -0.2], dtype=jnp.float64), lie_basis)
    l2 = rpj.Lie(jnp.array([0.0, -0.1, 0.4], dtype=jnp.float64), lie_basis)
    src = PiecewiseAbelianStream(
        _data=(l1, l2),
        _partition=partition,
        _lie_basis=lie_basis,
        _group_basis=group_basis,
    )
    stream = LieIncrementStream.from_stream(src, resolution=3)

    query = RealInterval(
        jnp.array([0.5], dtype=jnp.float64),
        jnp.array([1.5], dtype=jnp.float64),
        IntervalType.ClOpen,
    )

    actual_log = stream.log_signature(query)
    actual_sig = stream.signature(query)

    expected_query = RealInterval(0.5, 1.5, IntervalType.ClOpen)
    expected_log = stream.log_signature(expected_query)
    expected_sig = stream.signature(expected_query)

    assert jnp.allclose(actual_log.data, expected_log.data, atol=1e-6)
    assert jnp.allclose(actual_sig.data, expected_sig.data, atol=1e-6)


def test_batched_log_signature_and_signature_match_individual_queries():
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.zeros((16, lie_basis.size()), dtype=jnp.float32)
    cache = cache.at[:8, 0].set(jnp.arange(1, 9, dtype=jnp.float32))
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(0.0, 1.0, IntervalType.ClOpen),
        resolution=3,
    )

    query = RealInterval(
        jnp.array([0.0, 0.25, 0.5], dtype=jnp.float32),
        jnp.array([0.25, 0.75, 1.0], dtype=jnp.float32),
        IntervalType.ClOpen,
    )

    batched_log = stream.log_signature(query)
    batched_sig = stream.signature(query)

    for i in range(3):
        scalar_query = RealInterval(query.inf[i], query.sup[i], query.interval_type)
        assert jnp.allclose(
            batched_log.data[i], stream.log_signature(scalar_query).data
        )
        assert jnp.allclose(
            batched_sig.data[i], stream.signature(scalar_query).data
        )


def test_query_batch_dimensions_precede_stream_batch_dimensions():
    lie_basis = LieBasis(width=1, depth=1)
    resolution = 2
    stream_batch_dims = (2, 3)
    finest = jnp.arange(
        1,
        1 + (1 << resolution) * 2 * 3,
        dtype=jnp.float32,
    ).reshape(1 << resolution, *stream_batch_dims, lie_basis.size())
    next_level = finest.reshape(2, 2, *stream_batch_dims, 1).sum(axis=1)
    root = next_level.sum(axis=0, keepdims=True)
    zero = jnp.zeros((1, *stream_batch_dims, 1), dtype=jnp.float32)
    cache = jnp.concatenate((finest, next_level, root, zero), axis=0)
    stream = LieIncrementStream(cache, lie_basis, resolution=resolution)

    bucket_indices = jnp.array([[0, 1], [2, 3]], dtype=jnp.int32)
    query = RealInterval(
        bucket_indices / float(1 << resolution),
        (bucket_indices + 1) / float(1 << resolution),
        IntervalType.ClOpen,
    )

    result = stream.log_signature(query)

    assert result.data.shape == (2, 2, *stream_batch_dims, lie_basis.size())
    assert jnp.allclose(result.data, finest[bucket_indices])


def test_log_signature_without_interval_uses_support():
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.array([[1.0], [2.0], [3.0], [0.0]], dtype=jnp.float32)
    support = RealInterval(1.0, 3.0, IntervalType.ClOpen)
    stream = LieIncrementStream(cache, lie_basis, support=support, resolution=1)

    result = stream.log_signature()

    assert jnp.allclose(result.data, jnp.asarray([3.0], dtype=jnp.float32))


def test_log_signature_of_empty_interval_returns_zero():
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.array([[1.0], [2.0], [3.0], [0.0]], dtype=jnp.float32)
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(0.0, 1.0, IntervalType.ClOpen),
        resolution=1,
    )

    result = stream.log_signature(RealInterval(0.5, 0.5, IntervalType.ClOpen))

    assert jnp.allclose(result.data, rpj.Lie.zero(lie_basis).data)


@pytest.mark.parametrize("interval_type", list(IntervalType))
def test_log_signature_outside_support_returns_zero(interval_type):
    lie_basis = LieBasis(width=1, depth=1)
    stream_batch_dims = (2, 3)
    cache = jnp.ones((4, *stream_batch_dims, lie_basis.size()), dtype=jnp.float32)
    cache = cache.at[-1].set(0.0)
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(2.0, 4.0, IntervalType.ClOpen),
        resolution=1,
    )
    query = RealInterval(
        jnp.array([-2.0, 4.0, 5.0], dtype=jnp.float32),
        jnp.array([-1.0, 5.0, 8.0], dtype=jnp.float32),
        interval_type,
    )

    result = stream.log_signature(query)

    assert result.data.shape == (3, *stream_batch_dims, lie_basis.size())
    assert jnp.allclose(result.data, jnp.zeros_like(result.data))


@pytest.mark.parametrize(
    ("query", "clipped", "expected_data"),
    [
        (
            RealInterval(0.0, 2.1, IntervalType.ClOpen),
            RealInterval(1.0, 2.1, IntervalType.ClOpen),
            7.0,
        ),
        (
            RealInterval(1.9, 4.0, IntervalType.ClOpen),
            RealInterval(1.9, 3.0, IntervalType.ClOpen),
            12.0,
        ),
    ],
)
def test_log_signature_partially_overlapping_support_is_clipped(
    query,
    clipped,
    expected_data,
):
    lie_basis = LieBasis(width=1, depth=1)
    finest = jnp.array([[1.0], [2.0], [4.0], [8.0]], dtype=jnp.float32)
    cache = jnp.concatenate(
        (
            finest,
            jnp.array([[3.0], [12.0]], dtype=jnp.float32),
            jnp.array([[15.0], [0.0]], dtype=jnp.float32),
        ),
        axis=0,
    )
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(1.0, 3.0, IntervalType.ClOpen),
        resolution=2,
    )

    actual = stream.log_signature(query)
    expected = stream.log_signature(clipped)

    assert jnp.allclose(actual.data, expected.data)
    assert jnp.allclose(actual.data, jnp.asarray([expected_data]))


@pytest.mark.parametrize(
    ("inf", "sup", "expected"),
    [
        (0.100, 0.120, 0.0),
        (0.100, 0.125, 0.0),
        (0.125, 0.200, 2.5),
        (0.125, 0.130, 2.5),
        (0.249, 0.251, -1.25),
    ],
)
def test_log_signature_of_short_interval_uses_contained_endpoint(
    inf,
    sup,
    expected,
):
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.zeros((16, lie_basis.size()), dtype=jnp.float32)
    cache = cache.at[1, 0].set(2.5)
    cache = cache.at[2, 0].set(-1.25)
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(0.0, 1.0, IntervalType.ClOpen),
        resolution=3,
    )

    result = stream.log_signature(RealInterval(inf, sup, IntervalType.ClOpen))

    assert jnp.allclose(result.data, jnp.asarray([expected], dtype=jnp.float32))


def test_opencl_query_excludes_aligned_clopen_cache_endpoint():
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.zeros((16, lie_basis.size()), dtype=jnp.float32)
    cache = cache.at[2, 0].set(2.5)
    cache = cache.at[3, 0].set(-1.25)
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(0.0, 1.0, IntervalType.ClOpen),
        resolution=3,
    )

    result = stream.log_signature(
        RealInterval(0.25, 0.5, IntervalType.OpenCl)
    )

    assert jnp.allclose(result.data, jnp.asarray([-1.25], dtype=jnp.float32))


def test_opencl_query_does_not_nudge_unaligned_endpoint():
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.zeros((16, lie_basis.size()), dtype=jnp.float32)
    cache = cache.at[2, 0].set(2.5)
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(0.0, 1.0, IntervalType.ClOpen),
        resolution=3,
    )

    result = stream.log_signature(
        RealInterval(0.249, 0.251, IntervalType.OpenCl)
    )

    assert jnp.allclose(result.data, jnp.asarray([2.5], dtype=jnp.float32))


def test_batched_opencl_queries_match_inward_nudged_clopen_queries():
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.zeros((16, lie_basis.size()), dtype=jnp.float32)
    cache = cache.at[:8, 0].set(jnp.arange(1, 9, dtype=jnp.float32))
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(0.0, 1.0, IntervalType.ClOpen),
        resolution=3,
    )

    opencl = stream.log_signature(
        RealInterval(
            jnp.asarray([0.25, 0.20, 0.50]),
            jnp.asarray([0.75, 0.60, 0.55]),
            IntervalType.OpenCl,
        )
    )
    nudged_clopen = stream.log_signature(
        RealInterval(
            jnp.asarray([0.375, 0.20, 0.625]),
            jnp.asarray([0.75, 0.60, 0.55]),
            IntervalType.ClOpen,
        )
    )

    assert jnp.allclose(opencl.data, nudged_clopen.data)


def _build_l_shape_stream(t0, t1, increments):
    """Build a width-2 depth-2 LieIncrementStream from a 2D increment path.

    Timestamps are a linspace over ``[t0, t1]`` with a trailing zero increment.
    """
    lie_basis = LieBasis(width=2, depth=2)
    rows = list(increments) + [[0.0, 0.0]]
    ts = jnp.linspace(t0, t1, len(rows), dtype=jnp.float64)
    data = jnp.asarray(rows, dtype=jnp.float64)
    return LieIncrementStream.from_increments(
        timestamps=ts,
        data=data,
        resolution=3,
        input_data_basis=None,
        lie_basis=lie_basis,
        interval_type=IntervalType.ClOpen,
        time_dtype=jnp.float64.dtype,
    )


def test_from_increments_automatic_resolution_preserves_finest_level_data():
    lie_basis = LieBasis(width=1, depth=1)
    timestamps = jnp.array([1.0, 0.0, 0.75, 0.25, 0.5], dtype=jnp.float32)
    data = jnp.array([[5.0], [1.0], [4.0], [2.0], [3.0]], dtype=jnp.float32)

    stream = LieIncrementStream.from_increments(
        timestamps=timestamps,
        data=data,
        resolution=None,
        input_data_basis=None,
        lie_basis=lie_basis,
    )

    normalised_timestamps = (
        timestamps.astype(jnp.float32) - stream.support.inf
    ) / (stream.support.sup - stream.support.inf)
    buckets = jnp.floor(
        jnp.ldexp(normalised_timestamps, stream.resolution)
    ).astype(jnp.int32)
    finest = stream._cache[: 1 << stream.resolution, :]

    assert stream.batch_dims == ()
    assert jnp.unique(buckets).size == timestamps.size
    assert jnp.allclose(finest[buckets], data)


def test_from_increments_recovers_analytic_levy_area_on_nonunit_support():
    """Timestamps must be normalised onto the unit interval before bucketing.

    For an L-shaped path the width-2 depth-2 log-signature is
    ``[dx, dy, area]`` with signed Levy area +/-1/2.
    """
    right_then_up = _build_l_shape_stream(0.0, 2.0, [[1.0, 0.0], [0.0, 1.0]])
    up_then_right = _build_l_shape_stream(0.0, 2.0, [[0.0, 1.0], [1.0, 0.0]])

    rt = right_then_up.log_signature(right_then_up.support).data
    ur = up_then_right.log_signature(up_then_right.support).data

    assert jnp.allclose(rt, jnp.array([1.0, 1.0, 0.5]), atol=1e-6)
    assert jnp.allclose(ur, jnp.array([1.0, 1.0, -0.5]), atol=1e-6)
    # Order sensitivity: swapping the two increments flips the Levy area sign.
    assert not jnp.allclose(rt, ur, atol=1e-6)


def test_from_increments_subinterval_satisfies_chen_identity():
    lie_basis = LieBasis(width=2, depth=2)
    timestamps = jnp.linspace(0.0, 1.0, 9, dtype=jnp.float32)
    data = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.0, -0.25],
            [-0.5, 0.0],
            [0.0, 0.75],
            [0.25, 0.0],
            [0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    stream = LieIncrementStream.from_increments(
        timestamps=timestamps,
        data=data,
        resolution=4,
        input_data_basis=None,
        lie_basis=lie_basis,
    )
    support_length = stream.support.sup - stream.support.inf

    inf = stream.support.inf + 0.13 * support_length
    split = stream.support.inf + 0.53 * support_length
    sup = stream.support.inf + 0.89 * support_length
    whole = stream.log_signature(
        RealInterval(inf, sup, IntervalType.ClOpen)
    )
    left = stream.log_signature(
        RealInterval(inf, split, IntervalType.ClOpen)
    )
    right = stream.log_signature(
        RealInterval(split, sup, IntervalType.ClOpen)
    )

    combined = rpj.cbh(left, right, lie_basis=lie_basis)

    assert jnp.allclose(whole.data, combined.data, atol=1e-6)


def test_from_increments_is_invariant_to_timestamp_rescaling():
    """The signature depends on increment order, not the timestamp scale/offset.

    Building the same L-path over ``[0, 1]``, ``[0, 2]`` and an offset span
    ``[5, 9]`` must give an identical log-signature. 
    """
    reference = _build_l_shape_stream(0.0, 1.0, [[1.0, 0.0], [0.0, 1.0]])
    ref_data = reference.log_signature(reference.support).data

    for t0, t1 in [(0.0, 2.0), (5.0, 9.0), (-3.0, -1.0)]:
        stream = _build_l_shape_stream(t0, t1, [[1.0, 0.0], [0.0, 1.0]])
        data = stream.log_signature(stream.support).data
        assert jnp.allclose(data, ref_data, atol=1e-6)


def test_from_increments_supports_multiple_batched_input_streams():
    timestamps = [
        jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32),
        jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32),
    ]
    data = [
        jnp.array([[1.0], [2.0], [3.0]], dtype=jnp.float32),
        jnp.array([[-1.0], [0.5], [2.0]], dtype=jnp.float32),
    ]
    basis = LieBasis(width=1, depth=2)

    batched = LieIncrementStream.from_increments(
        timestamps=timestamps,
        data=data,
        resolution=3,
        input_data_basis=None,
        lie_basis=basis,
    )

    assert batched.batch_dims == (2,)
    actual = batched.log_signature(batched.support).data
    for i in range(2):
        single = LieIncrementStream.from_increments(
            timestamps=timestamps[i],
            data=data[i],
            resolution=3,
            input_data_basis=None,
            lie_basis=basis,
        )
        assert jnp.allclose(actual[i], single.log_signature(single.support).data)


def test_from_increments_preserves_batch_dims_of_single_data_array():
    timestamps = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    data = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3, 1)
    basis = LieBasis(width=1, depth=1)

    stream = LieIncrementStream.from_increments(
        timestamps=timestamps,
        data=data,
        resolution=2,
        input_data_basis=None,
        lie_basis=basis,
    )

    assert stream.batch_dims == (2, 3)
    assert stream.log_signature(stream.support).data.shape == (2, 3, basis.size())
    assert jnp.allclose(
        stream.log_signature(stream.support).data,
        data.sum(axis=0),
    )


def test_from_increments_preserves_leading_unit_data_batch_dimension():
    timestamps = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    data = jnp.arange(6, dtype=jnp.float32).reshape(3, 1, 2, 1)
    basis = LieBasis(width=1, depth=1)

    stream = LieIncrementStream.from_increments(
        timestamps=timestamps,
        data=data,
        resolution=2,
        input_data_basis=None,
        lie_basis=basis,
    )

    assert stream.batch_dims == (1, 2)
    assert stream.log_signature(stream.support).data.shape == (1, 2, basis.size())
    assert jnp.allclose(
        stream.log_signature(stream.support).data,
        data.sum(axis=0),
    )


def test_from_increments_sorts_each_input_by_timestamp():
    basis = LieBasis(width=2, depth=2)
    ordered_timestamps = jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32)
    ordered_data = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=jnp.float32
    )
    shuffled_timestamps = jnp.array([1.0, 0.0, 0.5], dtype=jnp.float32)
    shuffled_data = jnp.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32
    )

    ordered = LieIncrementStream.from_increments(
        timestamps=ordered_timestamps,
        data=ordered_data,
        resolution=3,
        input_data_basis=None,
        lie_basis=basis,
    )
    shuffled = LieIncrementStream.from_increments(
        timestamps=shuffled_timestamps,
        data=shuffled_data,
        resolution=3,
        input_data_basis=None,
        lie_basis=basis,
    )

    assert jnp.allclose(
        shuffled.log_signature(shuffled.support).data,
        ordered.log_signature(ordered.support).data,
        atol=1e-6,
    )


def test_from_increments_preserves_order_for_increments_in_one_bucket():
    basis = LieBasis(width=2, depth=2)
    timestamps = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    data = jnp.array(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=jnp.float32
    )

    stream = LieIncrementStream.from_increments(
        timestamps=timestamps,
        data=data,
        resolution=1,
        input_data_basis=None,
        lie_basis=basis,
    )

    expected = rpj.cbh(
        rpj.Lie(jnp.array([1.0, 0.0], dtype=jnp.float32), LieBasis(width=2, depth=1)),
        rpj.Lie(jnp.array([0.0, 1.0], dtype=jnp.float32), LieBasis(width=2, depth=1)),
        lie_basis=basis,
    )
    expected = rpj.cbh(
        expected,
        rpj.Lie.zero(basis),
        lie_basis=basis,
    )

    assert jnp.allclose(
        stream.log_signature(stream.support).data,
        expected.data,
        atol=1e-6,
    )


def test_from_increments_pads_data_to_the_input_basis():
    input_basis = LieBasis(width=2, depth=1)
    output_basis = LieBasis(width=2, depth=2)
    timestamps = jnp.array([0.0, 1.0], dtype=jnp.float32)
    data = jnp.array([[0.25, -0.5], [0.0, 0.0]], dtype=jnp.float32)

    stream = LieIncrementStream.from_increments(
        timestamps=timestamps,
        data=data,
        resolution=2,
        input_data_basis=input_basis,
        lie_basis=output_basis,
    )

    expected = rpj.Lie(data[0], input_basis).change_depth(output_basis.depth)
    assert stream.lie_basis == output_basis
    assert jnp.allclose(
        stream.log_signature(stream.support).data,
        expected.data,
        atol=1e-6,
    )


def test_from_increments_rejects_opencl_cache():
    with pytest.raises(ValueError, match="only supports ClOpen dyadic caches"):
        LieIncrementStream.from_increments(
            timestamps=jnp.array([0.0, 1.0], dtype=jnp.float32),
            data=jnp.array([[1.0], [0.0]], dtype=jnp.float32),
            resolution=1,
            input_data_basis=None,
            lie_basis=LieBasis(width=1, depth=1),
            interval_type=IntervalType.OpenCl,
        )


@pytest.mark.parametrize(
    ("timestamps", "data", "match"),
    [
        (
            jnp.array([0.0, 1.0], dtype=jnp.float32),
            jnp.array([[1.0]], dtype=jnp.float32),
            "Time dimension mismatch",
        ),
        (
            jnp.array([[0.0, 1.0]], dtype=jnp.float32),
            jnp.array([[1.0], [2.0]], dtype=jnp.float32),
            "timestamps must be held in 1D arrays",
        ),
        (
            [jnp.array([0.0, 1.0], dtype=jnp.float32)],
            [
                jnp.array([[1.0], [2.0]], dtype=jnp.float32),
                jnp.array([[3.0], [4.0]], dtype=jnp.float32),
            ],
            "same length",
        ),
    ],
)
def test_from_increments_rejects_invalid_shapes(timestamps, data, match):
    with pytest.raises(ValueError, match=match):
        LieIncrementStream.from_increments(
            timestamps=timestamps,
            data=data,
            resolution=2,
            input_data_basis=None,
            lie_basis=LieBasis(width=1, depth=2),
        )


def test_from_increments_rejects_inconsistent_batch_dimensions():
    timestamps = [
        jnp.array([0.0, 1.0], dtype=jnp.float32),
        jnp.array([0.0, 1.0], dtype=jnp.float32),
    ]
    data = [
        jnp.ones((2, 1), dtype=jnp.float32),
        jnp.ones((2, 2, 1), dtype=jnp.float32),
    ]

    with pytest.raises(ValueError, match="Batch dimension mismatch at index 1"):
        LieIncrementStream.from_increments(
            timestamps=timestamps,
            data=data,
            resolution=2,
            input_data_basis=None,
            lie_basis=LieBasis(width=1, depth=2),
        )
