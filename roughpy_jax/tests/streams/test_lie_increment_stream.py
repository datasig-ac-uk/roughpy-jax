import jax.numpy as jnp
import pytest

import roughpy_jax as rpj
from roughpy_jax.algebra import LieBasis, TensorBasis
from roughpy_jax.intervals import IntervalType, Partition, RealInterval
from roughpy_jax.streams import PiecewiseAbelianStream
from roughpy_jax.streams.lie_increment_stream import LieIncrementStream


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


def test_log_signature_rejects_nonsingleton_array_interval_endpoints():
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.zeros((4, lie_basis.size()), dtype=jnp.float32)
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(0.0, 1.0, IntervalType.ClOpen),
        resolution=1,
    )

    query = RealInterval(
        jnp.array([0.25, 0.5], dtype=jnp.float32),
        jnp.array([0.75, 0.5], dtype=jnp.float32),
        IntervalType.ClOpen,
    )

    with pytest.raises(ValueError, match="single-element endpoint arrays"):
        stream.log_signature(query)


def test_log_signature_without_interval_uses_support(monkeypatch):
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.zeros((4, lie_basis.size()), dtype=jnp.float32)
    support = RealInterval(1.0, 3.0, IntervalType.ClOpen)
    stream = LieIncrementStream(cache, lie_basis, support=support, resolution=1)
    captured = {}

    def fake_reparamterise(interval):
        captured["interval"] = interval
        return RealInterval(0.0, 0.0, IntervalType.ClOpen)

    monkeypatch.setattr(stream, "_reparamterise", fake_reparamterise)

    result = stream.log_signature()

    assert captured["interval"] == support
    assert jnp.allclose(result.data, rpj.Lie.zero(lie_basis).data)


def test_log_signature_of_empty_interval_returns_zero_without_query(monkeypatch):
    lie_basis = LieBasis(width=1, depth=1)
    cache = jnp.array([[1.0], [2.0], [3.0], [0.0]], dtype=jnp.float32)
    stream = LieIncrementStream(
        cache,
        lie_basis,
        support=RealInterval(0.0, 1.0, IntervalType.ClOpen),
        resolution=1,
    )

    def fail_reparamterise(_interval):
        raise AssertionError("empty intervals should return before reparametrisation")

    monkeypatch.setattr(stream, "_reparamterise", fail_reparamterise)

    result = stream.log_signature(RealInterval(0.5, 0.5, IntervalType.ClOpen))

    assert jnp.allclose(result.data, rpj.Lie.zero(lie_basis).data)


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
