import jax.numpy as jnp
import pytest

import roughpy_jax as rpj
import roughpy_jax.streams as streams
from roughpy_jax.intervals import IntervalType, Partition, RealInterval


class RecordingStream:
    def __init__(self):
        self.calls = []

    def log_signature(self, query):
        self.calls.append(("log_signature", query))
        return "log signature result"

    def signature(self, query):
        self.calls.append(("signature", query))
        return "signature result"


@pytest.mark.parametrize(
    ("function_name", "expected_result"),
    [
        ("log_signature", "log signature result"),
        ("signature", "signature result"),
    ],
)
def test_stream_function_delegates_interval_query(function_name, expected_result):
    stream = RecordingStream()
    query = RealInterval(0.25, 0.75, IntervalType.ClOpen)

    result = getattr(rpj, function_name)(stream, query)

    assert result == expected_result
    assert stream.calls == [(function_name, query)]


@pytest.mark.parametrize("function_name", ["log_signature", "signature"])
def test_stream_function_converts_partition_to_batched_interval(function_name):
    stream = RecordingStream()
    partition = Partition([0.0, 0.25, 1.0], IntervalType.ClOpen)

    getattr(rpj, function_name)(stream, partition)

    method_name, query = stream.calls[0]
    expected = partition.to_intervals()
    assert method_name == function_name
    assert isinstance(query, RealInterval)
    assert jnp.array_equal(query.inf, expected.inf)
    assert jnp.array_equal(query.sup, expected.sup)
    assert query.interval_type == expected.interval_type

