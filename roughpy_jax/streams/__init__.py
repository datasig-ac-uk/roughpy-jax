from __future__ import annotations

from roughpy_jax.intervals import Interval, Partition
from roughpy_jax.algebra import DenseFreeTensor, DenseLie

from .concepts import Stream, ValueStream, LieT, GroupT
from .lie_increment_stream import LieIncrementStream, dyadic_query
from .piecewise_abelian_stream import (
    PiecewiseAbelianStream,
    piecewise_abelian_stream_from_data,
    to_piecewise_abelian_stream,
)

__all__ = [
    "LieIncrementStream",
    "PiecewiseAbelianStream",
    "Stream",
    "ValueStream",
    "dyadic_query",
    "log_signature",
    "piecewise_abelian_stream_from_data",
    "signature",
    "simplify",
    "to_piecewise_abelian_stream",
]

def log_signature(stream: Stream[LieT, GroupT], query: Partition | Interval) -> LieT:
    """
    Compute the log-signature of a stream over an interval or partition.

    Essentially calls the `log_signature` method of the stream, but also handles
    partition queries by converting these to a batch of intervals.

    :param stream: The stream for which to compute the log-signature.
    :param query: The interval or partition over which to compute the log-signature.
    :return: The log-signature of the stream over the given query.
    """
    if isinstance(query, Partition):
        query = query.to_intervals()

    return stream.log_signature(query)


def signature(stream: Stream[LieT, GroupT], query: Partition | Interval) -> GroupT:
    """
    Compute the signature of a stream over an interval or partition.

    Essentially calls the `signature` method of the stream, but also handles
    partition queries by converting these to a batch of intervals.

    :param stream: The stream for which to compute the signature.
    :param query: The interval or partition over which to compute the signature.
    :return: The signature of the stream over the given query.
    """
    if isinstance(query, Partition):
        query = query.to_intervals()

    return stream.signature(query)


def simplify(stream: Stream[DenseLie, DenseFreeTensor], partition: Partition) -> PiecewiseAbelianStream:
    """
    Simplify a stream into a piecewise abelian stream.

    Alias of the `to_piecewise_abelian_stream` function to match the naming from RoughPy.

    :param stream: The stream to simplify.
    :param partition: The partition over which to simplify the stream.
    :return: The simplified piecewise abelian stream.
    """
    return to_piecewise_abelian_stream(stream, partition)
