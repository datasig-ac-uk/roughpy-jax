"""Public stream protocols, implementations, and query helpers."""

from __future__ import annotations

from roughpy_jax.algebra import DenseFreeTensor, DenseLie
from roughpy_jax.intervals import Interval, Partition

from .concepts import GroupT, LieT, Stream, ValueStream
from .lie_increment_stream import (
    LieIncrementStream,
    compute_separating_resolution,
    dyadic_query,
)
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
    "compute_separating_resolution",
    "dyadic_query",
    "log_signature",
    "piecewise_abelian_stream_from_data",
    "signature",
    "simplify",
    "to_piecewise_abelian_stream",
]


def log_signature(stream: Stream[LieT, GroupT], query: Partition | Interval) -> LieT:
    """Compute a stream's log-signature over an interval or partition.

    A partition is converted to its batch of consecutive intervals before
    dispatching to :meth:`Stream.log_signature`.

    Let ``Q`` be the batch shape obtained by broadcasting the query endpoints,
    and let ``D`` be ``stream.batch_dims``. The returned Lie element has batch
    shape ``Q + D``: query dimensions come first and intrinsic stream
    dimensions follow. For a scalar interval, ``Q`` is empty and the output
    retains only ``D``. For a partition with endpoint shape
    ``(*B, n_endpoints)``, ``Q`` is ``(*B, n_endpoints - 1)`` because every
    consecutive pair of endpoints forms a separate query. Empty dimensions in
    either batch are preserved.

    :param stream: Stream to query.
    :param query: Interval or partition over which to compute the log-signature.
    :return: Log-signature over ``query`` with batch shape ``Q + D``.
    """
    if isinstance(query, Partition):
        query = query.to_intervals()

    return stream.log_signature(query)


def signature(stream: Stream[LieT, GroupT], query: Partition | Interval) -> GroupT:
    """Compute a stream's signature over an interval or partition.

    A partition is converted to its batch of consecutive intervals before
    dispatching to :meth:`Stream.signature`.

    Let ``Q`` be the batch shape obtained by broadcasting the query endpoints,
    and let ``D`` be ``stream.batch_dims``. The returned group element has batch
    shape ``Q + D``: query dimensions come first and intrinsic stream
    dimensions follow. For a scalar interval, ``Q`` is empty and the output
    retains only ``D``. For a partition with endpoint shape
    ``(*B, n_endpoints)``, ``Q`` is ``(*B, n_endpoints - 1)`` because every
    consecutive pair of endpoints forms a separate query. Empty dimensions in
    either batch are preserved.

    :param stream: Stream to query.
    :param query: Interval or partition over which to compute the signature.
    :return: Signature over ``query`` with batch shape ``Q + D``.
    """
    if isinstance(query, Partition):
        query = query.to_intervals()

    return stream.signature(query)


def simplify(
    stream: Stream[DenseLie, DenseFreeTensor], partition: Partition
) -> PiecewiseAbelianStream:
    """Simplify a stream into a piecewise abelian stream.

    This is the RoughPy-compatible name for
    :func:`to_piecewise_abelian_stream`.

    :param stream: Stream to simplify.
    :param partition: Unbatched partition on which to sample the stream.
    :return: Piecewise abelian approximation over ``partition``.
    :raises ValueError: If ``partition`` has batch dimensions.
    """
    return to_piecewise_abelian_stream(stream, partition)
