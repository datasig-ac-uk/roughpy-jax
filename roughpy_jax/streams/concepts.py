"""Protocols defining the public interfaces of stream objects."""

import typing
from typing import Protocol, Self, TypeVar

from jax.typing import ArrayLike

from roughpy_jax.bases import Basis
from roughpy_jax.intervals import Interval

LieT = TypeVar("LieT", covariant=True)
GroupT = TypeVar("GroupT", covariant=True)
StreamValueT = TypeVar("StreamValueT", covariant=True)


@typing.runtime_checkable
class Stream(Protocol[LieT, GroupT]):
    """Describe a system through signatures over query intervals.

    A stream provides a signature and log-signature for each query interval.
    The standard implementation returns a group-like free tensor and a Lie
    element, but the protocol also permits other compatible Lie groups and
    algebras.

    Stream queries have two independent sources of batching. A stream may carry
    intrinsic batch dimensions ``D``, reported by :attr:`batch_dims`, and the lower
    and upper endpoints of a query interval may broadcast to a query batch shape
    ``Q``. Query batching forms the Cartesian product of these batches; it does not
    broadcast the query axes against the stream axes. The returned algebra element
    therefore has batch shape ``Q + D``, with the query dimensions first and the
    stream dimensions second. For the standard array-backed algebra types, the
    complete coefficient shape is ``Q + D + (basis_size,)``. A scalar interval has
    query batch shape ``()`` and consequently adds no dimensions to the result.
    """

    @property
    def lie_basis(self) -> Basis:
        """Return the basis of the stream's Lie algebra."""
        ...

    @property
    def group_basis(self) -> Basis:
        """Return the basis of the stream's group representation."""
        ...

    @property
    def support(self) -> Interval:
        """Return the support interval of the stream.

        Queries outside the support return the group identity and Lie zero.
        """
        ...

    @property
    def dtype(self):
        """Return the coefficient dtype produced by stream queries."""
        ...

    @property
    def batch_dims(self) -> tuple[int, ...]:
        """Return the intrinsic batch dimensions of the stream data.

        These dimensions do not include dimensions introduced by a batched query.
        Query dimensions precede these stream dimensions in values returned by
        :meth:`log_signature` and :meth:`signature`.
        """
        ...

    def __getitem__(self, index) -> Self:
        """Select from the intrinsic batch dimensions of the stream.

        Indexing applies only to the dimensions reported by
        :attr:`batch_dims`. Structural dimensions owned by an implementation,
        such as a piece or dyadic-cache dimension, and the trailing algebra
        coefficient dimension are preserved. Apart from protecting those
        dimensions, indexing follows JAX array semantics, including basic and
        advanced indexing and the insertion of new axes.

        An index may remove all batch dimensions and produce an unbatched
        stream, or produce a stream with an empty batch dimension.

        :param index: A JAX-compatible index into the intrinsic batch dimensions.
        :return: A stream of the same type containing the selected batch data.
        """
        ...

    def log_signature(self, interval: Interval) -> LieT:
        """Query the stream for the log signature over an interval.

        An interval containing no stream evolution, including one outside the
        support, returns the Lie zero.

        The endpoints may represent a batch of query intervals. If they broadcast
        to shape ``Q`` and the stream has batch shape ``D``, the returned Lie
        element has batch shape ``Q + D``. Interval endpoints are treated as
        non-differentiable query parameters.

        :param interval: Scalar or batched query interval.
        :return: A Lie element describing the stream over the interval.
        """
        ...

    def signature(self, interval: Interval) -> GroupT:
        """Query the stream for the signature over an interval.

        An interval containing no stream evolution, including one outside the
        support, returns the group identity.

        The endpoints may represent a batch of query intervals. If they broadcast
        to shape ``Q`` and the stream has batch shape ``D``, the returned group
        element has batch shape ``Q + D``. Interval endpoints are treated as
        non-differentiable query parameters.

        :param interval: Scalar or batched query interval.
        :return: A group element describing the stream over the interval.
        """
        ...


@typing.runtime_checkable
class ValueStream(Protocol[LieT, GroupT, StreamValueT]):
    """Track an evolving value together with its increment stream.

    A value stream pairs an increment stream with a base value at one boundary
    of its support. The value at a parameter is obtained by propagating that
    reference value using the intervening signature.

    For a tensor-valued stream, propagation may be left multiplication by the
    signature. More generally, implementations may apply another action of the
    signature on the value space.

    The propagation operation is an implementation detail exposed through
    :meth:`value_at`. :meth:`query` restricts the increment stream and updates
    the base value to the corresponding boundary of the requested interval.
    """

    @property
    def stream(self) -> Stream[LieT, GroupT]:
        """Return the underlying increment stream."""
        ...

    @property
    def base_value(self) -> StreamValueT:
        """Return the value at the stream's reference boundary."""
        ...

    def value_at(self, parameter: ArrayLike) -> StreamValueT:
        """Compute the stream value at a parameter value.

        The implementation propagates :attr:`base_value` to ``parameter`` using
        the signature of the underlying increment stream.

        :param parameter: Parameter value at which to evaluate the stream.
        :return: The propagated stream value at ``parameter``.
        """
        ...

    def query(self, interval: Interval) -> Self:
        """Query the value stream over an interval.

        The returned increment stream is restricted to ``interval``. Its base
        value is the original value propagated to the corresponding boundary
        of ``interval``.

        :param interval: Query interval.
        :return: Value stream with an updated base value and restricted
            increment stream.
        """
        ...
