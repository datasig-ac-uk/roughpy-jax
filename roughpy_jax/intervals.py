"""Interval representations and partition operations for stream queries."""

from __future__ import annotations

import enum
import typing
from dataclasses import FrozenInstanceError, dataclass
from typing import Any, Protocol, TypeVar

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

RealT = TypeVar("RealT")


class IntervalType(enum.IntEnum):
    """Describe which endpoint of an interval is closed."""

    ClOpen = 0
    """Left-closed and right-open: ``[inf, sup)``."""

    OpenCl = 1
    """Left-open and right-closed: ``(inf, sup]``."""


@typing.runtime_checkable
class Interval(Protocol):
    """Define the structural interface shared by interval representations.

    Endpoint arrays may be scalar or batched. Implementations expose one
    endpoint convention for the complete batch.
    """

    @property
    def interval_type(self) -> IntervalType:
        """Return the endpoint convention used by the interval."""
        ...

    @property
    def inf(self) -> Array:
        """Return the lower endpoint array."""
        ...

    @property
    def sup(self) -> Array:
        """Return the upper endpoint array."""
        ...

    @property
    def length(self) -> Array:
        """Return the nonnegative interval length array."""
        ...


class BaseInterval:
    """Provide shared formatting and length calculations for intervals."""

    # TODO: These don't need to be in a class, just have module-level functions for str, length, and intersection that
    # take Intervals as arguments. The only reason to have these in a class is if we want to use inheritance to share
    # code between different Interval implementations.

    @staticmethod
    def to_string(interval: Interval) -> str:
        """Format an interval using its endpoint convention.

        :param interval: Interval to format.
        :return: String containing the endpoints and matching brackets.
        """
        reprs = {
            IntervalType.ClOpen: "[{}, {})",
            IntervalType.OpenCl: "({}, {}]",
        }
        return reprs[interval.interval_type].format(interval.inf, interval.sup)

    @staticmethod
    def length(interval: Interval) -> Array:
        """Calculate the nonnegative length of an interval.

        :param interval: Interval whose length is required.
        :return: ``maximum(sup - inf, 0)`` with the broadcasted endpoint shape.
        """
        return jnp.maximum(0.0, jnp.asarray(interval.sup) - jnp.asarray(interval.inf))


def intersection(
    left_interval: Interval,
    right_interval: Interval,
) -> RealInterval | DyadicInterval | Partition:
    """Calculate the intersection of two interval representations.

    Two ordinary intervals produce a :class:`RealInterval`; two partitions
    produce the real interval spanned by their overlapping outer bounds. One
    partition and one other interval produce a clipped :class:`Partition`.
    Disjoint real bounds are represented with ``inf > sup`` and therefore have
    zero :attr:`Interval.length`.

    Dyadic intersection is not implemented. Two dyadic arguments raise
    :class:`NotImplementedError`, while mixing a dyadic and non-dyadic interval
    raises :class:`ValueError`.

    :param left_interval: First interval operand.
    :param right_interval: Second interval operand.
    :return: Intersection represented according to the operand types.
    :raises TypeError: If either operand does not satisfy :class:`Interval`, or
        if the endpoint conventions differ.
    :raises ValueError: If exactly one operand is dyadic.
    :raises NotImplementedError: If both operands are dyadic intervals.
    """
    if not isinstance(left_interval, Interval) or not isinstance(
        right_interval, Interval
    ):
        raise TypeError("Both arguments must be of type Interval")

    if left_interval.interval_type != right_interval.interval_type:
        raise TypeError("Both intervals must be of the same IntervalType")

    # Two dyadics → dyadic class method
    if isinstance(left_interval, DyadicInterval) and isinstance(
        right_interval, DyadicInterval
    ):
        return DyadicInterval.intersection(left_interval, right_interval)
    elif isinstance(left_interval, DyadicInterval) or isinstance(
        right_interval, DyadicInterval
    ):
        raise ValueError("Cannot intersect a DyadicInterval with a non-DyadicInterval")

    # Two partitions → real interval intersection
    if isinstance(left_interval, Partition) and isinstance(right_interval, Partition):
        return RealInterval.intersection(left_interval, right_interval)

    # One partition + one interval → Partition.truncate
    if isinstance(left_interval, Partition):
        return Partition.truncate(left_interval, right_interval)
    if isinstance(right_interval, Partition):
        return Partition.truncate(right_interval, left_interval)

    # Default: real interval intersection
    return RealInterval.intersection(left_interval, right_interval)


class Dyadic:
    """Represent values of the form ``k * 2**(-n)`` using integer arrays.

    ``k`` and ``n`` may be scalar or batched. They remain JAX arrays so that
    conversion can occur inside JAX transformations. When their shapes differ,
    normal JAX broadcasting applies during conversion.

    :param k: Integer numerator array.
    :param n: Integer exponent array.
    :raises TypeError: If either input has a non-integer dtype.
    """

    k: Array
    n: Array

    def __init__(self, k: ArrayLike, n: ArrayLike):
        k = jnp.asarray(k)
        n = jnp.asarray(n)

        if not jnp.issubdtype(k.dtype, jnp.integer):
            raise TypeError(f"Dyadic.k must be an integer array, got {k.dtype}: {k!r}")
        if not jnp.issubdtype(n.dtype, jnp.integer):
            raise TypeError(f"Dyadic.n must be an integer array, got {n.dtype}: {n!r}")

        object.__setattr__(self, "k", k)
        object.__setattr__(self, "n", n)

    def __setattr__(self, name: str, value: object) -> None:
        """Reject mutation of a dyadic value.

        :param name: Attribute name being assigned.
        :param value: Proposed attribute value.
        :raises FrozenInstanceError: On every attempted assignment.
        """
        raise FrozenInstanceError(f"cannot assign to field '{name}'")

    def __str__(self) -> str:
        """Return a representation containing the integer components."""
        return f"Dyadic(k={self.k}, n={self.n})"

    def __eq__(self, other: object) -> bool:
        """Compare dyadic components for exact array equality.

        :param other: Object to compare with this dyadic value.
        :return: Whether ``other`` has the same type, ``k``, and ``n`` arrays.
        """
        if type(self) is not type(other):
            return False
        return bool(
            jnp.array_equal(self.k, other.k) and jnp.array_equal(self.n, other.n)
        )

    def __jax_array__(self) -> Array:
        """Convert the dyadic components to a JAX floating-point array."""
        return jnp.ldexp(self.k, -self.n)


class DyadicInterval(Dyadic):
    """Represent unit-width dyadic intervals using integer components.

    For :attr:`IntervalType.ClOpen`, ``k`` identifies
    ``[k * 2**(-n), (k + 1) * 2**(-n))``. For
    :attr:`IntervalType.OpenCl`, it identifies
    ``((k - 1) * 2**(-n), k * 2**(-n)]``. Array components produce batched
    intervals, including batches with zero-length dimensions.

    :param k: Integer dyadic-location array.
    :param n: Integer dyadic-resolution array.
    :param interval_type: Endpoint convention shared by the interval batch.
    :raises TypeError: If ``k`` or ``n`` has a non-integer dtype.
    """

    _interval_type: IntervalType

    def __init__(
        self,
        k: ArrayLike,
        n: ArrayLike,
        interval_type: IntervalType = IntervalType.ClOpen,
    ):
        super().__init__(k, n)
        object.__setattr__(self, "_interval_type", interval_type)

    @property
    def interval_type(self) -> IntervalType:
        """Return the endpoint convention used by the interval."""
        return self._interval_type

    def __str__(self) -> str:
        """Format the interval using its endpoint convention."""
        return BaseInterval.to_string(self)

    def __eq__(self, other: object) -> bool:
        """Compare interval type and dyadic components for equality.

        :param other: Object to compare with this dyadic interval.
        :return: Whether ``other`` has the same interval type, ``k``, and ``n``.
        """
        return (
            isinstance(other, DyadicInterval)
            and self.interval_type == other.interval_type
            and bool(jnp.array_equal(self.k, other.k))
            and bool(jnp.array_equal(self.n, other.n))
        )

    @property
    def inf(self) -> Array:
        """Return the lower endpoint array."""
        k = self.k if self._interval_type == IntervalType.ClOpen else self.k - 1
        return jnp.ldexp(k, -self.n)

    @property
    def sup(self) -> Array:
        """Return the upper endpoint array."""
        k = (self.k + 1) if self._interval_type == IntervalType.ClOpen else self.k
        return jnp.ldexp(k, -self.n)

    @property
    def length(self) -> Array:
        """Return the nonnegative interval length array."""
        return BaseInterval.length(self)

    @classmethod
    def intersection(
        cls, left: DyadicInterval, right: DyadicInterval
    ) -> DyadicInterval:
        """Raise because dyadic intersection is not implemented.

        :param left: First dyadic interval.
        :param right: Second dyadic interval.
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError("DyadicInterval intersection is not implemented yet")


def _dyadic_tree_flatten(dyadic: Dyadic) -> tuple[tuple[Array, Array], None]:
    return (dyadic.k, dyadic.n), None


def _dyadic_tree_unflatten(_aux_data: None, children: tuple[Array, Array]) -> Dyadic:
    k, n = children
    return Dyadic(k, n)


def _dyadic_interval_tree_flatten(
    interval: DyadicInterval,
) -> tuple[tuple[Array, Array], IntervalType]:
    return (interval.k, interval.n), interval.interval_type


def _dyadic_interval_tree_unflatten(
    interval_type: IntervalType, children: tuple[Array, Array]
) -> DyadicInterval:
    k, n = children
    return DyadicInterval(k, n, interval_type)


jax.tree_util.register_pytree_node(Dyadic, _dyadic_tree_flatten, _dyadic_tree_unflatten)
jax.tree_util.register_pytree_node(
    DyadicInterval,
    _dyadic_interval_tree_flatten,
    _dyadic_interval_tree_unflatten,
)


@dataclass(frozen=True, init=False)
class RealInterval:
    """Represent a scalar or batched interval on the real line.

    The lower and upper endpoint inputs must have broadcast-compatible shapes.
    They are broadcast when the interval is constructed and stored as arrays
    with the same shape. Scalar arrays represent a single interval, while
    non-scalar arrays represent a batch of intervals. A single endpoint
    convention applies to the entire batch.

    ``RealInterval`` is a pytree with the endpoint arrays as dynamic leaves and
    the interval type as static metadata.

    Bounds are not required to be ordered. An element with ``inf >= sup`` has
    zero length and can represent an empty intersection. Empty batch dimensions
    are supported.

    :param _inf: Lower endpoint input.
    :param _sup: Upper endpoint input, broadcast-compatible with ``_inf``.
    :param _interval_type: Endpoint convention shared by the interval batch.
    """

    _inf: Array
    _sup: Array
    _interval_type: IntervalType

    def __init__(
        self,
        _inf: ArrayLike,
        _sup: ArrayLike,
        _interval_type: IntervalType,
    ):
        inf, sup = jnp.broadcast_arrays(jnp.asarray(_inf), jnp.asarray(_sup))
        object.__setattr__(self, "_inf", inf)
        object.__setattr__(self, "_sup", sup)
        object.__setattr__(self, "_interval_type", _interval_type)

    def __str__(self) -> str:
        """Format the interval using its endpoint convention."""
        return BaseInterval.to_string(self)

    def __hash__(self) -> int:
        """Return a hash for an unbatched interval.

        :return: Hash of the scalar endpoints and interval type.
        :raises TypeError: If either endpoint is not scalar.
        """
        if self._inf.ndim != 0 or self._sup.ndim != 0:
            raise TypeError("batched RealInterval objects are unhashable")
        return hash((self._inf.item(), self._sup.item(), self._interval_type))

    @property
    def interval_type(self) -> IntervalType:
        """Return the endpoint convention used by the interval."""
        return self._interval_type

    @property
    def inf(self) -> Array:
        """Return the lower endpoint array."""
        return self._inf

    @property
    def sup(self) -> Array:
        """Return the upper endpoint array."""
        return self._sup

    @property
    def length(self) -> Array:
        """Return ``maximum(sup - inf, 0)`` for each interval."""
        return BaseInterval.length(self)

    @classmethod
    def intersection(cls, left: Interval, right: Interval) -> RealInterval:
        """Compute an intersection from the overlapping endpoint bounds.

        The endpoint arrays follow JAX broadcasting rules. This low-level
        method assumes that the operands use compatible endpoint conventions;
        :func:`intersection` performs that validation for public calls.

        :param left: First interval operand.
        :param right: Second interval operand.
        :return: Real interval with lower bound ``maximum(left.inf, right.inf)``
            and upper bound ``minimum(left.sup, right.sup)``.
        """
        new_inf = jnp.maximum(left.inf, right.inf)
        new_sup = jnp.minimum(left.sup, right.sup)
        return RealInterval(new_inf, new_sup, left.interval_type)


RealInterval = jax.tree_util.register_dataclass(
    RealInterval,
    data_fields=["_inf", "_sup"],
    meta_fields=["_interval_type"],
)


@jax.tree_util.register_pytree_node_class
class Partition:
    """Represent a sorted partition of an interval.

    ``endpoints`` is stored as a JAX array. Its final axis contains the
    endpoints of each partition, while any preceding axes are batch
    dimensions. For example, an array with shape ``(batch, points)``
    represents ``batch`` partitions, each containing ``points`` endpoints.

    Endpoints are sorted when the partition is constructed. Duplicate
    endpoints are retained and represent empty subintervals. This is useful
    when combining partitions with different numbers of endpoints, since
    padding can be performed by repeating the final endpoint without changing
    the array shape.

    The partition is a JAX pytree. The endpoint array is its dynamic leaf and
    ``interval_type`` is static metadata.

    :param endpoints: Array-like endpoint values with shape
        ``(*batch_dims, n_endpoints)`` and at least two entries on the final
        axis. Empty batch dimensions are supported.
    :param interval_type: Endpoint convention shared by all subintervals.
    :raises ValueError: If the final axis contains fewer than two endpoints.

    """

    endpoints: Array
    interval_type: IntervalType

    def __init__(self, endpoints: ArrayLike, interval_type: IntervalType):
        endpoints = jnp.sort(jnp.asarray(endpoints))

        if endpoints.shape[-1] < 2:
            raise ValueError("Partition must have at least two endpoints")

        self.endpoints = endpoints
        self.interval_type = interval_type

    def __str__(self) -> str:
        """Return a compact representation using the outer endpoints."""
        inner = f"{self.inf}, ..., {self.sup}"
        if self.interval_type == IntervalType.ClOpen:
            return f"[{inner})"
        return f"({inner}]"

    @property
    def inf(self) -> Array:
        """Return the first endpoint for each batch element."""
        return self.endpoints[..., 0]

    @property
    def sup(self) -> Array:
        """Return the final endpoint for each batch element."""
        return self.endpoints[..., -1]

    @property
    def batch_dims(self) -> tuple[int, ...]:
        """Return the shape of the partition batch dimensions."""
        return self.endpoints.shape[:-1]

    @property
    def dtype(self) -> jnp.dtype:
        """Return the dtype of the endpoint array."""
        return self.endpoints.dtype

    def __len__(self) -> int:
        """Return the number of subintervals in each partition."""
        return self.endpoints.shape[-1] - 1

    @property
    def length(self) -> Array:
        """Return the outer interval length for each batch element."""
        return self.sup - self.inf

    def tree_flatten(self) -> tuple[Any, Any]:
        """Return dynamic children and static metadata for JAX pytree handling."""
        return (self.endpoints,), (self.interval_type,)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any):
        """Reconstruct a partition from its pytree representation.

        :param aux_data: Static tuple containing the endpoint convention.
        :param children: Dynamic tuple containing the endpoint array.
        :return: Reconstructed partition.
        """
        obj = cls.__new__(cls)
        obj.endpoints = children[0]
        obj.interval_type = aux_data[0]

        return obj

    def to_intervals(self) -> RealInterval:
        """Return all subintervals as one batched :class:`RealInterval`."""
        return RealInterval(
            self.endpoints[..., :-1], self.endpoints[..., 1:], self.interval_type
        )

    def truncate(self, other: Interval) -> Partition:
        """Clip every endpoint to the bounds of another interval.

        This does not change the size of the array, so any endpoints that
        lie outside the other interval are repeated. The partition batch shape
        and the broadcasted shape of the interval endpoints are broadcast
        together. In particular, truncating an unbatched partition by a batched
        interval produces a partition with the interval's batch shape.

        :param other: Interval providing the clipping bounds. Its endpoint
            convention must match the partition's convention.
        :return: Partition with clipped, sorted endpoints and the broadcasted
            batch shape.
        :raises ValueError: If the endpoint conventions differ or the batch
            shapes cannot be broadcast together.
        """
        if self.interval_type != other.interval_type:
            raise ValueError("Cannot truncate partitions with different interval type")

        endpoints = jnp.clip(self.endpoints, other.inf[..., None], other.sup[..., None])
        return Partition(endpoints, self.interval_type)

    def merge(self, other: Partition) -> Partition:
        """Interleave the endpoints from two partitions.

        This performs an interval union of the two domains spanned by the
        arguments, with contained infimum and supremum endpoints becoming new
        interior endpoints. The partitions must have identical batch shapes;
        merging does not broadcast batches because each pair of partitions must
        define one unambiguous combined subdivision.

        Duplicate endpoints are preserved.

        :param other: Partition whose endpoints should be merged with this one.
        :return: Sorted partition containing both sets of endpoints.
        :raises ValueError: If the endpoint conventions or batch shapes differ.
        """
        if self.interval_type != other.interval_type:
            raise ValueError("Cannot merge partitions with different interval types")
        if self.batch_dims != other.batch_dims:
            raise ValueError(
                "Cannot merge partitions with different batch dimensions: "
                f"{self.batch_dims} and {other.batch_dims}"
            )

        # The constructor will sort the endpoints
        endpoints = jnp.concatenate((self.endpoints, other.endpoints), axis=-1)
        return Partition(endpoints, self.interval_type)

    def to_real_interval(self) -> RealInterval:
        """Return the real interval spanned by the outer endpoints."""
        return RealInterval(self.inf, self.sup, self.interval_type)
