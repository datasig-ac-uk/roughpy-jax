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
    ClOpen = 0
    OpenCl = 1


@typing.runtime_checkable
class Interval(Protocol):
    """
    Representation of an interval with an unspecified type for mathematical
    computations or range definitions. This interface outlines the required
    methods for retrieving key properties of an interval.
    The purpose of this protocol is to define a constraint for other classes
    or structures that aim to represent intervals and their specific types,
    lower bounds (infimum), and upper bounds (supremum). Classes that adhere
    to this protocol should implement the specified methods to be considered
    compatible.
    :ivar interval_type: Type information of the interval. This attribute
        specifies the nature or classification of the interval, such as
        open, closed, or partially open/closed.
    :type interval_type: IntervalType
    :ivar inf: The lower bound (infimum) of the interval. This attribute
        represents the smallest value contained within the interval.
    :type inf: float
    :ivar sup: The upper bound (supremum) of the interval. This attribute
        represents the largest value contained within the interval.
    :type sup: float
    """

    @property
    def interval_type(self) -> IntervalType: ...

    @property
    def inf(self) -> Array: ...

    @property
    def sup(self) -> Array: ...

    @property
    def length(self) -> Array: ...


class BaseInterval:
    # TODO: These don't need to be in a class, just have module-level functions for str, length, and intersection that
    # take Intervals as arguments. The only reason to have these in a class is if we want to use inheritance to share
    # code between different Interval implementations.

    @staticmethod
    def to_string(interval: Interval) -> str:
        reprs = {
            IntervalType.ClOpen: "[{}, {})",
            IntervalType.OpenCl: "({}, {}]",
        }
        return reprs[interval.interval_type].format(interval.inf, interval.sup)

    @staticmethod
    def length(interval: Interval) -> Array:
        """
        Calculate the length of the interval.
        :return: The length of the interval, calculated as sup - inf.
        :rtype: float
        """
        return jnp.maximum(0.0, jnp.asarray(interval.sup) - jnp.asarray(interval.inf))


def intersection(
        left_interval: Interval,
        right_interval: Interval,
) -> RealInterval | DyadicInterval | Partition:
    """
    Calculate the intersection of two intervals, dispatching to the
    appropriate implementation based on the types of the arguments.

    - Two DyadicIntervals: delegates to DyadicInterval.intersection.
    - Two RealIntervals (or other plain Intervals): computes the
      intersection by bounds.
    - One Partition and one Interval: delegates to Partition.truncate.
    - Two Partitions: converts both to RealIntervals and computes the
      intersection by bounds.

    :param left_interval: The left interval.
    :param right_interval: The right interval.
    :return: The intersection.
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
    """
    Represents a dyadic number in mathematics.
    Dyadic numbers are numbers of the form k * (2^-n), where k is an integer and
    n is a non-negative integer.
    ``k`` and ``n`` are scalar JAX arrays. Keeping the components as arrays is
    important when dyadics are created or consumed inside JAX transformations.

    :ivar k: Integer component of the dyadic number.
    :type k: Array
    :ivar n: Exponent of 2 in the dyadic number.
    :type n: Array
    """

    def __init__(self, k: ArrayLike, n: ArrayLike) -> None:
        k = jnp.asarray(k)
        n = jnp.asarray(n)

        if not jnp.issubdtype(k.dtype, jnp.integer):
            raise TypeError(
                f"Dyadic.k must be an integer array, got {k.dtype}: {k!r}"
            )
        if not jnp.issubdtype(n.dtype, jnp.integer):
            raise TypeError(
                f"Dyadic.n must be an integer array, got {n.dtype}: {n!r}"
            )

        object.__setattr__(self, "k", k)
        object.__setattr__(self, "n", n)

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field '{name}'")

    def __str__(self) -> str:
        return f"Dyadic(k={self.k}, n={self.n})"

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        other = typing.cast(Dyadic, other)
        return bool(
            jnp.array_equal(self.k, other.k)
            and jnp.array_equal(self.n, other.n)
        )

    def __jax_array__(self) -> Array:
        return jnp.ldexp(self.k, -self.n)


class DyadicInterval(Dyadic):
    """
    This subclass represents a dyadic interval, and therefore conforms to the
    Interval protocol. Crucially this is a subclass of Dyadic becuase its
    endpoints (and width) are fully defined by the k and n parameters of Dyadic
    and whether the interval is closed or open on either end.
    """

    def __init__(
            self, k: ArrayLike, n: ArrayLike, interval_type: IntervalType = IntervalType.ClOpen
    ) -> None:
        super().__init__(k, n)
        object.__setattr__(self, "_interval_type", interval_type)

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    def __str__(self) -> str:
        return BaseInterval.to_string(self)

    def __eq__(self, other: object) -> bool:
        return (
                isinstance(other, DyadicInterval)
                and self.interval_type == other.interval_type
                and bool(jnp.array_equal(self.k, other.k))
                and bool(jnp.array_equal(self.n, other.n))
        )

    @property
    def inf(self) -> Array:
        k = self.k if self._interval_type == IntervalType.ClOpen else self.k - 1
        return jnp.ldexp(k, -self.n)

    @property
    def sup(self) -> Array:
        k = (self.k + 1) if self._interval_type == IntervalType.ClOpen else self.k
        return jnp.ldexp(k, -self.n)

    @property
    def length(self) -> Array:
        return BaseInterval.length(self)

    @classmethod
    def intersection(
            cls, left: DyadicInterval, right: DyadicInterval
    ) -> DyadicInterval:
        raise NotImplementedError("DyadicInterval intersection is not implemented yet")


def _dyadic_tree_flatten(dyadic: Dyadic):
    return (dyadic.k, dyadic.n), None


def _dyadic_tree_unflatten(_aux_data, children):
    return Dyadic(*children)


def _dyadic_interval_tree_flatten(interval: DyadicInterval):
    return (interval.k, interval.n), interval.interval_type


def _dyadic_interval_tree_unflatten(interval_type, children):
    return DyadicInterval(*children, interval_type)


jax.tree_util.register_pytree_node(
    Dyadic, _dyadic_tree_flatten, _dyadic_tree_unflatten
)
jax.tree_util.register_pytree_node(
    DyadicInterval,
    _dyadic_interval_tree_flatten,
    _dyadic_interval_tree_unflatten,
)


@dataclass(frozen=True)
class RealInterval:
    """A interval in the real line or collection thereof.

    The lower and upper endpoints are arrays with broadcast-compatible shapes.
    Scalar arrays represent a single interval, while non-scalar arrays represent
    a batch of intervals whose batch shape is the broadcast shape of the two
    endpoints. A single endpoint convention applies to the entire batch.

    ``RealInterval`` is a pytree with the endpoint arrays as dynamic leaves and
    the interval type as static metadata.

    :ivar inf: Lower endpoint array.
    :ivar sup: Upper endpoint array.
    :ivar interval_type: Shared endpoint convention for the interval batch.
    """

    _inf: Array
    _sup: Array
    _interval_type: IntervalType

    def __str__(self) -> str:
        return BaseInterval.to_string(self)

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> Array:
        return self._inf

    @property
    def sup(self) -> Array:
        return self._sup

    @property
    def length(self) -> Array:
        return BaseInterval.length(self)

    @classmethod
    def intersection(cls, left: Interval, right: Interval) -> RealInterval:
        """Compute the intersection of two intervals as a RealInterval."""
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
    """A sorted partition of an interval.

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

    Args:
        endpoints: Array-like endpoint values. The final axis must contain at
            least two endpoints.
        interval_type: Whether the intervals are left-closed/right-open or
            left-open/right-closed.

    Attributes:
        endpoints: Sorted endpoint array with shape ``(..., n_endpoints)``.
        interval_type: Shared endpoint convention for all represented
            intervals.
    """

    endpoints: Array
    interval_type: IntervalType

    def __init__(self, endpoints: ArrayLike, interval_type: IntervalType):
        """Construct a partition from array-like endpoints."""
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
        return (self.endpoints,), (self.interval_type,)

    @classmethod
    def tree_unflatten(cls, aux_data: Any, children: Any):
        obj = cls.__new__(cls)
        obj.endpoints = children[0]
        obj.interval_type = aux_data[0]

        return obj

    def to_intervals(self) -> RealInterval:
        """Return all subintervals as one batched :class:`RealInterval`."""
        return RealInterval(self.endpoints[..., :-1], self.endpoints[..., 1:], self.interval_type)

    def truncate(self, other: Interval) -> Partition:
        """
        Clip every endpoint to the bounds of ``other``.

        This does not change the size of the array, so any endpoints that
        lie outside the other interval are repeated.

        The interval types of the partition and the interval must match.
        """
        if self.interval_type != other.interval_type:
            raise ValueError("Cannot truncate partitions with different interval type")
        return Partition(jnp.clip(self.endpoints, other.inf, other.sup), self.interval_type)

    def merge(self, other: Partition) -> Partition:
        """
        Interleave the endpoints from two partitions.

        This performans an interval union of the two domains spanned by the
        arguments, and with the now contained inf and sup end points becoming new
        interior endpoints

        Duplicate endpoints are preserved.
        """
        if self.interval_type != other.interval_type:
            raise ValueError("Cannot merge partitions with different interval types")

        # The constructor will sort the endpoints
        endpoints = jnp.concatenate((self.endpoints, other.endpoints), axis=-1)
        return Partition(endpoints, self.interval_type)

    def to_real_interval(self) -> RealInterval:
        return RealInterval(self.inf, self.sup, self.interval_type)
