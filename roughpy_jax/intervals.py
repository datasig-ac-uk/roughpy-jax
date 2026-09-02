from __future__ import annotations

import enum
import typing
from dataclasses import FrozenInstanceError, dataclass
from typing import Protocol, TypeVar

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
        self, k: ArrayLike, n: ArrayLike, interval_type: IntervalType=IntervalType.ClOpen
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
    def inf(self) ->  Array:
        return self._inf

    @property
    def sup(self) ->  Array:
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


@dataclass(frozen=True)
class Partition:
    _endpoints: list
    _interval_type: IntervalType

    def __len__(self) -> int:
        return len(self._endpoints) - 1

    def __str__(self) -> str:
        return BaseInterval.to_string(self)

    @property
    def interval_type(self) -> IntervalType:
        return self._interval_type

    @property
    def inf(self) -> Array:
        return jnp.asarray(self._endpoints[0])

    @property
    def sup(self) -> Array:
        return jnp.asarray(self._endpoints[-1])

    @property
    def length(self) -> Array:
        return BaseInterval.length(self)

    def to_intervals(self) -> list[RealInterval]:
        """
        Convert the partition into a list of RealIntervals corresponding to
        the subintervals defined by the partition.
        :param partition: The Partition to convert.
        :type partition: Partition
        :return: A list of RealIntervals representing the subintervals of the partition.
        :rtype: list[RealInterval]
        """
        return [
            RealInterval(
                self._endpoints[i], self._endpoints[i + 1], self.interval_type
            )
            for i in range(len(self._endpoints) - 1)
        ]

    @staticmethod
    def to_real_interval(partition: Partition) -> RealInterval[RealT]:
        """
        Convert the partition to a RealInterval.
        :return: A RealInterval representing the partition.
        :rtype: RealInterval
        """
        return RealInterval(partition.inf, partition.sup, partition.interval_type)

    @classmethod
    def truncate(
        cls, partition: Partition, other: Interval
    ) -> Partition | RealInterval:
        """
        Calculate the intersection of this partition with another Interval.
        :param other: The other interval to intersect with.
        :type other: RealInterval
        :return: A new Partition representing the intersection, or a degenerate
        interval if there is no intersection.
        :rtype: Partition | RealInterval
        """
        # Here we convert to RealInterval to perform the intersection logic
        intermediate_itvl = cls.to_real_interval(partition)
        intersect_itvl = RealInterval.intersection(intermediate_itvl, other)
        if intersect_itvl.length == 0:
            return intersect_itvl  # Return the degenerate interval representing the empty intersection

        new_endpoints = []
        # 1) Add new inf if it is within bounds of old interval
        if partition.inf < intersect_itvl.inf:
            new_endpoints.append(intersect_itvl.inf)
        # 2) Include all inner points
        for ep in partition._endpoints:
            if intersect_itvl.inf <= ep <= intersect_itvl.sup:
                new_endpoints.append(ep)
        # 3) Add new sup if within bounds of old interval
        if intersect_itvl.sup < partition.sup:
            new_endpoints.append(intersect_itvl.sup)

        return cls(
            _endpoints=new_endpoints,
            _interval_type=partition.interval_type,
        )

    @classmethod
    def merge(cls, left: Partition, right: Partition) -> Partition | RealInterval:
        """
        Merge this partition with another Partition.
        :param other: The other partition to merge with.
        :type other: Partition
        :return: A new Partition representing the merged partitions.
        :rtype: Partition
        """
        if left.interval_type != right.interval_type:
            raise TypeError("Both partitions must be of the same IntervalType")

        left_itvl = cls.to_real_interval(left)
        right_itvl = cls.to_real_interval(right)
        inters = RealInterval.intersection(left_itvl, right_itvl)
        if inters.length == 0:
            return inters  # Return the degenerate interval representing the empty intersection

        new_endpoints = sorted(set(left._endpoints) | set(right._endpoints))
        return cls(
            _endpoints=new_endpoints,
            _interval_type=left.interval_type,
        )


Partition = jax.tree_util.register_dataclass(
    Partition,
    data_fields=["_endpoints"],
    meta_fields=["_interval_type"],
)
