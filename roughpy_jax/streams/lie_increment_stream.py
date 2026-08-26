import math
import dataclasses
from functools import partial
from collections.abc import Callable
from typing import TypeAlias, TypeVar, Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from roughpy_jax.algebra import (
    FreeTensor,
    Lie,
    LieBasis,
    TensorBasis,
    ft_exp,
    ft_fmexp,
    lie_to_tensor,
    to_log_signature,
)
from roughpy_jax.bases import to_tensor_basis, Basis
from roughpy_jax.intervals import (
    DyadicInterval,
    Interval,
    IntervalType,
    RealInterval,
    intersection,
)

from .concepts import Stream

T = TypeVar("T")


def _zero_lie(basis: LieBasis, batch_dims: tuple[int, ...], dtype: jnp.dtype) -> Lie:
    data = jnp.zeros((*batch_dims, basis.size()), dtype=dtype)
    return Lie(data, basis)


LeftT = TypeVar("LeftT")
RightT = TypeVar("RightT")
AccT = TypeVar("AccT")

DQInitT: TypeAlias = Callable[
    [int, int, int],
    AccT,
]
DQLeftGetterT: TypeAlias = Callable[[int, int, int], LeftT]
DQRightGetterT: TypeAlias = Callable[[int, int, int], RightT]
DQCombineT: TypeAlias = Callable[[LeftT, AccT, RightT], AccT]


def _resolve_short_case(
        inf_trim: int,
        sup_trim: int,
        inf_scaled: float,
        sup_scaled: float,
        is_clopen: bool,
) -> tuple[int, int]:
    if sup_trim < inf_trim:
        return inf_trim, inf_trim

    if is_clopen:
        k1 = inf_trim
        k2 = k1 + (0 if sup_trim == sup_scaled else 1)
        return k1, k2

    k2 = sup_trim
    k1 = k2 - (0 if inf_trim == inf_scaled else 1)

    return k1, k2


def _tree_where(mask: jax.Array, candidate, current):
    """Select query-batched pytree leaves while preserving their trailing axes."""

    def select(candidate_leaf, current_leaf):
        mask_shape = (*mask.shape, *((1,) * (candidate_leaf.ndim - mask.ndim)))
        return jnp.where(mask.reshape(mask_shape), candidate_leaf, current_leaf)

    return jax.tree.map(select, candidate, current)


@partial(jax.tree_util.register_dataclass, data_fields=["cache"],
         meta_fields=["cache_basis", "query_basis", "group_basis"])
@dataclasses.dataclass(frozen=True)
class _QueryContext:
    cache: jax.Array
    cache_basis: Basis
    query_basis: Basis
    group_basis: Basis


@partial(jax.jit,
         static_argnames=("resolution", 'init', 'get_left', 'get_right', 'combine'))
def _query_dyadic_cache(
        infs: jax.Array,
        sups: jax.Array,
        context: Any,
        *,
        resolution: int,
        init: Callable,
        get_left: Callable,
        get_right: Callable,
        combine: Callable
):
    inf_scaled = jnp.ldexp(infs, resolution)
    sup_scaled = jnp.ldexp(sups, resolution)
    inf_integer = jnp.ceil(inf_scaled).astype(jnp.int32)
    sup_integer = jnp.ceil(sup_scaled).astype(jnp.int32)

    empty = (sups <= infs) | (sup_integer <= inf_integer)

    scaled_length = jnp.ldexp(sup_scaled - inf_scaled, -resolution)
    scaled_length = jnp.where(empty, jnp.ones_like(scaled_length), scaled_length)
    _, exponent = jnp.frexp(scaled_length)
    coarse_resolution = 1 - exponent

    # There's three cases to handle here:
    #   1) the interval is non-empty and longer than 2^{-R}
    #   2) query interval is effectively empty
    #   3) query interval is shorter than the dyadic intervals of resolution R.
    # Each case has to be handled carefully. In the normal case, we do an expansion
    # starting from the largest contained dyadic outwards with at most R+1 steps.
    # For the empty case, the result should be the zero Lie. For the short case,
    # we return the Lie of the contained end-point if it exists and zero otherwise.
    # All three cases are computed simulataneously (needed for parallelism) with
    # the init function getting the correct value for the short case.
    short = ~empty & (coarse_resolution > resolution)
    normal = ~empty & ~short
    steps = jnp.where(normal, resolution - coarse_resolution, 0)

    step_scale = jnp.left_shift(jnp.ones_like(steps), steps)
    inf_working = (inf_integer + step_scale - 1) >> steps
    sup_working = sup_integer >> steps

    normal_k_left = inf_working
    normal_k_right = jnp.where(
        sup_working < inf_working,
        inf_working,
        sup_working,
    )
    initial_k_left = jnp.where(normal, normal_k_left, inf_integer)
    initial_k_right = jnp.where(
        normal,
        normal_k_right,
        jnp.where(short, sup_integer, inf_integer),
    )
    initial_resolution = jnp.where(normal, coarse_resolution, resolution)

    accumulator = init(
        context,
        initial_k_left,
        initial_k_right,
        initial_resolution
    )

    inf_difference = (inf_working << steps) - inf_integer
    sup_difference = sup_integer - (sup_working << steps)

    def expand(state, iteration):
        inf_cur, sup_cur, acc_cur = state
        active = normal & (iteration <= steps)
        bit_position = jnp.where(active, steps - iteration, 0)
        res_cur = jnp.where(active, coarse_resolution + iteration, resolution)

        inf_bit = jnp.where(active, (inf_difference >> bit_position) & 1, 0)
        sup_bit = jnp.where(active, (sup_difference >> bit_position) & 1, 0)

        left_k = (inf_cur << 1) - inf_bit
        right_k = sup_cur << 1

        left = get_left(context, left_k, res_cur, inf_bit)
        right = get_right(context, right_k, res_cur, sup_bit)

        candidate = combine(context, left, acc_cur, right)
        acc_next = _tree_where(active, candidate, acc_cur)

        inf_next = jnp.where(active, left_k, inf_cur)
        sup_next = jnp.where(active, right_k + sup_bit, sup_cur)

        return (inf_next, sup_next, acc_next), None

    iterations = jnp.arange(1, resolution + 1, dtype=jnp.int32)
    (_, _, accumulator), _ = jax.lax.scan(
        expand,
        (inf_working, sup_working, accumulator),
        iterations
    )

    return accumulator


def _dyadic_tree_index(cache: jax.Array, k: jax.Array, n: jax.Array) -> jax.Array:
    level_start = cache.shape[0] - jnp.left_shift(1, n + 1)
    return level_start + k


def _dyadic_tree_get_lie(context: _QueryContext, k: jax.Array, n: jax.Array, digit: jax.Array) -> Lie:
    zero_index = context.cache.shape[0] - 1
    index = jnp.where(digit != 0, _dyadic_tree_index(context.cache, k, n), zero_index)
    return Lie(context.cache[index], context.cache_basis)


def _dyadic_query_init_lie(context: _QueryContext, k1: jax.Array, k2: jax.Array, n: jax.Array) -> Lie:
    zero_index = context.cache.shape[0] - 1
    index = jnp.where(k1 == k2, zero_index, _dyadic_tree_index(context.cache, k1, n))
    result = Lie(context.cache[index], context.cache_basis)

    if context.query_basis is not context.cache_basis:
        result = result.change_depth(context.query_basis.depth)

    return result


def _tree_lie_combine(context: _QueryContext, left: Lie, accumulator: Lie, right: Lie) -> Lie:
    tensor_basis = context.group_basis
    result = FreeTensor.identity(tensor_basis, dtype=accumulator.data.dtype, batch_dims=accumulator.data.shape[:-1])
    result = ft_fmexp(result, lie_to_tensor(left), out_basis=tensor_basis)
    result = ft_fmexp(result, lie_to_tensor(accumulator), out_basis=tensor_basis)
    result = ft_fmexp(result, lie_to_tensor(right), out_basis=tensor_basis)
    return to_log_signature(result, context.query_basis)


def dyadic_query(
        query: Interval,
        resolution: int,
        init: DQInitT,
        get_left: DQLeftGetterT,
        get_right: DQRightGetterT,
        combine: DQCombineT,
        cache_interval_type: IntervalType = IntervalType.ClOpen,
) -> AccT:
    is_clopen = cache_interval_type == IntervalType.ClOpen

    inf_scaled = math.ldexp(query.inf, resolution)
    sup_scaled = math.ldexp(query.sup, resolution)
    inf = math.ceil(inf_scaled)
    sup = math.floor(sup_scaled)

    # If query and cache interval types differ, nudge the endpoint that is
    # excluded by the query but included by the cache inward at dyadic points.
    if query.interval_type != cache_interval_type:
        if query.interval_type == IntervalType.OpenCl and is_clopen:
            if inf_scaled == inf:
                inf += 1
                inf_scaled = inf
        elif query.interval_type == IntervalType.ClOpen and not is_clopen:
            if sup_scaled == sup:
                sup -= 1
                sup_scaled = sup

    if sup < inf:
        return init(inf, inf, resolution)

    effective_width = math.ldexp(sup_scaled - inf_scaled, -resolution)
    _, exponent = math.frexp(effective_width)
    coarse_resolution = 1 - exponent

    # When the query interval is smaller than the shortest dyadics provided here then special care is needed
    # We have to determine if the included end of any max-resolution interval is contained in the query interval.
    # This should be the case if the rounded inf and sup are different, in which case it is just a matter of
    # selecting the one that lies inside the interval. Which endpoint this is depends on the direction of rounding.
    if coarse_resolution > resolution:
        k1, k2 = _resolve_short_case(inf, sup, inf_scaled, sup_scaled, is_clopen)
        return init(k1, k2, resolution)

    steps = resolution - coarse_resolution
    inf_working = (inf + ((1 << steps) - 1)) >> steps
    sup_working = sup >> steps

    if sup_working < inf_working:
        return init(inf_working, inf_working, coarse_resolution)

    result = init(inf_working, sup_working, coarse_resolution)

    for i in range(1, steps + 1):
        j = steps - i
        r = coarse_resolution + i

        inf_bit = (inf >> j) & 1
        sup_bit = (sup >> j) & 1

        # The update of working values needs to be handled with care. This depends on whether
        # the dyadic intervals are open on the right or left. In the first case (clopen) the
        # value passed as the first argument to get_right should be 2*sup_working and the
        # new bit should be added after this call. For get_left, the first argument should
        # be the fully updated 2*inf_working - inf_bit. In the second case, the reverse
        # holds: inf_working has a two-stage update and sup_working has a one-stage update.
        if is_clopen:
            left_k = (inf_working << 1) - inf_bit
            right_k = sup_working << 1
            left = get_left(left_k, r, inf_bit)
            right = get_right(right_k, r, sup_bit)
            inf_working = left_k
            sup_working = right_k + sup_bit
        else:
            left_k = inf_working << 1
            right_k = (sup_working << 1) + sup_bit
            left = get_left(left_k, r, inf_bit)
            right = get_right(right_k, r, sup_bit)
            inf_working = left_k - inf_bit
            sup_working = right_k

        result = combine(left, result, right)

    return result


def _make_finest_increment_body(state, current, *, input_lie_basis, cache_lie_basis, tensor_basis):
    current_bucket, current_acc, current_out = state
    bucket, data = current

    current_tensor = lie_to_tensor(Lie(data, input_lie_basis))

    def same_bucket():
        return ft_fmexp(current_acc, current_tensor, out_basis=tensor_basis), current_out

    def next_bucket():
        next_acc = ft_exp(
            current_tensor.change_depth(tensor_basis.depth),
            out_basis=tensor_basis,
        )

        logsig = to_log_signature(current_acc, cache_lie_basis)
        next_out = current_out.at[current_bucket, ...].set(logsig.data)

        return next_acc, next_out

    acc, out = jax.lax.cond(bucket == current_bucket, same_bucket, next_bucket)

    return (bucket, acc, out), None


@partial(jax.jit, static_argnames=("resolution", "cache_lie_basis", "input_lie_basis"))
def _make_finest_increment_level(buckets: jax.Array, data: jax.Array, *, resolution: int,
                                 cache_lie_basis: LieBasis, input_lie_basis: LieBasis) -> jax.Array:
    num_buckets = 1 << resolution
    tensor_basis = to_tensor_basis(cache_lie_basis)

    order = jnp.argsort(buckets, stable=True)
    buckets = buckets[order]
    data = data[order, ...]

    time_dim, *batch_dims, data_dim = data.shape

    assert time_dim == buckets.shape[0]

    lie_dim = cache_lie_basis.size()
    output = jnp.zeros(
        (num_buckets, *batch_dims, lie_dim),
        dtype=data.dtype,
    )
    accumulator = FreeTensor.identity(tensor_basis, dtype=data.dtype, batch_dims=tuple(batch_dims))
    first_bucket = buckets[0]

    body_fn = partial(
        _make_finest_increment_body,
        tensor_basis=tensor_basis,
        cache_lie_basis=cache_lie_basis,
        input_lie_basis=input_lie_basis
    )

    (final_bucket, final_acc, final_out), _ = jax.lax.scan(
        body_fn,
        (first_bucket, accumulator, output),
        (buckets, data)
    )

    final_logsig = to_log_signature(final_acc, cache_lie_basis)

    return final_out.at[final_bucket, ...].set(final_logsig.data)


@partial(jax.jit, static_argnames=("resolution", "cache_lie_basis"))
def _extend_from_finest_level(finest: jax.Array, *, resolution: int, cache_lie_basis: LieBasis) -> jax.Array:
    levels = [finest]

    for i in range(resolution):
        left = Lie(levels[i][0::2, ...], cache_lie_basis)
        right = Lie(levels[i][1::2, ...], cache_lie_basis)

        next_level = ft_exp(lie_to_tensor(left))
        next_level = ft_fmexp(next_level, lie_to_tensor(right))

        levels.append(to_log_signature(next_level, cache_lie_basis).data)

    zero = jnp.zeros(
        (1, *finest.shape[1:]),
        dtype=finest.dtype
    )
    levels.append(zero)

    return jnp.concatenate(levels, axis=0)


class LieIncrementStream(Stream[Lie, FreeTensor]):
    """
    Stream backed by a contiguous cache of dyadic log-signatures.

    The cache is a JAX array with shape (2^(R+1), ..., LieDim), where the
    cache axis packs log-signatures over dyadic intervals of lengths between
    2^-R and 1 in steps of 2. The final element of the cache axis is unused
    by the geometric series of dyadic intervals and should be zero.
    """

    @staticmethod
    def _cache_length_from_resolution(resolution: int) -> int:
        return 1 << (int(resolution) + 1)

    def __init__(
            self,
            cache: jnp.ndarray,
            lie_basis: LieBasis,
            resolution: int,
            support: Interval | None = None,
            group_basis: TensorBasis | None = None,
            interval_type: IntervalType = IntervalType.ClOpen,
    ):
        if cache.ndim < 2:
            raise ValueError("cache must have shape (cache_length, ..., lie_dim)")

        lie_dim = int(lie_basis.size())
        if cache.shape[-1] != lie_dim:
            raise ValueError(
                f"cache lie dimension mismatch: expected {lie_dim}, got {cache.shape[-1]}"
            )

        cache_length = int(cache.shape[0])
        expected_length = self._cache_length_from_resolution(resolution)
        if cache_length != expected_length:
            raise ValueError(
                f"cache length mismatch for resolution {resolution}: "
                f"expected {expected_length}, got {cache_length}"
            )

        self._cache = cache
        self._lie_basis = lie_basis
        self._group_basis = group_basis or to_tensor_basis(lie_basis)
        self._support = support or RealInterval(0.0, 1.0, IntervalType.ClOpen)
        self._resolution = int(resolution)
        self._interval_type = interval_type
        self._zero_index = cache_length - 1

    @staticmethod
    def _stream_to_cache(
            stream: Stream,
            resolution: int,
            interval_type: IntervalType = IntervalType.ClOpen,
    ) -> jnp.ndarray:

        inf = stream.support.inf
        scale_factor = stream.support.sup - inf

        def reparam(di):
            return RealInterval(
                di.inf * scale_factor + inf,
                di.sup * scale_factor + inf,
                interval_type,
            )

        def f(k, r):
            di = DyadicInterval(k, r, interval_type)
            query = reparam(di)
            return stream.log_signature(query)

        finest = jnp.stack(
            [f(k, resolution).data for k in range(1 << resolution)],
            axis=0,
        )

        return _extend_from_finest_level(
            finest,
            resolution=resolution,
            cache_lie_basis=stream.lie_basis,
        )

    @classmethod
    def from_stream(cls: type[T], stream: Stream[Lie, FreeTensor], resolution: int) -> T:
        lie_basis = stream.lie_basis
        group_basis = stream.group_basis
        support = stream.support

        if resolution <= 0:
            raise ValueError(f"resolution must be positive, got {resolution}")

        if (fun := getattr(stream, "__dyadic_cache__", None)) is not None:
            cache = jnp.asarray(fun(resolution))
        else:
            cache = cls._stream_to_cache(stream, resolution)  # ty: ignore[unresolved-attribute]

        new_stream = cls(
            cache=cache,
            lie_basis=lie_basis,
            support=support,
            group_basis=group_basis,
            resolution=resolution,
        )

        new_stream.__base_stream__ = stream  # ty: ignore[unresolved-attribute]

        return new_stream

    @classmethod
    def from_increments(
            cls: type[T],
            timestamps: ArrayLike | list[jax.Array],
            data: ArrayLike | list[jax.Array],
            *,
            resolution: int | None,
            input_data_basis: LieBasis | None,
            lie_basis: LieBasis | None,
            interval_type=IntervalType.ClOpen,
            data_dtype: jnp.dtype | None = None,
            time_dtype: jnp.dtype = jnp.float32.dtype,
            dyadic_integer_type: jnp.dtype = jnp.int32.dtype,
            **kwargs,
    ) -> T:

        if isinstance(timestamps, list):
            time_arrays = [jnp.asarray(ts) for ts in timestamps]
        else:
            time_arrays = [jnp.asarray(timestamps)]

        if isinstance(data, list):
            data_arrays = [jnp.asarray(ds) for ds in data]
        else:
            data_arrays = [jnp.asarray(data)]

        if not time_arrays or not data_arrays:
            raise ValueError("timestamps and data cannot be empty")

        if not len(time_arrays) == len(data_arrays):
            raise ValueError("timestamps and data must be the same length")

        time_lens = []
        mins = []
        maxs = []
        for ts in time_arrays:
            if ts.ndim != 1:
                raise ValueError("timestamps must be held in 1D arrays")

            time_lens.append(ts.shape[0])
            mins.append(jnp.min(ts))
            maxs.append(jnp.max(ts))

        ds = data_arrays[0]
        if ds.ndim < 2:
            raise ValueError("data arrays must be at least 2D")

        dt_dim, *batch_dims, lie_dim = ds.shape
        if dt_dim != time_lens[0]:
            raise ValueError(
                f"Time dimension mismatch at index 0: expected {time_lens[0]}, got {dt_dim}"
            )

        dtypes = [ds.dtype]
        for i, (ds, expected_dt) in enumerate(
                zip(data_arrays[1:], time_lens[1:], strict=True), start=1
        ):
            if ds.ndim < 2:
                raise ValueError("data arrays must be at least 2D")

            dt_dim, *b_dims, l_dim = ds.shape
            if dt_dim != expected_dt:
                raise ValueError(
                    f"Time dimension mismatch at index {i}: expected {expected_dt}, got {dt_dim}"
                )

            if b_dims != batch_dims:
                raise ValueError(
                    f"Batch dimension mismatch at index {i}: expected {batch_dims}, got {b_dims}"
                )

            dtypes.append(ds.dtype)

            if input_data_basis is not None:
                basis_size = input_data_basis.size()
                if l_dim > basis_size:
                    raise ValueError(
                        f"data dimension {l_dim} is incompatible with the specified data basis with size {basis_size}"
                    )
            elif l_dim != lie_dim:
                raise ValueError(
                    f"unable to determine appropriate data basis: inconsistent data dimensions at index {i}"
                )

        dtype = data_dtype or jnp.result_type(*dtypes)
        input_data_basis = input_data_basis or LieBasis(width=lie_dim, depth=1)
        basis_size = input_data_basis.size()

        for i in range(len(data_arrays)):
            *shape, l_dim = data_arrays[i].shape
            padding = [(0, 0)] * len(shape) + [(0, basis_size - l_dim)]
            data_arrays[i] = jnp.pad(data_arrays[i].astype(dtype), padding)

        if lie_basis is None:
            lie_basis = LieBasis(width=input_data_basis.width, depth=2)

        # Now sort out the support and scale the data
        if interval_type == IntervalType.ClOpen:
            sup = jnp.nextafter(max(maxs), jnp.inf)
            inf = min(mins)
        else:  # interval_type == IntervalType.OpenCl:
            sup = max(maxs)
            inf = jnp.nextafter(min(mins), -jnp.inf)

        support = RealInterval(inf, sup, interval_type)

        # Adjust the timestamps so they lie in the unit interval
        sf = time_dtype.type(sup - inf)
        shift = time_dtype.type(inf)
        time_arrays = [(ts.astype(time_dtype) - shift) / sf for ts in time_arrays]

        if resolution is None:
            min_diff = min(jnp.min(jnp.diff(ts, axis=-1)) for ts in time_arrays)
            _, exp = jnp.frexp(min_diff)
            resolution = int(1 - exp)

        tensor_basis = to_tensor_basis(lie_basis)

        rounder = jnp.floor if interval_type == IntervalType.ClOpen else jnp.ceil
        k_arrays = [
            rounder(jnp.ldexp(ts, resolution)).astype(dyadic_integer_type)
            for ts in time_arrays
        ]

        base = jnp.stack([
            _make_finest_increment_level(ks, ds,
                                         resolution=resolution,
                                         cache_lie_basis=lie_basis,
                                         input_lie_basis=input_data_basis)
            for ks, ds in zip(k_arrays, data_arrays)
        ], axis=1)

        cache = _extend_from_finest_level(base, resolution=resolution, cache_lie_basis=lie_basis)

        return cls(
            cache,
            lie_basis,
            resolution,
            support=support,
            group_basis=tensor_basis,
            **kwargs,
        )

    @property
    def lie_basis(self) -> LieBasis:
        return self._lie_basis

    @property
    def group_basis(self) -> TensorBasis:
        return self._group_basis

    @property
    def support(self) -> Interval:
        return self._support

    @property
    def dtype(self):
        return self._cache.dtype

    @property
    def batch_dims(self) -> tuple[int, ...]:
        return self._cache.shape[1:-1]

    @property
    def resolution(self) -> int:
        return self._resolution

    def _zero_log_signature(self) -> Lie:
        return Lie(self._cache[-1, ...], self._lie_basis)

    def _query_dyadic(self, k: int, n: int) -> Lie:
        level_start = (1 << (self._resolution + 1)) - (1 << (n + 1))
        return Lie(self._cache[level_start + k, ...], self._lie_basis)

    def _reparamterise(self, interval: Interval) -> RealInterval:
        inf = self.support.inf
        length = self.support.sup - self.support.inf

        return RealInterval(
            (interval.inf - inf) / length,
            (interval.sup - inf) / length,
            interval.interval_type,
        )

    #
    # def _query_init(self, k1: int, k2: int, n: int) -> Lie:
    #     if k1 == k2:
    #         return self._zero_log_signature()
    #
    #     if self._interval_type == IntervalType.ClOpen:
    #         return self._query_dyadic(k1, n)
    #
    #     return self._query_dyadic(k2, n)
    #
    # def _query_get(self, k: int, n: int, digit: int) -> Lie:
    #     if not digit:
    #         return self._zero_log_signature()
    #
    #     return self._query_dyadic(k, n)
    #
    # def _query_combine(self, left: Lie, acc: Lie, right: Lie) -> Lie:
    #     ft_result = FreeTensor.identity(
    #         self._group_basis, dtype=acc.data.dtype, batch_dims=self._cache.shape[1:-1]
    #     )
    #
    #     ft_result = ft_fmexp(ft_result, lie_to_tensor(left))
    #     ft_result = ft_fmexp(ft_result, lie_to_tensor(acc))
    #     ft_result = ft_fmexp(ft_result, lie_to_tensor(right))
    #
    #     return to_log_signature(ft_result)

    def log_signature(self, interval: Interval | None = None) -> Lie:
        """
        Compute the log signature over an interval.

        Whilst intervals do support batching as arrays, and lie increment
        streams may be amenable to batched log-signature calculation, this
        functionality is not yet enabled. For now, only single intervals
        will be accepted by this method. This may change in a future release.
        """
        if interval is None:
            interval = self._support

        inf = jnp.asarray(interval.inf)
        sup = jnp.asarray(interval.sup)
        # if inf.size != 1 or sup.size != 1:
        #     raise ValueError(
        #         "LieIncrementStream only supports scalar interval endpoints "
        #         "or single-element endpoint arrays"
        #     )
        # if inf.shape or sup.shape:
        #     interval = RealInterval(inf.reshape(()), sup.reshape(()), interval.interval_type)

        clipped_inf = jnp.clip(inf, self._support.inf, self._support.sup)
        clipped_sup = jnp.clip(sup, self._support.inf, self._support.sup)
        # query_interval = intersection(interval, self.support)
        # if jnp.all(query_interval.length == 0):
        #     return self._zero_log_signature()

        # reparam_query = self._reparamterise(query_interval)
        reparam_inf = (clipped_inf - self._support.inf) / (self._support.sup - self._support.inf)
        reparam_sup = (clipped_sup - self._support.inf) / (self._support.sup - self._support.inf)

        context = _QueryContext(self._cache, self._lie_basis, self._lie_basis, self._group_basis)

        result = _query_dyadic_cache(
            reparam_inf,
            reparam_sup,
            context,
            resolution=self._resolution,
            init=_dyadic_query_init_lie,
            get_left=_dyadic_tree_get_lie,
            get_right=_dyadic_tree_get_lie,
            combine=_tree_lie_combine,
        )

        # result = dyadic_query(
        #     reparam_query,
        #     self._resolution,
        #     self._query_init,
        #     self._query_get,
        #     self._query_get,
        #     self._query_combine,
        #     self._interval_type,
        # )

        return result

    def signature(
            self,
            interval: Interval | None = None,
    ) -> FreeTensor:
        """
        Compute the signature over an interval.

        Whilst intervals do support batching as arrays, and lie increment
        streams may be amenable to batched signature calculation, this
        functionality is not yet enabled. For now, only single intervals
        will be accepted by this method. This may change in a future release.
        """
        log_sig = self.log_signature(interval)
        tensor = lie_to_tensor(log_sig)
        return ft_exp(tensor, out_basis=self._group_basis)
