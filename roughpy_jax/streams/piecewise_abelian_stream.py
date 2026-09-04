from dataclasses import dataclass
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp

from roughpy_jax.algebra import (
    DenseFreeTensor,
    DenseLie,
    FreeTensor,
    ft_fmexp,
    lie_to_tensor,
    to_log_signature,
    to_signature,
)
from roughpy_jax.bases import Basis, LieBasis, TensorBasis, check_basis_compat, to_tensor_basis
from roughpy_jax.intervals import Interval, Partition, RealInterval

from .concepts import Stream


@partial(
    jax.tree_util.register_dataclass,
    data_fields=["_data", "_partition"],
    meta_fields=["_lie_basis", "_group_basis"],
)
@dataclass(frozen=True)
class PiecewiseAbelianStream(Stream[DenseLie, DenseFreeTensor]):
    """A stream whose log-signature is linear on each partition interval.

    The partition divides one shared time domain into consecutive intervals. The
    Lie element at position ``i`` is the log-signature increment over interval
    ``i``. Within that interval, an increment over a subinterval is obtained by
    scaling the stored Lie element by the proportion of the interval covered.
    Increments from successive pieces are combined in chronological order using
    the tensor product, or equivalently the Campbell--Baker--Hausdorff product
    after returning to the Lie algebra.

    A piecewise abelian stream must be defined over an unbatched partition. A
    batched partition represents several different subdivisions of time, whose
    intervals would require different corresponding piece data rather than a
    single sequence of Lie increments. Batching is instead supported in the Lie
    data: the coefficient array has a leading piece axis followed by the stream
    batch dimensions and the Lie coordinate axis.

    Query intervals may also be batched. Query batch dimensions precede the
    stream-data batch dimensions in the result. Thus a query with batch shape
    ``Q`` against stream data with batch shape ``D`` produces coefficient data
    with shape ``Q + D + (basis_size,)``.

    Args:
        _data: Lie increment coefficients with shape
            ``(n_pieces, *batch_dims, lie_basis_size)``.
        _partition: The unbatched partition defining the temporal pieces.
        _lie_basis: Basis used for Lie-valued increments and log-signatures.
        _group_basis: Tensor basis used to combine increments and form
            signatures.
    """

    _data: jax.Array
    _partition: Partition
    _lie_basis: LieBasis
    _group_basis: TensorBasis

    def __post_init__(self):
        """Validate the piecewise abelian stream."""
        if self._data.ndim < 2:
            raise ValueError(
                "Piecewise abelian stream data must have shape "
                "(n_pieces, ..., lie_basis_size)."
            )

        if self._data.shape[0] != len(self._partition):
            raise ValueError(
                f"Data length {self._data.shape[0]} must match number "
                f"of intervals in partition {len(self._partition)}."
            )

        if self._data.shape[-1] != self._lie_basis.size():
            raise ValueError(
                f"Data Lie dimension {self._data.shape[-1]} must match "
                f"Lie basis size {self._lie_basis.size()}."
            )

        if len(self._partition.batch_dims) > 0:
            raise ValueError(
                "A piecewise abelian stream cannot be defined over a batched partition."
            )

        check_basis_compat(self._lie_basis, self._group_basis, exact=True)

    @property
    def lie_basis(self) -> Basis:
        """Return the Lie basis."""
        return self._lie_basis

    @property
    def group_basis(self) -> Basis:
        """Return the group basis."""
        return self._group_basis

    @property
    def support(self) -> Interval:
        """Return the support interval."""
        return RealInterval(
            _inf=self._partition.inf,
            _sup=self._partition.sup,
            _interval_type=self._partition.interval_type,
        )

    @property
    def dtype(self):
        """Return the coefficient dtype of the stream values."""
        return self._data.dtype

    @property
    def batch_dims(self) -> tuple[int, ...]:
        """Return the leading batch dimensions of the stream values."""
        return self._data.shape[1:-1]

    @jax.jit
    def log_signature(self, interval: Interval) -> DenseLie:
        """Compute the log-signature over one or more query intervals.

        For batched interval endpoints, query dimensions precede any batch
        dimensions carried by the stream data in the returned coefficients.

        Args:
            interval: Scalar or batched interval over which to query the stream.

        Returns:
            The log-signature over ``interval`` in the stream's Lie basis.
        """
        inf, sup = jnp.broadcast_arrays(
            jnp.asarray(interval.inf),
            jnp.asarray(interval.sup),
        )

        P = len(self._partition)
        partition_intervals = self._partition.to_intervals()
        query_dims = (1,) * inf.ndim
        partition_inf = partition_intervals.inf.reshape((P,) + query_dims)
        partition_sup = partition_intervals.sup.reshape((P,) + query_dims)

        query_inf = inf[None, ...]
        query_sup = sup[None, ...]

        begin = jnp.maximum(partition_inf, query_inf)
        end = jnp.minimum(partition_sup, query_sup)

        overlap = jnp.maximum(0.0, end - begin)
        length = partition_sup - partition_inf
        pos_length = length > 0
        length = jnp.where(pos_length, length, 1.0)

        data = self._data
        scale_factors = jnp.where(pos_length, overlap / length, 0.0).astype(
            data.dtype
        )

        # The batch dimensions on the path data to be CBH should be
        #   (P, Q1, ..., Qk, D1, ..., Dm, L)
        # where Q1, ..., Qk are the batch dimensions on the query interval and
        # D1, ..., Dm are the batch dimensions of the stream data. We insert some
        # extra dimensions to facilitate this specific broadcast.
        scale_factors_extra_dims = (1,) * (data.ndim - 1)
        data_extra_dims = (1,) * (scale_factors.ndim - 1)

        scale_factors = scale_factors.reshape(scale_factors.shape + scale_factors_extra_dims)
        data = data.reshape((P,) + data_extra_dims + data.shape[1:])

        path_data = scale_factors * data

        initial = FreeTensor.identity(self._group_basis, dtype=data.dtype, batch_dims=path_data.shape[1:-1])

        def combine(carry, piece_data):
            piece = DenseLie(piece_data, self._lie_basis)
            update = ft_fmexp(carry, lie_to_tensor(piece), self._group_basis)
            return update, None

        result_tensor, _ = jax.lax.scan(combine, initial, path_data)
        return to_log_signature(result_tensor)

    @jax.jit
    def signature(self, interval: Interval) -> DenseFreeTensor:
        """Compute the signature over one or more query intervals.

        For batched interval endpoints, query dimensions precede any batch
        dimensions carried by the stream data in the returned coefficients.

        Args:
            interval: Scalar or batched interval over which to query the stream.

        Returns:
            The signature over ``interval`` in the stream's group basis.
        """
        log_sig = self.log_signature(interval)
        return to_signature(log_sig, tensor_basis=self._group_basis)


def to_piecewise_abelian_stream(
        stream: Stream[DenseLie, DenseFreeTensor], partition: Partition
) -> PiecewiseAbelianStream:
    """Approximate a stream by a piecewise abelian stream on a partition.

    The source stream is queried over every interval of ``partition`` in one
    batched call. Each resulting log-signature becomes the Lie increment for the
    corresponding piece of the new stream. Consequently, the converted stream
    has the same increment as ``stream`` over every complete partition interval,
    while its behaviour inside each interval is the piecewise abelian
    interpolation of that increment.

    The partition must be unbatched because it defines one sequence of temporal
    pieces shared by all stream-data batches. Batch dimensions already carried by
    ``stream`` are preserved in every increment of the converted stream.

    Args:
        stream: Source stream whose partition increments will be sampled. Its
            ``log_signature`` method must accept the batched interval returned by
            :meth:`Partition.to_intervals`.
        partition: Unbatched partition defining the pieces of the converted
            stream.

    Returns:
        A piecewise abelian stream over ``partition`` using the source stream's
        Lie and group bases.

    Raises:
        ValueError: If ``partition`` has batch dimensions.
    """
    if len(partition.batch_dims) > 0:
        raise ValueError("batched partitions for piecewise abelian streams are not supported")

    intervals = partition.to_intervals()
    log_sigs = stream.log_signature(intervals)

    new_stream = PiecewiseAbelianStream(
        log_sigs.data,
        partition,
        stream.lie_basis,  # ty: ignore[invalid-argument-type]
        stream.group_basis,  # ty: ignore[invalid-argument-type]
    )

    return new_stream


def piecewise_abelian_stream_from_data(data: DenseLie | tuple[DenseLie, ...] | list[DenseLie],
                                       partition: Partition) -> PiecewiseAbelianStream:
    """
    Construct a PiecewiseAbelianStream from Lie data and partition
    :param data: Lie data with shape (P, ..., L)
    :param partition: partition with P intervals (must not be batched)
    :return: New PiecewiseAbelianStream instance
    """
    partition_size = len(partition)
    if len(partition.batch_dims) > 0:
        raise ValueError("batched partitions for piecewise abelian streams are not supported")

    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            raise ValueError("data must not be empty")

        lie_basis = data[0].basis
        check_basis_compat(*(lie.basis for lie in data), exact=True, same_type=True)
        lie_data = jnp.stack([item for item in data])
    else:
        lie_basis = data.basis
        lie_data = data.data

    partition_check, *batch_dims, lie_dim = lie_data.shape

    if partition_check != partition_size:
        raise ValueError("data shape does not match partition size")

    if lie_dim != lie_basis.size():
        raise ValueError("data shape does not match Lie basis size")

    group_basis = to_tensor_basis(lie_basis)

    return PiecewiseAbelianStream(lie_data, partition, lie_basis, group_basis)
