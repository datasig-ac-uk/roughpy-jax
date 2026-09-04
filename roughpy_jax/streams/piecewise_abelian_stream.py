from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from roughpy_jax.algebra import (
    DenseFreeTensor,
    DenseLie,
    FreeTensor,
    ft_fmexp,
    lie_to_tensor,
    to_log_signature,
    to_signature,
)
from roughpy_jax.bases import Basis, LieBasis, TensorBasis
from roughpy_jax.intervals import Interval, Partition, RealInterval, intersection

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
    data: every piece may contain the same leading batch dimensions while sharing
    the one partition.

    Query intervals may also be batched. Query batch dimensions precede the
    stream-data batch dimensions in the result. Thus a query with batch shape
    ``Q`` against stream data with batch shape ``D`` produces coefficient data
    with shape ``Q + D + (basis_size,)``.

    Args:
        _data: One Lie increment for each interval in ``_partition``. All
            increments must use the same basis and have the same batch shape and
            dtype.
        _partition: The unbatched partition defining the temporal pieces.
        _lie_basis: Basis used for Lie-valued increments and log-signatures.
        _group_basis: Tensor basis used to combine increments and form
            signatures.
    """

    _data: tuple[DenseLie, ...]
    _partition: Partition
    _lie_basis: LieBasis
    _group_basis: TensorBasis

    def __post_init__(self):
        """Validate the piecewise abelian stream."""
        if len(self._data) != len(self._partition):
            raise ValueError(
                f"Data length {len(self._data)} must match number "
                f"of intervals in partition {len(self._partition)}."
            )

        if len(self._partition.batch_dims) > 0:
            raise ValueError(
                "A piecewise abelian stream cannot be defined over a batched partition."
            )

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
        return self._data[0].data.dtype

    @property
    def batch_dims(self) -> tuple[int, ...]:
        """Return the leading batch dimensions of the stream values."""
        return self._data[0].data.shape[:-1]

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
        inf = jnp.asarray(interval.inf)
        sup = jnp.asarray(interval.sup)

        P = len(self._partition)
        partition_intervals = self._partition.to_intervals()
        partition_inf = partition_intervals.inf.reshape((P,) + (1,) * interval.inf.ndim)
        partition_sup = partition_intervals.sup.reshape((P,) + (1,) * interval.sup.ndim)

        query_inf = inf[None, ...]
        query_sup = sup[None, ...]

        begin = jnp.maximum(partition_inf, query_inf)
        end = jnp.minimum(partition_sup, query_sup)

        overlap = jnp.maximum(0.0, end - begin)
        length = partition_sup - partition_inf
        pos_length = length > 0
        length = jnp.where(pos_length, length, 1.0)

        scale_factors = jnp.where(pos_length, overlap / length, 0.0).astype(self._data[0].dtype)

        # TODO: This is unnecessary once we replace the PAS internals to use an array
        # instead of a tuple of Lie
        data = jnp.stack([piece.data for piece in self._data])

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

    # TODO: This will be fixed once the internals of PiecewiseAbelianStream have been streamlined
    data = tuple(DenseLie(log_sigs.data[i, ...], log_sigs.basis) for i in range(len(partition)))
    new_stream = PiecewiseAbelianStream(
        data,
        partition,
        stream.lie_basis,  # ty: ignore[invalid-argument-type]
        stream.group_basis,  # ty: ignore[invalid-argument-type]
    )

    return new_stream
