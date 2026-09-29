"""Shared storage and batch utilities for dense algebra elements.

Dense algebra coefficient arrays have shape ``(*batch_shape, basis.size())``.
The final dimension is always the algebra coordinate; every preceding
dimension, including a zero-length dimension, belongs to the batch shape.
"""

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Generic, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

from roughpy_jax.bases import BasisT, TensorBasis, result_basis

AlgebraT = TypeVar("AlgebraT", bound="DenseAlgebra[Any]")


def get_batch_shape(operand) -> tuple[int, ...]:
    """Return the batch shape associated with an operand.

    Dense algebra objects contribute their ``batch_shape`` property, i.e. the
    leading dimensions before the trailing algebra coordinate. Plain arrays are
    treated as pure batch data and therefore contribute their full shape.

    :param operand: Dense algebra object or array-like operand.
    :return: Batch shape for the operand.
    """
    if (batch_shape := getattr(operand, "batch_shape", None)) is not None:
        return batch_shape

    return jnp.shape(operand)


def get_common_batch_shape(*operands) -> tuple[int, ...]:
    """Validate that all operands share the same batch shape and return it.

    Dense algebra objects contribute the leading dimensions before the trailing
    algebra coordinate. Plain arrays contribute their full shape. This helper
    performs exact equality validation; it does not apply broadcasting rules.

    :param operands: Dense algebra objects or array-like operands to validate.
    :return: The batch shape common to all operands.
    :raises ValueError: If no operands are supplied or any operand has a
        different batch shape.
    """
    if not operands:
        raise ValueError("expected at least one operand")

    first, *rem = operands
    batch_shape = get_batch_shape(first)

    for i, operand in enumerate(rem, start=1):
        operand_batch_shape = get_batch_shape(operand)
        if operand_batch_shape != batch_shape:
            raise ValueError(
                f"incompatible batch shape in argument at index {i}:"
                f" expected {batch_shape} but got {operand_batch_shape}"
            )

    return batch_shape


def broadcast_to_batch_shape(
    data: jax.typing.ArrayLike,
    batch_shape: tuple[int, ...],
    *,
    core_dims: int = 1,
) -> jax.Array:
    """Reshape data for broadcasting over a target batch shape and core dimensions.

    Scalars and exact prefixes of ``batch_shape`` are reshaped by appending
    singleton dimensions, followed by ``core_dims`` singleton core dimensions.
    This deliberately implements prefix broadcasting rather than general
    NumPy broadcasting. Inputs already shaped as
    ``batch_shape + (1,) * core_dims`` are returned unchanged.

    :param data: Array-like input to reshape.
    :param batch_shape: Target batch shape.
    :param core_dims: Number of trailing singleton core dimensions required in
        the result.
    :return: Reshaped array suitable for broadcasting over an algebra value.
    :raises ValueError: If the input shape is neither a prefix of
        ``batch_shape`` nor ``batch_shape + (1,) * core_dims``.
    """
    data = jnp.asarray(data)
    data_shape = data.shape

    ds_len = len(data_shape)
    bs_len = len(batch_shape)

    if ds_len <= bs_len and data_shape == batch_shape[:ds_len]:
        new_shape = data_shape + (1,) * (bs_len - ds_len) + (1,) * core_dims
    elif ds_len == bs_len + core_dims and data_shape[:-core_dims] == batch_shape:
        if data_shape[-core_dims:] != (1,) * core_dims:
            raise ValueError(
                f"batch shape {batch_shape} is incompatible with data shape {data_shape}"
            )
        new_shape = data_shape
    else:
        raise ValueError(
            f"batch shape {batch_shape} is incompatible with data shape {data_shape}"
        )

    return jnp.reshape(data, new_shape)


def _pad_final_dim(data: jax.Array, size: int) -> jax.Array:
    """Pad the trailing algebra dimension of ``data`` with zeros up to ``size``.

    This is used when combining dense algebra elements of different truncation
    depths by embedding the shallower coefficient array into the deeper basis.

    :param data: Coefficient array to pad.
    :param size: Target size of the trailing algebra dimension.
    :return: ``data`` with zeros appended along the final dimension.
    """
    pad_width = [(0, 0)] * (len(data.shape) - 1) + [(0, size - data.shape[-1])]
    return jnp.pad(data, pad_width)


def _algebra_add(
    a: AlgebraT, b: AlgebraT, *, impl: Callable[[jax.Array, jax.Array], jax.Array]
) -> AlgebraT:
    """Apply a pointwise binary operation to two compatible dense algebra objects.

    Addition and subtraction between dense algebra elements are implemented by
    first checking width and batch-shape compatibility, then promoting the
    shallower operand to the deeper basis by zero-padding its trailing algebra
    dimension.

    :param a: Left operand.
    :param b: Right operand.
    :param impl: Elementwise array implementation such as ``jnp.add``.
    :return: Result of applying ``impl`` in the deeper of the two bases.
    """
    cls = type(a)

    if not issubclass(type(b), cls):
        return NotImplemented

    if a.basis.width != b.basis.width:
        raise ValueError("basis widths must match for addition")

    get_common_batch_shape(a, b)

    if a.basis.depth >= b.basis.depth:
        result_basis = a.basis
        a_data = a.data
        b_data = _pad_final_dim(b.data, a.basis.size())
    else:
        result_basis = b.basis
        a_data = _pad_final_dim(a.data, b.basis.size())
        b_data = b.data

    result_data = impl(a_data, b_data)
    return cls(result_data, result_basis)


def _algebra_scalar_multiply(a: AlgebraT, s: jax.typing.ArrayLike) -> AlgebraT:
    """Multiply a dense algebra element by a scalar-like value.

    :param a: Dense algebra operand.
    :param s: Scalar-like multiplier.
    :return: Scaled algebra element in the same basis as ``a``.
    """
    cls = type(a)
    scalar = jnp.asarray(s)
    ext_scalar = broadcast_to_batch_shape(scalar, a.batch_shape)
    result_data = jnp.multiply(a.data, ext_scalar)
    return cls(result_data, a.basis)


def _redepth_data(data: jax.Array, new_alg_dim: int) -> jax.Array:
    """Resize the trailing algebra dimension by truncating or zero-padding.

    This helper is used when changing the truncation depth of a dense algebra
    element. Shrinking the algebra dimension drops higher-order coordinates,
    while increasing it appends zeros for the newly introduced basis elements.

    :param data: Coefficient array to resize.
    :param new_alg_dim: Target size of the trailing algebra dimension.
    :return: Resized coefficient array.
    """
    shape = data.shape

    if new_alg_dim < shape[-1]:
        return data[..., :new_alg_dim]

    pad_dims = [(0, 0)] * (len(shape) - 1) + [(0, new_alg_dim - shape[-1])]
    return jnp.pad(data, pad_dims)


@jax.tree_util.register_pytree_node_class
class DenseAlgebra(Generic[BasisT]):
    """Provide shared dense storage for batched algebra elements.

    This class is intended to be subclassed to define concrete dense
    algebra types such as ``DenseLie``, ``DenseFreeTensor``, and
    ``DenseShuffleTensor``. It centralises storage, basic arithmetic, pytree
    registration, and simple basis-changing utilities shared by those classes.

    Coefficient data has shape ``(*batch_shape, basis.size())``. The basis is
    static pytree metadata, while the coefficient array is the dynamic leaf.

    :param data: Coefficients with shape
        ``(*batch_shape, basis.size())``. Empty batch dimensions are supported.
    :param basis: Basis describing the trailing coefficient dimension.
    :raises ValueError: If ``data`` is scalar or its final dimension does not
        equal ``basis.size()``.
    """

    data: jax.Array
    basis: BasisT

    DualVector: ClassVar[type["DenseAlgebra"]]

    def __init__(self, data: jax.typing.ArrayLike, basis: BasisT):
        self.basis = basis
        self.data = jnp.asarray(data)

        if not self.data.shape:
            raise ValueError("data must have at least one dimension")

        if not basis.size() == self.data.shape[-1]:
            raise ValueError(
                f"basis size must match data dimension, expected {basis.size()} but got {self.data.shape[-1]}"
            )

    @property
    def dtype(self):
        """Return the coefficient-array dtype."""
        return self.data.dtype

    @property
    def shape(self):
        """Return the complete coefficient-array shape."""
        return self.data.shape

    @property
    def batch_shape(self):
        """Return the leading batch dimensions of the coefficient array."""
        return self.data.shape[:-1]

    @property
    def dimension(self):
        """Return the size of the trailing algebra-coordinate dimension."""
        return self.data.shape[-1]

    def change_depth(self: AlgebraT, new_depth: int) -> AlgebraT:
        """Re-express the element in the same basis family at a new depth.

        If ``new_depth`` matches the current depth, the element is returned
        unchanged. Otherwise, a new basis of the same concrete type and width is
        constructed and the coefficient data are resized accordingly.

        :param new_depth: Target truncation depth.
        :return: Algebra element represented in the new basis.
        """
        if new_depth == self.basis.depth:
            return self

        algebra_cls = type(self)
        basis_cls = type(self.basis)

        new_basis = basis_cls(self.basis.width, new_depth)
        new_size = new_basis.size()

        return algebra_cls(_redepth_data(self.data, new_size), new_basis)

    def __array__(self, dtype=None, copy=None):
        """Convert the coefficient data to a NumPy array.

        :param dtype: Optional NumPy dtype for the result.
        :param copy: Whether NumPy must copy the coefficient data.
        :return: NumPy representation of the coefficient array.
        """
        return np.asarray(self.data, dtype=dtype, copy=copy)

    def __numpy_dtype__(self):
        """Return the NumPy dtype corresponding to the coefficient dtype."""
        return np.dtype(self.data.dtype)

    def __getitem__(self: AlgebraT, index) -> AlgebraT:
        """Select a sub-batch while preserving the algebra coordinate axis.

        The index is interpreted exclusively against :attr:`batch_shape`; the
        trailing algebra-coordinate dimension is protected by an appended full
        slice. All other indexing behavior, including advanced indexing and
        out-of-bounds handling, follows JAX array semantics. This includes
        allowing selections that produce a zero-length batch dimension.

        :param index: JAX-compatible index into the algebra's batch dimensions.
        :return: An algebra of the same concrete type and basis containing the
            selected sub-batch.
        """
        batch_index = index if isinstance(index, tuple) else (index,)
        new_data = self.data[*batch_index, slice(None)]

        return type(self)(new_data, self.basis)

    @classmethod
    def _equal(
        cls,
        left: AlgebraT,
        right: AlgebraT,
        *,
        equal_nan: bool = False,
    ) -> jax.Array:
        """Compare dense coefficient arrays for exact equality.

        :param left: First algebra to compare.
        :param right: Second algebra to compare.
        :param equal_nan: Whether corresponding NaN coefficients compare equal.
        :return: Boolean array with the broadcasted batch shape.
        """
        left_data, right_data = jnp.broadcast_arrays(left.data, right.data)
        matches = left_data == right_data
        matches = matches | (
            jnp.asarray(equal_nan) & jnp.isnan(left_data) & jnp.isnan(right_data)
        )
        return jnp.all(matches, axis=-1)

    @classmethod
    def _allclose(
        cls,
        left: AlgebraT,
        right: AlgebraT,
        *,
        rtol: jax.typing.ArrayLike = 1e-5,
        atol: jax.typing.ArrayLike = 1e-8,
        equal_nan: bool = False,
    ) -> jax.Array:
        """Compare dense coefficient arrays using numerical tolerances.

        :param left: First algebra to compare.
        :param right: Second algebra to compare.
        :param rtol: Relative tolerance for coefficient comparisons.
        :param atol: Absolute tolerance for coefficient comparisons.
        :param equal_nan: Whether corresponding NaN coefficients compare equal.
        :return: Boolean array with the broadcasted batch shape.
        """
        left_data, right_data = jnp.broadcast_arrays(left.data, right.data)
        matches = jnp.isclose(
            left_data,
            right_data,
            rtol=rtol,
            atol=atol,
            equal_nan=False,
        )
        matches = matches | (
            jnp.asarray(equal_nan) & jnp.isnan(left_data) & jnp.isnan(right_data)
        )
        return jnp.all(matches, axis=-1)

    @classmethod
    def _astype(
        cls: type[AlgebraT],
        algebra: AlgebraT,
        dtype: jax.typing.DTypeLike,
    ) -> AlgebraT:
        """Convert dense coefficient storage to a new dtype.

        :param algebra: Algebra whose coefficients should be converted.
        :param dtype: Target coefficient dtype.
        :return: Converted algebra with its concrete type and basis preserved.
        """
        return cls(algebra.data.astype(dtype), algebra.basis)

    def __add__(self, other):
        """Add a dense algebra element of the same concrete type.

        :param other: Right-hand algebra operand.
        :return: Sum in the deeper operand basis, or ``NotImplemented`` for an
            unsupported operand type.
        """
        if isinstance(other, type(self)):
            return _algebra_add(self, other, impl=jnp.add)
        return NotImplemented

    def __sub__(self, other):
        """Subtract a dense algebra element of the same concrete type.

        :param other: Right-hand algebra operand.
        :return: Difference in the deeper operand basis, or ``NotImplemented``
            for an unsupported operand type.
        """
        if isinstance(other, type(self)):
            return _algebra_add(self, other, impl=jnp.subtract)
        return NotImplemented

    def __mul__(self, other):
        """Multiply the coefficient data by a scalar-like value.

        :param other: Scalar or prefix-batched multiplier.
        :return: Scaled algebra element, or ``NotImplemented`` for another
            algebra operand.
        """
        if isinstance(other, DenseAlgebra):
            return NotImplemented
        return _algebra_scalar_multiply(self, jnp.asarray(other))

    def __rmul__(self, other):
        """Multiply the coefficient data by a scalar-like value.

        :param other: Scalar or prefix-batched multiplier.
        :return: Scaled algebra element, or ``NotImplemented`` for another
            algebra operand.
        """
        if isinstance(other, DenseAlgebra):
            return NotImplemented
        return _algebra_scalar_multiply(self, jnp.asarray(other))

    def __truediv__(self, other):
        """Divide the coefficient data by a scalar-like value.

        :param other: Scalar or prefix-batched divisor.
        :return: Scaled algebra element, or ``NotImplemented`` for another
            algebra operand.
        """
        if isinstance(other, DenseAlgebra):
            return NotImplemented
        return _algebra_scalar_multiply(self, 1 / jnp.asarray(other))

    def tree_flatten(self):
        """Return dynamic children and static metadata for JAX pytree handling."""
        return (self.data,), (self.basis,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct an algebra element from its pytree representation.

        :param aux_data: Static tuple containing the algebra basis.
        :param children: Dynamic tuple containing the coefficient array.
        :return: Reconstructed dense algebra element.
        """
        obj = cls.__new__(cls)
        obj.data = children[0]
        obj.basis = aux_data[0]
        return obj

    @classmethod
    def zero(
        cls: type[AlgebraT],
        basis: BasisT,
        dtype: jax.typing.DTypeLike = jnp.float32,
        batch_dims: tuple[int, ...] = tuple(),
        device: jax.Device | None = None,
    ) -> AlgebraT:
        """Construct the additive identity in the given basis.

        The returned element has coefficient array shape
        ``batch_dims + (basis.size(),)`` and contains zeros in every basis
        coordinate. The optional batch dimensions allow a batch of zero
        elements to be created in a single call.

        :param basis: Basis in which the zero element should live.
        :param dtype: Data type used for the coefficient array.
        :param batch_dims: Optional leading batch dimensions. Zero-length
            dimensions are supported.
        :param device: Optional device on which to create the coefficient array.
        :return: A zero element of ``cls`` in ``basis``.
        """
        shape = (*batch_dims, basis.size())
        zero_data = jnp.zeros(dtype=jnp.dtype(dtype), shape=shape, device=device)
        return cls(zero_data, basis)

    @classmethod
    def stack(
        cls: type[AlgebraT],
        algebras: Sequence[AlgebraT],
        axis: int = 0,
        dtype: jax.typing.DTypeLike | None = None,
    ) -> AlgebraT:
        """Implement representation-specific storage for :func:`algebra.stack`.

        This class method is an implementation hook and is not intended to be
        called directly. Use :func:`roughpy_jax.algebra.stack` instead so that
        the operands, batch shapes, and axis are validated before dispatching
        to the appropriate representation.

        :param algebras: Dense algebra objects whose coefficient data will be
            stacked.
        :param axis: Batch axis at which to insert the new dimension.
        :param dtype: Optional data type for the resulting coefficient array.
        :return: A dense algebra object containing the stacked coefficients.
        """
        bases = [algebra.basis for algebra in algebras]
        basis = result_basis(*bases, strategy="max_depth")

        basis_size = basis.size()
        new_data = jnp.stack(
            [_redepth_data(algebra.data, basis_size) for algebra in algebras],
            axis=axis,
            dtype=dtype,
        )
        return cls(new_data, basis)

    @classmethod
    def concatenate(
        cls: type[AlgebraT],
        algebras: Sequence[AlgebraT],
        axis: int = 0,
        dtype: jax.typing.DTypeLike | None = None,
    ) -> AlgebraT:
        """Implement representation-specific storage for :func:`algebra.concatenate`.

        This class method is an implementation hook and is not intended to be
        called directly. Use :func:`roughpy_jax.algebra.concatenate` instead so
        that the operands, batch shapes, and axis are validated before dispatching
        to the appropriate representation.

        :param algebras: Dense algebra objects whose coefficient data will be
            concatenated.
        :param axis: Existing batch axis along which to concatenate.
        :param dtype: Optional data type for the resulting coefficient array.
        :return: A dense algebra object containing the concatenated coefficients.
        """
        bases = [algebra.basis for algebra in algebras]
        basis = result_basis(*bases, strategy="max_depth")

        basis_size = basis.size()
        data = [_redepth_data(algebra.data, basis_size) for algebra in algebras]
        new_data = jnp.concatenate(data, axis=axis, dtype=dtype)
        return cls(new_data, basis)


DenseAlgebra.DualVector = DenseAlgebra


@jax.tree_util.register_pytree_node_class
class DenseTensor(DenseAlgebra[TensorBasis]):
    """Represent a tensor-algebra element in dense coordinates.

    This base class supplies the multiplicative identity shared by concrete
    free- and shuffle-tensor representations.
    """

    @classmethod
    def identity(
        cls: type[AlgebraT],
        basis: TensorBasis,
        dtype: jax.typing.DTypeLike = jnp.float32,
        batch_dims: tuple[int, ...] = tuple(),
        device: jax.Device | None = None,
    ) -> AlgebraT:
        """Construct the multiplicative identity in a tensor basis.

        The returned coefficient array has shape
        ``batch_dims + (basis.size(),)``. Its empty-word coordinate is one and
        every other coordinate is zero.

        :param basis: Tensor basis in which the identity should live.
        :param dtype: Data type used for the coefficient array.
        :param batch_dims: Optional leading batch dimensions. Zero-length
            dimensions are supported.
        :param device: Optional device on which to create the coefficient array.
        :return: Identity element of ``cls`` in ``basis``.
        """
        shape = (*batch_dims, basis.size())
        data = jnp.zeros(dtype=jnp.dtype(dtype), shape=shape, device=device)
        data = data.at[..., 0].set(1)
        return cls(data, basis)


DenseTensor.DualVector = DenseTensor


def zero_like(algebra: AlgebraT, dtype: jax.typing.DTypeLike | None = None) -> AlgebraT:
    """Construct a zero element with the same type, shape, and basis as ``algebra``.

    The returned object keeps the basis metadata and concrete dense algebra
    class of the input while replacing all coefficients with zeros. An
    optional ``dtype`` may be supplied to override the data type of the
    resulting coefficient array.

    :param algebra: Dense algebra element whose structure should be copied.
    :param dtype: Optional data type for the zero coefficient array.
    :return: Zero element matching the input algebra structure.
    """
    data = jnp.zeros_like(algebra.data, dtype=dtype)
    return type(algebra)(data, algebra.basis)


def identity_like(
    tensor: AlgebraT, dtype: jax.typing.DTypeLike | None = None
) -> AlgebraT:
    """Construct a multiplicative identity matching a tensor's structure.

    The concrete type, basis, and batch shape are preserved. The empty-word
    coordinate is set to one and every other coefficient is set to zero.

    :param tensor: Tensor whose type, basis, and batch shape should be copied.
    :param dtype: Optional coefficient dtype. Defaults to ``tensor.dtype``.
    :return: Multiplicative identity matching ``tensor``.
    """
    data = jnp.zeros_like(tensor.data, dtype=dtype)
    data = data.at[..., 0].set(1)
    return type(tensor)(data, tensor.basis)


def to_dual(algebra: DenseAlgebra) -> DenseAlgebra:
    """Map an algebra element to its isomorphic dual-space representation.

    This reinterprets ``algebra`` in the corresponding dual space using the
    dual basis associated with the same truncated algebra. In the truncated
    algebras used in roughpy-jax (free, shuffle, and Lie), this map is
    represented by changing the concrete algebra type while leaving the
    coefficient data and basis metadata unchanged.

    This operation is only valid when ``algebra.DualVector`` identifies an
    isomorphic dual representation using compatible basis coordinates.

    :param algebra: Algebra element to reinterpret in the dual space.
    :return: The same coefficients viewed in ``algebra.DualVector``.
    """
    return algebra.DualVector(algebra.data, algebra.basis)
