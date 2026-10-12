#ifndef ROUGHPY_JAX_XLA_HELPERS_XLA_COMMON_HPP
#define ROUGHPY_JAX_XLA_HELPERS_XLA_COMMON_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>

#include <rpp/support/error.hpp>

#include "xla_includes.hpp"

#define RPY_XLA_SUCCESS_OR_RETURN(stat)                                        \
    do {                                                                       \
        const auto status = (stat);                                            \
        if (!status.success()) {                                               \
            return status;                                                     \
        }                                                                      \
    }                                                                          \
    while (0)

namespace rpy::jax {

namespace ffi = xla::ffi;

// Preserve the RPP error message while translating its status to XLA FFI.
template <typename Payload>
ffi::Error map_err(const rpp::Error<Payload>& err) noexcept {
    using rpp::ErrorCode;
    ffi::ErrorCode code;
    switch (err.code()) {
    case ErrorCode::Success:
        return ffi::Error::Success();
    case ErrorCode::Cancelled:
        code = ffi::ErrorCode::kCancelled;
        break;
    case ErrorCode::Unknown:
        code = ffi::ErrorCode::kUnknown;
        break;
    case ErrorCode::InvalidArgument:
        code = ffi::ErrorCode::kInvalidArgument;
        break;
    case ErrorCode::Timeout:
        code = ffi::ErrorCode::kDeadlineExceeded;
        break;
    case ErrorCode::OutOfResources:
        code = ffi::ErrorCode::kResourceExhausted;
        break;
    case ErrorCode::ContractViolation:
        code = ffi::ErrorCode::kFailedPrecondition;
        break;
    case ErrorCode::OutOfBounds:
        code = ffi::ErrorCode::kOutOfRange;
        break;
    case ErrorCode::NotImplemented:
        code = ffi::ErrorCode::kUnimplemented;
        break;
    case ErrorCode::Internal:
        code = ffi::ErrorCode::kInternal;
        break;
    default:
        code = ffi::ErrorCode::kUnknown;
        break;
    }
    return ffi::Error{code, std::string(err.message())};
}

// Platforms specialise this tag for their additional scalar types. An omitted
// accumulation type selects the default for the storage dtype. Dispatch checks
// availability before instantiating a functor that uses Scalar and Accum.
template <ffi::DataType DType, typename AccumType = void>
struct ScalarTag {
    static constexpr auto tag = DType;
    static constexpr bool available = false;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F32, AccumType> {
    static constexpr bool available = true;
    static constexpr auto tag = ffi::DataType::F32;
    using Scalar = float;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F64, AccumType> {
    static constexpr bool available = true;
    static constexpr auto tag = ffi::DataType::F64;
    using Scalar = double;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, double, AccumType>;
};

using DegreeBeginIndex =
    std::conditional_t<(sizeof(std::ptrdiff_t) == sizeof(int64_t)),
                       int64_t,
                       int32_t>;
using DegreeBeginSpan = ffi::Span<const DegreeBeginIndex>;

template <typename T>
constexpr auto cast_db_array(const T* db_in) noexcept -> const std::ptrdiff_t* {
    static_assert(sizeof(T) == sizeof(std::ptrdiff_t), "mismatch integer size");
    static_assert(std::is_signed_v<T> == std::is_signed_v<std::ptrdiff_t>,
                  "different signedness");
    return reinterpret_cast<const std::ptrdiff_t*>(db_in);
}
template <typename Basis>
auto data_size_to_degree(const Basis& basis, int32_t degree) ->
    typename Basis::Index {
    if (degree < 0) {
        return 0;
    }
    if (degree >= basis.depth) {
        return basis.size();
    }
    return basis.degree_begin[degree + 1];
}

template <typename... BufferType>
bool all_buffers_match_type(xla::ffi::DataType expected_type,
                            const BufferType&... buffer) {
    return ((buffer.element_type() == expected_type) && ...);
}

namespace detail {

inline ffi::Span<const int64_t> shape(ffi::AnyBuffer const& arg) noexcept {
    return arg.dimensions();
}
inline ffi::Span<const int64_t>
shape(const ffi::Result<ffi::AnyBuffer>& arg) noexcept {
    return arg->dimensions();
}
} // namespace detail

// Compare buffer batch shapes against an explicit shape. This also handles
// scalar outputs, whose entire shape is the batch shape of tensor inputs.
template <typename... Buffers>
ffi::Error check_batch_dims(ffi::Span<const int64_t> batch_dims,
                            size_t ncore_dims,
                            const Buffers&... buffers) noexcept {
    const auto matches = [&](const auto& buffer) {
        const auto dims = detail::shape(buffer);
        return dims.size() >= ncore_dims &&
            dims.size() - ncore_dims == batch_dims.size() &&
            std::equal(batch_dims.begin(), batch_dims.end(), dims.begin());
    };
    if (!(matches(buffers) && ...)) {
        return ffi::Error::InvalidArgument("mismatched batch dimensions");
    }
    return ffi::Error::Success();
}

// Core dimensions may differ between buffers; batch ranks and dimensions must
// match exactly. Validation does not compute the number of batches.
template <typename Head, typename... Tail>
ffi::Error check_batch_dims(size_t ncore_dims,
                            const Head& head,
                            const Tail&... tail) noexcept {
    const auto dims = detail::shape(head);
    if (dims.size() < ncore_dims) {
        return ffi::Error::InvalidArgument(
            "buffer rank is smaller than core rank");
    }
    return check_batch_dims(
        dims.first(dims.size() - ncore_dims), ncore_dims, tail...);
}

// Requires a buffer whose rank and batch shape have already been validated.
template <typename Buffer>
int64_t get_batch_size(size_t ncore_dims, const Buffer& buffer) noexcept {
    const auto dims = detail::shape(buffer);
    const auto batch_dims = dims.first(dims.size() - ncore_dims);
    return std::accumulate(
        batch_dims.begin(), batch_dims.end(), int64_t{1}, std::multiplies<>{});
}

template <typename Buffer, typename Basis>
ffi::Error check_data_degree(Buffer& buf,
                             const Basis& basis,
                             int32_t degree,
                             int dim = -1) {
    if (degree > basis.depth) {
        return ffi::Error::InvalidArgument("degree exceeds basis depth");
    }

    const auto buf_shape = detail::shape(buf);
    if (dim < 0) {
        dim = buf_shape.size() + dim;
    }
    if (dim < 0 || static_cast<size_t>(dim) >= buf_shape.size()) {
        return ffi::Error::InvalidArgument("invalid dimension for buffer");
    }

    const auto val = buf_shape[dim];
    const auto data_size = data_size_to_degree(basis, degree);

    if (val < data_size) {
        return ffi::Error::InvalidArgument(
            "data dimension is too small for specified degree");
    }

    return ffi::Error::Success();
}

namespace detail {

template <typename ScalarTag,
          template <typename> class Functor,
          typename... Args>
std::enable_if_t<ScalarTag::available, ffi::Error>
exec_functor(Args&&... args) noexcept {
    using Fn = Functor<ScalarTag>;
    return Fn::eval(std::forward<Args>(args)...);
}

template <typename ScalarTag, template <typename> class Functor>
ffi::Error exec_functor(...) noexcept {
    return ffi::Error::InvalidArgument("unsupported dtype");
}

} // namespace detail

template <template <typename> class Functor, typename... Args>
ffi::Error select_type_and_go(ffi::DataType dtype, Args&&... args) noexcept {
    switch (dtype) {
    case ffi::DataType::F32:
        return detail::exec_functor<ScalarTag<ffi::DataType::F32>, Functor>(
            std::forward<Args>(args)...);
    case ffi::DataType::F64:
        return detail::exec_functor<ScalarTag<ffi::DataType::F64>, Functor>(
            std::forward<Args>(args)...);
    case ffi::DataType::F16:
        return detail::exec_functor<ScalarTag<ffi::DataType::F16>, Functor>(
            std::forward<Args>(args)...);
    case ffi::DataType::BF16:
        return detail::exec_functor<ScalarTag<ffi::DataType::BF16>, Functor>(
            std::forward<Args>(args)...);
    case ffi::DataType::F8E5M2:
        return detail::exec_functor<ScalarTag<ffi::DataType::F8E5M2>, Functor>(
            std::forward<Args>(args)...);
    case ffi::DataType::F8E4M3:
        return detail::exec_functor<ScalarTag<ffi::DataType::F8E4M3>, Functor>(
            std::forward<Args>(args)...);
    case ffi::DataType::F8E4M3FN:
        return detail::exec_functor<ScalarTag<ffi::DataType::F8E4M3FN>,
                                    Functor>(std::forward<Args>(args)...);
    case ffi::DataType::F8E4M3B11FNUZ:
        return detail::exec_functor<ScalarTag<ffi::DataType::F8E4M3B11FNUZ>,
                                    Functor>(std::forward<Args>(args)...);
    case ffi::DataType::F8E5M2FNUZ:
        return detail::exec_functor<ScalarTag<ffi::DataType::F8E5M2FNUZ>,
                                    Functor>(std::forward<Args>(args)...);
    case ffi::DataType::F8E4M3FNUZ:
        return detail::exec_functor<ScalarTag<ffi::DataType::F8E4M3FNUZ>,
                                    Functor>(std::forward<Args>(args)...);
    case ffi::DataType::F8E3M4:
        return detail::exec_functor<ScalarTag<ffi::DataType::F8E3M4>, Functor>(
            std::forward<Args>(args)...);
    case ffi::DataType::F4E2M1FN:
        return detail::exec_functor<ScalarTag<ffi::DataType::F4E2M1FN>,
                                    Functor>(std::forward<Args>(args)...);
    case ffi::DataType::F8E8M0FNU:
        return detail::exec_functor<ScalarTag<ffi::DataType::F8E8M0FNU>,
                                    Functor>(std::forward<Args>(args)...);

    default:
        return ffi::Error::InvalidArgument("unsupported dtype");
    }
}

} // namespace rpy::jax

#endif // ROUGHPY_JAX_XLA_HELPERS_XLA_COMMON_HPP
