#ifndef PLATFORMS_CUDA_SRC_CUDA_XLA_HEADERS_HPP
#define PLATFORMS_CUDA_SRC_CUDA_XLA_HEADERS_HPP


#include <cstdint>
#include <type_traits>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wattributes"
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"
#pragma GCC diagnostic pop

#define RPY_XLA_SUCCESS_OR_RETURN(stat)                                        \
do {                                                                       \
const auto status = (stat);                                            \
if (!status.success()) { return status; }                              \
} while (0)


namespace rpy::jax::cuda {

namespace ffi = xla::ffi;

using DegreeBeginIndex = std::conditional_t<(sizeof(std::ptrdiff_t) == sizeof(int64_t)),
                                            int64_t, int32_t>;
using DegreeBeginSpan = ffi::Span<const DegreeBeginIndex>;


template <typename T>
/// This is a solution to a problem that shouldn't exist. Unfortunately, only
/// long long (=int64_t) is a registered type in the XLA decoder logic, and on
/// MacOs ptrdiff_t resolves to a long, even though this is the same size it is
/// a different type. This function just casts the pointer to a ptrdiff_t, after
/// checking it is the right size.
constexpr auto cast_db_array(const T* db_in) noexcept -> const std::ptrdiff_t* {
    static_assert(sizeof(T) == sizeof(std::ptrdiff_t), "mismatch integer size");
    static_assert(std::is_signed_v<T> == std::is_signed_v<std::ptrdiff_t>, "different signedness");
    return reinterpret_cast<const std::ptrdiff_t*>(db_in);
}


template <typename Basis>
auto data_size_to_degree(const Basis& basis, int32_t degree) -> typename Basis::Index
{
    if (degree < 0) {
        return 0;
    }
    if (degree >= basis.depth) {
        return basis.size();
    }
    return basis.degree_begin[degree + 1];
}

template <typename... BufferType>
bool all_buffers_match_type(xla::ffi::DataType expected_type, const BufferType&... buffer) {
    return ((buffer.element_type() == expected_type) && ...);
}



namespace detail {

inline ffi::Span<const int64_t> shape(ffi::AnyBuffer const& arg) noexcept
{
    return arg.dimensions();
}
inline ffi::Span<const int64_t> shape(ffi::Result<ffi::AnyBuffer>& arg) noexcept
{
    return arg->dimensions();
}
}


template <typename Buffer, typename Basis>
ffi::Error check_data_degree(Buffer& buf, const Basis& basis, int32_t degree, int dim=-1)
{
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
                "data dimension is too small for specified degree"
        );
    }

    return ffi::Error::Success();
}

template <typename ScalarTag>
typename ScalarTag::Scalar const* buffer_to_pointer(ffi::AnyBuffer buffer) noexcept {
    using ResultPtr = typename ScalarTag::Scalar const*;
    return static_cast<ResultPtr>(buffer.untyped_data());
}

template <typename ScalarTag>
typename ScalarTag::Scalar* buffer_to_pointer(ffi::Result<ffi::AnyBuffer> buffer) noexcept {
    using ResultPtr = typename ScalarTag::Scalar*;
    return static_cast<ResultPtr>(buffer->untyped_data());
}


} // namespace rpy::jax::cuda


#endif //PLATFORMS_CUDA_SRC_CUDA_XLA_HEADERS_HPP
