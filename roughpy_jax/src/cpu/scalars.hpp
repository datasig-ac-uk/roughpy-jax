#ifndef ROUGHPY_JAX_CPU_SCALARS_HPP
#define ROUGHPY_JAX_CPU_SCALARS_HPP

#include <type_traits>

#include <Eigen/Core>

#include <xla_helpers/xla_common.hpp>

namespace rpy::jax {

template <typename AccumType>
struct ScalarTag<ffi::DataType::F16, AccumType> {
    static constexpr bool available = true;
    static constexpr auto tag = ffi::DataType::F16;
    using Scalar = Eigen::half;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::BF16, AccumType> {
    static constexpr bool available = true;
    static constexpr auto tag = ffi::DataType::BF16;
    using Scalar = Eigen::bfloat16;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

// FFI buffers contain raw 16-bit coefficients, with no per-element padding.
static_assert(sizeof(Eigen::half) == sizeof(uint16_t));
static_assert(sizeof(Eigen::bfloat16) == sizeof(uint16_t));
static_assert(alignof(Eigen::half) == alignof(uint16_t));
static_assert(alignof(Eigen::bfloat16) == alignof(uint16_t));
static_assert(std::is_trivially_copyable_v<Eigen::half>);
static_assert(std::is_trivially_copyable_v<Eigen::bfloat16>);

} // namespace rpy::jax

#endif // ROUGHPY_JAX_CPU_SCALARS_HPP
