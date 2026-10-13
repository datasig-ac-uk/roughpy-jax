#ifndef ROUGHPY_JAX_CUDA_SCALARS_HPP
#define ROUGHPY_JAX_CUDA_SCALARS_HPP

#include <type_traits>


#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <xla_helpers/xla_common.hpp>

namespace rpy::jax {

// Specialisations of the shared tag must live in its declaring namespace.
template <typename AccumType>
struct ScalarTag<ffi::DataType::F16, AccumType> {
    static constexpr bool available = true;
    static constexpr auto tag = ffi::DataType::F16;
    using Scalar = __half;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::BF16, AccumType> {
    static constexpr bool available = true;
    static constexpr auto tag = ffi::DataType::BF16;
    using Scalar = __nv_bfloat16;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

// CUDA FP8 wrappers have storage mappings, but the kernels do not yet
// support their arithmetic. Keep them unavailable to scalar dispatch.
template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E4M3, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E4M3;
    using Scalar = __nv_fp8_e4m3;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E4M3FN, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E4M3FN;
    using Scalar = __nv_fp8_e4m3;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E4M3B11FNUZ, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E4M3B11FNUZ;
    using Scalar = __nv_fp8_e4m3;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E4M3FNUZ, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E4M3FNUZ;
    using Scalar = __nv_fp8_e4m3;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E5M2, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E5M2;
    using Scalar = __nv_fp8_e5m2;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E5M2FNUZ, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E5M2FNUZ;
    using Scalar = __nv_fp8_e5m2;
    using Accum =
        std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

} // namespace rpy::jax

#endif // ROUGHPY_JAX_CUDA_SCALARS_HPP
