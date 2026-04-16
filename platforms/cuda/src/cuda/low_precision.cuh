#ifndef PLATFORMS_CUDA_SRC_CUDA_LOW_PRECISION_CUH
#define PLATFORMS_CUDA_SRC_CUDA_LOW_PRECISION_CUH

#include "xla_headers.hpp"

#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace rpy::jax::cuda {


template <xla::ffi::DataType DType>
struct ScalarTag;

template <>
struct ScalarTag<xla::ffi::DataType::F16>
{
    static constexpr auto tag = xla::ffi::DataType::F16;
    using Scalar = __half;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::BF16>
{
    static constexpr auto tag = xla::ffi::DataType::BF16;
    using Scalar = __nv_bfloat16;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::F8E4M3>
{
    static constexpr auto tag = xla::ffi::DataType::F8E4M3;
    using Scalar = __nv_fp8_e4m3;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::F8E4M3FN>
{
    static constexpr auto tag = xla::ffi::DataType::F8E4M3FN;
    using Scalar = __nv_fp8_e4m3;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::F8E4M3B11FNUZ>
{
    static constexpr auto tag = xla::ffi::DataType::F8E4M3B11FNUZ;
    using Scalar = __nv_fp8_e4m3;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::F8E4M3FNUZ>
{
    static constexpr auto tag = xla::ffi::DataType::F8E4M3FNUZ;
    using Scalar = __nv_fp8_e4m3;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::F8E5M2>
{
    static constexpr auto tag = xla::ffi::DataType::F8E5M2;
    using Scalar = __nv_fp8_e5m2;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::F8E5M2FNUZ>
{
    static constexpr auto tag = xla::ffi::DataType::F8E5M2FNUZ;
    using Scalar = __nv_fp8_e5m2;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::F4E2M1FN>
{
    static constexpr auto tag = xla::ffi::DataType::F4E2M1FN;
    using Scalar = __nv_fp4_e2m1;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::F32>
{
    static constexpr auto tag = xla::ffi::DataType::F32;
    using Scalar = float;
    using Accum = float;
};

template <>
struct ScalarTag<xla::ffi::DataType::F64>
{
    static constexpr auto tag = xla::ffi::DataType::F64;
    using Scalar = double;
    using Accum = double;
};

} // namespace rpy::jax::cuda


#endif //PLATFORMS_CUDA_SRC_CUDA_LOW_PRECISION_CUH
