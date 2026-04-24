#include "dense_st_fma.h"

#include "xla_headers.hpp"

namespace ffi = xla::ffi;

namespace rpy::jax::cuda {

namespace {

ffi::Error unimplemented() noexcept
{
    return {ffi::ErrorCode::kInternal, "not implemented"};
}

} // namespace

ffi::Error cuda_dense_st_fma_impl(
    cudaStream_t,
    ffi::Result<ffi::AnyBuffer>,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    int32_t,
    int32_t,
    DegreeBeginSpan,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t
) noexcept
{
    return unimplemented();
}

ffi::Error cuda_dense_st_mul_impl(
    cudaStream_t,
    ffi::Result<ffi::AnyBuffer>,
    ffi::AnyBuffer,
    ffi::AnyBuffer,
    int32_t,
    int32_t,
    DegreeBeginSpan,
    int32_t,
    int32_t,
    int32_t,
    int32_t
) noexcept
{
    return unimplemented();
}

} // namespace rpy::jax::cuda

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_st_fma,
    rpy::jax::cuda::cuda_dense_st_fma_impl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::cuda::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("a_max_deg")
        .Attr<int32_t>("b_max_deg")
        .Attr<int32_t>("c_max_deg")
        .Attr<int32_t>("b_min_deg")
        .Attr<int32_t>("c_min_deg")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_st_mul,
    rpy::jax::cuda::cuda_dense_st_mul_impl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::cuda::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("lhs_max_deg")
        .Attr<int32_t>("rhs_max_deg")
        .Attr<int32_t>("lhs_min_deg")
        .Attr<int32_t>("rhs_min_deg")
);
