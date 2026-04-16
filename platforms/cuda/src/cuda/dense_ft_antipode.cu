#include "dense_ft_antipode.h"

#include "xla_headers.hpp"

namespace ffi = xla::ffi;

namespace rpy::jax::cuda {

ffi::Error cuda_dense_ft_antipode_impl(
    cudaStream_t,
    ffi::Result<ffi::AnyBuffer>,
    ffi::AnyBuffer,
    int32_t,
    int32_t,
    DegreeBeginSpan,
    int32_t,
    bool
) noexcept
{
    return {ffi::ErrorCode::kInternal, "not implemented"};
}

} // namespace rpy::jax::cuda

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_ft_antipode,
    rpy::jax::cuda::cuda_dense_ft_antipode_impl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::cuda::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("arg_max_deg")
        .Attr<bool>("no_sign")
);
