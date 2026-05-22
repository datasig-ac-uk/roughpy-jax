#include "dense_ft_log.h"

#include <algorithm>
#include <functional>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/gpu/block/operations/intermediate/ft_log.hpp>

#include "batch.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"
#include "xla_headers.hpp"

namespace ffi = xla::ffi;

namespace rpy::jax::cuda {
namespace {

struct DenseFTLogStaticArgs {
    HostTensorBasis basis;
    int32_t max_degree;
};

template <typename Tag>
struct DenseFTLogFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer arg,
                           DenseFTLogStaticArgs static_args,
                           cudaStream_t stream) noexcept {
        const auto tensor_size = static_args.basis.size();
        const auto out_shape = out->dimensions();
        const auto n_tensors =
            std::accumulate(out_shape.begin(),
                            out_shape.end() - 1,
                            1LL,
                            std::multiplies<>{});

        return select_strategy<Accum>(tensor_size, [&](auto strategy) {
            using LaunchCfg = typename decltype(strategy)::LaunchConfig;
            return rpp::ops::ft_log(
                strategy,
                LaunchCfg{stream},
                make_tensor_batch<Scalar>(out, 0, static_args.basis.depth),
                make_tensor_batch<Scalar>(arg, 0, static_args.max_degree),
                static_args.basis,
                n_tensors);
        });
    }
};

} // namespace

ffi::Error cuda_dense_ft_log_impl(cudaStream_t stream,
                                  ffi::Result<ffi::AnyBuffer> out,
                                  ffi::AnyBuffer arg,
                                  int32_t width,
                                  int32_t depth,
                                  DegreeBeginSpan degree_begin,
                                  int32_t max_degree) noexcept {
    DenseFTLogStaticArgs static_args{
        HostTensorBasis{
            static_cast<HostTensorBasis::Degree>(width),
            static_cast<HostTensorBasis::Degree>(depth),
            cast_db_array(degree_begin.begin())},
        max_degree};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(arg, static_args.basis, max_degree));

    if (!all_buffers_match_type(out->element_type(), arg)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    return select_type_and_go<DenseFTLogFunctor>(
        out->element_type(), out, arg, static_args, stream);
}

} // namespace rpy::jax::cuda

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_ft_log,
    rpy::jax::cuda::cuda_dense_ft_log_impl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::cuda::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("arg_max_deg"));
