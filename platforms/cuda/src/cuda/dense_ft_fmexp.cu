#include "dense_ft_fmexp.h"

#include <algorithm>
#include <functional>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/gpu/block/operations/intermediate/ft_fmexp.hpp>

#include "batch.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"
#include "xla_headers.hpp"

namespace ffi = xla::ffi;

namespace rpy::jax::cuda {
namespace {

struct DenseFTFMExpStaticArgs {
    HostTensorBasis basis;
    int32_t mul_max_deg;
    int32_t exp_max_deg;
    int32_t mul_min_deg;
    int32_t exp_min_deg;
};

template <typename Tag>
struct DenseFTFMExpFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer multiplier,
                           ffi::AnyBuffer exponent,
                           DenseFTFMExpStaticArgs static_args,
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
            return rpp::ops::ft_fmexp(
                strategy,
                LaunchCfg{stream},
                make_tensor_batch<Scalar>(out, 0, static_args.basis.depth),
                make_tensor_batch<Scalar>(multiplier,
                                          static_args.mul_min_deg,
                                          static_args.mul_max_deg),
                make_tensor_batch<Scalar>(exponent,
                                          static_args.exp_min_deg,
                                          static_args.exp_max_deg),
                static_args.basis,
                n_tensors);
        });
    }
};

} // namespace

ffi::Error cuda_dense_ft_fmexp_impl(cudaStream_t stream,
                                    ffi::Result<ffi::AnyBuffer> out,
                                    ffi::AnyBuffer multiplier,
                                    ffi::AnyBuffer exponent,
                                    int32_t width,
                                    int32_t depth,
                                    DegreeBeginSpan degree_begin,
                                    int32_t mul_max_deg,
                                    int32_t exp_max_deg,
                                    int32_t mul_min_deg,
                                    int32_t exp_min_deg) noexcept {
    DenseFTFMExpStaticArgs static_args{
        HostTensorBasis{
            static_cast<HostTensorBasis::Degree>(width),
            static_cast<HostTensorBasis::Degree>(depth),
            cast_db_array(degree_begin.begin())},
        mul_max_deg,
        exp_max_deg,
        mul_min_deg,
        exp_min_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(multiplier, static_args.basis, mul_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(exponent, static_args.basis, exp_max_deg));

    if (!all_buffers_match_type(out->element_type(), multiplier, exponent)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    return select_type_and_go<DenseFTFMExpFunctor>(out->element_type(),
                                                   out,
                                                   multiplier,
                                                   exponent,
                                                   static_args,
                                                   stream);
}

} // namespace rpy::jax::cuda

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_ft_fmexp,
    rpy::jax::cuda::cuda_dense_ft_fmexp_impl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::cuda::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("mul_max_deg")
        .Attr<int32_t>("exp_max_deg")
        .Attr<int32_t>("mul_min_deg")
        .Attr<int32_t>("exp_min_deg"));
