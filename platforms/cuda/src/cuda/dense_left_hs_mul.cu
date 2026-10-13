#include "dense_left_hs_mul.h"


#include <rpp/basis/tensor_basis.hpp>
#include <rpp/gpu/block/operations/basic/left_hs_mul.hpp>

#include <xla_helpers/batch.hpp>
#include <xla_helpers/static_args.hpp>
#include <xla_helpers/xla_common.hpp>
#include "output_buffer.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"

namespace rpy::jax::cuda {
namespace {


template <typename Tag>
struct DenseLeftHSMulFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer integrator,
                           ffi::AnyBuffer integrand,
                           DenseLeftHSMulStaticArgs static_args,
                           cudaStream_t stream) noexcept {
        const auto tensor_size = static_args.basis.size();
        const auto n_tensors = get_batch_size(1, out);

        RPY_XLA_SUCCESS_OR_RETURN(zero_result_buffer(out, stream));

        return select_strategy<Accum>(tensor_size, [&](auto strategy) {
            using LaunchCfg = typename decltype(strategy)::LaunchConfig;
            return rpp::ops::left_hs_mul(strategy,
                                         LaunchCfg{stream},
                                         make_tensor_batch<Scalar, DefaultArch>(
                                             out, 0, static_args.basis.depth),
                                         make_tensor_batch<Scalar, DefaultArch>(
                                             integrator,
                                             static_args.integrator_min_degree,
                                             static_args.integrator_max_degree),
                                         make_tensor_batch<Scalar, DefaultArch>(
                                             integrand,
                                             static_args.integrand_min_degree,
                                             static_args.integrand_max_degree),
                                         static_args.basis,
                                         n_tensors);
        });
    }
};

} // namespace

ffi::Error cuda_dense_left_hs_mul_impl(cudaStream_t stream,
                                       ffi::Result<ffi::AnyBuffer> out,
                                       ffi::AnyBuffer integrator,
                                       ffi::AnyBuffer integrand,
                                       int32_t width,
                                       int32_t depth,
                                       DegreeBeginSpan degree_begin,
                                       int32_t integrator_max_deg,
                                       int32_t integrand_max_deg,
                                       int32_t integrator_min_deg,
                                       int32_t integrand_min_deg) noexcept {
    DenseLeftHSMulStaticArgs static_args{
        HostTensorBasis{static_cast<HostTensorBasis::Degree>(width),
                        static_cast<HostTensorBasis::Degree>(depth),
                        cast_db_array(degree_begin.begin())},
        integrator_max_deg,
        integrand_max_deg,
        integrator_min_deg,
        integrand_min_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(integrator, static_args.basis, integrator_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(integrand, static_args.basis, integrand_max_deg));
    if (!all_buffers_match_type(out->element_type(), integrator, integrand)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, out, integrator, integrand));

    RPY_XLA_SUCCESS_OR_RETURN(
        check_batch_size<DefaultArch>(get_batch_size(1, out)));

    return select_type_and_go<DenseLeftHSMulFunctor>(
        out->element_type(), out, integrator, integrand, static_args, stream);
}

} // namespace rpy::jax::cuda

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_left_hs_mul,
    rpy::jax::cuda::cuda_dense_left_hs_mul_impl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("integrator_max_deg")
        .Attr<int32_t>("integrand_max_deg")
        .Attr<int32_t>("integrator_min_deg")
        .Attr<int32_t>("integrand_min_deg"));
