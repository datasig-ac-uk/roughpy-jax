#include "dense_ft_adj_lmul.h"


#include <rpp/basis/tensor_basis.hpp>
#include <rpp/gpu/block/operations/basic/ft_adj_lmul.hpp>


#include <xla_helpers/batch.hpp>
#include <xla_helpers/static_args.hpp>
#include <xla_helpers/xla_common.hpp>
#include "output_buffer.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"

namespace rpy::jax::cuda {

namespace {


template <typename Tag>
struct DenseFtAdjLMulFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;


    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer op,
                           ffi::AnyBuffer arg,
                           DenseFTAdjLMulStaticArgs static_args,
                           cudaStream_t stream) noexcept {
        const auto tensor_size = static_args.basis.size();

        const auto n_tensors = get_batch_size(1, out);

        RPY_XLA_SUCCESS_OR_RETURN(zero_result_buffer(out, stream));

        return select_strategy<Accum>(tensor_size, [&](auto strategy) {
            using LaunchCfg = typename decltype(strategy)::LaunchConfig;
            return rpp::ops::ft_adj_lmul(
                strategy,
                LaunchCfg{stream},
                make_tensor_batch<Scalar, DefaultArch>(
                    out, 0, static_args.basis.depth),
                make_tensor_batch<Scalar, DefaultArch>(
                    op, 0, static_args.op_max_degree),
                make_tensor_batch<Scalar, DefaultArch>(
                    arg, 0, static_args.arg_max_degree),
                static_args.basis,
                n_tensors);
        });
    }
};


ffi::Error cuda_dense_ft_adj_lmul_impl(cudaStream_t stream,
                                       ffi::Result<ffi::AnyBuffer> out,
                                       ffi::AnyBuffer op,
                                       ffi::AnyBuffer arg,
                                       int32_t width,
                                       int32_t depth,
                                       DegreeBeginSpan degree_begin,
                                       int32_t op_max_deg,
                                       int32_t arg_max_deg) noexcept {
    DenseFTAdjLMulStaticArgs static_args{
        HostTensorBasis{static_cast<HostTensorBasis::Degree>(width),
                        static_cast<HostTensorBasis::Degree>(depth),
                        cast_db_array(degree_begin.begin())},
        op_max_deg,
        arg_max_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(op, static_args.basis, op_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(arg, static_args.basis, arg_max_deg));
    auto elt_type = out->element_type();
    if (!all_buffers_match_type(elt_type, op, arg)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, out, op, arg));

    RPY_XLA_SUCCESS_OR_RETURN(
        check_batch_size<DefaultArch>(get_batch_size(1, out)));

    return select_type_and_go<DenseFtAdjLMulFunctor>(
        elt_type, out, op, arg, static_args, stream);
}


} // namespace

} // namespace rpy::jax::cuda

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_ft_adj_lmul,
    rpy::jax::cuda::cuda_dense_ft_adj_lmul_impl,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("op_max_deg")
        .Attr<int32_t>("arg_max_deg"));
