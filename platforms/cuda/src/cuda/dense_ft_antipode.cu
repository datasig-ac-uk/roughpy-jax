#include "dense_ft_antipode.h"


#include <rpp/basis/basis_pack.hpp>
#include <rpp/basis/tensor_basis.hpp>
#include <rpp/gpu/block/operations/basic/tensor_antipode.hpp>
#include <rpp/gpu/block/operations/basic/tensor_reflect.hpp>
#include <rpp/operations/basic/tensor_antipode.hpp>
#include <rpp/operations/basic/tensor_reflect.hpp>


#include <xla_helpers/batch.hpp>
#include <xla_helpers/static_args.hpp>
#include <xla_helpers/xla_common.hpp>
#include "output_buffer.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"

namespace rpy::jax::cuda {

namespace {


template <typename Tag>
struct DenseFTAntipodeFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer arg,
                           DenseFTAntipodeStaticArgs static_args,
                           cudaStream_t stream) noexcept {
        const auto tensor_size = static_args.basis.size();
        const auto n_tensors = get_batch_size(1, out);

        RPY_XLA_SUCCESS_OR_RETURN(zero_result_buffer(out, stream));

        return select_strategy<Accum>(tensor_size, [&](auto strategy) {
            using LaunchCfg = typename decltype(strategy)::LaunchConfig;
            if (static_args.no_sign) {
                return rpp::ops::tensor_reflect(
                    strategy,
                    LaunchCfg{stream},
                    make_tensor_batch<Scalar, DefaultArch>(
                        out, 0, static_args.basis.depth),
                    make_tensor_batch<Scalar, DefaultArch>(
                        arg, 0, static_args.arg_max_degree),
                    static_args.basis,
                    n_tensors);
            }
            return rpp::ops::tensor_antipode(
                strategy,
                LaunchCfg{stream},
                make_tensor_batch<Scalar, DefaultArch>(
                    out, 0, static_args.basis.depth),
                make_tensor_batch<Scalar, DefaultArch>(
                    arg, 0, static_args.arg_max_degree),
                static_args.basis,
                n_tensors);
        });
    }
};

ffi::Error cuda_dense_ft_antipode_impl(cudaStream_t stream,
                                       ffi::Result<ffi::AnyBuffer> out,
                                       ffi::AnyBuffer arg,
                                       int32_t width,
                                       int32_t depth,
                                       DegreeBeginSpan degree_begin,
                                       int32_t max_degree,
                                       bool no_sign) noexcept {
    DenseFTAntipodeStaticArgs static_args{
        HostTensorBasis{static_cast<HostTensorBasis::Degree>(width),
                        static_cast<HostTensorBasis::Degree>(depth),
                        cast_db_array(degree_begin.begin())},
        max_degree,
        no_sign};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(arg, static_args.basis, max_degree));
    if (!all_buffers_match_type(out->element_type(), arg)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, out, arg));

    RPY_XLA_SUCCESS_OR_RETURN(
        check_batch_size<DefaultArch>(get_batch_size(1, out)));

    return select_type_and_go<DenseFTAntipodeFunctor>(
        out->element_type(), out, arg, static_args, stream);
}
} // namespace

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
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("arg_max_deg")
        .Attr<bool>("no_sign"));
