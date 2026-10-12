#include "dense_st_mul.h"


#include <rpp/basis/tensor_basis.hpp>
#include <rpp/gpu/block/operations/basic/st_mul.hpp>

#include <xla_helpers/batch.hpp>
#include <xla_helpers/static_args.hpp>
#include <xla_helpers/xla_common.hpp>
#include "output_buffer.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"

namespace rpy::jax::cuda {
namespace {


template <typename Tag>
struct DenseSTMulFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer lhs,
                           ffi::AnyBuffer rhs,
                           DenseSTMulStaticArgs static_args,
                           cudaStream_t stream) noexcept {
        const auto tensor_size = static_args.basis.size();
        const auto n_tensors = get_batch_size(1, out);

        RPY_XLA_SUCCESS_OR_RETURN(zero_result_buffer(out, stream));

        return select_strategy<Accum>(tensor_size, [&](auto strategy) {
            using LaunchCfg = typename decltype(strategy)::LaunchConfig;
            return rpp::ops::st_mul(strategy,
                                    LaunchCfg{stream},
                                    make_tensor_batch<Scalar, DefaultArch>(
                                        out, 0, static_args.basis.depth),
                                    make_tensor_batch<Scalar, DefaultArch>(
                                        lhs,
                                        static_args.lhs_min_degree,
                                        static_args.lhs_max_degree),
                                    make_tensor_batch<Scalar, DefaultArch>(
                                        rhs,
                                        static_args.rhs_min_degree,
                                        static_args.rhs_max_degree),
                                    static_args.basis,
                                    n_tensors);
        });
    }
};

} // namespace

ffi::Error cuda_dense_st_mul_impl(cudaStream_t stream,
                                  ffi::Result<ffi::AnyBuffer> out,
                                  ffi::AnyBuffer lhs,
                                  ffi::AnyBuffer rhs,
                                  int32_t width,
                                  int32_t depth,
                                  DegreeBeginSpan degree_begin,
                                  int32_t lhs_max_deg,
                                  int32_t rhs_max_deg,
                                  int32_t lhs_min_deg,
                                  int32_t rhs_min_deg) noexcept {
    DenseSTMulStaticArgs static_args{
        HostTensorBasis{static_cast<HostTensorBasis::Degree>(width),
                        static_cast<HostTensorBasis::Degree>(depth),
                        cast_db_array(degree_begin.begin())},
        lhs_max_deg,
        rhs_max_deg,
        lhs_min_deg,
        rhs_min_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(lhs, static_args.basis, lhs_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(rhs, static_args.basis, rhs_max_deg));
    if (!all_buffers_match_type(out->element_type(), lhs, rhs)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, out, lhs, rhs));

    RPY_XLA_SUCCESS_OR_RETURN(
        check_batch_size<DefaultArch>(get_batch_size(1, out)));

    return select_type_and_go<DenseSTMulFunctor>(
        out->element_type(), out, lhs, rhs, static_args, stream);
}

} // namespace rpy::jax::cuda

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
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("lhs_max_deg")
        .Attr<int32_t>("rhs_max_deg")
        .Attr<int32_t>("lhs_min_deg")
        .Attr<int32_t>("rhs_min_deg"));
