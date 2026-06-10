#include "dense_ft_fma.h"

#include <algorithm>
#include <functional>

#include <rpp/basis/tensor_basis.hpp>
#include <rpp/gpu/block/operations/basic/ft_fma.hpp>
#include <rpp/gpu/block/operations/basic/ft_mul.hpp>

#include "batch.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"
#include "xla_headers.hpp"

using namespace rpy::jax::cuda;

namespace {
namespace ffi = xla::ffi;

struct DenseFTFmaStaticArgs {
    HostTensorBasis basis;
    int32_t a_max_degree;
    int32_t b_max_degree;
    int32_t c_max_degree;
    int32_t b_min_degree = 0;
    int32_t c_min_degree = 0;
};

template <typename Tag>
struct DenseFTFmaFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer a,
                           ffi::AnyBuffer b,
                           ffi::AnyBuffer c,
                           DenseFTFmaStaticArgs static_args,
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
            return rpp::ops::ft_fma(
                strategy,
                LaunchCfg{stream},
                make_tensor_batch<Scalar>(out, 0, static_args.basis.depth),
                make_tensor_batch<Scalar>(a, 0, static_args.a_max_degree),
                make_tensor_batch<Scalar>(b,
                                          static_args.b_min_degree,
                                          static_args.b_max_degree),
                make_tensor_batch<Scalar>(c,
                                          static_args.c_min_degree,
                                          static_args.c_max_degree),
                static_args.basis,
                n_tensors);
        });
    }

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer lhs,
                           ffi::AnyBuffer rhs,
                           DenseFTFmaStaticArgs static_args,
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
            return rpp::ops::ft_mul(
                strategy,
                LaunchCfg{stream},
                make_tensor_batch<Scalar>(out, 0, static_args.basis.depth),
                make_tensor_batch<Scalar>(lhs,
                                          static_args.b_min_degree,
                                          static_args.b_max_degree),
                make_tensor_batch<Scalar>(rhs,
                                          static_args.c_min_degree,
                                          static_args.c_max_degree),
                static_args.basis,
                n_tensors);
        });
    }
};

ffi::Error cuda_dense_ft_fma_impl(cudaStream_t stream,
                                  ffi::Result<ffi::AnyBuffer> out,
                                  ffi::AnyBuffer a,
                                  ffi::AnyBuffer b,
                                  ffi::AnyBuffer c,
                                  int32_t width,
                                  int32_t depth,
                                  DegreeBeginSpan degree_begin,
                                  int32_t a_max_deg,
                                  int32_t b_max_deg,
                                  int32_t c_max_deg,
                                  int32_t b_min_deg,
                                  int32_t c_min_deg) noexcept {
    DenseFTFmaStaticArgs static_args{
        HostTensorBasis{
            static_cast<HostTensorBasis::Degree>(width),
            static_cast<HostTensorBasis::Degree>(depth),
            cast_db_array(degree_begin.begin())},
        a_max_deg,
        b_max_deg,
        c_max_deg,
        b_min_deg,
        c_min_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(a, static_args.basis, a_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(b, static_args.basis, b_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(c, static_args.basis, c_max_deg));

    if (!all_buffers_match_type(out->element_type(), a, b, c)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    return select_type_and_go<DenseFTFmaFunctor>(
        out->element_type(), out, a, b, c, static_args, stream);
}

ffi::Error cuda_dense_ft_mul_impl(cudaStream_t stream,
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
    DenseFTFmaStaticArgs static_args{
        HostTensorBasis{
            static_cast<HostTensorBasis::Degree>(width),
            static_cast<HostTensorBasis::Degree>(depth),
            cast_db_array(degree_begin.begin())},
        0,
        lhs_max_deg,
        rhs_max_deg,
        lhs_min_deg,
        rhs_min_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(lhs, static_args.basis, lhs_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(rhs, static_args.basis, rhs_max_deg));

    if (!all_buffers_match_type(out->element_type(), lhs, rhs)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    return select_type_and_go<DenseFTFmaFunctor>(
        out->element_type(), out, lhs, rhs, static_args, stream);
}
} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_ft_fma,
    cuda_dense_ft_fma_impl,
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
        .Attr<int32_t>("c_min_deg"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_ft_mul,
    cuda_dense_ft_mul_impl,
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
        .Attr<int32_t>("rhs_min_deg"));
