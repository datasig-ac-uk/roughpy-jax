#include "dense_ft_adj_mul.h"


#include <algorithm>
#include <functional>


#include <rpp/basis/tensor_basis.hpp>
#include <rpp/gpu/block/operations/basic/ft_adj_lmul.hpp>
#include <rpp/gpu/block/operations/basic/ft_adj_rmul.hpp>


#include "scalars.hpp"
#include "select_strategy.cuh"
#include "batch.cuh"
#include "xla_headers.hpp"

using namespace rpy::jax::cuda;

namespace {
namespace ffi = xla::ffi;


struct DenseFtAdjMulStaticArgs {
    HostTensorBasis basis;
    int32_t op_max_deg;
    int32_t arg_max_deg;
};

template<typename Tag>
struct DenseFtAdjLMulFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;


    static ffi::Error eval(
        ffi::Result<ffi::AnyBuffer> out,
        ffi::AnyBuffer op,
        ffi::AnyBuffer arg,
        DenseFtAdjMulStaticArgs static_args,
        cudaStream_t stream
    ) noexcept {
        const auto tensor_size = static_args.basis.size();

        const auto elt_type = out->element_type();

        const auto out_shape = out->dimensions();
        const auto n_tensors = std::accumulate(
            out_shape.begin(), out_shape.end() - 1, 1LL,
            std::multiplies<>{});

        return select_strategy<Accum>(tensor_size, [&](auto strategy) {
            using LaunchCfg = typename decltype(strategy)::LaunchConfig;
            return rpp::ops::ft_adj_lmul(
                strategy,
                LaunchCfg { stream },
                make_tensor_batch<Scalar>(out, 0, static_args.basis.depth),
                make_tensor_batch<Scalar>(op, 0, static_args.op_max_deg),
                make_tensor_batch<Scalar>(arg, 0, static_args.arg_max_deg),
                static_args.basis,
                n_tensors
                );
        });
    }
};


template<typename Tag>
struct DenseFTAdjRMulFunctor {
    using Accum = typename Tag::Accum;
    using Scalar = typename Tag::Scalar;

    static ffi::Error eval(
        ffi::Result<ffi::AnyBuffer> out,
        ffi::AnyBuffer op,
        ffi::AnyBuffer arg,
        DenseFtAdjMulStaticArgs static_args,
        cudaStream_t stream
    ) {
        const auto tensor_size = static_args.basis.size();

        const auto elt_type = out->element_type();

        const auto out_shape = out->dimensions();
        const auto n_tensors = std::accumulate(
            out_shape.begin(), out_shape.end() - 1, 1LL,
            std::multiplies<>{});

        return select_strategy<Accum>(tensor_size, [&](auto strategy) {
            using LaunchCfg = typename decltype(strategy)::LaunchConfig;
            return rpp::ops::ft_adj_rmul(
                strategy,
                LaunchCfg { stream },
                make_tensor_batch<Scalar>(out, 0, static_args.basis.depth),
                make_tensor_batch<Scalar>(op, 0, static_args.op_max_deg),
                make_tensor_batch<Scalar>(arg, 0, static_args.arg_max_deg),
                static_args.basis,
                n_tensors
                );
        });
    }
};

ffi::Error cuda_dense_ft_adj_lmul_impl(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::AnyBuffer op,
    ffi::AnyBuffer arg,
    int32_t width,
    int32_t depth,
    DegreeBeginSpan degree_begin,
    int32_t op_max_deg,
    int32_t arg_max_deg
) noexcept {
    DenseFtAdjMulStaticArgs static_args{
        HostTensorBasis{
            static_cast<HostTensorBasis::Degree>(width),
            static_cast<HostTensorBasis::Degree>(depth),
            cast_db_array(degree_begin.begin())},
        op_max_deg,
        arg_max_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(op, static_args.basis, op_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(arg, static_args.basis, arg_max_deg));

    auto elt_type = out->element_type();
    if (!all_buffers_match_type(elt_type, op, arg)) {
        return ffi::Error::InvalidArgument("all tensors should have the same data type");
    }

    return select_type_and_go<DenseFtAdjLMulFunctor>(
        elt_type,
        out,
        op,
        arg,
        static_args,
        stream
    );
}


ffi::Error cuda_dense_ft_adj_rmul_impl(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::AnyBuffer op,
    ffi::AnyBuffer arg,
    int32_t width,
    int32_t depth,
    DegreeBeginSpan degree_begin,
    int32_t op_max_deg,
    int32_t arg_max_deg
) noexcept {
    DenseFtAdjMulStaticArgs static_args{
        HostTensorBasis{
            static_cast<HostTensorBasis::Degree>(width),
            static_cast<HostTensorBasis::Degree>(depth),
            cast_db_array(degree_begin.begin())},
        op_max_deg,
        arg_max_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(op, static_args.basis, op_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(arg, static_args.basis, arg_max_deg));

    auto elt_type = out->element_type();
    if (!all_buffers_match_type(elt_type, op, arg)) {
        return ffi::Error::InvalidArgument("all tensors should have the same data type");
    }

    return select_type_and_go<DenseFTAdjRMulFunctor>(
        elt_type,
        out,
        op,
        arg,
        static_args,
        stream
    );
}
} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_ft_adj_lmul,
    cuda_dense_ft_adj_lmul_impl,
    xla::ffi::Ffi::Bind()
    .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
    .Ret<xla::ffi::AnyBuffer>()
    .Arg<xla::ffi::AnyBuffer>()
    .Arg<xla::ffi::AnyBuffer>()
    .Attr<int32_t>("width")
    .Attr<int32_t>("depth")
    .Attr<rpy::jax::cuda::DegreeBeginSpan>("degree_begin")
    .Attr<int32_t>("op_max_deg")
    .Attr<int32_t>("arg_max_deg")
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_ft_adj_rmul,
    cuda_dense_ft_adj_rmul_impl,
    xla::ffi::Ffi::Bind()
    .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
    .Ret<xla::ffi::AnyBuffer>()
    .Arg<xla::ffi::AnyBuffer>()
    .Arg<xla::ffi::AnyBuffer>()
    .Attr<int32_t>("width")
    .Attr<int32_t>("depth")
    .Attr<rpy::jax::cuda::DegreeBeginSpan>("degree_begin")
    .Attr<int32_t>("op_max_deg")
    .Attr<int32_t>("arg_max_deg")
);
