#include "dense_ft_adj_mul.h"


#include <algorithm>
#include <functional>


#include "error_handling.cuh"
#include "low_precision.cuh"
#include "basis_support.cuh"
#include "xla_headers.hpp"

#include "kernels/dense_ft_antipode.cuh"
#include "kernels/dense_ft_adj_mul.cuh"


using namespace rpy::jax::cuda;

namespace {
namespace ffi = xla::ffi;

using TensorBasis = rpp::StandardTensorBasis;
using Degree = typename TensorBasis::Degree;
using Index = typename TensorBasis::Index;


struct DenseFtAdjMulStaticArgs {
    TensorBasis basis;
    int32_t op_max_deg;
    int32_t arg_max_deg;
};

template <typename Tag>
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

        unsigned block ;
        if (tensor_size < 64) {
            block = 32;
        } else if (tensor_size >= 64) {
            block = 64;
        } else if (tensor_size >= 128) {
            block = 128;
        } else {
            block = 256;
        }

        const auto elt_type = out->element_type();

        const auto out_shape = out->dimensions();
        const auto n_tensors = std::accumulate(
            out_shape.begin(), out_shape.end() - 1, 1LL,
            std::multiplies<>{});

        const auto grid = static_cast<unsigned>(std::min(n_tensors, 2LL<<28));

        TensorBasisConverter<Degree, Index> converted(static_args.basis, stream);
        if (!converted) {
            return cuda_error_to_xla_error(converted.error);
        }

        auto* out_ptr = buffer_to_pointer<Tag>(out);
        auto const* op_ptr = buffer_to_pointer<Tag>(op);
        const Index op_stride = op.dimensions().back();
        auto const* arg_ptr = buffer_to_pointer<Tag>(arg);
        const Index arg_stride = op.dimensions().back();

        dense_ft_adj_lmul_kernel<Tag><<<grid, block, 0, stream>>>(
            out_ptr,
            op_ptr,
            op_stride,
            arg_ptr,
            arg_stride,
            converted.d_basis,
            static_args.op_max_deg,
            static_args.arg_max_deg,
            0,0,
            n_tensors
            );

        return cuda_error_to_xla_error(cudaGetLastError());
    }
};


template <typename Tag>
struct DenseDTAdjRMulFunctor : DenseFtAdjLMulFunctor<Tag> {

    static ffi::Error eval(
        ffi::Result<ffi::AnyBuffer> out,
        ffi::AnyBuffer op,
        ffi::AnyBuffer arg,
        DenseFtAdjMulStaticArgs static_args,
        cudaStream_t stream
        ) {
        const auto tensor_size = static_args.basis.size();

        unsigned block ;
        if (tensor_size < 64) {
            block = 32;
        } else if (tensor_size >= 64) {
            block = 64;
        } else if (tensor_size >= 128) {
            block = 128;
        } else {
            block = 256;
        }


        const auto out_shape = out->dimensions();
        const auto n_tensors = std::accumulate(
            out_shape.begin(), out_shape.end() - 1, 1LL,
            std::multiplies<>{});

        const auto grid = static_cast<unsigned>(std::min(n_tensors, 2LL<<28));

        TensorBasisConverter<Degree, Index> converted(static_args.basis, stream);
        if (!converted) {
            return cuda_error_to_xla_error(converted.error);
        }


        auto* out_ptr = buffer_to_pointer<Tag>(out);
        auto const* op_ptr = buffer_to_pointer<Tag>(op);
        const Index op_stride = op.dimensions().back();
        auto const* arg_ptr = buffer_to_pointer<Tag>(arg);
        const Index arg_stride = op.dimensions().back();

        dense_ft_adj_rmul_kernel<Tag><<<grid, block, 0, stream>>>(
            out_ptr,
            op_ptr,
            op_stride,
            arg_ptr,
            arg_stride,
            converted.d_basis,
            static_args.op_max_deg,
            static_args.arg_max_deg,
            0,0,
            n_tensors
            );

        return cuda_error_to_xla_error(cudaGetLastError());

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
) noexcept
{
    DenseFtAdjMulStaticArgs static_args {
        TensorBasis { width, depth, cast_db_array(degree_begin.begin()) },
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
) noexcept
{
    DenseFtAdjMulStaticArgs static_args {
        TensorBasis { width, depth, cast_db_array(degree_begin.begin()) },
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
