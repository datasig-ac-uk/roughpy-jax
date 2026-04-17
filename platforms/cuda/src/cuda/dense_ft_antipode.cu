#include "dense_ft_antipode.h"

#include <rpp/basis.hpp>

#include "xla_headers.hpp"

#include "basis_support.cuh"
#include "error_handling.cuh"
#include "low_precision.cuh"

#include "kernels/dense_ft_antipode.cuh"

using namespace rpy::jax::cuda;

namespace {
namespace ffi = xla::ffi;

using TensorBasis = rpp::StandardTensorBasis;
using Degree = typename TensorBasis::Degree;
using Index = typename TensorBasis::Index;

struct DenseFTAntipodeStaticArgs {
    TensorBasis basis;
    int32_t max_degree;
    bool no_sign;
};

template <typename Tag>
struct DenseFTAntipodeFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(
        ffi::Result<ffi::AnyBuffer> out,
        ffi::AnyBuffer arg,
        DenseFTAntipodeStaticArgs static_args,
        cudaStream_t stream
        ) noexcept {
        const auto tensor_size = static_args.basis.size();

        unsigned block = 32;
        if (tensor_size >= 64) {
            block = 64;
        } else if (tensor_size >= 128) {
            block = 128;
        } else {
            block = 256;
        }

        auto out_shape = out->dimensions();
        const auto n_tensors = std::accumulate(
            out_shape.begin(), out_shape.end() - 1, 1LL, std::multiplies<>{});

        const auto grid = static_cast<unsigned>(std::min(n_tensors, 2LL << 28));

        TensorBasisConverter<Degree, Index> converted(static_args.basis, stream);
        if (!converted) {
            return cuda_error_to_xla_error(converted.error, "failed to convert tensor basis");
        }

        auto* out_ptr = buffer_to_pointer<Tag>(out);
        auto const* arg_ptr = buffer_to_pointer<Tag>(arg);
        Index arg_stride = arg.dimensions().back();

        if (!static_args.no_sign) {
            dense_ft_antipode_general_kernel<Tag, true><<<grid, block, 0, stream>>>(
                out_ptr,
                arg_ptr,
                converted.d_basis,
                static_args.max_degree
                );
        } else {
            dense_ft_antipode_general_kernel<Tag, false><<<grid, block, 0, stream>>>(
                out_ptr,
                arg_ptr,
                converted.d_basis,
                static_args.max_degree
                );
        }

        return cuda_error_to_xla_error(cudaGetLastError());
    }

};

ffi::Error cuda_dense_ft_antipode_impl(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::AnyBuffer arg,
    int32_t width,
    int32_t depth,
    DegreeBeginSpan degree_begin,
    int32_t max_degree,
    bool no_sign
) noexcept
{

    DenseFTAntipodeStaticArgs static_args {
        rpp::StandardTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        max_degree,
        no_sign
    };

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(arg, static_args.basis, max_degree));

    if (!all_buffers_match_type(out->element_type(), arg)) {
        return ffi::Error::InvalidArgument("all tensors should have the same data type");
    }

    return select_type_and_go<DenseFTAntipodeFunctor>(out->element_type(), out, arg, static_args, stream);
}

} // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_ft_antipode,
    cuda_dense_ft_antipode_impl,
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
