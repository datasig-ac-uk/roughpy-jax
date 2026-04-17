#include "dense_ft_fma.h"

#include <algorithm>
#include <functional>

#include "error_handling.cuh"
#include "low_precision.cuh"
#include "basis_support.cuh"

#include "kernels/dense_ft_fma.cuh"


using namespace rpy::jax::cuda;


namespace {
namespace ffi = xla::ffi;

using TensorBasis = rpp::StandardTensorBasis;
using Degree = typename TensorBasis::Degree;
using Index = typename TensorBasis::Index;

struct DenseFTFmaStaticArgs {
    rpp::StandardTensorBasis basis;
    int32_t a_max_degree;
    int32_t b_max_degree;
    int32_t c_max_degree;

    int32_t b_min_degree = 0;
    int32_t c_min_degree = 0;
};

template<typename Tag>
struct DenseFTFmaFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(
        ffi::Result<ffi::AnyBuffer> out,
        ffi::AnyBuffer a,
        ffi::AnyBuffer b,
        ffi::AnyBuffer c,
        AnyScalar alpha, AnyScalar beta,
        DenseFTFmaStaticArgs static_args,
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

        const auto out_shape = out->dimensions();
        const auto n_tensors = std::accumulate(
            out_shape.begin(), out_shape.end()-1, 1LL, std::multiplies<>{});


        const auto grid = static_cast<unsigned>(std::min(n_tensors, 2LL << 28));

        auto alpha_decoded = cast_scalar<Accum>(alpha);
        if (alpha_decoded.has_error()) { return alpha_decoded.error(); }

        auto beta_decoded = cast_scalar<Accum>(beta);
        if (beta_decoded.has_error()) { return beta_decoded.error(); }

        TensorBasisConverter<Degree, Index> converted(static_args.basis, stream);
        if (!converted) {
            return cuda_error_to_xla_error(converted.error, "failed to convert tensor basis");
        }

        auto* out_ptr = buffer_to_pointer<Tag>(out);
        auto* a_ptr = buffer_to_pointer<Tag>(a);
        Index a_stride = a.dimensions().back();
        auto* b_ptr = buffer_to_pointer<Tag>(b);
        Index b_stride = b.dimensions().back();
        auto* c_ptr = buffer_to_pointer<Tag>(c);
        Index c_stride = c.dimensions().back();

        dense_ft_fma_general_kernel<Tag><<<grid, block, 0, stream>>>(
            out_ptr,
            a_ptr,
            a_stride,
            b_ptr,
            b_stride,
            c_ptr,
            c_stride,
            alpha_decoded.value(), beta_decoded.value(),
            converted.d_basis,
            static_args.a_max_degree,
            static_args.b_max_degree,
            static_args.c_max_degree,
            static_args.b_min_degree,
            static_args.c_min_degree,
            n_tensors
        );

        return cuda_error_to_xla_error(cudaGetLastError());
    }

    static ffi::Error eval(
        ffi::Result<ffi::AnyBuffer> out,
        ffi::AnyBuffer b,
        ffi::AnyBuffer c,
        AnyScalar alpha, AnyScalar beta,
        DenseFTFmaStaticArgs static_args,
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

        const auto out_shape = out->dimensions();
        const auto n_tensors = std::accumulate(
            out_shape.begin(), out_shape.end(), 1LL, std::multiplies<>{});


        const auto grid = static_cast<unsigned>(std::min(n_tensors, 2LL << 28));

        auto alpha_decoded = cast_scalar<Accum>(alpha);
        if (alpha_decoded.has_error()) { return alpha_decoded.error(); }

        auto beta_decoded = cast_scalar<Accum>(beta);
        if (beta_decoded.has_error()) { return beta_decoded.error(); }

        TensorBasisConverter<Degree, Index> converted(static_args.basis, stream);
        if (!converted) {
            return cuda_error_to_xla_error(converted.error, "failed to convert tensor basis");
        }


        auto* out_ptr = buffer_to_pointer<Tag>(out);
        Scalar* a_ptr = nullptr;
        Index a_stride = out->dimensions().back();
        auto* b_ptr = buffer_to_pointer<Tag>(b);
        Index b_stride = b.dimensions().back();
        auto* c_ptr = buffer_to_pointer<Tag>(c);
        Index c_stride = c.dimensions().back();

        dense_ft_fma_general_kernel<Tag><<<grid, block, 0, stream>>>(
            out_ptr,
            a_ptr,
            a_stride,
            b_ptr,
            b_stride,
            c_ptr,
            c_stride,
            alpha_decoded.value(),
            beta_decoded.value(),
            converted.d_basis,
            static_args.a_max_degree,
            static_args.b_max_degree,
            static_args.c_max_degree,
            static_args.b_min_degree,
            static_args.c_min_degree,
            n_tensors
        );

        return cuda_error_to_xla_error(cudaGetLastError());
    }
};


ffi::Error cuda_dense_ft_fma_impl(
    cudaStream_t stream,
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
    int32_t c_min_deg
) noexcept {
    DenseFTFmaStaticArgs static_args{
        rpp::StandardTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        a_max_deg,
        b_max_deg,
        c_max_deg,
        b_min_deg,
        c_min_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(out, static_args.basis, depth)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(a, static_args.basis, a_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(b, static_args.basis, b_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(c, static_args.basis, c_max_deg)
    );

    if (!all_buffers_match_type(out->element_type(), a, b, c)) {
        return ffi::Error::InvalidArgument("all tensors should have the same data type");
    }

    const float alpha_val = 1.0f;
    const float beta_val = 1.0f;
    AnyScalar alpha(alpha_val);
    AnyScalar beta(beta_val);

    return select_type_and_go<DenseFTFmaFunctor>(
        out->element_type(),
        out,
        a,
        b,
        c,
        alpha, beta,
        static_args,
        stream
    );
}

ffi::Error cuda_dense_ft_mul_impl(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::AnyBuffer lhs,
    ffi::AnyBuffer rhs,
    int32_t width,
    int32_t depth,
    DegreeBeginSpan degree_begin,
    int32_t lhs_max_deg,
    int32_t rhs_max_deg,
    int32_t lhs_min_deg,
    int32_t rhs_min_deg
) noexcept {
    DenseFTFmaStaticArgs static_args{
        rpp::StandardTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        0,
        lhs_max_deg,
        rhs_max_deg,
        lhs_min_deg,
        rhs_min_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(out, static_args.basis, depth)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(lhs, static_args.basis, lhs_max_deg)
    );
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(rhs, static_args.basis, rhs_max_deg)
    );

    if (!all_buffers_match_type(out->element_type(), lhs, rhs)) {
        return ffi::Error::InvalidArgument("all tensors should have the same data type");
    }

    const float alpha_val = 0.0f;
    const float beta_val = 1.0f;
    AnyScalar alpha(alpha_val);
    AnyScalar beta(beta_val);

    return select_type_and_go<DenseFTFmaFunctor>(
        out->element_type(),
        out,
        lhs,
        rhs,
        alpha, beta,
        static_args,
        stream
    );
}
}


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
    .Attr<int32_t>("c_min_deg")
);


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
    .Attr<int32_t>("rhs_min_deg")
);
