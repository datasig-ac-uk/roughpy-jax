#include "dense_st_fma.h"
#include "output_buffer.hpp"

#include <rpp/architecture.hpp>
#include <xla_helpers/batch.hpp>

#include <rpp/cpu/single_thread/operations/basic/st_fma.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>
#include <rpp/operations/basic/st_fma.hpp>

#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cpu {

struct DenseSTFmaStaticArgs {
    HostTensorBasis basis;
    int32_t a_max_degree;
    int32_t b_max_degree;
    int32_t c_max_degree;
    int32_t b_min_degree = 0;
    int32_t c_min_degree = 0;
};

template <typename Tag>
struct DenseSTFmaFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer a,
                           ffi::AnyBuffer b,
                           ffi::AnyBuffer c,
                           DenseSTFmaStaticArgs static_args) noexcept {
        using Arch = rpp::arch::DefaultArchitecture;

        using Strategy =
            rpp::cpu::strategies::SingleThreadStrategy<Accum, Arch>;

        zero_result_buffer(out);
        auto out_batch =
            make_tensor_batch<Scalar, Arch>(out, 0, static_args.basis.depth);
        auto a_batch =
            make_tensor_batch<Scalar, Arch>(a, 0, static_args.a_max_degree);
        auto b_batch = make_tensor_batch<Scalar, Arch>(
            b, static_args.b_min_degree, static_args.b_max_degree);
        auto c_batch = make_tensor_batch<Scalar, Arch>(
            c, static_args.c_min_degree, static_args.c_max_degree);

        const auto n_tensors = get_batch_size(1, out);

        const auto status = rpp::ops::st_fma(Strategy{},
                                             typename Strategy::LaunchConfig{},
                                             out_batch,
                                             a_batch,
                                             b_batch,
                                             c_batch,
                                             static_args.basis,
                                             n_tensors);

        return map_err(status);
    }
};

ffi::Error cpu_dense_st_fma_impl(ffi::Result<ffi::AnyBuffer> out,
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
                                 int32_t c_min_deg) {

    DenseSTFmaStaticArgs static_args{
        HostTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        a_max_deg,
        b_max_deg,
        c_max_deg,
        b_min_deg,
        c_min_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(a, static_args.basis, a_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(b, static_args.basis, b_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(c, static_args.basis, c_max_deg));

    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, out, a, b, c));
    if (!all_buffers_match_type(out->element_type(), a, b, c)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    RPY_XLA_SUCCESS_OR_RETURN(check_batch_size<rpp::arch::DefaultArchitecture>(
        get_batch_size(1, out)));

    return select_type_and_go<DenseSTFmaFunctor>(
        out->element_type(), out, a, b, c, static_args);
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_st_fma,
    rpy::jax::cpu::cpu_dense_st_fma_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("a_max_deg")
        .Attr<int32_t>("b_max_deg")
        .Attr<int32_t>("c_max_deg")
        .Attr<int32_t>("b_min_deg")
        .Attr<int32_t>("c_min_deg")

);
