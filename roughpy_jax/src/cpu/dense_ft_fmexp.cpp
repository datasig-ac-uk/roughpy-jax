#include "dense_ft_fmexp.h"
#include "output_buffer.hpp"

#include <rpp/architecture.hpp>
#include <rpp/cpu/single_thread/operations/intermediate/ft_fmexp.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>
#include <xla_helpers/batch.hpp>
#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cpu {

struct DenseFTFMExpStaticArgs {
    HostTensorBasis basis;
    int32_t mul_max_degree;
    int32_t exp_max_degree;
    int32_t mul_min_degree;
    int32_t exp_min_degree;
};

template <typename Tag>
struct DenseFTFMExpFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer multiplier,
                           ffi::AnyBuffer exponent,
                           DenseFTFMExpStaticArgs static_args) noexcept {
        using Arch = rpp::arch::DefaultArchitecture;
        using Strategy =
            rpp::cpu::strategies::SingleThreadStrategy<Accum, Arch>;
        zero_result_buffer(out);
        auto out_batch =
            make_tensor_batch<Scalar, Arch>(out, 0, static_args.basis.depth);
        auto mul_batch = make_tensor_batch<Scalar, Arch>(
            multiplier, static_args.mul_min_degree, static_args.mul_max_degree);
        auto exp_batch = make_tensor_batch<Scalar, Arch>(
            exponent, static_args.exp_min_degree, static_args.exp_max_degree);
        return map_err(rpp::ops::ft_fmexp(Strategy{},
                                          typename Strategy::LaunchConfig{},
                                          out_batch,
                                          mul_batch,
                                          exp_batch,
                                          static_args.basis,
                                          get_batch_size(1, out)));
    }
};

ffi::Error cpu_dense_ft_fmexp_impl(ffi::Result<ffi::AnyBuffer> result,
                                   ffi::AnyBuffer multiplier,
                                   ffi::AnyBuffer exponent,
                                   int32_t width,
                                   int32_t depth,
                                   DegreeBeginSpan degree_begin,
                                   int32_t mul_max_deg,
                                   int32_t exp_max_deg,
                                   int32_t mul_min_deg,
                                   int32_t exp_min_deg) {
    DenseFTFMExpStaticArgs static_args{
        HostTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        mul_max_deg,
        exp_max_deg,
        mul_min_deg,
        exp_min_deg};

    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(result, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(multiplier, static_args.basis, mul_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(exponent, static_args.basis, exp_max_deg));

    RPY_XLA_SUCCESS_OR_RETURN(
        check_batch_dims(1, result, multiplier, exponent));
    if (!all_buffers_match_type(result->element_type(), multiplier, exponent)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    RPY_XLA_SUCCESS_OR_RETURN(check_batch_size<rpp::arch::DefaultArchitecture>(
        get_batch_size(1, result)));
    return select_type_and_go<DenseFTFMExpFunctor>(
        result->element_type(), result, multiplier, exponent, static_args);
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_fmexp,
    rpy::jax::cpu::cpu_dense_ft_fmexp_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("mul_max_deg")
        .Attr<int32_t>("exp_max_deg")
        .Attr<int32_t>("mul_min_deg")
        .Attr<int32_t>("exp_min_deg"));
