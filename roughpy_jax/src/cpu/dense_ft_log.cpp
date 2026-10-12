#include "dense_ft_log.h"

#include <rpp/architecture.hpp>
#include <rpp/cpu/single_thread/operations/intermediate/ft_log.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>
#include <xla_helpers/batch.hpp>
#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cpu {

struct DenseFTLogStaticArgs {
    HostTensorBasis basis;
    int32_t arg_max_degree;
};

template <typename Tag>
struct DenseFTLogFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer arg,
                           DenseFTLogStaticArgs static_args) noexcept {
        using Arch = rpp::arch::DefaultArchitecture;
        using Strategy =
            rpp::cpu::strategies::SingleThreadStrategy<Accum, Arch>;
        auto out_batch =
            make_tensor_batch<Scalar, Arch>(out, 0, static_args.basis.depth);
        auto arg_batch =
            make_tensor_batch<Scalar, Arch>(arg, 0, static_args.arg_max_degree);
        return map_err(rpp::ops::ft_log(Strategy{},
                                        typename Strategy::LaunchConfig{},
                                        out_batch,
                                        arg_batch,
                                        static_args.basis,
                                        get_batch_size(1, out)));
    }
};

ffi::Error cpu_dense_ft_log_impl(ffi::Result<ffi::AnyBuffer> result,
                                 ffi::AnyBuffer arg,
                                 int32_t width,
                                 int32_t depth,
                                 DegreeBeginSpan degree_begin,
                                 int32_t arg_max_deg) {
    DenseFTLogStaticArgs static_args{
        HostTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        arg_max_deg};

    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(result, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(arg, static_args.basis, arg_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, result, arg));
    if (!all_buffers_match_type(result->element_type(), arg)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_size<rpp::arch::DefaultArchitecture>(
        get_batch_size(1, result)));
    return select_type_and_go<DenseFTLogFunctor>(
        result->element_type(), result, arg, static_args);
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_log,
    rpy::jax::cpu::cpu_dense_ft_log_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("arg_max_deg"));
