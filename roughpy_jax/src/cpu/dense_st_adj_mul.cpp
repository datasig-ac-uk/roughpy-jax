#include "dense_st_adj_mul.h"

#include <rpp/architecture.hpp>
#include <rpp/cpu/single_thread/operations/basic/st_adj_mul.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>
#include <xla_helpers/batch.hpp>
#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cpu {

struct DenseSTAdjMulStaticArgs {
    HostTensorBasis basis;
    int32_t op_max_deg;
    int32_t arg_max_deg;
};

template <typename Tag>
struct DenseSTAdjMulFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer op,
                           ffi::AnyBuffer arg,
                           DenseSTAdjMulStaticArgs static_args) noexcept {
        using Arch = rpp::arch::DefaultArchitecture;
        using Strategy =
            rpp::cpu::strategies::SingleThreadStrategy<Accum, Arch>;
        auto out_batch =
            make_tensor_batch<Scalar, Arch>(out, 0, static_args.basis.depth);
        auto op_batch =
            make_tensor_batch<Scalar, Arch>(op, 0, static_args.op_max_deg);
        auto arg_batch =
            make_tensor_batch<Scalar, Arch>(arg, 0, static_args.arg_max_deg);

        return map_err(rpp::ops::st_adj_mul(Strategy{},
                                            typename Strategy::LaunchConfig{},
                                            out_batch,
                                            op_batch,
                                            arg_batch,
                                            static_args.basis,
                                            get_batch_size(1, out)));
    }
};

ffi::Error cpu_dense_st_adj_mul_impl(ffi::Result<ffi::AnyBuffer> out,
                                     ffi::AnyBuffer op,
                                     ffi::AnyBuffer arg,
                                     int32_t width,
                                     int32_t depth,
                                     DegreeBeginSpan degree_begin,
                                     int32_t op_max_deg,
                                     int32_t arg_max_deg) noexcept {
    DenseSTAdjMulStaticArgs static_args{
        HostTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        op_max_deg,
        arg_max_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(op, static_args.basis, op_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(arg, static_args.basis, arg_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, out, op, arg));
    if (!all_buffers_match_type(out->element_type(), op, arg)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_size<rpp::arch::DefaultArchitecture>(
        get_batch_size(1, out)));
    return select_type_and_go<DenseSTAdjMulFunctor>(
        out->element_type(), out, op, arg, static_args);
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_st_adj_mul,
    rpy::jax::cpu::cpu_dense_st_adj_mul_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("op_max_deg")
        .Attr<int32_t>("arg_max_deg"));
