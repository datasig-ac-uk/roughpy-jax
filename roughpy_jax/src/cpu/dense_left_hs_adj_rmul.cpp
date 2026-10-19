#include "dense_left_hs_adj_rmul.h"
#include "output_buffer.hpp"
#include "scalars.hpp"

#include <rpp/architecture.hpp>
#include <rpp/cpu/single_thread/operations/basic/left_hs_adj_rmul.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>
#include <tuple>
#include <xla_helpers/batch.hpp>
#include <xla_helpers/static_args.hpp>
#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cpu {


template <typename Tag>
struct DenseLeftHSAdjRMulFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer integrator,
                           ffi::AnyBuffer integrand,
                           DenseLeftHSAdjRMulStaticArgs static_args) noexcept {
        using Arch = rpp::arch::DefaultArchitecture;
        using Strategy =
            rpp::cpu::strategies::SingleThreadStrategy<Accum, Arch>;
        zero_result_buffer(out);
        auto out_batch =
            make_tensor_batch<Scalar, Arch>(out, 0, static_args.basis.depth);
        auto integrator_batch =
            make_tensor_batch<Scalar, Arch>(integrator,
                                            static_args.integrator_min_degree,
                                            static_args.integrator_max_degree);
        auto integrand_batch =
            make_tensor_batch<Scalar, Arch>(integrand,
                                            static_args.integrand_min_degree,
                                            static_args.integrand_max_degree);

        // The current CPU convenience wrapper forwards beta, while the RPP
        // operation accepts only the three views. Launch the unit-coefficient
        // operation directly until those interfaces are aligned upstream.
        using Op = rpp::ops::LeftHSAdjRMul<Strategy>;
        return map_err(Strategy{}.template launch<Op>(
            typename Strategy::LaunchConfig{},
            std::make_tuple(out_batch, integrator_batch, integrand_batch),
            rpp::basis::make_basis_pack(static_args.basis),
            get_batch_size(1, out)));
    }
};

ffi::Error cpu_dense_left_hs_adj_rmul_impl(ffi::Result<ffi::AnyBuffer> out,
                                           ffi::AnyBuffer integrator,
                                           ffi::AnyBuffer integrand,
                                           int32_t width,
                                           int32_t depth,
                                           DegreeBeginSpan degree_begin,
                                           int32_t integrator_max_deg,
                                           int32_t integrand_max_deg,
                                           int32_t integrator_min_deg,
                                           int32_t integrand_min_deg) {
    DenseLeftHSAdjRMulStaticArgs static_args{
        HostTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        integrator_max_deg,
        integrand_max_deg,
        integrator_min_deg,
        integrand_min_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(integrator, static_args.basis, integrator_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(integrand, static_args.basis, integrand_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, out, integrator, integrand));
    if (!all_buffers_match_type(out->element_type(), integrator, integrand)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_size<rpp::arch::DefaultArchitecture>(
        get_batch_size(1, out)));
    return select_type_and_go<DenseLeftHSAdjRMulFunctor>(
        out->element_type(), out, integrator, integrand, static_args);
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_left_hs_adj_rmul,
    rpy::jax::cpu::cpu_dense_left_hs_adj_rmul_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("integrator_max_deg")
        .Attr<int32_t>("integrand_max_deg")
        .Attr<int32_t>("integrator_min_deg")
        .Attr<int32_t>("integrand_min_deg"));
