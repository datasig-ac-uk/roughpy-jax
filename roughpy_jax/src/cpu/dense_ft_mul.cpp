#include "dense_ft_mul.h"

#include <algorithm>
#include <rpp/architecture.hpp>
#include <rpp/cpu/single_thread/operations/basic/ft_mul.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>
#include <xla_helpers/batch.hpp>
#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cpu {

struct DenseFTMulStaticArgs {
    HostTensorBasis basis;
    int32_t lhs_max_degree;
    int32_t rhs_max_degree;
    int32_t lhs_min_degree;
    int32_t rhs_min_degree;
};

template <typename Tag>
struct DenseFTMulFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer lhs,
                           ffi::AnyBuffer rhs,
                           DenseFTMulStaticArgs static_args) noexcept {
        using Arch = rpp::arch::DefaultArchitecture;
        using Strategy =
            rpp::cpu::strategies::SingleThreadStrategy<Accum, Arch>;
        auto out_batch =
            make_tensor_batch<Scalar, Arch>(out, 0, static_args.basis.depth);
        auto lhs_batch = make_tensor_batch<Scalar, Arch>(
            lhs, static_args.lhs_min_degree, static_args.lhs_max_degree);
        auto rhs_batch = make_tensor_batch<Scalar, Arch>(
            rhs, static_args.rhs_min_degree, static_args.rhs_max_degree);

        // RPP's CPU multiplication currently reads the output with a zero
        // coefficient. Initialise it so indeterminate values and NaNs cannot
        // leak.
        std::fill_n(out->typed_data<Scalar>(), out->element_count(), Scalar{});
        return map_err(rpp::ops::ft_mul(Strategy{},
                                        typename Strategy::LaunchConfig{},
                                        out_batch,
                                        lhs_batch,
                                        rhs_batch,
                                        static_args.basis,
                                        get_batch_size(1, out)));
    }
};

ffi::Error cpu_dense_ft_mul_impl(ffi::Result<ffi::AnyBuffer> out,
                                 ffi::AnyBuffer lhs,
                                 ffi::AnyBuffer rhs,
                                 int32_t width,
                                 int32_t depth,
                                 DegreeBeginSpan degree_begin,
                                 int32_t lhs_max_deg,
                                 int32_t rhs_max_deg,
                                 int32_t lhs_min_deg,
                                 int32_t rhs_min_deg) {
    DenseFTMulStaticArgs static_args{
        HostTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        lhs_max_deg,
        rhs_max_deg,
        lhs_min_deg,
        rhs_min_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(lhs, static_args.basis, lhs_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(rhs, static_args.basis, rhs_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, out, lhs, rhs));
    if (!all_buffers_match_type(out->element_type(), lhs, rhs)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_size<rpp::arch::DefaultArchitecture>(
        get_batch_size(1, out)));
    return select_type_and_go<DenseFTMulFunctor>(
        out->element_type(), out, lhs, rhs, static_args);
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_mul,
    rpy::jax::cpu::cpu_dense_ft_mul_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("lhs_max_deg")
        .Attr<int32_t>("rhs_max_deg")
        .Attr<int32_t>("lhs_min_deg")
        .Attr<int32_t>("rhs_min_deg"));
