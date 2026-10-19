#include "dense_ft_antipode.h"
#include "output_buffer.hpp"
#include "scalars.hpp"

#include <rpp/architecture.hpp>
#include <rpp/cpu/single_thread/operations/basic/tensor_generalised_antipode.hpp>
#include <rpp/cpu/single_thread/strategy.hpp>
#include <xla_helpers/batch.hpp>
#include <xla_helpers/static_args.hpp>
#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cpu {


template <typename Tag>
struct DenseFTAntipodeFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer arg,
                           DenseFTAntipodeStaticArgs static_args) noexcept {
        using Arch = rpp::arch::DefaultArchitecture;
        using Strategy =
            rpp::cpu::strategies::SingleThreadStrategy<Accum, Arch>;
        using Policy = rpp::ops::TensorAntipodeSigningPolicy;
        zero_result_buffer(out);
        auto out_batch =
            make_tensor_batch<Scalar, Arch>(out, 0, static_args.basis.depth);
        auto arg_batch =
            make_tensor_batch<Scalar, Arch>(arg, 0, static_args.arg_max_degree);
        const auto n_tensors = get_batch_size(1, out);

        if (static_args.no_sign) {
            return map_err(
                rpp::ops::tensor_generalised_antipode<Policy::NoSigning>(
                    Strategy{},
                    typename Strategy::LaunchConfig{},
                    out_batch,
                    arg_batch,
                    static_args.basis,
                    n_tensors));
        }
        return map_err(
            rpp::ops::tensor_generalised_antipode<Policy::SignByDegree>(
                Strategy{},
                typename Strategy::LaunchConfig{},
                out_batch,
                arg_batch,
                static_args.basis,
                n_tensors));
    }
};

ffi::Error cpu_dense_ft_antipode_impl(ffi::Result<ffi::AnyBuffer> result,
                                      ffi::AnyBuffer arg,
                                      int32_t width,
                                      int32_t depth,
                                      DegreeBeginSpan degree_begin,
                                      int32_t max_deg,
                                      bool no_sign) {
    DenseFTAntipodeStaticArgs static_args{
        HostTensorBasis{width, depth, cast_db_array(degree_begin.begin())},
        max_deg,
        no_sign};

    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(result, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(
        check_data_degree(arg, static_args.basis, max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_dims(1, result, arg));
    if (!all_buffers_match_type(result->element_type(), arg)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }
    RPY_XLA_SUCCESS_OR_RETURN(check_batch_size<rpp::arch::DefaultArchitecture>(
        get_batch_size(1, result)));
    return select_type_and_go<DenseFTAntipodeFunctor>(
        result->element_type(), result, arg, static_args);
}

} // namespace rpy::jax::cpu

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_dense_ft_antipode,
    rpy::jax::cpu::cpu_dense_ft_antipode_impl,
    xla::ffi::Ffi::Bind()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("arg_max_deg")
        .Attr<bool>("no_sign"));
