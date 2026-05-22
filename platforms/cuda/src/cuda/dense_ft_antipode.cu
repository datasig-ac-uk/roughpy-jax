#include "dense_ft_antipode.h"

#include <algorithm>
#include <functional>

#include <rpp/basis/basis_pack.hpp>
#include <rpp/basis/tensor_basis.hpp>
#include <rpp/operations/base_operation.hpp>

#include "batch.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"
#include "xla_headers.hpp"

using namespace rpy::jax::cuda;

namespace {
namespace ffi = xla::ffi;

struct DenseFTAntipodeStaticArgs {
    HostTensorBasis basis;
    int32_t max_degree;
    bool no_sign;
};

template <typename Strategy, bool NoSign>
class DenseFTAntipodeCompatOp : public rpp::ops::BaseOperation<Strategy> {
public:
    using Context = typename Strategy::Context;
    using Index = typename Strategy::Index;
    using Degree = typename Strategy::Degree;

    template <typename TensorOut, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx,
                               TensorOut& out,
                               TensorArg const& arg) const noexcept {
        using Scalar = typename TensorOut::value_type;
        using Accum = typename Strategy::Accum;
        auto const& basis = out.basis();

        for (Index elt_idx = ctx.thread_rank(); elt_idx < arg.size();
             elt_idx += ctx.num_threads()) {
            const Degree degree = basis.degree(elt_idx);
            const Index degree_begin = basis.start_of_degree(degree);
            const Index rev_idx =
                basis.reverse_index(elt_idx - degree_begin, degree) + degree_begin;

            if constexpr (NoSign) {
                out[rev_idx] = arg[elt_idx];
            } else {
                const Accum sign = (degree % 2 == 0) ? Accum{1} : Accum{-1};
                out[rev_idx] = static_cast<Scalar>(Accum{arg[elt_idx]} * sign);
            }
        }
    }
};

template <typename Tag>
struct DenseFTAntipodeFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer arg,
                           DenseFTAntipodeStaticArgs static_args,
                           cudaStream_t stream) noexcept {
        const auto tensor_size = static_args.basis.size();
        const auto out_shape = out->dimensions();
        const auto n_tensors =
            std::accumulate(out_shape.begin(),
                            out_shape.end() - 1,
                            1LL,
                            std::multiplies<>{});

        return select_strategy<Accum>(tensor_size, [&](auto strategy) {
            using LaunchCfg = typename decltype(strategy)::LaunchConfig;
            if (static_args.no_sign) {
                using Op = DenseFTAntipodeCompatOp<decltype(strategy), true>;
                return strategy.template launch<Op>(
                    LaunchCfg{stream},
                    std::make_tuple(
                        make_tensor_batch<Scalar>(out, 0, static_args.max_degree),
                        make_tensor_batch<Scalar>(arg, 0, static_args.max_degree)),
                    rpp::basis::make_basis_pack(static_args.basis),
                    n_tensors);
            }
            using Op = DenseFTAntipodeCompatOp<decltype(strategy), false>;
            return strategy.template launch<Op>(
                LaunchCfg{stream},
                std::make_tuple(
                    make_tensor_batch<Scalar>(out, 0, static_args.max_degree),
                    make_tensor_batch<Scalar>(arg, 0, static_args.max_degree)),
                rpp::basis::make_basis_pack(static_args.basis),
                n_tensors);
        });
    }
};

ffi::Error cuda_dense_ft_antipode_impl(cudaStream_t stream,
                                       ffi::Result<ffi::AnyBuffer> out,
                                       ffi::AnyBuffer arg,
                                       int32_t width,
                                       int32_t depth,
                                       DegreeBeginSpan degree_begin,
                                       int32_t max_degree,
    bool no_sign) noexcept {
    DenseFTAntipodeStaticArgs static_args{
        HostTensorBasis{
            static_cast<HostTensorBasis::Degree>(width),
            static_cast<HostTensorBasis::Degree>(depth),
            cast_db_array(degree_begin.begin())},
        max_degree,
        no_sign};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(arg, static_args.basis, max_degree));

    if (!all_buffers_match_type(out->element_type(), arg)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    return select_type_and_go<DenseFTAntipodeFunctor>(
        out->element_type(), out, arg, static_args, stream);
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
        .Attr<bool>("no_sign"));
