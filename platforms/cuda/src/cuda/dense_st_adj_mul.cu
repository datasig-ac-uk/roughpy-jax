#include "dense_st_adj_mul.h"

#include <algorithm>
#include <functional>

#include <rpp/basis/basis_pack.hpp>
#include <rpp/basis/tensor_basis.hpp>
#include <rpp/operations/base_operation.hpp>

#include "batch.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"
#include "xla_headers.hpp"

namespace ffi = xla::ffi;

namespace rpy::jax::cuda {
namespace {

struct DenseSTAdjMulStaticArgs {
    HostTensorBasis basis;
    int32_t op_max_deg;
    int32_t arg_max_deg;
};

template <typename Strategy>
class DenseSTAdjMulCompatOp : public rpp::ops::BaseOperation<Strategy> {
public:
    using Context = typename Strategy::Context;
    using Accum = typename Strategy::Accum;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

    template <typename TensorOut, typename TensorOp, typename TensorArg>
    RPP_DEVICE void operator()(Context const& ctx,
                               TensorOut& out,
                               TensorOp const& op,
                               TensorArg const& arg) const noexcept {
        using Scalar = typename TensorOut::value_type;
        auto const& basis = out.basis();

        const auto arg_begin = basis.start_of_degree(arg.min_degree());
        const auto arg_end = basis.end_of_degree(arg.max_degree());

        for (Index out_idx = ctx.thread_rank(); out_idx < out.size();
             out_idx += ctx.num_threads()) {
            Accum sum{0};

            for (Index elt_idx = arg_begin; elt_idx < arg_end; ++elt_idx) {
                const auto elt_degree = basis.degree(elt_idx);
                Letter letters[Strategy::Architecture::max_depth];
                basis.unpack_index_to_letters(
                    letters,
                    elt_degree,
                    elt_idx - basis.start_of_degree(elt_degree));

                const Accum arg_val{arg[elt_idx]};
                for (Bitmask mask{0}; mask < (Bitmask{1} << elt_degree); ++mask) {
                    Index op_idx;
                    Index candidate_out_idx;
                    Degree op_deg;
                    Degree out_deg;
                    basis.pack_masked_index(letters,
                                            elt_degree,
                                            mask,
                                            op_deg,
                                            op_idx,
                                            out_deg,
                                            candidate_out_idx);
                    op_idx += basis.start_of_degree(op_deg);
                    candidate_out_idx += basis.start_of_degree(out_deg);

                    if (candidate_out_idx == out_idx && op.has_degree(op_deg)) {
                        sum += arg_val * Accum{op[op_idx]};
                    }
                }
            }

            out[out_idx] = static_cast<Scalar>(sum);
        }
    }
};

template <typename Tag>
struct DenseSTAdjMulFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer op,
                           ffi::AnyBuffer arg,
                           DenseSTAdjMulStaticArgs static_args,
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
            using Op = DenseSTAdjMulCompatOp<decltype(strategy)>;
            return strategy.template launch<Op>(
                LaunchCfg{stream},
                std::make_tuple(
                    make_tensor_batch<Scalar>(out, 0, static_args.basis.depth),
                    make_tensor_batch<Scalar>(op, 0, static_args.op_max_deg),
                    make_tensor_batch<Scalar>(arg, 0, static_args.arg_max_deg)),
                rpp::basis::make_basis_pack(static_args.basis),
                n_tensors);
        });
    }
};

} // namespace

ffi::Error cuda_dense_st_adj_mul(cudaStream_t stream,
                                 ffi::Result<ffi::AnyBuffer> out,
                                 ffi::AnyBuffer op,
                                 ffi::AnyBuffer arg,
                                 int32_t width,
                                 int32_t depth,
                                 DegreeBeginSpan degree_begin,
                                 int32_t op_max_deg,
                                 int32_t arg_max_deg) noexcept {
    DenseSTAdjMulStaticArgs static_args{
        HostTensorBasis{
            static_cast<HostTensorBasis::Degree>(width),
            static_cast<HostTensorBasis::Degree>(depth),
            cast_db_array(degree_begin.begin())},
        op_max_deg,
        arg_max_deg};

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(out, static_args.basis, depth));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(op, static_args.basis, op_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(arg, static_args.basis, arg_max_deg));

    if (!all_buffers_match_type(out->element_type(), op, arg)) {
        return ffi::Error::InvalidArgument(
            "all tensors should have the same data type");
    }

    return select_type_and_go<DenseSTAdjMulFunctor>(
        out->element_type(), out, op, arg, static_args, stream);
}

} // namespace rpy::jax::cuda

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_st_adj_mul,
    rpy::jax::cuda::cuda_dense_st_adj_mul,
    xla::ffi::Ffi::Bind()
        .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
        .Ret<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Arg<xla::ffi::AnyBuffer>()
        .Attr<int32_t>("width")
        .Attr<int32_t>("depth")
        .Attr<rpy::jax::cuda::DegreeBeginSpan>("degree_begin")
        .Attr<int32_t>("op_max_deg")
        .Attr<int32_t>("arg_max_deg"));
