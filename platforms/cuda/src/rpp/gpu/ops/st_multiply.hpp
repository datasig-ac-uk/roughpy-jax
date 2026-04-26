#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_ST_MULTIPLY_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_ST_MULTIPLY_HPP

#include <algorithm>

#include <rpp/config.h>

#include <rpp/operations.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/gpu/strategies.hpp>

namespace rpp::ops {

template<typename Scalar_, typename Accum_, unsigned BlockSize, typename Architecture, typename Basis>
class STMultiply<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis>
    : Vector<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>;
    using Scalar = Scalar_;
    using Accum = Accum_;
    using Basis = Basis;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;
    using Letter = typename Strategy::Letter;
    using Bitmask = typename Strategy::Bitmask;

public:
    using Context = typename Strategy::Context;
    using SharedMemory = int;

private:

    RPP_DEVICE
    static Accum multiply_loop_with_degree(
        Context const& ctx,
        Index elt_idx,
        Degree elt_degree,
        Basis const& basis,
        dense::DenseTensorView<Scalar const*, Basis> const& lhs,
        dense::DenseTensorView<Scalar const*, Basis> const& rhs
    ) noexcept {

        const auto idx = elt_idx - basis.start_of_degree(elt_degree);
        Letter letters[Strategy::max_depth];
        basis.unpack_index_to_letters(
            letters,
            elt_degree,
            elt_idx
            );

        Accum acc { 0 };

        for (Bitmask mask { 0 }; mask < Bitmask{ 1 << elt_degree }; ++mask) {

            Index left_idx, right_idx;
            Degree left_deg, right_deg;
            basis.pack_masked_index(
                letters,
                elt_degree,
                mask,
                left_deg,
                left_idx,
                right_deg,
                right_idx
                );
            left_idx += basis.start_of_degree(left_deg);
            right_idx += basis.start_of_degree(right_deg);

            if (lhs.has_degree(left_deg) && rhs.has_degree(right_deg)) {
                const Accum left_val { lhs[left_idx] };
                const Accum right_val { rhs[right_idx] };
                acc += left_val * right_val;
            }
        }

        return acc;
    }

    RPP_DEVICE static void multiply_loop(
        Context const& ctx,
        Index elt_idx,
        Basis const& basis,
        dense::DenseTensorView<Scalar const*, Basis> const& lhs,
        dense::DenseTensorView<Scalar const*, Basis> const& rhs
        ) noexcept {
        const auto degree = basis.degree(elt_idx);
        return multiply_loop_with_degree(ctx, elt_idx, degree, lhs, rhs);
    }


public:
    RPP_DEVICE
    static void st_multiply(
        Context const& ctx,
        dense::DenseTensorView<Scalar*, Basis>& out,
        dense::DenseTensorView<Scalar const*, Basis> const& lhs,
        dense::DenseTensorView<Scalar const*, Basis> const& rhs,
        Accum beta
        ) noexcept {
        auto const& basis = out.basis();

        for (Index elt_idx = ctx.thread_rank(); elt_idx < out.size(); elt_idx += ctx.num_threads()) {
            auto acc = multiply_loop(ctx, elt_idx, basis, lhs, rhs);

            out[elt_idx] = static_cast<Scalar>(beta * acc);
        }
    }


    RPP_DEVICE
    static void st_fma(
        Context const& ctx,
        dense::DenseTensorView<Scalar*, Basis>& a,
        dense::DenseTensorView<Scalar const*, Basis> const& b,
        dense::DenseTensorView<Scalar const*, Basis> const& c,
        Accum alpha = Accum { 1 },
        Accum beta = Accum { 1 }
        ) noexcept {

        auto const& basis = a.basis();

        for (Index elt_idx = ctx.thread_rank(); elt_idx < a.size(); elt_idx += ctx.num_threads()) {
            auto acc = multiply_loop(ctx, elt_idx, basis, b, c);

            acc *= beta;
            const Accum a_val { a[elt_idx] };

            a[elt_idx] = static_cast<Scalar>(alpha * a_val + acc);
        }
    }

    RPP_DEVICE
    static void st_fma(
        Context const& ctx,
        dense::DenseTensorView<Scalar*, Basis>& out,
        dense::DenseTensorView<Scalar const*, Basis> const& a,
        dense::DenseTensorView<Scalar const*, Basis> const& b,
        dense::DenseTensorView<Scalar const*, Basis> const& c,
        Accum alpha = Accum { 1 },
        Accum beta = Accum { 1 }
        ) noexcept {
        auto const& basis = a.basis();

        for (Index elt_idx = ctx.thread_rank(); elt_idx < out.size(); elt_idx += ctx.num_threads()) {
            const auto degree = basis.degree(elt_idx);
            auto acc = multiply_loop_with_degree(ctx, elt_idx, degree, basis, b, c);

            acc *= beta;
            Accum a_val { 0 };
            if (a.has_degree(degree)) {
                a_val = a[elt_idx];
            }

            a[elt_idx] = static_cast<Scalar>(alpha * a_val + acc);
        }
    }



    RPP_DEVICE static void st_adj_mul(
        Context const& ctx,
        dense::DenseTensorView<Scalar*, Basis>& out,
        dense::DenseTensorView<Scalar const*, Basis> const& op,
        dense::DenseTensorView<Scalar const*, Basis> const& arg
        ) noexcept {
        auto const& basis = out.basis();

        STMultiply::set_zero(ctx, out);

        ctx.sync();

        for (Index elt_idx = ctx.thread_rank(); elt_idx<arg.size(); elt_idx += ctx.num_threads()) {
            const auto elt_degree = basis.degree(elt_idx);

            Letter letters[Strategy::max_depth];
            basis.unpack_index_to_letters(
                letters,
                elt_degree,
                elt_idx);

            const Accum arg_val { arg[elt_idx] };

            for (Bitmask mask { 0 }; mask < Bitmask{ 1 << elt_degree }; ++mask) {
                Index op_idx, out_idx;
                Degree op_deg, out_deg;
                basis.pack_masked_index(
                    letters,
                    elt_degree,
                    mask,
                    op_deg,
                    op_idx,
                    out_deg,
                    out_idx
                    );
                op_idx += basis.start_of_degree(op_deg);
                out_idx += basis.start_of_degree(out_deg);


                //TODO: This strategy won't work in general
                if (op.has_degree(op_deg)) {
                    const auto contrib = arg_val * Accum{ op[op_idx] };
                    atomicAdd(out + out_idx, static_cast<Scalar>(contrib));
                }
            }
        }

    }

};

} // namespace rpp::ops

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_ST_MULTIPLY_HPP
