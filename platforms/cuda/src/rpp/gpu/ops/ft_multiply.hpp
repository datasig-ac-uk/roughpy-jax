#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_MULTIPLY_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_MULTIPLY_HPP

#include <algorithm>
#include <functional>

#include <rpp/config.h>

#include <rpp/operations.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/gpu/strategies.hpp>


namespace rpp::ops {
template<typename Scalar_, typename Accum_, unsigned MaxBlockSize, typename Architecture, typename Basis>
class FTMultiply<gpu::strategies::BlockStrategy<Scalar_, Accum_, MaxBlockSize, Architecture>, Basis> {
    using Strategy = gpu::strategies::BlockStrategy<Scalar_, Accum_, MaxBlockSize, Architecture>;
    using Scalar = Scalar_;
    using Accum = Accum_;
    using Basis = Basis;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    using Context = typename Strategy::Context;

    union SharedMemory {
        typename Strategy::BlockReduceArray reduce;
    };

private:
    RPP_DEVICE static Accum multiply_loop_with_degree(
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c,
        Index elt_idx,
        Degree degree,
        Basis const &basis
    ) {
        Accum acc{0};
        if (c.min_degree() == 0 && b.max_degree() >= degree) {
            const Accum lhs_val{b[elt_idx]};
            const Accum rhs_val{c[0]};
            acc += lhs_val * rhs_val;
        }

        const auto rhs_min_deg = std::max(1, std::max<Degree>(degree - b.max_degree(), c.min_degree()));
        const auto rhs_max_deg = std::min<Degree>(degree - b.min_degree(), c.max_degree());

        auto splitter = basis.splitter(rhs_min_deg);
        const Index idx = elt_idx - basis.degree_begin[degree];
        for (Index rhs_deg = std::max(rhs_min_deg, 1); rhs_deg <= std::min(degree - 1, rhs_max_deg); ++rhs_deg) {
            const auto lhs_deg = degree - rhs_deg;
            const auto lhs_idx = idx / splitter;
            const auto rhs_idx = idx % splitter;

            const Accum l_elt{b[basis.degree_begin[lhs_deg] + lhs_idx]};
            const Accum r_elt{c[basis.degree_begin[rhs_deg] + rhs_idx]};

            acc += l_elt * r_elt;

            splitter *= basis.width;
        }

        if (rhs_max_deg == degree) {
            const Accum lhs_val{b[0]};
            const Accum rhs_val{c[elt_idx]};
            acc += lhs_val * rhs_val;
        }

        return acc;
    }

    RPP_DEVICE static Accum multiply_loop(
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c,
        Index elt_idx,
        Basis const &basis
    ) noexcept {
        const auto degree = basis.degree(elt_idx);
        return multiply_loop_with_degree(b, c, elt_idx, degree, basis);
    }

public:
    RPP_DEVICE static void ft_fma(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> const &out,
        dense::DenseTensorView<Scalar const *, Basis> const &a,
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c,
        Accum alpha,
        Accum beta
    ) noexcept {
        auto const &basis = a.basis();

        for (Index elt_idx = ctx.thread_rank(); elt_idx < a.size(); elt_idx += ctx.num_threads()) {
            const auto degree = basis.degree(elt_idx);
            auto acc = multiply_loop_with_degree(b, c, elt_idx, degree, basis);

            acc *= beta;
            Accum a_val{0};
            if (a.has_degree(degree)) {
                a_val = Accum{a[elt_idx]};
            }

            out[elt_idx] = static_cast<Scalar>(alpha * a_val + acc);
        }
    }

    RPP_DEVICE static void ft_mul(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> const &out,
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c
    ) noexcept {
        auto const &basis = b.basis();

        for (Index elt_idx = ctx.thread_rank(); elt_idx < b.size(); elt_idx += ctx.num_threads()) {
            auto acc = multiply_loop(b, c, elt_idx, basis);
            out[elt_idx] = static_cast<Scalar>(acc);
        }
    }

    RPP_DEVICE static void ft_inplace_mul(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> const &lhs,
        dense::DenseTensorView<Scalar const *, Basis> const &rhs) noexcept {
        auto const &basis = lhs.basis();
        const auto low_range_degree = Strategy::low_range_degree(ctx, basis);

        // High pass
        for (Degree out_deg = basis.depth; out_deg > low_range_degree; --out_deg) {
            for (auto elt_idx = basis.start_of_degree(out_deg); elt_idx < basis.end_of_degree(out_deg); ++elt_idx) {
                auto acc = multiply_loop_with_degree(lhs, rhs, elt_idx, out_deg, basis);
                lhs[elt_idx] = static_cast<Scalar>(acc);
            }
            ctx.sync();
        }

        // Low pass
        auto elt_idx = std::min<Index>(ctx.thread_rank());
        const auto active = elt_idx < basis.end_of_degree(low_range_degree);
        Accum acc{0};
        if (active) {
            const auto degree = basis.degree_linear(elt_idx);
            acc = multiply_loop_with_degree(lhs, rhs, elt_idx, degree, basis);
        }

        ctx.sync();
        if (active) {
            lhs[elt_idx] = static_cast<Scalar>(acc);
        }
    }

    RPP_DEVICE static void ft_inplace_mul(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> const &lhs,
        dense::DenseTensorView<Scalar const *, Basis> const &rhs,
        Accum beta
    ) noexcept {
        auto const &basis = lhs.basis();
        const auto low_range_degree = ctx.low_range_degree(basis);

        // High pass
        for (Degree out_deg = basis.depth; out_deg > low_range_degree; --out_deg) {
            for (auto elt_idx = basis.start_of_degree(out_deg); elt_idx < basis.end_of_degree(out_deg); ++elt_idx) {
                auto acc = multiply_loop_with_degree(lhs, rhs, elt_idx, out_deg, basis);
                lhs[elt_idx] = static_cast<Scalar>(beta * acc);
            }
            ctx.sync();
        }

        // Low pass
        auto elt_idx = std::min<Index>(ctx.thread_rank());
        const auto active = elt_idx < basis.end_of_degree(low_range_degree);
        Accum acc{0};
        if (active) {
            const auto degree = basis.degree_linear(elt_idx);
            acc = multiply_loop_with_degree(lhs, rhs, elt_idx, degree, basis);
        }

        ctx.sync();
        if (active) {
            lhs[elt_idx] = static_cast<Scalar>(beta * acc);
        }
    }

private:
    template<InplaceFMAType FMAType>
    RPP_DEVICE static void inplace_fma(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> const &a,
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) noexcept {
        auto const &basis = a.basis();
        const auto low_range_degree = ctx.low_range_degree(basis);

        // High pass
        for (Degree out_deg = basis.depth; out_deg > low_range_degree; --out_deg) {
            for (auto elt_idx = basis.start_of_degree(out_deg); elt_idx < basis.end_of_degree(out_deg); ++elt_idx) {
                if constexpr (FMAType == InplaceFMAType::AEqualsABPlusC) {
                    auto acc = multiply_loop_with_degree(a, b, elt_idx, out_deg, basis);
                    acc *= beta;
                    Accum c_val{0};
                    if (c.has_degree(out_deg)) {
                        c_val = Accum{c[elt_idx]};
                    }
                    a[elt_idx] = static_cast<Scalar>(alpha * c_val + acc);
                } else if constexpr (FMAType == InplaceFMAType::AEqualsBAPlusC) {
                    auto acc = multiply_loop_with_degree(b, a, elt_idx, out_deg, basis);
                    acc *= beta;
                    Accum c_val{0};
                    if (c.has_degree(out_deg)) {
                        c_val = Accum{c[elt_idx]};
                    }
                    a[elt_idx] = static_cast<Scalar>(alpha * c_val + acc);
                } else if constexpr (FMAType == InplaceFMAType::AEqualsBCPlusA) {
                    auto acc = multiply_loop_with_degree(b, c, elt_idx, out_deg, basis);
                    acc *= beta;
                    const Accum a_val{a[elt_idx]};
                    a[elt_idx] = static_cast<Scalar>(alpha * a_val + acc);
                } else {
                    RPP_UNREACHABLE();
                }
            }
            ctx.sync();
        }

        // Low pass
        auto elt_idx = ctx.thread_rank();
        const auto active = elt_idx < basis.end_of_degree(low_range_degree);

        Accum acc{0};
        if (active) {
            const auto degree = basis.degree_linear(elt_idx);
            if constexpr (FMAType == InplaceFMAType::AEqualsABPlusC) {
                acc = multiply_loop_with_degree(a, b, elt_idx, degree, basis);
                acc *= beta;
                acc += alpha * Accum{c[elt_idx]};
            } else if constexpr (FMAType == InplaceFMAType::AEqualsBAPlusC) {
                acc = multiply_loop_with_degree(b, a, elt_idx, degree, basis);
                acc *= beta;
                acc += alpha * Accum{c[elt_idx]};
            } else if constexpr (FMAType == InplaceFMAType::AEqualsBCPlusA) {
                acc = multiply_loop_with_degree(b, c, elt_idx, degree, basis);
                acc *= beta;
                acc += alpha * Accum{a[elt_idx]};
            } else {
                RPP_UNREACHABLE();
            }
        }

        ctx.sync();
        if (active) {
            a[elt_idx] = static_cast<Scalar>(beta * acc);
        }
    }

public:
    RPP_DEVICE static void ft_fma(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> const &a,
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) noexcept {
        auto const &basis = a.basis();

        for (Index elt_idx = ctx.thread_rank(); elt_idx < a.size(); elt_idx += ctx.num_threads()) {
            auto acc = multiply_loop(b, c, elt_idx, basis);

            acc *= beta;
            const Accum a_val{a[elt_idx]};

            a[elt_idx] = static_cast<Scalar>(alpha * a_val + acc);
        }
    }

    RPP_DEVICE static void ft_inplace_fma_123(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> const &a,
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) noexcept {
        inplace_fma<InplaceFMAType::AEqualsABPlusC>(ctx, a, b, c, alpha, beta);
    }

    RPP_DEVICE static void ft_inplace_fma_213(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> const &a,
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) noexcept {
        inplace_fma<InplaceFMAType::AEqualsBAPlusC>(ctx, a, b, c, alpha, beta);
    }

    RPP_DEVICE static void ft_inplace_fma_231(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> const &a,
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
    ) noexcept {
        ft_fma(ctx, a, b, c, alpha, beta);
    }

private:
    RPP_DEVICE static Accum adjoint_degree_zero_reduce(
        Context const &ctx,
        dense::DenseTensorView<Scalar const *, Basis> const &op,
        dense::DenseTensorView<Scalar const *, Basis> const &arg,
        Basis const &basis
    ) noexcept {
        const auto deg_min = std::max(op.min_degree(), arg.min_degree());
        const auto deg_max = std::min(op.max_degree(), arg.max_degree());

        Accum val{0};
        for (Index i = basis.start_of_degree(deg_min); i < basis.end_of_degree(deg_max); i += ctx.num_threads()) {
            const Accum op_val{op[i]};
            const Accum arg_val{op[i]};

            val += op_val * arg_val;
        }

        return ctx.reduce(val, std::plus<Accum>{});
    }

    template<typename Fn>
    RPP_DEVICE static Accum adjoint_low_degree_reduce(
        Context const &ctx,
        dense::DenseTensorView<Scalar const *, Basis> const &op,
        dense::DenseTensorView<Scalar const *, Basis> const &arg,
        const Degree op_degree_min,
        const Degree op_degree_max,
        Basis const &basis,
        Fn &&arg_idx
    ) noexcept {
        Accum val{0};
        const auto begin = basis.start_of_degree(op_degree_min);
        const auto end = basis.end_of_degree(op_degree_max);

        for (Index i = begin; i < end; i += ctx.num_threads()) {
            const Accum op_val{op[i]};
            const Accum arg_val{op[arg_idx(i)]};

            val += op_val * arg_val;
        }

        return ctx.reduce(val, std::plus<Accum>{});
    }

public:
    RPP_DEVICE static void ft_adj_lmul(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> &out,
        dense::DenseTensorView<Scalar const *, Basis> const &op,
        dense::DenseTensorView<Scalar const *, Basis> const &arg
    ) noexcept {
        auto const &basis = out.basis();

        // Degree zero output
        if (op.min_degree() == 0) {
            out[0] = static_cast<Scalar>(adjoint_low_degree_reduce(
                ctx,
                op,
                arg,
                std::max(op.min_degree(), arg.min_degree()),
                std::min(op.max_degree(), arg.max_degree()),
                basis,
                [](Index i) { return i; }
            ));
        }

        Degree out_deg = std::max(op.min_degree(), 1);

        const auto small_size_threshold = ctx.num_threads() / 4;
        for (; out_deg < op.max_degree() && basis.size_of_degree(out_deg) <= small_size_threshold; ++out_deg) {
            const auto deg_1_op_min_deg = std::max(op.min_degree(), arg.min_degree()-1);
            const auto deg_1_op_max_deg = std::min(op.max_degree(), arg.max_degree() - 1);
            Index splitter = basis.size_of_degree(out_deg);

            for (Index suffix=basis.start_of_degre(out_deg); suffix<basis.end_of_degree(out_deg); ++suffix) {
                const auto val = adjoint_low_degree_reduce(
                    ctx,
                    op,
                    arg,
                    deg_1_op_min_deg,
                    deg_1_op_max_deg,
                    basis,
                    [&suffix, &splitter](Index i) { return i * splitter + suffix; }
                    );
                out[suffix] = static_cast<Scalar>(val);
            }
        }


        for (Index elt_idx = basis.start_of_degree(out_deg) + ctx.thread_rank(); elt_idx < out.size(); elt_idx += ctx.num_threads()) {
            const auto elt_degree = basis.degree(elt_idx);

            const auto op_min_deg = std::max(arg.min_degree() - elt_degree, op.min_degree());
            const auto op_max_deg = std::min(arg.max_degree() - elt_degree, op.max_degree());

            for (auto op_deg = op_min_deg; op_deg < op_max_deg; ++op_deg) {
                const auto begin = basis.start_of_degree(op_deg);
                const auto end = basis.end_of_degree(op_deg);
                const auto stride = end - begin;

                Accum elt { 0 };
                for (Index op_idx = basis.start_of_degree(op_deg); op_idx<basis.end_of_degree(op_deg); ++op_idx) {
                    Accum op_val { op[op_idx] };
                    Accum arg_val { arg[op_idx*stride + elt_idx] };
                    elt += op_val * arg_val;
                }
                out[elt_idx] = static_cast<Scalar>(elt);
            }
        }
    }

    RPP_DEVICE static void ft_adj_rmul(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> &out,
        dense::DenseTensorView<Scalar const *, Basis> const &op,
        dense::DenseTensorView<Scalar const *, Basis> const &arg
    ) noexcept {
        auto const& basis = out.basis();

        // Degree zero output
        if (op.min_degree() == 0) {
            out[0] = static_cast<Scalar>(adjoint_low_degree_reduce(
                ctx,
                op,
                arg,
                std::max(op.min_degree(), arg.min_degree()),
                std::min(op.max_degree(), arg.max_degree()),
                basis,
                [](Index i) { return i; }
            ));
        }

        const auto out_deg_min = std::max(1, out.min_degree());
        const auto out_deg_max = out.max_degree();

        auto const begin = basis.start_of_degree(out_deg_min);
        auto const end = basis.end_of_degree(out_deg_max);
        for (Index elt_idx = begin + ctx.thread_rank(); elt_idx < end; elt_idx += ctx.num_threads()) {
            const auto degree = basis.degree(elt_idx);
            const auto op_min_deg = std::max(op.min_degree(), arg.min_degree() - degree);
            const auto op_max_deg = std::min(op.max_degree(), arg.max_degree() - degree);

            Accum elt {0};
            for (Degree op_deg=op_min_deg; op_deg <= op_max_deg; ++op_deg) {
                const auto arg_stride = basis.size_of_degree(op_deg);
                for (Index op_idx = basis.start_of_degree(op_deg); op_idx < basis.end_of_degree(op_deg); ++op_idx) {
                    Accum arg_val { arg[elt_idx * arg_stride + op_idx] };
                    Accum op_val { op[op_idx] };
                    elt += arg_val * op_val;
                }
            }

            out[elt_idx] = static_cast<Scalar>(elt);
        }
    }
};
} // namespace rpp::ops


#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_MULTIPLY_HPP
