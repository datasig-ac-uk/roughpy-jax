#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_MULTIPLY_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_MULTIPLY_HPP

#include <algorithm>

#include <rpp/config.h>

#include <rpp/operations.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/gpu/strategies.hpp>


namespace rpp::ops {
template<typename Scalar_, typename Accum_, unsigned BlockSize, typename Architecture, typename Basis>
class FTMultiply<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis> {
    using Strategy = gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>;
    using Scalar = Scalar_;
    using Accum = Accum_;
    using Basis = Basis;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    using Context = typename Strategy::Context;
    using SharedMemory = int;

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
            auto acc = multiply_loop(b, c, elt_idx, basis);

            acc *= beta;
            const Accum a_val{a[elt_idx]};

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
        auto const& basis = lhs.basis();
        const auto low_range_degree = Strategy::low_range_degree(ctx, basis);

        // High pass
        for (Degree out_deg=basis.depth; out_deg>low_range_degree; --out_deg) {
            for (auto elt_idx=basis.start_of_degree(out_deg); elt_idx<basis.end_of_degree(out_deg); ++elt_idx) {
                auto acc = multiply_loop_with_degree(lhs, rhs, elt_idx, out_deg, basis);
                lhs[elt_idx] = static_cast<Scalar>(acc);
            }
            ctx.sync();
        }

        // Low pass
        auto elt_idx = std::min<Index>(ctx.thread_rank());
        const auto active = elt_idx < basis.end_of_degree(low_range_degree);
        Accum acc { 0 };
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
        auto const& basis = lhs.basis();
        const auto low_range_degree = Strategy::low_range_degree(ctx, basis);

        // High pass
        for (Degree out_deg=basis.depth; out_deg>low_range_degree; --out_deg) {
            for (auto elt_idx=basis.start_of_degree(out_deg); elt_idx<basis.end_of_degree(out_deg); ++elt_idx) {
                auto acc = multiply_loop_with_degree(lhs, rhs, elt_idx, out_deg, basis);
                lhs[elt_idx] = static_cast<Scalar>(beta * acc);
            }
            ctx.sync();
        }

        // Low pass
        auto elt_idx = std::min<Index>(ctx.thread_rank());
        const auto active = elt_idx < basis.end_of_degree(low_range_degree);
        Accum acc { 0 };
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

    template <InplaceFMAType FMAType>
    RPP_DEVICE static void inplace_fma(
        Context const& ctx,
        dense::DenseTensorView<Scalar *, Basis> const &a,
        dense::DenseVectorView<Scalar const *, Basis> const &b,
        dense::DenseVectorView<Scalar const *, Basis> const &c,
        Accum alpha = Accum{1},
        Accum beta = Accum{1}
        ) noexcept {
        auto const &basis = a.basis();
        const auto low_range_degree = Strategy::low_range_degree(ctx, a.basis());

        // High pass
        for (Degree out_deg=basis.depth; out_deg>low_range_degree; --out_deg) {
            for (auto elt_idx=basis.start_of_degree(out_deg); elt_idx<basis.end_of_degree(out_deg); ++elt_idx) {
                if constexpr (FMAType == InplaceFMAType::AEqualsABPlusC) {
                    auto acc = multiply_loop(a, b, elt_idx, out_deg, basis);
                    acc *= beta;
                    const Accum c_val { c[elt_idx] };
                    a[elt_idx] = static_cast<Scalar>(alpha * c_val + acc);
                } else if constexpr (FMAType == InplaceFMAType::AEqualsBAPlusC) {
                    auto acc = multiply_loop(b, a, elt_idx, out_deg, basis);
                    acc *= beta;
                    const Accum c_val { c[elt_idx] };
                    a[elt_idx] = static_cast<Scalar>(alpha * c_val + acc);
                } else if constexpr (FMAType == InplaceFMAType::AEqualsBCPlusA) {
                    auto acc = multiply_loop(b, c, elt_idx, out_deg, basis);
                    acc *= beta;
                    const Accum a_val { a[elt_idx] };
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

        Accum acc { 0 };
        if (active) {
            const auto degree = basis.degree_linear(elt_idx);
            if constexpr (FMAType == InplaceFMAType::AEqualsABPlusC) {
                acc = multiply_loop(a, b, elt_idx, degree, basis);
                acc *= beta;
                acc += alpha * Accum { c[elt_idx] };
            } else if constexpr (FMAType == InplaceFMAType::AEqualsBAPlusC) {
                acc = multiply_loop(b, a, elt_idx, degree, basis);
                acc *= beta;
                acc += alpha * Accum { c[elt_idx] };
            } else if constexpr (FMAType == InplaceFMAType::AEqualsBCPlusA) {
                acc = multiply_loop(b, c, elt_idx, degree, basis);
                acc *= beta;
                acc += alpha * Accum { a[elt_idx] };
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
};



} // namespace rpp::ops


#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_MULTIPLY_HPP
