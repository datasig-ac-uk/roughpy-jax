#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_EXP_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_EXP_HPP

#include <rpp/config.h>

#include <rpp/operations.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/gpu/strategies.hpp>

#include <rpp/gpu/ops/ft_basic.hpp>
#include <rpp/gpu/ops/ft_multiply.hpp>


namespace rpp::ops {
template<typename Scalar_, typename Accum_, unsigned BlockSize, typename Architecture, typename Basis>
class FTExpAndLog<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis>
        : FTBasic<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis>,
          FTMultiply<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis> {
    using Strategy = gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>;

    using Scalar = Scalar_;
    using Accum = Accum_;
    using Basis = Basis;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    using Context = typename Strategy::Context;
    using SharedMemory = int;

    RPP_DEVICE static void ft_exp(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> &out,
        dense::DenseTensorView<Scalar const *, Basis> const &arg
    ) noexcept {
        auto &basis = out.basis();

        FTExpAndLog::set_identity(ctx, out);
        constexpr Accum one{1};

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;

            ctx.sync();

            FTExpAndLog::ft_inplace_mul(
                ctx,
                out,
                arg.truncate(1, max_degree),
                divisor
            );

            FTExpAndLog::add_identity(out);
        }
    }

    RPP_DEVICE static void ft_fmexp(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> &out,
        dense::DenseTensorView<Scalar const *, Basis> const &multiplier,
        dense::DenseTensorView<Scalar const *, Basis> const &exponent
    ) noexcept {
        auto &basis = out.basis();

        FTExpAndLog::assign(ctx, out, multiplier);
        constexpr Accum one{1};

        for (Degree d = basis.depth; d > 0; --d) {
            const auto max_degree = basis.depth - d + 1;
            const Accum divisor = one / d;

            ctx.sync();

            FTExpAndLog::ft_inplace_fma_123(
                ctx,
                out,
                exponent.truncate(1, max_degree),
                multiplier,
                one,
                divisor
            );
        }
    }

    RPP_DEVICE static void ft_log(
        Context const &ctx,
        dense::DenseTensorView<Scalar *, Basis> &out,
        dense::DenseTensorView<Scalar const *, Basis> const &arg
    ) noexcept {
        auto &basis = out.basis();

        FTExpAndLog::set_zero(ctx, out);
        constexpr Accum one{1};

        for (Degree d = basis.depth; d > 0; --d) {
            Accum val = one / d;
            if (d % 2 == 0) {
                val = -val;
            }

            if (ctx.thread_rank() == 0) {
                Accum unit{out[0]};
                out[0] = static_cast<Scalar>(unit + val);
            }

            ctx.sync();

            FTExpAndLog::ft_inplace_mul(
                ctx,
                out,
                arg.truncate(1, d)
            );
        }
    }
};
} // namespace rpp::ops

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_EXP_HPP
