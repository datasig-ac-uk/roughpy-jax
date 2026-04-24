#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_ADJ_MULTIPLY_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_ADJ_MULTIPLY_HPP

#include <rpp/config.h>

#include <rpp/operations.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/gpu/strategies.hpp>


namespace rpp::ops {


template<typename Scalar_, typename Accum_, unsigned BlockSize, typename Architecture, typename Basis>
class FTAdjointMultiply<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis> {
    using Strategy = gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>;
    using Scalar = Scalar_;
    using Accum = Accum_;
    using Basis = Basis;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    using Context = typename Strategy::Context;
    using SharedMemory = int;

    RPP_DEVICE static void ft_adj_lmul(
        Context const& ctx,
        dense::DenseTensorView<Scalar*, Basis>& out,
        dense::DenseTensorView<Scalar const*, Basis> const& op,
        dense::DenseTensorView<Scalar const*, Basis> const& arg
        ) noexcept {

    }

    RPP_DEVICE static void ft_adj_rmul(
        Context const& ctx,
        dense::DenseTensorView<Scalar*, Basis>& out,
        dense::DenseTensorView<Scalar const*, Basis> const& op,
        dense::DenseTensorView<Scalar const*, Basis> const& arg
        ) noexcept {

    }

};

} // namespace rpp::ops

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_ADJ_MULTIPLY_HPP
