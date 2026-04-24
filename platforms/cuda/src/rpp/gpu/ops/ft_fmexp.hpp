#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_FMEXP_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_FMEXP_HPP

#include <rpp/config.h>

#include <rpp/operations.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/gpu/strategies.hpp>

#include <rpp/gpu/ops/ft_basic.hpp>
#include <rpp/gpu/ops/ft_multiply.hpp>


namespace rpp::ops {


template<typename Scalar_, typename Accum_, unsigned BlockSize, typename Architecture, typename Basis>
class FTFMExp<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis>
    : FTBasic<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis>,
      FTMultiply<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis>
{
    using Strategy = gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>;
    using Scalar = Scalar_;
    using Accum = Accum_;
    using Basis = Basis;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    using Context = typename Strategy::Context;
    using SharedMemory = int;

    RPP_DEVICE static void ft_fmexp(
        Context const& ctx,
        dense::DenseTensorView<Scalar*, Basis>& out,
        dense::DenseTensorView<Scalar const*, Basis> const& arg
        ) noexcept {


    }

};


} // namespace rpp::ops
#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_FMEXP_HPP
