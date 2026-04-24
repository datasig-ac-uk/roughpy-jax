#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_ANTIPODE_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_ANTIPODE_HPP

#include <rpp/config.h>

#include <rpp/basis.hpp>
#include <rpp/operations.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/gpu/strategies.hpp>

namespace rpp::ops {


template<typename Scalar_, typename Accum_, unsigned BlockSize, typename Architecture, typename Basis>
class Antipode<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis> {
    using Strategy = gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>;
    using Scalar = Scalar_;
    using Accum = Accum_;
    using Basis = Basis;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    using Context = typename Strategy::Context;
    using SharedMemory = int;

    RPP_DEVICE static void antipode(
        Context const& ctx,
        dense::DenseTensorView<Scalar*, Basis>& out,
        dense::DenseTensorView<Scalar const*, Basis> const& arg
        ) noexcept
    {}

    RPP_DEVICE static void reflect(
        Context const& ctx,
        dense::DenseTensorView<Scalar*, Basis>& out,
        dense::DenseTensorView<Scalar const*, Basis> const& arg
        ) noexcept
    {}

};




}

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_ANTIPODE_HPP
