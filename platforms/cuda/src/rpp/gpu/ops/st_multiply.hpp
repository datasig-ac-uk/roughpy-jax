#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_ST_MULTIPLY_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_ST_MULTIPLY_HPP

#include <algorithm>

#include <rpp/config.h>

#include <rpp/operations.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/gpu/strategies.hpp>

namespace rpp::ops {

template<typename Scalar_, typename Accum_, unsigned BlockSize, typename Architecture, typename Basis>
class STMultiply<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis> {
    using Strategy = gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>;
    using Scalar = Scalar_;
    using Accum = Accum_;
    using Basis = Basis;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    using Context = typename Strategy::Context;
    using SharedMemory = int;

};

} // namespace rpp::ops

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_ST_MULTIPLY_HPP
