#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_BASIC_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_BASIC_HPP


#include <rpp/operations.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/dense/views.hpp>

#include <rpp/gpu/ops/vector.hpp>

namespace rpp::ops {

template<typename Scalar_, typename Accum_, unsigned BlockSize, typename Architecture, typename Basis>
class FTBasic<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>, Basis>
    : Vector<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>>
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

    RPP_DEVICE static void add_identity(
        Context const& ctx,
        dense::DenseTensorView<Scalar *, Basis>& tensor,
        Accum scalar = Accum{1}
        ) noexcept {
        if (ctx.thread_rank() == 0) {
            Accum elt { tensor[0] };
            tensor[0] = static_cast<Scalar>(elt + scalar);
        }
    }

    RPP_DEVICE static void set_identity(
        Context const& ctx,
        dense::DenseTensorView<Scalar *, Basis>& tensor,
        Accum scalar = Accum { 1 }
        ) noexcept {
        FTBasic::set_identity(ctx, tensor);
        if (ctx.thread_rank() == 0) {
            tensor[0] = static_cast<Scalar>(scalar);
        }
    }
};

}// namespace rpp::ops



#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_FT_BASIC_HPP
