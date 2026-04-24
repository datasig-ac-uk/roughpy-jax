#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_OPS_VECTOR_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_OPS_VECTOR_HPP

#include <rpp/config.h>

#include <rpp/operations.hpp>
#include <rpp/gpu/strategies.hpp>
#include <rpp/dense/views.hpp>

namespace rpp::ops {

template<typename Scalar_, typename Accum_, unsigned BlockSize, typename Architecture>
class Vector<gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>> {
    using Strategy = gpu::strategies::BlockStrategy<Scalar_, Accum_, BlockSize, Architecture>;
    using Scalar = Scalar_;
    using Accum = Accum_;

    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

public:
    using Context = typename Strategy::Context;
    using SharedMemory = int;

    template <typename Basis>
    RPP_DEVICE static void set_constant(
        Context const& ctx,
        dense::DenseVectorView<Scalar*, Basis>& tensor,
        Scalar const& value
        ) noexcept {
        for (Index i=ctx.thread_rank(); i<tensor.size(); i+=ctx.num_threads()) {
            tensor[i] = static_cast<Scalar>(value);
        }
    }

    template <typename Basis>
    RPP_DEVICE static void set_zero(
        Context const& ctx,
        dense::DenseVectorView<Scalar *, Basis>& tensor
        ) {
        set_constant(ctx, tensor, Accum{0});
    }

    template <typename Basis>
    RPP_DEVICE static void assign(
        Context const& ctx,
        dense::DenseVectorView<Scalar *, Basis>& out,
        dense::DenseVectorView<Scalar *, Basis> const& arg
        ) noexcept {
        for (Index i=ctx.thread_rank(); i<out.size(); i+=ctx.num_threads()) {
            out[i] = arg[i];
        }
    }

};

} // namespace rpp::ops

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_OPS_VECTOR_HPP
