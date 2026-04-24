#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_STRATEGIES_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_STRATEGIES_HPP

#include <cstdint>
#include <cstddef>


#include <cuda_runtime.h>

#include <rpp/config.h>
#include <rpp/gpu/architecture.hpp>


namespace rpp::gpu::strategies {

namespace detail {

template <typename Strategy_>
class BlockContext {
    using Strategy = Strategy_;

    std::byte* smem_ptr_;

public:
    explicit constexpr BlockContext(std::byte* smem_ptr) noexcept
        : smem_ptr_(smem_ptr)
    {}

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned thread_rank() noexcept {
        return threadIdx.x;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned num_threads() noexcept {
        return blockDim.x;
    }

    RPP_DEVICE static void sync() noexcept {
        __syncthreads();
    }

    template <typename SharedMemory>
    RPP_DEVICE SharedMemory& shared_memory() noexcept {
        return *reinterpret_cast<SharedMemory*>(smem_ptr_);
    }
};




} // namespace detail

template <typename Scalar_, typename Accum_, unsigned BlockSize = 32, typename Architecture_=arch::DefaultArchitecture>
struct BlockStrategy {
    using Scalar = Scalar_;
    using Accum = Accum_;
    using Architecture = Architecture_;
    using Size = typename Architecture::Size;
    using Index = typename Architecture::Index;
    using Degree = typename Architecture::Degree;

    using Context = detail::BlockContext<BlockStrategy>;

    static constexpr unsigned block_size = BlockSize;
    static constexpr unsigned warp_size = Architecture::warp_size;

    template <typename Basis>
    RPP_HOST_DEVICE static constexpr Degree low_range_degree(Context context& ctx, Basis const& basis) noexcept {
        Degree result = 0;
        while (result <= basis.depth && basis.degree_begin[result+1] < block_size) {
            ++result;
        }
        return result;
    }

};




} // namespace rpp::gpu::strategies

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_STRATEGIES_HPP
