#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_STRATEGIES_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_STRATEGIES_HPP

#include <cstdint>
#include <cstddef>
#include <type_traits>


#include <cuda_runtime.h>

#include <rpp/config.h>
#include <rpp/gpu/architecture.hpp>


namespace rpp::gpu::strategies {

namespace detail {

template <typename Strategy_>
class BlockContext {
    using Strategy = Strategy_;
    using Accum = typename Strategy::Accum;
    using BlockReduceArray = typename Strategy::BlockReduceArray;
    using Degree = typename Strategy::Degree;
    using Index = typename Strategy::Index;

    std::byte* smem_ptr_;

public:
    explicit constexpr BlockContext(std::byte* smem_ptr) noexcept
        : smem_ptr_(smem_ptr)
    {}

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned thread_rank() noexcept {
        return threadIdx.x;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned warp_lane() noexcept {
        return threadIdx.x % Strategy::warp_size;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned warp_idx() noexcept {
        return threadIdx.x / Strategy::warp_size;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned num_threads() noexcept {
        return blockDim.x;
    }

    RPP_DEVICE RPP_NODISCARD static constexpr unsigned num_warps() noexcept {
        return blockDim.x / Strategy::warp_size;
    }

    RPP_DEVICE static void sync() noexcept {
        __syncthreads();
    }

    template <typename SharedMemory>
    RPP_DEVICE SharedMemory& shared_memory() noexcept {
        return *reinterpret_cast<SharedMemory*>(smem_ptr_);
    }

    template <typename Fn>
    RPP_DEVICE static Accum warp_reduce(Accum val, Fn&& fn) noexcept {
        for (unsigned i = Strategy_::warp_size / 2; i > 0; i /= 2) {
            val = fn(val, __shfl_down_sync(0xffffffffu, val, i));
        }
        return val;
    }

    template <typename Fn>
    RPP_DEVICE static Accum reduce(Accum val, Fn&& fn) noexcept {
        auto& smem = shared_memory<BlockReduceArray>();
        val = warp_reduce(val, fn);

        const auto widx = warp_idx();
        const auto wlane = warp_lane();
        if (wlane == 0) {
            smem[widx] = val;
        }
        sync();

        Accum block_sum { 0 };
        if (widx == 0) {
            if (wlane < Strategy::warp_count) {
                block_sum = smem[wlane];
            }
            block_sum = warp_reduce(block_sum, fn);
        }

        return block_sum;
    }

    template <typename Basis>
    RPP_DEVICE Degree low_range_degree(Basis const& basis) const noexcept {
        Degree result = 0;
        const auto threads = static_cast<Index>(num_threads());
        while (result <= basis.depth && basis.end_of_degree(result) < threads) {
            ++result;
        }
        return result;
    }
};

struct GroupConfig {
    unsigned thread_rank_: 4; // 4 (0 <= x <= 31)
    unsigned num_threads_: 5; // 9 (0 <= x <= 32, power of 2)
    unsigned num_threads_pow_: 3; // 12 (0 <= x <= 5)
    unsigned group_lane_: 7; // 19 (0 <= x < 256)
};

template <typename Strategy_>
class SmallCoopGroupContext {
    using Strategy = Strategy_;

    uint8_t* smem_ptr;
    GroupConfig config;
    unsigned group_mask;

public:
    RPP_DEVICE explicit constexpr SmallCoopGroupContext(unsigned group_size_pow)
    {
        unsigned

    }

    RPP_DEVICE constexpr unsigned thread_rank() const noexcept {
        return config.thread_rank_;
    }

    RPP_DEVICE constexpr unsigned num_threads() const noexcept {
        return config.num_threads_;
    }

    RPP_DEVICE void sync() const noexcept {
        __syncwarp(group_mask);
    }

    RPP_DEVICE static unsigned warp_lane() noexcept {
        return threadIdx.x % Strategy::warp_size;
    }

    RPP_DEVICE static unsigned warp_idx() noexcept {
        return threadIdx.x / Strategy::warp_size;
    }



    template <typename SharedMemory>
    RPP_DEVICE decltype(auto) shared_memory() const noexcept {
        if constexpr (std::is_pointer_v<SharedMemory>) {
            return reinterpret_cast<SharedMemory>(smem_ptr);
        } else {
            return *reinterpret_cast<SharedMemory*>(smem_ptr);
        }
    }

};


} // namespace detail

template <typename Scalar_, typename Accum_, unsigned MaxBlockSize = 256, typename Architecture_=arch::DefaultArchitecture>
struct BlockStrategy {
    using Scalar = Scalar_;
    using Accum = Accum_;
    using Architecture = Architecture_;
    using Size = typename Architecture::Size;
    using Index = typename Architecture::Index;
    using Degree = typename Architecture::Degree;
    using Letter = typename Architecture::Letter;
    using Bitmask = typename Architecture::Bitmask;

    using Context = detail::BlockContext<BlockStrategy>;

    static constexpr unsigned max_block_size = MaxBlockSize;
    static constexpr unsigned warp_size = Architecture::warp_size;
    static constexpr unsigned max_warp_count = (max_block_size + warp_size - 1) / warp_size;

    using BlockReduceArray = Accum[max_warp_count];
};






} // namespace rpp::gpu::strategies

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_STRATEGIES_HPP
