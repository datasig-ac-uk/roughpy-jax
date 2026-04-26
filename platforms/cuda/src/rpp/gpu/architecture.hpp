#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_ARCHITECTURE_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_ARCHITECTURE_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <rpp/architecture.hpp>

namespace rpp::gpu::arch {

namespace detail {

using BitmaskBase = unsigned;

template <unsigned MaxDepth>
inline constexpr size_t kBitmaskArraySize = (MaxDepth + 8*sizeof(BitmaskBase) - 1) / (8*sizeof(BitmaskBase));


}

template <typename Size_, typename Letter_=uint8_t, unsigned MaxDepth=30>
struct GPUArchitecture : public ::rpp::arch::Architecture<Size_> {
    static constexpr unsigned warp_size = 32;

    using Letter = Letter_;
    using Bitmask = unsigned;

    static constexpr unsigned max_width = std::numeric_limits<Letter>::max();
    static constexpr unsigned max_depth = MaxDepth;
};

using NativeArchitecture = GPUArchitecture<std::size_t>;
using Architecture32 = GPUArchitecture<std::uint32_t>;
using Architecture64 = GPUArchitecture<std::uint64_t>;


using DefaultArchitecture = Architecture32;


};

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_ARCHITECTURE_HPP
