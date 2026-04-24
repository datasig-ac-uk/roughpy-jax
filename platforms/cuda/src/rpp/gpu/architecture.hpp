#ifndef PLATFORMS_CUDA_SRC_RPP_GPU_ARCHITECTURE_HPP
#define PLATFORMS_CUDA_SRC_RPP_GPU_ARCHITECTURE_HPP

#include <cstddef>
#include <cstdint>

#include <rpp/architecture.hpp>

namespace rpp::gpu::arch {

template <typename Size_>
struct GPUArchitecture : public ::rpp::arch::Architecture<Size_> {
    static constexpr unsigned warp_size = 32;
};

using NativeArchitecture = GPUArchitecture<std::size_t>;
using Architecture32 = GPUArchitecture<std::uint32_t>;
using Architecture64 = GPUArchitecture<std::uint64_t>;


using DefaultArchitecture = Architecture32;


};

#endif //PLATFORMS_CUDA_SRC_RPP_GPU_ARCHITECTURE_HPP
