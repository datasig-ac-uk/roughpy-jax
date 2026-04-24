#ifndef PLATFORMS_CUDA_SRC_RPP_ARCHITECTURE_HPP
#define PLATFORMS_CUDA_SRC_RPP_ARCHITECTURE_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rpp::arch {

template <typename Size_>
struct Architecture {
    using Size = Size_;
    using Index = std::make_signed_t<Size>;

    using Degree = int32_t;
};


using NativeArchitecture = Architecture<std::size_t>;
using Architecture32 = Architecture<std::uint32_t>;
using Architecture64 = Architecture<std::uint64_t>;

}

#endif //PLATFORMS_CUDA_SRC_RPP_ARCHITECTURE_HPP
