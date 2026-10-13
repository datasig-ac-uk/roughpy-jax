#ifndef ROUGHPY_JAX_CUDA_SELECT_STRATEGY_CUH
#define ROUGHPY_JAX_CUDA_SELECT_STRATEGY_CUH


#include <rpp/architecture.hpp>
#include <rpp/gpu/architecture.hpp>
#include <rpp/gpu/block/strategy.hpp>

#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cuda {
using DefaultArch = rpp::gpu::arch::DefaultArchitecture;



template<typename Accum, typename Arch=DefaultArch, typename Index,typename F>
xla::ffi::Error select_strategy(Index problem_size, F&& func) noexcept {
    if (problem_size < 64) {
        using Strategy = rpp::gpu::strategies::BlockStrategy<Accum, 32, 256, Arch>;
        return map_err(func(Strategy{32}));
    }
    if (problem_size < 128) {
        using Strategy = rpp::gpu::strategies::BlockStrategy<Accum, 64, 256, Arch>;
        return map_err(func(Strategy{64}));
    }
    if (problem_size < 256) {
        using Strategy = rpp::gpu::strategies::BlockStrategy<Accum, 128, 256, Arch>;
        return map_err(func(Strategy{128}));
    }
    using Strategy = rpp::gpu::strategies::BlockStrategy<Accum, 256, 256, Arch>;
    return map_err(func(Strategy{256}));
}
} // namespace rpy::jax::cuda


#endif //ROUGHPY_JAX_CUDA_SELECT_STRATEGY_CUH
