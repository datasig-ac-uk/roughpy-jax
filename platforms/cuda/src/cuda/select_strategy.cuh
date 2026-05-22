#ifndef PLATFORMS_CUDA_SRC_CUDA_SELECT_STRATEGY_CUH
#define PLATFORMS_CUDA_SRC_CUDA_SELECT_STRATEGY_CUH


#include <rpp/architecture.hpp>
#include <rpp/support/error.hpp>
#include <rpp/gpu/architecture.hpp>
#include <rpp/gpu/block/strategy.hpp>

#include "xla_headers.hpp"

namespace rpy::jax::cuda {
using DefaultArch = rpp::gpu::arch::DefaultArchitecture;

namespace detail {
template<typename Payload>
xla::ffi::Error map_err(rpp::Error<Payload> &&err) noexcept {
    using rpp::ErrorCode;
    using XLAErr = xla::ffi::Error;
    using XLAEC = xla::ffi::ErrorCode;

    switch (err.code()) {
        case ErrorCode::Success: return XLAErr::Success();
        case ErrorCode::Cancelled: return XLAErr{XLAEC::kCancelled, std::string(err.message())};
        case ErrorCode::Unknown: return XLAErr{XLAEC::kUnknown, std::string(err.message())};
        case ErrorCode::Internal: return XLAErr{XLAEC::kInternal, std::string(err.message())};
        case ErrorCode::Timeout: return XLAErr{XLAEC::kDeadlineExceeded, std::string(err.message())};
        case ErrorCode::OutOfResources: return XLAErr{XLAEC::kResourceExhausted, std::string(err.message())};
        case ErrorCode::ContractViolation: return XLAErr{XLAEC::kFailedPrecondition, std::string(err.message())};
        case ErrorCode::OutOfBounds: return XLAErr{XLAEC::kOutOfRange, std::string(err.message())};
        case ErrorCode::NotImplemented: return XLAErr{XLAEC::kUnimplemented, std::string(err.message())};
        default:
            return XLAErr{XLAEC::kUnknown, std::string(err.message())};
    }
}


} // namespace detail

template<typename Accum, typename Arch=DefaultArch, typename Index,typename F>
xla::ffi::Error select_strategy(Index problem_size, F&& func) noexcept {
    if (problem_size < 64) {
        using Strategy = rpp::gpu::strategies::BlockStrategy<Accum, 32, 256, Arch>;
        return detail::map_err(func(Strategy{32}));
    }
    if (problem_size < 128) {
        using Strategy = rpp::gpu::strategies::BlockStrategy<Accum, 64, 256, Arch>;
        return detail::map_err(func(Strategy{64}));
    }
    if (problem_size < 256) {
        using Strategy = rpp::gpu::strategies::BlockStrategy<Accum, 128, 256, Arch>;
        return detail::map_err(func(Strategy{128}));
    }
    using Strategy = rpp::gpu::strategies::BlockStrategy<Accum, 256, 256, Arch>;
    return detail::map_err(func(Strategy{256}));
}
} // namespace rpy::jax::cuda


#endif //PLATFORMS_CUDA_SRC_CUDA_SELECT_STRATEGY_CUH
