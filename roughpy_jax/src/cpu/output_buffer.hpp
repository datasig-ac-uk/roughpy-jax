#ifndef ROUGHPY_JAX_CPU_OUTPUT_BUFFER_HPP
#define ROUGHPY_JAX_CPU_OUTPUT_BUFFER_HPP

#include <cstring>
#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cpu {

// Clear the complete output allocation, including coefficients outside an
// operation's natural non-zero range. Empty batches may have null data
// pointers.
inline void zero_result_buffer(ffi::Result<ffi::AnyBuffer> result) noexcept {
    const auto size = result->size_bytes();
    if (size != 0) {
        std::memset(result->untyped_data(), 0, size);
    }
}

} // namespace rpy::jax::cpu

#endif // ROUGHPY_JAX_CPU_OUTPUT_BUFFER_HPP
