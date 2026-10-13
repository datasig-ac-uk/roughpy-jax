#ifndef ROUGHPY_JAX_CUDA_OUTPUT_BUFFER_CUH
#define ROUGHPY_JAX_CUDA_OUTPUT_BUFFER_CUH

#include <cuda_runtime_api.h>
#include <string>
#include <xla_helpers/xla_common.hpp>

namespace rpy::jax::cuda {

// Enqueue clearing on the execution stream so the following kernel observes
// zeroed output without synchronising the host or any unrelated streams.
inline ffi::Error zero_result_buffer(ffi::Result<ffi::AnyBuffer> result,
                                     cudaStream_t stream) noexcept {
    const auto size = result->size_bytes();
    if (size == 0) {
        return ffi::Error::Success();
    }
    const auto status =
        cudaMemsetAsync(result->untyped_data(), 0, size, stream);
    if (status != cudaSuccess) {
        return ffi::Error::Internal(std::string("cudaMemsetAsync failed: ") +
                                    cudaGetErrorString(status));
    }
    return ffi::Error::Success();
}

} // namespace rpy::jax::cuda

#endif // ROUGHPY_JAX_CUDA_OUTPUT_BUFFER_CUH
