#ifndef PLATFORMS_CUDA_SRC_CUDA_ERROR_HANDLING_CUH
#define PLATFORMS_CUDA_SRC_CUDA_ERROR_HANDLING_CUH

#include "xla_headers.hpp"

#include <cuda_runtime_api.h>

#include <string_view>


namespace rpy::jax::cuda {

xla::ffi::ErrorCode cuda_error_to_xla_error_code(cudaError_t error) noexcept;

xla::ffi::Error cuda_error_to_xla_error(cudaError_t error) noexcept;

xla::ffi::Error cuda_error_to_xla_error(cudaError_t error, std::string_view context) noexcept;

} // namespace rpy::jax::cuda

#endif // PLATFORMS_CUDA_SRC_CUDA_ERROR_HANDLING_CUH
