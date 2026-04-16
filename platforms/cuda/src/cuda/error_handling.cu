#include "error_handling.cuh"

#include <string>


namespace rpy::jax::cuda {

xla::ffi::ErrorCode cuda_error_to_xla_error_code(cudaError_t error) noexcept
{
    switch (error) {
        case cudaSuccess:
            return xla::ffi::ErrorCode::kOk;

        case cudaErrorInvalidValue:
            [[fallthrough]];
        case cudaErrorInvalidPitchValue:
            [[fallthrough]];
        case cudaErrorInvalidSymbol:
            [[fallthrough]];
        case cudaErrorInvalidMemcpyDirection:
            [[fallthrough]];
        case cudaErrorInvalidDeviceFunction:
            [[fallthrough]];
        case cudaErrorInvalidConfiguration:
            [[fallthrough]];
        case cudaErrorInvalidDevice:
            [[fallthrough]];
        case cudaErrorInvalidResourceHandle:
            [[fallthrough]];
        case cudaErrorInvalidKernelImage:
            [[fallthrough]];
        case cudaErrorInvalidSource:
            [[fallthrough]];
        case cudaErrorInvalidPtx:
            [[fallthrough]];
        case cudaErrorInvalidGraphicsContext:
            [[fallthrough]];
        case cudaErrorUnsupportedLimit:
            [[fallthrough]];
        case cudaErrorDuplicateVariableName:
            [[fallthrough]];
        case cudaErrorDuplicateTextureName:
            [[fallthrough]];
        case cudaErrorDuplicateSurfaceName:
            [[fallthrough]];
        case cudaErrorDevicesUnavailable:
            [[fallthrough]];
        case cudaErrorNoDevice:
            return xla::ffi::ErrorCode::kInvalidArgument;

        case cudaErrorMemoryAllocation:
            [[fallthrough]];
        case cudaErrorLaunchOutOfResources:
            return xla::ffi::ErrorCode::kResourceExhausted;

        case cudaErrorInsufficientDriver:
            [[fallthrough]];
        case cudaErrorSystemNotReady:
            [[fallthrough]];
        case cudaErrorSystemDriverMismatch:
            [[fallthrough]];
        case cudaErrorCompatNotSupportedOnDevice:
            return xla::ffi::ErrorCode::kUnavailable;

        case cudaErrorNotSupported:
            return xla::ffi::ErrorCode::kUnimplemented;

        default:
            return xla::ffi::ErrorCode::kInternal;
    }
}

xla::ffi::Error cuda_error_to_xla_error(cudaError_t error) noexcept
{
    if (error == cudaSuccess) {
        return xla::ffi::Error::Success();
    }
    return {cuda_error_to_xla_error_code(error), cudaGetErrorString(error)};
}

xla::ffi::Error cuda_error_to_xla_error(cudaError_t error, std::string_view context) noexcept
{
    if (error == cudaSuccess) {
        return xla::ffi::Error::Success();
    }

    if (context.empty()) {
        return cuda_error_to_xla_error(error);
    }

    std::string message(context);
    message += ": ";
    message += cudaGetErrorString(error);
    return {cuda_error_to_xla_error_code(error), std::move(message)};
}

} // namespace rpy::jax::cuda
