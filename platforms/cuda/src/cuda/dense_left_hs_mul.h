#ifndef PLATFORMS_CUDA_SRC_CUDA_DENSE_LEFT_HS_MUL_H
#define PLATFORMS_CUDA_SRC_CUDA_DENSE_LEFT_HS_MUL_H

#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

XLA_FFI_Error* cuda_dense_left_hs_mul(XLA_FFI_CallFrame*);

#ifdef __cplusplus
}
#endif

#endif // PLATFORMS_CUDA_SRC_CUDA_DENSE_LEFT_HS_MUL_H
