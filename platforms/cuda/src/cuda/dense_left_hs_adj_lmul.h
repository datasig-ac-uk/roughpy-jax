#ifndef PLATFORMS_CUDA_SRC_CUDA_DENSE_LEFT_HS_ADJ_LMUL_H
#define PLATFORMS_CUDA_SRC_CUDA_DENSE_LEFT_HS_ADJ_LMUL_H

#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

XLA_FFI_Error* cuda_dense_left_hs_adj_lmul(XLA_FFI_CallFrame*);

#ifdef __cplusplus
}
#endif

#endif // PLATFORMS_CUDA_SRC_CUDA_DENSE_LEFT_HS_ADJ_LMUL_H
