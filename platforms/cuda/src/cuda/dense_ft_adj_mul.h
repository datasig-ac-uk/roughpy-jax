#ifndef PLATFORMS_CUDA_SRC_CUDA_DENSE_FT_ADJ_MUL_H
#define PLATFORMS_CUDA_SRC_CUDA_DENSE_FT_ADJ_MUL_H

#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

XLA_FFI_Error* cuda_dense_ft_adj_lmul(XLA_FFI_CallFrame*);
XLA_FFI_Error* cuda_dense_ft_adj_rmul(XLA_FFI_CallFrame*);

#ifdef __cplusplus
}
#endif

#endif // PLATFORMS_CUDA_SRC_CUDA_DENSE_FT_ADJ_MUL_H
