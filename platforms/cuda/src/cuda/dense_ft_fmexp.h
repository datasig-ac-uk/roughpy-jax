#ifndef PLATFORMS_CUDA_SRC_CUDA_DENSE_FT_FMEXP_H
#define PLATFORMS_CUDA_SRC_CUDA_DENSE_FT_FMEXP_H

#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

XLA_FFI_Error* cuda_dense_ft_fmexp(XLA_FFI_CallFrame*);

#ifdef __cplusplus
}
#endif

#endif // PLATFORMS_CUDA_SRC_CUDA_DENSE_FT_FMEXP_H
