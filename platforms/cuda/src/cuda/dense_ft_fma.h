#ifndef PLATFORMS_CUDA_SRC_CUDA_DENSE_FT_FMA_H
#define PLATFORMS_CUDA_SRC_CUDA_DENSE_FT_FMA_H

#include <xla/ffi/api/c_api.h>


#ifdef __cplusplus
extern "C" {
#endif

XLA_FFI_Error* cuda_dense_ft_fma(XLA_FFI_CallFrame*);
XLA_FFI_Error* cuda_dense_ft_mul(XLA_FFI_CallFrame*);


#ifdef __cplusplus
}
#endif



#endif //PLATFORMS_CUDA_SRC_CUDA_DENSE_FT_FMA_H
