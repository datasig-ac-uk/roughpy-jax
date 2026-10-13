#ifndef ROUGHPY_JAX_CPU_DENSE_LEFT_HS_MUL_H
#define ROUGHPY_JAX_CPU_DENSE_LEFT_HS_MUL_H

#include <xla/ffi/api/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

XLA_FFI_Error* cpu_dense_left_hs_mul(XLA_FFI_CallFrame*);

#ifdef __cplusplus
}
#endif

#endif // ROUGHPY_JAX_CPU_DENSE_LEFT_HS_MUL_H
