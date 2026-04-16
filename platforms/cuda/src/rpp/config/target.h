#ifndef RPP_CONFIG_TARGET_H
#define RPP_CONFIG_TARGET_H

#include <rpp/config/compiler.h>

#if defined(__CUDACC__) && defined(__CUDA_ARCH__)
#define RPP_HOST __host__
#define RPP_DEVICE __device__
#define RPP_HOST_DEVICE __host__ __device__
#else
#define RPP_HOST
#define RPP_DEVICE
#define RPP_HOST_DEVICE
#endif


#endif //RPP_CONFIG_TARGET_H
