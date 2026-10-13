#ifndef ROUGHPY_JAX_XLA_HELPERS_STATIC_ARGS_HPP
#define ROUGHPY_JAX_XLA_HELPERS_STATIC_ARGS_HPP

#include <cstdint>
#include <rpp/basis/tensor_basis.hpp>

namespace rpy::jax {

using HostTensorBasis = rpp::basis::StandardTensorBasis;

// Shared internal operation contracts. CPU and CUDA use these exact types;
// execution resources such as CUDA streams are passed separately.

struct DenseFTAdjLMulStaticArgs {
    HostTensorBasis basis;
    int32_t op_max_degree;
    int32_t arg_max_degree;
};

struct DenseFTAdjRMulStaticArgs {
    HostTensorBasis basis;
    int32_t op_max_degree;
    int32_t arg_max_degree;
};

struct DenseFTAntipodeStaticArgs {
    HostTensorBasis basis;
    int32_t arg_max_degree;
    bool no_sign;
};

struct DenseFTExpStaticArgs {
    HostTensorBasis basis;
    int32_t arg_max_degree;
};

struct DenseFTFmaStaticArgs {
    HostTensorBasis basis;
    int32_t a_max_degree;
    int32_t b_max_degree;
    int32_t c_max_degree;
    int32_t b_min_degree = 0;
    int32_t c_min_degree = 0;
};

struct DenseFTFMExpStaticArgs {
    HostTensorBasis basis;
    int32_t mul_max_degree;
    int32_t exp_max_degree;
    int32_t mul_min_degree;
    int32_t exp_min_degree;
};

struct DenseFTLogStaticArgs {
    HostTensorBasis basis;
    int32_t arg_max_degree;
};

struct DenseFTMulStaticArgs {
    HostTensorBasis basis;
    int32_t lhs_max_degree;
    int32_t rhs_max_degree;
    int32_t lhs_min_degree = 0;
    int32_t rhs_min_degree = 0;
};

struct DenseSTAdjMulStaticArgs {
    HostTensorBasis basis;
    int32_t op_max_degree;
    int32_t arg_max_degree;
};

struct DenseSTFmaStaticArgs {
    HostTensorBasis basis;
    int32_t a_max_degree;
    int32_t b_max_degree;
    int32_t c_max_degree;
    int32_t b_min_degree = 0;
    int32_t c_min_degree = 0;
};

struct DenseSTMulStaticArgs {
    HostTensorBasis basis;
    int32_t lhs_max_degree;
    int32_t rhs_max_degree;
    int32_t lhs_min_degree = 0;
    int32_t rhs_min_degree = 0;
};

struct DenseTensorPairingStaticArgs {
    HostTensorBasis basis;
    int32_t fun_max_degree;
    int32_t arg_max_degree;
    int32_t fun_min_degree;
    int32_t arg_min_degree;
};

} // namespace rpy::jax

#endif // ROUGHPY_JAX_XLA_HELPERS_STATIC_ARGS_HPP
