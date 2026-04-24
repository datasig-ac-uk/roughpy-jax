#ifndef PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_ANTIPODE_CUH
#define PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_ANTIPODE_CUH

#include <cuda/std/algorithm>
#include <rpp/basis.hpp>

namespace rpy::jax::cuda {

template <typename Index, typename Degree>
RPP_HOST_DEVICE constexpr Index reverse_index(Index arg, Degree width, Degree degree) noexcept {
    Index result = 0;
    for (; degree > 0; --degree) {
        result *= width;
        result += arg % width;
        arg /= width;
    }
    return result;
}


template <typename ScalarTag, bool Sign>
__global__ void dense_ft_antipode_general_kernel(
    typename ScalarTag::Scalar* __restrict__ out,
    typename ScalarTag::Scalar const* arg,
    const rpp::StandardTensorBasis::Index arg_stride,
    const rpp::StandardTensorBasis basis,
    const typename rpp::StandardTensorBasis::Degree max_degree,
    const typename rpp::StandardTensorBasis::Index n_tensors,
    ) {
    using Scalar = typename ScalarTag::Scalar;
    using Accum = typename ScalarTag::Accum;
    using Index = typename rpp::StandardTensorBasis::Index;
    using Degree = typename rpp::StandardTensorBasis::Degree;

    for (Index tensor_idx=blockIdx.x; tensor_idx < n_tensors; tensor_idx += gridDim.x) {
        const auto tensor_size = basis.size();
        auto* this_out = out + tensor_idx * basis.size();
        auto const* this_arg = arg + tensor_idx * arg_stride;

        for (Index elt_idx = threadIdx.x; elt_idx < tensor_size; elt_idx += blockDim.x) {
            auto degree = basis.degree(elt_idx);

            Accum elt { 0.f };
            if (degree <= max_degree) {
                elt = Accum{this_arg[elt_idx]};
            }

            auto out_idx = reverse_index(elt_idx, basis.width, degree);

            if constexpr (Sign) {
                this_out[elt_idx] = Scalar{-elt};
            } else {
                this_out[elt_idx] = Scalar{elt};
            }
        }
    }

}



}// namespace rpy::jax::cuda



#endif //PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_ANTIPODE_CUH
