#ifndef PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_ADJ_MUL_CUH
#define PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_ADJ_MUL_CUH

#include <cuda/std/algorithm>
#include <rpp/basis.hpp>

namespace rpy::jax::cuda {

template <typename ScalarTag>
__global__ void dense_ft_adj_lmul(
    typename ScalarTag::Scalar* __restrict__ out,
    typename ScalarTag::Scalar const* __restrict__ op,
    const typename rpp::StandardTensorBasis::Index op_stride,
    typename ScalarTag::Scalar const* __restrict__ arg,
    const typename rpp::StandardTensorBasis::Index arg_stride,
    const rpp::StandardTensorBasis basis,
    const typename rpp::StandardTensorBasis::Degree op_max_degree,
    const typename rpp::StandardTensorBasis::Degree arg_max_degree,
    const typename rpp::StandardTensorBasis::Degree op_min_degree,
    const typename rpp::StandardTensorBasis::Degree arg_min_degree,
    const typename rpp::StandardTensorBasis::Index n_tensors
    ) {
    using Index = typename rpp::StandardTensorBasis::Index;
    using Degree = typename rpp::StandardTensorBasis::Degree;
    using Accum = typename ScalarTag::Accum;
    using Scalar = typename ScalarTag::Scalar;

    const auto& db = basis.degree_begin;

    const auto arg_max_deg= ::cuda::std::min(arg_max_degree - op_min_degree, basis.depth);
    const auto arg_min_deg= ::cuda::std::max(arg_min_degree - op_max_degree, 0);

    for (Index tensor_idx = blockIdx.x; tensor_idx < n_tensors; tensor_idx += gridDim.x) {
        const auto tensor_size = basis.size();
        auto* this_out = out + tensor_idx * tensor_size;
        auto const* this_op = op + tensor_idx * op_stride;
        auto const* this_arg = arg + tensor_idx * arg_stride;

        for (Index elt_idx = threadIdx.x; elt_idx < tensor_size; elt_idx += blockDim.x) {
            const auto degree = basis.degree(elt_idx);

            /*
             * We have to loop over all the prefix words afforded by op. The coefficients
             * of the op word u and the argument word uw contribute to the output word w.
             * The degree of the argument word is deg(u) + deg(w), so the range of u that
             * should be considered is between op_min_degree and arg_max_degree - degree.
             */

            const auto op_min_deg = ::cuda::std::max(op_min_deg, arg_min_deg - degree);
            const auto op_max_deg = ::cuda::std::min(arg_max_degree - degree, op_max_degree);

            Accum elt{0};
            Index shift = 1;
            for (Degree d=0; d<op_min_deg; ++d) {
                shift *= basis.width;
            }

            for (Degree op_deg=op_min_deg; op_deg < op_max_deg; ++op_deg) {




            }



        }
    }

}

} // namespace rpy::jax::cuda


#endif //PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_ADJ_MUL_CUH
