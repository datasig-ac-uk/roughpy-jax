#ifndef PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_FMA_CUH
#define PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_FMA_CUH


#include <cuda/std/algorithm>
#include <rpp/basis.hpp>



namespace rpy::jax::cuda {


template<typename ScalarTag>
__global__ void dense_ft_fma_general_kernel(
    typename ScalarTag::Scalar *__restrict__ out,
    typename ScalarTag::Scalar const *__restrict__ a,
    const typename rpp::StandardTensorBasis::Index a_stride,
    typename ScalarTag::Scalar const *__restrict__ b,
    const typename rpp::StandardTensorBasis::Index b_stride,
    typename ScalarTag::Scalar const *__restrict__ c,
    const typename rpp::StandardTensorBasis::Index c_stride,
    typename ScalarTag::Accum alpha,
    typename ScalarTag::Accum beta,
    const rpp::StandardTensorBasis basis,
    const typename rpp::StandardTensorBasis::Degree a_max_deg,
    const typename rpp::StandardTensorBasis::Degree b_max_deg,
    const typename rpp::StandardTensorBasis::Degree c_max_deg,
    const typename rpp::StandardTensorBasis::Degree b_min_deg,
    const typename rpp::StandardTensorBasis::Degree c_min_deg,
    const typename rpp::StandardTensorBasis::Index n_tensors
) {
    using Index = typename rpp::StandardTensorBasis::Index;
    using Degree = typename rpp::StandardTensorBasis::Degree;
    using Accum = typename ScalarTag::Accum;
    using Scalar = typename ScalarTag::Scalar;

    auto const &db = basis.degree_begin;

    for (Index tensor_idx = blockIdx.x; tensor_idx < n_tensors; tensor_idx += gridDim.x) {
        const auto tensor_size = basis.size();
        auto *this_out = out + tensor_idx * tensor_size;
        auto const *this_a = (a == nullptr ? out : a) + tensor_idx * a_stride;
        auto const *this_b = b + tensor_idx * b_stride;
        auto const *this_c = c + tensor_idx * c_stride;

        for (Index elt_idx = threadIdx.x; elt_idx < tensor_size; elt_idx += blockDim.x) {
            const auto degree = basis.degree(elt_idx);

            const auto lhs_max_deg = ::cuda::std::min<Degree>(degree - c_min_deg, b_max_deg);
            const auto lhs_min_deg = ::cuda::std::max<Degree>(b_min_deg, degree - c_max_deg);
            const auto out_idx = elt_idx - db[degree];

            Accum accum{0};
            for (Degree lhs_deg = lhs_min_deg; lhs_deg <= lhs_max_deg; lhs_deg++) {
                const auto rhs_deg = degree - lhs_deg;

                const auto splitter = db[rhs_deg + 1] - db[rhs_deg];
                const auto lhs_idx = out_idx / splitter;
                const auto rhs_idx = out_idx % splitter;

                Accum lhs_elt{this_b[db[lhs_deg] + lhs_idx]};
                Accum rhs_elt{this_c[db[rhs_deg] + rhs_idx]};

                accum += lhs_elt * rhs_elt;
            }

            accum *= beta;
            if (alpha != Accum{0} && degree <= a_max_deg) {
                accum += alpha * Accum{this_a[elt_idx]};
            }

            this_out[elt_idx] = Scalar(accum);
        }
    }
}

}



#endif //PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_FMA_CUH
