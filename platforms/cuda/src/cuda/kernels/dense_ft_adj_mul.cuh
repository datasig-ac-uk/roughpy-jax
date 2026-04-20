#ifndef PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_ADJ_MUL_CUH
#define PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_ADJ_MUL_CUH


#include <cuda/std/algorithm>

#include <rpp/basis.hpp>


namespace rpy::jax::cuda {


namespace detail {
//TODO: Maybe later we can replace these with the block reduce functions from cooperative groups

template<typename Accum, unsigned WarpSize = 32>
RPP_FORCEINLINE
RPP_DEVICE Accum warp_reduce_sum(Accum elt) {
    for (unsigned i = WarpSize / 2; i > 0; i /= 2) {
        elt += __shfl_down_sync(0xFFFFFFFF, elt, i, WarpSize);
    }
    return elt;
}

template<typename Accum, unsigned WarpSize = 32>
RPP_FORCEINLINE
RPP_DEVICE Accum block_reduce_sum(Accum elt, Accum *const&scratch) {
    const auto warp_idx = threadIdx.x % WarpSize;
    const auto warp_lane = threadIdx.x / WarpSize;
    const auto warp_count = (blockDim.x + WarpSize - 1) / WarpSize;

    auto warp_elt = warp_reduce_sum<Accum, WarpSize>(elt);
    if (warp_idx == 0) {
        scratch[warp_lane] = warp_elt;
    }
    __syncthreads();

    Accum result{0};
    if (warp_lane == 0) {
        if (warp_idx < warp_count) {
            result = scratch[warp_idx];
        }

        result = warp_reduce_sum<Accum, WarpSize>(result);
    }
    return result;
}

template<typename Accum, typename Scalar, typename Index>
RPP_FORCEINLINE
RPP_DEVICE void ft_adj_mul_level_0_reduce(
    Scalar *const&out,
    Scalar const *const&op,
    Scalar const *const&arg,
    Index const &begin,
    Index const &end,
    Accum *scratch
) {
    Accum elt{0};
    for (Index idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
        Accum op_val{op[idx]};
        Accum arg_val{arg[idx]};

        elt += op_val * arg_val;
    }

    const auto block_sum = block_reduce_sum(elt, scratch);
    if (threadIdx.x == 0) {
        out[0] = static_cast<Scalar>(block_sum);
    }
}
} // namespace detail

template<typename ScalarTag>
__global__ void dense_ft_adj_lmul(
    typename ScalarTag::Scalar *__restrict__ out,
    typename ScalarTag::Scalar const *__restrict__ op,
    const typename rpp::StandardTensorBasis::Index op_stride,
    typename ScalarTag::Scalar const *__restrict__ arg,
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

    // ReSharper disable once CppTooWideScope
    __shared__ Accum dense_ft_adj_lmul_scratch[8];

    const auto &db = basis.degree_begin;
    // ReSharper disable once CppTooWideScope
    constexpr Degree cooperative_levels = 1;

    for (Index tensor_idx = blockIdx.x; tensor_idx < n_tensors; tensor_idx += gridDim.x) {
        const auto tensor_size = basis.size();
        auto *this_out = out + tensor_idx * tensor_size;
        auto const *this_op = op + tensor_idx * op_stride;
        auto const *this_arg = arg + tensor_idx * arg_stride;

        /*
        * The degree-zero term of the output is essentially the dot-product of the op
        * and arg. This needs a different strategy from the rest of the operation,
        * namely a block-cooperative reduction over the common degree-range of op
        * and arg. This needs a little shared memory.
        */
        detail::ft_adj_mul_level_0_reduce(
            this_out, this_op, this_arg,
            db[::cuda::std::max(op_min_degree, arg_min_degree)],
            db[::cuda::std::min(op_max_degree, arg_max_degree)],
            dense_ft_adj_lmul_scratch
        );

        /*
         * The degree 1, and possibly higher degree, terms are similar. Here, the shape
         * of the computation is "long and thin", with relatively few output values computed from
         * a large number of op/arg values. The obvious cut-off is when the number of outputs
         * to be computed becomes larger than the size of the block, at which point the other
         * strategy becomes a productive use of resources (see below). The challenge here is
         * either dealing with strided access for the arg values, or dealing with the block-wide
         * asynchronous accumulations.
         *
         * If there are performance gains to be had, it will be in optimising this (these)
         * computations. In some very specific situations, vectorised accesses might help.
         *
         * This is left as a loop to leave open the possibility of refactoring for more
         * levels later, and to separate the blocks.
         */
        auto stride = static_cast<Index>(basis.width);
        for (Degree coop_deg=1; coop_deg <= cooperative_levels; ++coop_deg) {
            const auto op_min_deg = ::cuda::std::max(op_min_degree, arg_max_degree - coop_deg);
            const auto op_max_deg = ::cuda::std::min(op_max_degree, arg_max_degree - coop_deg);

            for (Index elt_idx=db[coop_deg]; elt_idx < db[coop_deg + 1]; ++elt_idx) {
                Accum elt { 0 };
                for (Index op_idx=db[op_min_deg] + threadIdx.x; op_idx < db[op_max_deg + 1]; op_idx += blockDim.x) {
                    Accum op_val { this_op[op_idx] };
                    Accum arg_val { this_arg[op_idx * stride + elt_idx] };

                    elt += op_val * arg_val;
                }

                const auto block_sum = detail::block_reduce_sum(elt, dense_ft_adj_lmul_scratch);
                if (threadIdx.x == 0) {
                    this_out[elt_idx] = static_cast<Scalar>(block_sum);
                }
            }
            stride *= basis.width;
        }


        for (Index elt_idx = db[cooperative_levels+1] +  threadIdx.x;
            elt_idx < tensor_size;
            elt_idx += blockDim.x) {
            const auto degree = basis.degree(elt_idx);

            /*
             * We have to loop over all the prefix words afforded by op. The coefficients
             * of the op word u and the argument word uw contribute to the output word w.
             * The degree of the argument word is deg(u) + deg(w), so the range of u that
             * should be considered is between op_min_degree and arg_max_degree - degree.
             */
            const auto op_min_deg = ::cuda::std::max( arg_min_degree - degree, op_min_degree);
            const auto op_max_deg = ::cuda::std::min(arg_max_degree - degree, op_max_degree);

            Accum elt{0};
            Index shift = stride;
            for (Degree d = cooperative_levels+1; d < op_min_deg; ++d) {
                shift *= basis.width;
            }

            for (auto op_idx = db[op_min_deg]; op_idx < db[op_max_deg + 1]; ++op_idx) {
                Accum op_val{this_op[op_idx]};
                Accum arg_val{this_arg[op_idx * shift + elt_idx]};
                elt += op_val * arg_val;
            }

            this_out[elt_idx] = static_cast<Scalar>(elt);
        }
    }
}


template<typename ScalarTag>
__global__ void dense_ft_adj_rmul(
    typename ScalarTag::Scalar *__restrict__ out,
    typename ScalarTag::Scalar const *__restrict__ op,
    const typename rpp::StandardTensorBasis::Index op_stride,
    typename ScalarTag::Scalar const *__restrict__ arg,
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

    // ReSharper disable once CppTooWideScope
    constexpr Degree cooperative_levels = 1;
    // ReSharper disable once CppTooWideScope
    __shared__ Accum dense_ft_adj_rmul_scratch[8];

    auto const& db = basis.degree_begin;

    for (Index tensor_idx=blockIdx.x; tensor_idx < n_tensors; tensor_idx += gridDim.x) {
        const auto tensor_size = basis.size();
        auto* this_out = out + tensor_idx * tensor_size;
        auto const* this_op = op + tensor_idx * op_stride;
        auto const* this_arg = arg + tensor_idx * arg_stride;

        /*
         * The degree-zero term of the output is essentially the dot-product of the op
         * and arg. This needs a different strategy from the rest of the operation,
         * namely a block-cooperative reduction over the common degree-range of op
         * and arg. This needs a little shared memory.
         */
        detail::ft_adj_mul_level_0_reduce(
            this_out, this_op, this_arg,
            db[::cuda::std::max(op_min_degree, arg_min_degree)],
            db[::cuda::std::min(op_max_degree, arg_max_degree)],
            dense_ft_adj_rmul_scratch
        );


        for (Degree coop_deg=1; coop_deg <= cooperative_levels; ++coop_deg) {
            const auto op_min_deg = ::cuda::std::max(op_min_degree, arg_max_degree - coop_deg);
            const auto op_max_deg = ::cuda::std::min(op_max_degree, arg_max_degree - coop_deg);

            for (Index elt_idx=db[coop_deg]; elt_idx < db[coop_deg + 1]; ++elt_idx) {
                Accum elt { 0 };

                Index stride = 1;
                for (Degree op_deg = op_min_deg; op_deg <= op_max_deg; ++op_deg) {
                    for (Index op_idx = db[op_deg] + threadIdx.x; op_idx < db[op_deg + 1]; op_idx += blockDim.x) {
                        Accum arg_val { this_arg[elt_idx * stride + op_idx]};
                        Accum op_val { this_op[op_idx] };
                        elt += arg_val * op_val;
                    }

                    stride *= basis.width;
                }

                const auto block_sum = detail::block_reduce_sum(elt, dense_ft_adj_rmul_scratch);
                if (threadIdx.x == 0) {
                    this_out[elt_idx] = static_cast<Scalar>(block_sum);
                }
            }
        }


        for (Index elt_idx = db[cooperative_levels+1] + threadIdx.x; elt_idx < tensor_size; elt_idx += blockDim.x) {
            const auto degree = basis.degree(elt_idx);
            const auto op_min_deg = ::cuda::std::max(arg_min_degree - degree, op_min_degree);
            const auto op_max_deg = ::cuda::std::min(arg_max_degree - degree, op_max_degree);

            Accum elt{0};
            Index arg_shift = 1;

            for (Degree op_deg = op_min_deg; op_deg <= op_max_deg; ++op_deg) {
                for (Index op_idx = db[op_deg]; op_idx < db[op_deg + 1]; ++op_idx) {
                    Accum arg_val { this_arg[elt_idx * arg_shift + op_idx] };
                    Accum op_val { this_op[op_idx] };
                    elt += arg_val * op_val;
                }

                arg_shift *= basis.width;
            }

            this_out[elt_idx] = static_cast<Scalar>(elt);
        }
    }

}


} // namespace rpy::jax::cuda


#endif //PLATFORMS_CUDA_SRC_CUDA_KERNELS_DENSE_FT_ADJ_MUL_CUH
