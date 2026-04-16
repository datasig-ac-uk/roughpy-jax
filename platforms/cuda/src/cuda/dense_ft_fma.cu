#include "dense_ft_fma.cuh"

#include <algorithm>
#include <functional>

#include "low_precision.cuh"


using namespace rpy::jax::cuda;

namespace {

using TensorBasis = rpp::StandardTensorBasis;
using Degree = typename TensorBasis::Degree;
using Index = typename TensorBasis::Index;

struct DenseFTFmaStaticArgs {
    rpp::StandardTensorBasis basis;
    int32_t a_max_degree;
    int32_t b_max_degree;
    int32_t c_max_degree;

    int32_t b_min_degree = 0;
    int32_t c_min_degree = 0;

    bool zero_out_a = false;
};


template <typename ScalarTag>
__global__ void dense_ft_fma_general_kernel(
    typename ScalarTag::Scalar* __restrict__ out,
    typename ScalarTag::Scalar const* __restrict__ a,
    typename ScalarTag::Scalar const* __restrict__ b,
    typename ScalarTag::Scalar const* __restrict__ c,
    typename ScalarTag::Scalar alpha,
    typename ScalarTag::Scalar beta,
    rpp::StandardTensorBasis basis,
    Degree a_max_deg, Degree b_max_deg, Degree c_max_deg,
    Degree b_min_deg, Degree c_min_deg,
    Index n_tensors
    ) {
    using Accum = typename ScalarTag::Accum;

    Accum alpha0 { alpha };
    Accum beta0 { beta };
    auto const& db = basis.degree_begin;

    for (Index tensor_idx=blockIdx.x; tensor_idx<n_tensors; tensor_idx+=gridDim.x) {
        const auto tensor_size = basis.size();
        auto* this_out = out + tensor_idx * tensor_size;
        auto const* this_a = (a == nullptr ? out : a) + tensor_idx * tensor_size;
        auto const* this_b = b + tensor_idx * tensor_size;
        auto const* this_c = c + tensor_idx * tensor_size;

        for (Index elt_idx=threadIdx.x; elt_idx < tensor_size; elt_idx+=blockDim.x) {
            const auto degree = basis.degree(elt_idx);

            const auto lhs_max_deg = std::min<Degree>(degree - c_min_deg, b_max_deg);
            const auto lhs_min_deg = std::max<Degree>(b_min_deg, degree - c_max_deg);
            const auto out_idx = elt_idx - db[degree];

            Accum accum { 0 };
            for (Degree lhs_deg=lhs_min_deg; lhs_deg<=lhs_max_deg; lhs_deg++) {
                const auto rhs_deg = degree - lhs_deg;

                const auto splitter = db[rhs_deg+1] - db[rhs_deg];
                const auto lhs_idx = out_idx / splitter;
                const auto rhs_idx = out_idx % splitter;

                Accum lhs_elt { this_b[db[lhs_deg] + lhs_idx] };
                Accum rhs_elt { this_c[db[rhs_deg] + rhs_idx] };

                accum += lhs_elt * rhs_elt;
            }

            accum *= beta0;
            if (degree <= a_max_deg) {
                accum += alpha0 * a[elt_idx];
            }

            this_out[elt_idx] = Scalar(accum);
        }
    }
}


template <xla::ffi::DataType DType>
inline xla::ffi::Error dense_ft_fma_general_4_arg(
    xla::ffi::Result<xla::ffi::AnyBuffer> out,
    xla::ffi::AnyBuffer a,
    xla::ffi::AnyBuffer b,
    xla::ffi::AnyBuffer c,
    DenseFTFmaStaticArgs&& static_args,
    cudaStream_t stream
    ) {
    using Tag = ScalarTag<DType>;
    using Native = xla::ffi::NativeType<DType>;

    const auto tensor_size = static_args.basis.size();

    unsigned block = 32;
    if (tensor_size >= 64) {
        block = 64;
    } else if (tensor_size >= 128) {
        block = 128;
    } else {
        block = 256;
    }

    const auto out_shape = out->dimensions();
    const auto n_tensors = std::accumulate(
        out_shape.begin(), out_shape.end(), 1LL, std::multiplies<> {});


    const auto grid = static_cast<unsigned>(std::min(n_tensors, 65536LL));

    typename Tag::Scalar alpha { 1.0f };
    typename Tag::Scalar beta { 1.0f };

    const size_t db_bytes = sizeof(Index) * static_args.basis.size();

    Index* d_degree_begin;
    auto ret = cudaMallocAsync(&d_degree_begin, db_bytes, stream);
    if (ret != cudaSuccess) {
        return {xla::ffi::ErrorCode::kInternal, cudaGetErrorString(ret)};
    }

    ret = cudaMemcpyAsync(d_degree_begin, static_args.basis.degree_begin, db_bytes, cudaMemcpyHostToDevice, stream);
    if (ret != cudaSuccess) {
        cudaFreeAsync(d_degree_begin, stream);
        return {xla::ffi::ErrorCode::kInternal, cudaGetErrorString(ret)};
    }

    rpp::StandardTensorBasis d_basis { static_args.basis.width, static_args.basis.depth, d_degree_begin};

    dense_ft_fma_general_kernel<Tag><<<grid, block, 0, stream>>>(
        out->typed_data<Native>,
        a.typed_data<Native>,
        b.typed_data<Native>,
         c.typed_data<Native>,
         alpha, beta,
         d_basis,
         static_args.a_max_degree,
         static_args.b_max_degree,
         static_args.c_max_degree,
         static_args.b_min_degree,
         static_args.c_min_degree
        );

    cudaFreeAsync(d_degree_begin, stream);
    if ((ret = cudaGetLastError()) != cudaSuccess) {
        return { xla::ffi::ErrorCode::kInternal, cudaGetErrorString(ret)};
    }

    return xla::ffi::Error::Success();
}



}




