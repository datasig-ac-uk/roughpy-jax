#ifndef PLATFORMS_CUDA_SRC_CUDA_BASIS_SUPPORT_CUH
#define PLATFORMS_CUDA_SRC_CUDA_BASIS_SUPPORT_CUH

#include <vector>

#include <rpp/basis.hpp>
#include <rpp/integer_maths.hpp>

#include "xla_headers.hpp"


namespace rpy::jax::cuda {

namespace internal {

template <typename Dst, typename Src>
RPP_FORCEINLINE cudaError_t convert_copy_to_device(Dst* d_dst, Src const* h_src, size_t n_elts, cudaStream_t stream) {
    if constexpr (std::is_same_v<Dst, Src>) {
        return cudaMemcpyAsync(d_dst, h_src, sizeof(Dst)*n_elts, cudaMemcpyHostToDevice, stream);
    } else {
        std::vector<Dst> tmp(h_src, h_src + n_elts);
        return cudaMemcpyAsync(d_dst, tmp.data(), sizeof(Dst)*n_elts, cudaMemcpyHostToDevice, stream);
    }
}

} // namespace internal

template<typename Degree_, typename Index_>
struct TensorBasisConverter {
    using Basis = rpp::TensorBasis<Degree_, Index_>;
    Basis d_basis;
    cudaStream_t stream;
    cudaError_t error;

    template<typename TensorBasis>
    explicit TensorBasisConverter(TensorBasis const &h_basis, cudaStream_t stream) noexcept;

    ~TensorBasisConverter() noexcept;

    constexpr bool operator!() const noexcept {
        return error != cudaSuccess;
    }
};


template<typename Degree_, typename Index_>
template<typename TensorBasis>
TensorBasisConverter<Degree_, Index_>::TensorBasisConverter(TensorBasis const &h_basis, cudaStream_t stream_) noexcept
    : d_basis{h_basis.width, h_basis.depth, nullptr}, stream(stream_), error(cudaSuccess) {
    const size_t db_bytes = sizeof(Index_) * (h_basis.depth + 2);

    Index_* d_db;
    error = cudaMallocAsync(&d_db, db_bytes, stream);
    if (error == cudaSuccess) {
        error = internal::convert_copy_to_device(d_db, h_basis.degree_begin, h_basis.depth + 2,
                                                 stream);
        d_basis.degree_begin = d_db;
    }
}

template<typename Degree_, typename Index_>
TensorBasisConverter<Degree_, Index_>::~TensorBasisConverter() noexcept {
    if (d_basis.degree_begin != nullptr) {
        cudaFreeAsync(const_cast<Index_*>(d_basis.degree_begin), stream);
    }
}



template <typename Degree_, typename Index_>
struct LieBasisConverter {
    using Basis = rpp::LieBasis<Degree_, Index_>;

    Basis d_basis;
    cudaStream_t stream;
    cudaError_t error;

    template <typename LieBasis>
    explicit LieBasisConverter(LieBasis const& h_basis, cudaStream_t stream) noexcept;

    ~LieBasisConverter() noexcept;

};


template<typename Degree_, typename Index_>
template<typename LieBasis>
LieBasisConverter<Degree_, Index_>::LieBasisConverter(LieBasis const &h_basis, cudaStream_t stream) noexcept
    : d_basis{h_basis.width, h_basis.depth, nullptr, nullptr},  stream(stream), error(cudaSuccess)
{
    const auto db_size = h_basis.depth + 2;
    const auto basis_size = h_basis.true_size();

    const auto data_offset = rpp::align_up(db_size, alignof(Index_));

    const auto bytes = sizeof(Index_) * (data_offset + basis_size);

    Index_* d_ptr;
    error = cudaMallocAsync(&d_ptr, bytes, stream);
    if (error != cudaSuccess) { return; }
    auto* d_data = d_ptr + data_offset;

    d_basis.degree_begin = d_ptr;
    d_basis.data = d_data;

    error = internal::convert_copy_to_device(d_ptr, h_basis.degree_begin, db_size, stream);
    if (error != cudaSuccess) { return; }

    error = internal::convert_copy_to_device(d_data, h_basis.data, basis_size, stream);
}

template<typename Degree_, typename Index_>
LieBasisConverter<Degree_, Index_>::~LieBasisConverter() noexcept {
    if (d_basis.degree_begin != nullptr) {
        cudaFreeAsync(const_cast<Index_*>(d_basis.degree_begin), stream);
    }
}
} // namespace rpy::jax::cuda

#endif //PLATFORMS_CUDA_SRC_CUDA_BASIS_SUPPORT_CUH
