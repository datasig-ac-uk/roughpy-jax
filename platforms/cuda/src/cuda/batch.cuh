#ifndef PLATFORMS_CUDA_SRC_CUDA_BATCH_CUH
#define PLATFORMS_CUDA_SRC_CUDA_BATCH_CUH

#include <limits>

#include <rpp/gpu/architecture.hpp>
#include <rpp/views/batch.hpp>
#include <rpp/views/dense_tensor_view.hpp>
#include <rpp/views/dense_lie_view.hpp>
#include <rpp/views/scalar_view.hpp>


#include "xla_headers.hpp"

namespace rpy::jax::cuda {
template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
using GMemPtr = typename Arch::template GMemPtr<T>;

template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
using TensorView = rpp::DenseTensorView<
    GMemPtr<T, Arch>,
    rpp::basis::TensorBasis<Arch>
>;

template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
using LieView = rpp::DenseLieView<
    GMemPtr<T, Arch>,
    rpp::basis::LieBasis<Arch>
>;

template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
using ScalarView = rpp::ScalarView<GMemPtr<T, Arch>>;

template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
using TensorBatch = rpp::Batch<
    TensorView<T, Arch>,
    rpp::layouts::StrideLayout<typename Arch::Index>
>;

template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
using LieBatch = rpp::Batch<
    LieView<T, Arch>,
    rpp::layouts::StrideLayout<typename Arch::Index>
>;

template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
using ScalarBatch = rpp::Batch<ScalarView<T, Arch>,
    rpp::layouts::NoStrideLayout
>;

// Batch counts are computed with 64-bit host integers, but the GPU operation
// launch narrows them to the architecture's index type.
template<typename Arch=rpp::gpu::arch::DefaultArchitecture, typename Size>
ffi::Error check_batch_size(Size batch_size) noexcept {
    using Index = typename Arch::Index;

    constexpr auto max_batch_size = std::numeric_limits<Index>::max();
    if (batch_size > max_batch_size) {
        return ffi::Error::InvalidArgument(
            "batch size exceeds the CUDA kernel index range");
    }

    return ffi::Error::Success();
}

template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
auto make_tensor_batch(
    xla::ffi::AnyBuffer buffer,
    typename Arch::Degree min_degree = 0,
    typename Arch::Degree max_degree = -1
) noexcept {
    return TensorBatch<const T, Arch>{
        GMemPtr<const T, Arch>(static_cast<const T *>(buffer.untyped_data())),
        rpp::layouts::StrideLayout<typename Arch::Index>{
            static_cast<typename Arch::Index>(buffer.dimensions().back())
        },
        {rpp::basis::TensorBasisTag{}, min_degree, max_degree}
    };
}

template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
auto make_tensor_batch(
    xla::ffi::Result<xla::ffi::AnyBuffer> buffer,
    typename Arch::Degree min_degree = 0,
    typename Arch::Degree max_degree = -1
) noexcept {
    return TensorBatch<T, Arch>{
        GMemPtr<T, Arch>(static_cast<T *>(buffer->untyped_data())),
        rpp::layouts::StrideLayout<typename Arch::Index>{
            static_cast<typename Arch::Index>(buffer->dimensions().back())
        },
        {rpp::basis::TensorBasisTag{}, min_degree, max_degree}
    };
}


template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
auto make_lie_batch(
    xla::ffi::AnyBuffer buffer,
    typename Arch::Degree min_degree = 0,
    typename Arch::Degree max_degree = -1
) noexcept {
    return LieBatch<const T, Arch>{
        GMemPtr<const T, Arch>(static_cast<const T *>(buffer.untyped_data())),
        rpp::layouts::StrideLayout<typename Arch::Index>{
            static_cast<typename Arch::Index>(buffer.dimensions().back())
        },
        {rpp::basis::LieBasisTag{}, min_degree, max_degree}
    };
}

template<typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
auto make_lie_batch(
    xla::ffi::Result<xla::ffi::AnyBuffer> buffer,
    typename Arch::Degree min_degree = 0,
    typename Arch::Degree max_degree = -1
) noexcept {
    return LieBatch<T, Arch>{
        GMemPtr<T, Arch>(static_cast<T *>(buffer->untyped_data())),
        rpp::layouts::StrideLayout<typename Arch::Index>{
            static_cast<typename Arch::Index>(buffer->dimensions().back())
        },
        {rpp::basis::LieBasisTag{}, min_degree, max_degree}
    };
}

template <typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
auto make_scalar_batch(xla::ffi::AnyBuffer buffer) noexcept {
    return ScalarBatch<T, Arch>{
        GMemPtr<const T, Arch>(static_cast<T const*>(buffer.untyped_data())),
        rpp::layouts::NoStrideLayout{},
        {}
    };
}

template <typename T, typename Arch=rpp::gpu::arch::DefaultArchitecture>
auto make_scalar_batch(ffi::Result<ffi::AnyBuffer> buffer) noexcept {
    return ScalarBatch<T, Arch>{
        GMemPtr<T, Arch>(static_cast<T *>(buffer->untyped_data())),
        rpp::layouts::NoStrideLayout{},
        {}
    };
}

} // namespace rpy::jax::cuda


#endif //PLATFORMS_CUDA_SRC_CUDA_BATCH_CUH
