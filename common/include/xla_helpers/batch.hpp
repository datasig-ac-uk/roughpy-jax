#ifndef ROUGHPY_JAX_XLA_HELPERS_BATCH_HPP
#define ROUGHPY_JAX_XLA_HELPERS_BATCH_HPP

#include <limits>

#include <rpp/architecture.hpp>
#include <rpp/basis/lie_basis.hpp>
#include <rpp/basis/tensor_basis.hpp>
#include <rpp/views/batch.hpp>
#include <rpp/views/dense_lie_view.hpp>
#include <rpp/views/dense_tensor_view.hpp>
#include <rpp/views/scalar_view.hpp>


#include "xla_common.hpp"

namespace rpy::jax {

template <typename T, typename Arch>
using Ptr = typename Arch::template Ptr<T>;

template <typename T, typename Arch>
using TensorView =
    rpp::DenseTensorView<Ptr<T, Arch>, rpp::basis::TensorBasis<Arch>>;

template <typename T, typename Arch>
using LieView = rpp::DenseLieView<Ptr<T, Arch>, rpp::basis::LieBasis<Arch>>;

template <typename T, typename Arch>
using ScalarView = rpp::ScalarView<Ptr<T, Arch>>;

template <typename T, typename Arch>
using TensorBatch =
    rpp::Batch<TensorView<T, Arch>,
               rpp::layouts::StrideLayout<typename Arch::Index>>;

template <typename T, typename Arch>
using LieBatch = rpp::Batch<LieView<T, Arch>,
                            rpp::layouts::StrideLayout<typename Arch::Index>>;

template <typename T, typename Arch>
using ScalarBatch =
    rpp::Batch<ScalarView<T, Arch>, rpp::layouts::NoStrideLayout>;

// Batch counts are computed with 64-bit host integers, but operations
// use the selected architecture's index type.
template <typename Arch, typename Size>
ffi::Error check_batch_size(Size batch_size) noexcept {
    using Index = typename Arch::Index;

    constexpr auto max_batch_size = std::numeric_limits<Index>::max();
    if (batch_size > max_batch_size) {
        return ffi::Error::InvalidArgument(
            "batch size exceeds the architecture index range");
    }

    return ffi::Error::Success();
}

template <typename T, typename Arch>
auto make_tensor_batch(xla::ffi::AnyBuffer buffer,
                       typename Arch::Degree min_degree = 0,
                       typename Arch::Degree max_degree = -1) noexcept {
    return TensorBatch<const T, Arch>{
        Ptr<const T, Arch>(static_cast<const T*>(buffer.untyped_data())),
        rpp::layouts::StrideLayout<typename Arch::Index>{
            static_cast<typename Arch::Index>(buffer.dimensions().back())},
        {rpp::basis::TensorBasisTag{}, min_degree, max_degree}};
}

template <typename T, typename Arch>
auto make_tensor_batch(xla::ffi::Result<xla::ffi::AnyBuffer> buffer,
                       typename Arch::Degree min_degree = 0,
                       typename Arch::Degree max_degree = -1) noexcept {
    return TensorBatch<T, Arch>{
        Ptr<T, Arch>(static_cast<T*>(buffer->untyped_data())),
        rpp::layouts::StrideLayout<typename Arch::Index>{
            static_cast<typename Arch::Index>(buffer->dimensions().back())},
        {rpp::basis::TensorBasisTag{}, min_degree, max_degree}};
}


template <typename T, typename Arch>
auto make_lie_batch(xla::ffi::AnyBuffer buffer,
                    typename Arch::Degree min_degree = 0,
                    typename Arch::Degree max_degree = -1) noexcept {
    return LieBatch<const T, Arch>{
        Ptr<const T, Arch>(static_cast<const T*>(buffer.untyped_data())),
        rpp::layouts::StrideLayout<typename Arch::Index>{
            static_cast<typename Arch::Index>(buffer.dimensions().back())},
        {rpp::basis::LieBasisTag{}, min_degree, max_degree}};
}

template <typename T, typename Arch>
auto make_lie_batch(xla::ffi::Result<xla::ffi::AnyBuffer> buffer,
                    typename Arch::Degree min_degree = 0,
                    typename Arch::Degree max_degree = -1) noexcept {
    return LieBatch<T, Arch>{
        Ptr<T, Arch>(static_cast<T*>(buffer->untyped_data())),
        rpp::layouts::StrideLayout<typename Arch::Index>{
            static_cast<typename Arch::Index>(buffer->dimensions().back())},
        {rpp::basis::LieBasisTag{}, min_degree, max_degree}};
}

template <typename T, typename Arch>
auto make_scalar_batch(xla::ffi::AnyBuffer buffer) noexcept {
    return ScalarBatch<const T, Arch>{
        Ptr<const T, Arch>(static_cast<T const*>(buffer.untyped_data())),
        rpp::layouts::NoStrideLayout{},
        {}};
}

template <typename T, typename Arch>
auto make_scalar_batch(ffi::Result<ffi::AnyBuffer> buffer) noexcept {
    return ScalarBatch<T, Arch>{
        Ptr<T, Arch>(static_cast<T*>(buffer->untyped_data())),
        rpp::layouts::NoStrideLayout{},
        {}};
}

} // namespace rpy::jax


#endif // ROUGHPY_JAX_XLA_HELPERS_BATCH_HPP
