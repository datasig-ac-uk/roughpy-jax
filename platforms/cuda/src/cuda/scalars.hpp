#ifndef ROUGHPY_JAX_CUDA_SCALARS_HPP
#define ROUGHPY_JAX_CUDA_SCALARS_HPP

#include <type_traits>
#include <utility>

#include <rpp/config.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <xla_helpers/xla_common.hpp>

namespace rpy::jax {

// Specialisations of the shared tag must live in its declaring namespace.
template <typename AccumType>
struct ScalarTag<ffi::DataType::F16, AccumType> {
    static constexpr bool available = true;
    static constexpr auto tag = ffi::DataType::F16;
    using Scalar = __half;
    using Accum = std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::BF16, AccumType> {
    static constexpr bool available = true;
    static constexpr auto tag = ffi::DataType::BF16;
    using Scalar = __nv_bfloat16;
    using Accum = std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

// CUDA FP8 wrappers have storage mappings, but the kernels do not yet
// support their arithmetic. Keep them unavailable to scalar dispatch.
template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E4M3, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E4M3;
    using Scalar = __nv_fp8_e4m3;
    using Accum = std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E4M3FN, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E4M3FN;
    using Scalar = __nv_fp8_e4m3;
    using Accum = std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E4M3B11FNUZ, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E4M3B11FNUZ;
    using Scalar = __nv_fp8_e4m3;
    using Accum = std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E4M3FNUZ, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E4M3FNUZ;
    using Scalar = __nv_fp8_e4m3;
    using Accum = std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E5M2, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E5M2;
    using Scalar = __nv_fp8_e5m2;
    using Accum = std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

template <typename AccumType>
struct ScalarTag<ffi::DataType::F8E5M2FNUZ, AccumType> {
    static constexpr bool available = false;
    static constexpr auto tag = ffi::DataType::F8E5M2FNUZ;
    using Scalar = __nv_fp8_e5m2;
    using Accum = std::conditional_t<std::is_void_v<AccumType>, float, AccumType>;
};

namespace cuda {

template<typename T>
inline constexpr ffi::DataType native_dtype_v = ffi::DataType::INVALID;

template<>
inline constexpr ffi::DataType native_dtype_v<__half> = ffi::DataType::F16;

template<>
inline constexpr ffi::DataType native_dtype_v<__nv_bfloat16> = ffi::DataType::BF16;

template<>
inline constexpr ffi::DataType native_dtype_v<__nv_fp8_e4m3> = ffi::DataType::F8E4M3;

template<>
inline constexpr ffi::DataType native_dtype_v<__nv_fp8_e5m2> = ffi::DataType::F8E5M2;

template<>
inline constexpr ffi::DataType native_dtype_v<float> = ffi::DataType::F32;

template<>
inline constexpr ffi::DataType native_dtype_v<double> = ffi::DataType::F64;

class AnyScalar {
    void const *data_;
    ffi::DataType dtype_;

public:
    template<typename T>
    constexpr explicit AnyScalar(T const &scalar_val)
        : data_(static_cast<void const *>(&scalar_val)),
          dtype_(native_dtype_v<std::decay_t<T> >) {
        static_assert(native_dtype_v<std::decay_t<T> > != ffi::DataType::INVALID,
                      "unsupported scalar type");
    }

    constexpr explicit AnyScalar(void const *data, ffi::DataType dtype)
        : data_(data), dtype_(dtype) {
    }

    [[nodiscard]]
    constexpr ffi::DataType dtype() const noexcept { return dtype_; }

    template<typename T>
    [[nodiscard]]
    ffi::ErrorOr<std::add_const_t<T> *> ptr() const noexcept {
        if (native_dtype_v<std::decay_t<T> > != dtype_) [[unlikely]] {
            return {ffi::Error::InvalidArgument( "unexpected scalar decoding")};
        }
        return static_cast<std::add_const_t<T>*>(data_);
    }
};


inline ffi::ErrorOr<AnyScalar> make_scalar(ffi::AnyBuffer buffer) noexcept {
    if (buffer.element_count() != 1) {
        return {ffi::Error::InvalidArgument("expected a scalar, got numerous values")};
    }
    return AnyScalar(buffer.untyped_data(), buffer.element_type());
}


namespace internal {
template<typename T, typename S>
ffi::ErrorOr<T> map_cast_err(ffi::ErrorOr<S> val) noexcept {
    if (val.has_error()) [[unlikely]] { return {val.error()}; }
    return T(*val.value());
}
} // namespace internal

template<typename T>
ffi::ErrorOr<T> cast_scalar(AnyScalar const &value) noexcept {
    switch (value.dtype()) {
        case ffi::DataType::F16:
            return internal::map_cast_err<T>(value.ptr<__half>());
        case ffi::DataType::BF16:
            return internal::map_cast_err<T>(value.ptr<__nv_bfloat16>());
        case ffi::DataType::F8E4M3:
            return internal::map_cast_err<T>(value.ptr<__nv_fp8_e4m3>());
        case ffi::DataType::F8E4M3FN:
            return internal::map_cast_err<T>(value.ptr<__nv_fp8_e4m3>());
        case ffi::DataType::F8E4M3B11FNUZ:
            return internal::map_cast_err<T>(value.ptr<__nv_fp8_e4m3>());
        case ffi::DataType::F8E4M3FNUZ:
            return internal::map_cast_err<T>(value.ptr<__nv_fp8_e4m3>());
        case ffi::DataType::F8E5M2:
            return internal::map_cast_err<T>(value.ptr<__nv_fp8_e5m2>());
        case ffi::DataType::F8E5M2FNUZ:
            return internal::map_cast_err<T>(value.ptr<__nv_fp8_e5m2>());
        case ffi::DataType::F32:
            return internal::map_cast_err<T>(value.ptr<float>());
        case ffi::DataType::F64:
            return internal::map_cast_err<T>(value.ptr<double>());
        default:
            return {ffi::Error(ffi::ErrorCode::kInvalidArgument, "unsupported scalar type")};
    }
}


} // namespace cuda
} // namespace rpy::jax

#endif //ROUGHPY_JAX_CUDA_SCALARS_HPP
