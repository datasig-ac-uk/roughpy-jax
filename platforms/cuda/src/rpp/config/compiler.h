#ifndef RPP_CONFIG_COMPILER_H
#define RPP_CONFIG_COMPILER_H

#define RPP_VERSION_ENCODE(major, minor, patch) \
    (((major) * 1000000) + ((minor) * 1000) + (patch))

#if defined(__clang__)
#define RPP_COMPILER_CLANG 1
#define RPP_COMPILER_CLANG_VERSION \
    RPP_VERSION_ENCODE(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
#define RPP_COMPILER_CLANG 0
#define RPP_COMPILER_CLANG_VERSION 0
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define RPP_COMPILER_MSVC 1
#define RPP_COMPILER_MSVC_VERSION _MSC_VER
#else
#define RPP_COMPILER_MSVC 0
#define RPP_COMPILER_MSVC_VERSION 0
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define RPP_COMPILER_GCC 1
#define RPP_COMPILER_GCC_VERSION \
    RPP_VERSION_ENCODE(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#else
#define RPP_COMPILER_GCC 0
#define RPP_COMPILER_GCC_VERSION 0
#endif

#if defined(__CUDACC__)
#define RPP_COMPILER_CUDA 1
#define RPP_COMPILER_CUDA_VERSION \
    RPP_VERSION_ENCODE(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__)
#else
#define RPP_COMPILER_CUDA 0
#define RPP_COMPILER_CUDA_VERSION 0
#endif

#if defined(__CUDA_ARCH__)
#define RPP_CUDA_DEVICE_COMPILE 1
#define RPP_CUDA_ARCH __CUDA_ARCH__
#else
#define RPP_CUDA_DEVICE_COMPILE 0
#define RPP_CUDA_ARCH 0
#endif

#define RPP_COMPILER_VERSION \
    (RPP_COMPILER_CLANG ? RPP_COMPILER_CLANG_VERSION : \
    (RPP_COMPILER_MSVC ? RPP_COMPILER_MSVC_VERSION : \
    (RPP_COMPILER_GCC ? RPP_COMPILER_GCC_VERSION : 0)))

#if defined(_MSVC_LANG)
#define RPP_CXX_STANDARD _MSVC_LANG
#else
#define RPP_CXX_STANDARD __cplusplus
#endif

#define RPP_CXX_11 (RPP_CXX_STANDARD >= 201103L)
#define RPP_CXX_14 (RPP_CXX_STANDARD >= 201402L)
#define RPP_CXX_17 (RPP_CXX_STANDARD >= 201703L)
#define RPP_CXX_20 (RPP_CXX_STANDARD >= 202002L)
#define RPP_CXX_23 (RPP_CXX_STANDARD >= 202302L)

#if defined(__has_cpp_attribute)
#define RPP_HAS_CPP_ATTRIBUTE(attr) __has_cpp_attribute(attr)
#else
#define RPP_HAS_CPP_ATTRIBUTE(attr) 0
#endif

#if defined(__has_attribute)
#define RPP_HAS_ATTRIBUTE(attr) __has_attribute(attr)
#else
#define RPP_HAS_ATTRIBUTE(attr) 0
#endif

#if defined(__has_builtin)
#define RPP_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#define RPP_HAS_BUILTIN(builtin) 0
#endif

#if defined(__has_feature)
#define RPP_HAS_FEATURE(feature) __has_feature(feature)
#else
#define RPP_HAS_FEATURE(feature) 0
#endif

#if defined(__has_extension)
#define RPP_HAS_EXTENSION(extension) __has_extension(extension)
#else
#define RPP_HAS_EXTENSION(extension) 0
#endif

#if defined(__has_warning)
#define RPP_HAS_WARNING(warning) __has_warning(warning)
#else
#define RPP_HAS_WARNING(warning) 0
#endif

#if defined(__is_identifier)
#define RPP_IS_IDENTIFIER(token) __is_identifier(token)
#else
#define RPP_IS_IDENTIFIER(token) 1
#endif

#if RPP_HAS_CPP_ATTRIBUTE(nodiscard)
#define RPP_NODISCARD [[nodiscard]]
#else
#define RPP_NODISCARD
#endif

#if RPP_HAS_CPP_ATTRIBUTE(maybe_unused)
#define RPP_MAYBE_UNUSED [[maybe_unused]]
#elif RPP_HAS_ATTRIBUTE(unused) || RPP_COMPILER_GCC || RPP_COMPILER_CLANG
#define RPP_MAYBE_UNUSED __attribute__((unused))
#else
#define RPP_MAYBE_UNUSED
#endif

#if RPP_HAS_CPP_ATTRIBUTE(fallthrough)
#define RPP_FALLTHROUGH [[fallthrough]]
#elif RPP_HAS_ATTRIBUTE(fallthrough) || RPP_COMPILER_GCC || RPP_COMPILER_CLANG
#define RPP_FALLTHROUGH __attribute__((fallthrough))
#else
#define RPP_FALLTHROUGH ((void)0)
#endif

#if RPP_HAS_CPP_ATTRIBUTE(no_unique_address)
#define RPP_NO_UNIQUE_ADDRESS [[no_unique_address]]
#else
#define RPP_NO_UNIQUE_ADDRESS
#endif

#if RPP_HAS_CPP_ATTRIBUTE(likely)
#define RPP_LIKELY [[likely]]
#define RPP_UNLIKELY [[unlikely]]
#else
#define RPP_LIKELY
#define RPP_UNLIKELY
#endif

#if RPP_COMPILER_MSVC
#define RPP_FORCEINLINE __forceinline
#elif RPP_HAS_ATTRIBUTE(always_inline) || RPP_COMPILER_GCC || RPP_COMPILER_CLANG
#define RPP_FORCEINLINE inline __attribute__((always_inline))
#else
#define RPP_FORCEINLINE inline
#endif

#if RPP_COMPILER_MSVC
#define RPP_NOINLINE __declspec(noinline)
#elif RPP_HAS_ATTRIBUTE(noinline) || RPP_COMPILER_GCC || RPP_COMPILER_CLANG
#define RPP_NOINLINE __attribute__((noinline))
#else
#define RPP_NOINLINE
#endif

#if RPP_COMPILER_MSVC
#define RPP_RESTRICT __restrict
#elif defined(__CUDACC__) || RPP_COMPILER_GCC || RPP_COMPILER_CLANG
#define RPP_RESTRICT __restrict__
#else
#define RPP_RESTRICT
#endif

#if RPP_HAS_ATTRIBUTE(visibility) && (RPP_COMPILER_GCC || RPP_COMPILER_CLANG)
#define RPP_EXPORT __attribute__((visibility("default")))
#define RPP_HIDDEN __attribute__((visibility("hidden")))
#elif RPP_COMPILER_MSVC
#define RPP_EXPORT __declspec(dllexport)
#define RPP_HIDDEN
#else
#define RPP_EXPORT
#define RPP_HIDDEN
#endif

#if RPP_COMPILER_MSVC
#define RPP_DEBUG_BREAK() __debugbreak()
#elif RPP_HAS_BUILTIN(__builtin_debugtrap)
#define RPP_DEBUG_BREAK() __builtin_debugtrap()
#elif RPP_HAS_BUILTIN(__builtin_trap) || RPP_COMPILER_GCC || RPP_COMPILER_CLANG
#define RPP_DEBUG_BREAK() __builtin_trap()
#else
#define RPP_DEBUG_BREAK() ((void)0)
#endif

#if RPP_HAS_BUILTIN(__builtin_expect) || RPP_COMPILER_GCC || RPP_COMPILER_CLANG
#define RPP_PREDICT_TRUE(expr) (__builtin_expect(!!(expr), 1))
#define RPP_PREDICT_FALSE(expr) (__builtin_expect(!!(expr), 0))
#else
#define RPP_PREDICT_TRUE(expr) (!!(expr))
#define RPP_PREDICT_FALSE(expr) (!!(expr))
#endif

#if RPP_HAS_BUILTIN(__builtin_unreachable) || RPP_COMPILER_GCC || RPP_COMPILER_CLANG
#define RPP_UNREACHABLE() __builtin_unreachable()
#elif RPP_COMPILER_MSVC
#define RPP_UNREACHABLE() __assume(0)
#else
#define RPP_UNREACHABLE() ((void)0)
#endif

#if RPP_CXX_17
#define RPP_IF_CONSTEXPR if constexpr
#else
#define RPP_IF_CONSTEXPR if
#endif

#if defined(__cpp_constexpr)
#define RPP_HAS_CONSTEXPR __cpp_constexpr
#else
#define RPP_HAS_CONSTEXPR 0
#endif

#if defined(__cpp_if_consteval)
#define RPP_HAS_IF_CONSTEVAL __cpp_if_consteval
#else
#define RPP_HAS_IF_CONSTEVAL 0
#endif

#if defined(__cpp_lib_is_constant_evaluated) || RPP_HAS_BUILTIN(__builtin_is_constant_evaluated)
#define RPP_HAS_IS_CONSTANT_EVALUATED 1
#else
#define RPP_HAS_IS_CONSTANT_EVALUATED 0
#endif


#endif //RPP_CONFIG_COMPILER_H
