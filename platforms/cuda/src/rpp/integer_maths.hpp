#ifndef RPP_INTEGER_MATHS_HPP
#define RPP_INTEGER_MATHS_HPP

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <rpp/config.h>


namespace rpp {

template <typename I>
RPP_HOST_DEVICE constexpr bool is_pow_2(I val) noexcept {
    return (val > 0) && (val & (val - 1)) == I{0};
}


template <typename T, typename S>
RPP_HOST_DEVICE
constexpr T align_up(T value, S alignment) noexcept {
    return (value + static_cast<T>(alignment) - 1) & ~(static_cast<T>(alignment) - 1);
}

} // namespace rpp


#endif //RPP_INTEGER_MATHS_HPP
