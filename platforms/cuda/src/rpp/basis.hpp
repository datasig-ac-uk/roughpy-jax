#ifndef RPP_BASIS_HPP
#define RPP_BASIS_HPP

#include <cstdint>
#include <utility>
#include <algorithm>

#include <rpp/config.h>


namespace rpp {

namespace detail {

template <typename Degree_, typename Index_>
struct GradedBasis {
    using Degree = Degree_;
    using Index = Index_;

    Degree width;
    Degree depth;
    Index const *degree_begin;

    GradedBasis(Degree width_, Degree depth_, Index const *degree_begin_) noexcept
        : width(width_), depth(depth_), degree_begin(degree_begin_) {
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index size() const noexcept {
        return degree_begin[depth + 1];
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index true_size() const noexcept { return size(); }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index start_of_degree(Degree d) const noexcept {
        return degree_begin[d];
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index end_of_degree(Degree d) const noexcept {
        return degree_begin[d+1];
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index size_of_degree(Degree d) const noexcept {
        return degree_begin[d+1] - degree_begin[d];
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Degree degree(Index idx) const noexcept {
        Degree diff = this->depth + 1;
        Degree pos = 0;
        while (diff > 0) {
            const Degree half = diff / 2;
            const Degree new_pos = pos + half;

            if (this->degree_begin[new_pos] <= idx) {
                pos = new_pos + 1;
                diff -= half + 1;
            } else {
                diff = half;
            }
        }
        return pos - 1;
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Degree degree_linear(Index idx) const noexcept {
        Degree result = 0;
        while (result <= depth && degree_begin[result] <= idx) {
            ++result;
        }
        return result - 1;
    }

};
} // namespace internal


template <typename Degree_, typename Index_>
struct TensorBasis : detail::GradedBasis<Degree_, Index_> {
    using Base = detail::GradedBasis<Degree_, Index_>;

    using Degree = typename Base::Degree;
    using Index = typename Base::Index;

    using Base::Base;
    using Base::size;
    using Base::degree;

    RPP_HOST RPP_NODISCARD
    constexpr TensorBasis truncate(Degree new_depth) const noexcept {
        return {
            this->width, std::min(this->depth, new_depth),
            this->degree_begin
        };
    }

    template<typename Letter>
    RPP_HOST_DEVICE void unpack_index_to_letters(
        Letter *letter_array,
        Degree degree,
        Index index
    ) noexcept {
        for (Degree d = 0; d < degree; ++d) {
            letter_array[d] = static_cast<Letter>(index % this->width);
            index /= this->width;
        }
    }

    template<typename Letter, typename BitMask>
    RPP_HOST_DEVICE
    void pack_masked_index(Letter const *letters,
                           Degree degree,
                           BitMask const &bitmask,
                           Degree &lhs_deg,
                           Index &lhs_idx,
                           Degree &rhs_deg,
                           Index &rhs_idx
    ) const noexcept {
        for (; degree >= 0; --degree) {
            if (bitmask[degree]) {
                ++lhs_deg;
                lhs_idx = lhs_idx * this->width + letters[degree];
            } else {
                ++rhs_deg;
                rhs_idx = rhs_idx * this->width + letters[degree];
            }
        }
    }
};

template <typename Degree_, typename Index_>
struct LieBasis : detail::GradedBasis<Degree_, Index_> {
    using Base = detail::GradedBasis<Degree_, Index_>;
    using Degree = typename Base::Degree;
    using Index = typename Base::Index;

    Index const *data;

    using Base::degree;

    explicit constexpr LieBasis(Degree width, Degree depth, Index const *degree_begin,
                                Index const *data_)
        : Base{width, depth, degree_begin}, data{data_} {
    }


    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr LieBasis truncate(Degree new_depth) const noexcept {
        return {
            this->width, std::min(this->depth, new_depth),
            this->degree_begin
        };
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index size() const noexcept {
        if (this->width <= 1 || this->depth == 0) { return 0; }
        return Base::size() - 1;
    }

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index true_size() const noexcept { return Base::size(); }
};


using StandardTensorBasis = TensorBasis<std::int32_t, std::ptrdiff_t>;
using StandardLieBasis = LieBasis<std::int32_t, std::ptrdiff_t>;
} // namespace rpp


#endif //RPP_BASIS_HPP
