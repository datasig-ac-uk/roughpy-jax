#ifndef RPP_BASIS_HPP
#define RPP_BASIS_HPP

#include <cstdint>
#include <utility>
#include <algorithm>

#include <rpp/config.h>


namespace rpp {

namespace internal {

template <typename Degree_, typename Index_>
struct GradedBasis {
    using Degree = Degree_;
    using Index = Index_;

    Degree width;
    Degree depth;
    Index const* degree_begin;

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr Index size() const noexcept {
        return degree_begin[depth + 1];
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
};



} // namespace internal


template <typename Degree_, typename Index_>
struct TensorBasis : internal::GradedBasis<Degree_, Index_> {
    using Base = internal::GradedBasis<Degree_, Index_>;

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

    template <typename Letter>
    RPP_HOST_DEVICE void unpack_index_to_letters(
        Letter* letter_array,
        Degree degree,
        Index index
        ) noexcept {
        for (Degree d=0; d<degree; ++d) {
            letter_array[d] = static_cast<Letter>(index % this->width);
            index /= this->width;
        }
    }

    template <typename Letter, typename BitMask>
    RPP_HOST_DEVICE
    void pack_masked_index(Letter const* letters,
        Degree degree,
        BitMask const& bitmask,
        Degree& lhs_deg,
        Index& lhs_idx,
        Degree& rhs_deg,
        Index& rhs_idx
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
struct LieBasis : internal::GradedBasis<Degree_, Index_> {
    using Base = internal::GradedBasis<Degree_, Index_>;
    using Degree = typename Base::Degree;
    using Index = typename Base::Index;

    using Base::Base;
    using Base::size;
    using Base::degree;

    Index const* data;

    RPP_HOST_DEVICE RPP_NODISCARD
    constexpr LieBasis truncate(Degree new_depth) const noexcept {
        return {
            this->width, std::min(this->depth, new_depth),
            this->degree_begin
        };
    }

};


using StandardTensorBasis = TensorBasis<std::int32_t, std::ptrdiff_t>;
using StandardLieBasis = LieBasis<std::int32_t, std::ptrdiff_t>;



} // namespace rpp



#endif //RPP_BASIS_HPP
