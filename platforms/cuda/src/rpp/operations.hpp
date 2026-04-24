#ifndef PLATFORMS_CUDA_SRC_RPP_OPERATIONS_HPP
#define PLATFORMS_CUDA_SRC_RPP_OPERATIONS_HPP


namespace rpp::ops {

enum class InplaceFMAType {
    AEqualsBCPlusA, // a <- b*c + a
    AEqualsABPlusC, // a <- a*b + c
    AEqualsBAPlusC  // a <- b*a + c
};

template <typename Strategy>
class Vector;

template <typename Strategy, typename Basis>
class FTBasic;

template <typename Strategy, typename Basis>
class FTMultiply;

template <typename Strategy, typename Basis>
class FTAdjointMultiply;

template <typename Strategy, typename Basis>
class Antipode;

template <typename Strategy, typename Basis>
class STMultiply;

template <typename Strategy, typename Basis>
class STAdjointMultiply;

template <typename Strategy, typename Basis>
class FTExp;

template <typename Strategy, typename Basis>
class FTFMExp;

template <typename Strategy, typename Basis>
class FTLog;

} // namespace rpp


#endif //PLATFORMS_CUDA_SRC_RPP_OPERATIONS_HPP
