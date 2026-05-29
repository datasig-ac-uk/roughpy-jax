#include "dense_tensor_pairing.h"


#include <rpp/basis/tensor_basis.hpp>
#include <rpp/gpu/block/operations/basic/tensor_pairing.hpp>

#include "batch.cuh"
#include "scalars.hpp"
#include "select_strategy.cuh"
#include "xla_headers.hpp"


namespace ffi = xla::ffi;


namespace rpy::jax::cuda {
struct DenseTensorPairingStaticArgs {
    HostTensorBasis basis;
    int32_t fun_max_degree;
    int32_t arg_max_degree;
    int32_t fun_min_degree;
    int32_t arg_min_degree;
};

template<typename Tag>
struct DenseTensorPairingFunctor {
    using Scalar = typename Tag::Scalar;
    using Accum = typename Tag::Accum;

    static ffi::Error eval(ffi::Result<ffi::AnyBuffer> out,
                           ffi::AnyBuffer func,
                           ffi::AnyBuffer arg,
                           DenseTensorPairingStaticArgs static_args,
                           cudaStream_t stream
    ) noexcept {
        const auto tensor_size = static_args.basis.size();
        auto out_shape = out->dimensions();
        const auto batch_size = std::accumulate(out_shape.begin(), out_shape.end(), 1LL, std::multiplies<int64_t>());

        return select_strategy<Accum>(tensor_size, [&](auto strategy) {
            using LaunchCfg = typename decltype(strategy)::LaunchConfig;
            return rpp::ops::tensor_pairing(strategy, LaunchCfg{stream},
                                            make_scalar_batch<Scalar>(out),
                                            make_tensor_batch<Scalar>(func, static_args.fun_max_degree,
                                                                      static_args.fun_min_degree),
                                            make_tensor_batch<Scalar>(arg, static_args.arg_min_degree,
                                                                      static_args.arg_max_degree),
                                            static_args.basis,
                                            batch_size
            );
        });
    }
};


ffi::Error cuda_dense_tensor_pairing(
    cudaStream_t stream,
    ffi::Result<ffi::AnyBuffer> out,
    ffi::AnyBuffer functional,
    ffi::AnyBuffer argument,
    int32_t width,
    int32_t depth,
    DegreeBeginSpan degree_begin,
    int32_t fun_max_deg,
    int32_t arg_max_deg,
    int32_t fun_min_deg,
    int32_t arg_min_deg
) noexcept {
    DenseTensorPairingStaticArgs static_args{
        HostTensorBasis{
            width,
            depth,
            cast_db_array(degree_begin.begin())
        },
        fun_max_deg, arg_max_deg, fun_min_deg, arg_min_deg
    };

    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(functional, static_args.basis, fun_max_deg));
    RPY_XLA_SUCCESS_OR_RETURN(check_data_degree(argument, static_args.basis, arg_max_deg));

    if (!all_buffers_match_type(out->element_type(), functional, argument)) {
        return ffi::Error::InvalidArgument(
            "all arguments should have the same data type");
    }

    return select_type_and_go<DenseTensorPairingFunctor>(
        out->element_type(), out, functional, argument, static_args, stream);
}
} // namespace rpy::jax::cuda


XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cuda_dense_tensor_pairing,
    rpy::jax::cuda::cuda_dense_tensor_pairing,
    xla::ffi::Ffi::Bind()
    .Ctx<xla::ffi::PlatformStream<cudaStream_t>>()
    .Ret<xla::ffi::AnyBuffer>()
    .Arg<xla::ffi::AnyBuffer>()
    .Attr<int32_t>("width")
    .Attr<int32_t>("depth")
    .Attr<rpy::jax::cuda::DegreeBeginSpan>("degree_begin")
    .Attr<int32_t>("fun_max_deg")
    .Attr<int32_t>("arg_max_deg")
    .Attr<int32_t>("fun_min_deg")
    .Attr<int32_t>("arg_min_deg")
);
