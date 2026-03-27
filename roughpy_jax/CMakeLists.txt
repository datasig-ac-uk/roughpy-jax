
target_sources(RoughPy_JaxModule PRIVATE
    src/cpu/batching_loop.hpp
    src/cpu/dense_ft_adj_mul.cpp
    src/cpu/dense_ft_adj_mul.h
    src/cpu/dense_ft_fma.cpp
    src/cpu/dense_ft_exp.cpp
    src/cpu/dense_ft_log.cpp
    src/cpu/dense_ft_fmexp.cpp
    src/cpu/dense_ft_antipode.cpp
    src/cpu/dense_st_fma.cpp
    src/cpu/dense_st_adj_mul.cpp
    src/cpu/dense_st_adj_mul.h
    src/cpu/xla_common.cpp
    src/roughpy_jax_module.c
)



