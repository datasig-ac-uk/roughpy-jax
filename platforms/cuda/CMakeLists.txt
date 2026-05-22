cmake_minimum_required(VERSION 3.30)

project(RoughPy_JAX_CUDA VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

option(RPJ_CUDA_INPLACE_BUILD "Build the plugin into the local package directory" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(Python_FIND_VIRTUAL_ENVIRONMENT FIRST)
find_package(
        Python 3.11 REQUIRED COMPONENTS Development.Module Interpreter ${SKBUILD_SABI_COMPONENT}
)
find_package(CUDAToolkit REQUIRED)
find_package(RoughPathPrimitives CONFIG REQUIRED COMPONENTS GPU)


set(RPJ_CUDA_PACKAGE_SUFFIX "cuda${RPJ_CUDA_TOOLKIT_MAJOR}")
set(RPJ_CUDA_PACKAGE_DIR "roughpy_jax_${RPJ_CUDA_PACKAGE_SUFFIX}")
set(RPJ_CUDA_MODULE_NAME "_rpy_jax_${RPJ_CUDA_PACKAGE_SUFFIX}")

execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "from jax import ffi; print(ffi.include_dir())"
        RESULT_VARIABLE XLA_DIR_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE XLA_DIR
        ERROR_VARIABLE XLA_DIR_ERROR
)
if (NOT XLA_DIR_RESULT EQUAL 0 OR XLA_DIR STREQUAL "")
    message(FATAL_ERROR "Failed to locate JAX FFI headers: ${XLA_DIR_ERROR}")
endif ()

add_library(XLA::FFI INTERFACE IMPORTED)
set_target_properties(XLA::FFI PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${XLA_DIR}")


if (DEFINED SKBUILD_SABI_VERSION)
    python_add_library(RoughPyJAXCudaPlugin MODULE WITH_SOABI USE_SABI ${SKBUILD_SABI_VERSION})
else ()
    python_add_library(RoughPyJAXCudaPlugin MODULE WITH_SOABI)
endif ()

set_target_properties(
        RoughPyJAXCudaPlugin
        PROPERTIES
        OUTPUT_NAME "${RPJ_CUDA_MODULE_NAME}"
)


target_sources(RoughPyJAXCudaPlugin PRIVATE
        src/cuda/dense_ft_adj_mul.cu
        src/cuda/dense_ft_adj_mul.h
        src/cuda/dense_ft_antipode.cu
        src/cuda/dense_ft_antipode.h
        src/cuda/dense_ft_exp.cu
        src/cuda/dense_ft_exp.h
        src/cuda/dense_ft_fma.h
        src/cuda/dense_ft_fma.cu
        src/cuda/dense_ft_fmexp.cu
        src/cuda/dense_ft_fmexp.h
        src/cuda/dense_ft_log.cu
        src/cuda/dense_ft_log.h
        src/cuda/dense_st_adj_mul.cu
        src/cuda/dense_st_adj_mul.h
        src/cuda/dense_st_fma.cu
        src/cuda/dense_st_fma.h
        src/cuda/xla_headers.hpp
        src/cuda/scalars.hpp
        src/cuda/select_strategy.cuh
        src/stub.cpp
        src/module.c
)

target_link_libraries(RoughPyJAXCudaPlugin PRIVATE CUDA::cudart)



target_include_directories(
        RoughPyJAXCudaPlugin
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_compile_definitions(
        RoughPyJAXCudaPlugin
        PRIVATE
        RPJ_CUDA_TOOLKIT_MAJOR=${RPJ_CUDA_TOOLKIT_MAJOR}
)

target_link_libraries(
        RoughPyJAXCudaPlugin
        PRIVATE
        Python::Module
        RPP::RPP_GPU
        #    CUDA::cudart
        XLA::FFI
)

if (RPJ_CUDA_INPLACE_BUILD)
    set_target_properties(
            RoughPyJAXCudaPlugin
            PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${RPJ_CUDA_PACKAGE_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/${RPJ_CUDA_PACKAGE_DIR}"
    )
endif ()

install(
        TARGETS RoughPyJAXCudaPlugin
        LIBRARY DESTINATION ${RPJ_CUDA_PACKAGE_DIR}
        RUNTIME DESTINATION ${RPJ_CUDA_PACKAGE_DIR}
)

if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${RPJ_CUDA_PACKAGE_DIR}")
    install(
            DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${RPJ_CUDA_PACKAGE_DIR}/"
            DESTINATION ${RPJ_CUDA_PACKAGE_DIR}
            FILES_MATCHING
            PATTERN "*.py"
            PATTERN "*.pyi"
            PATTERN "py.typed"
    )
endif ()
