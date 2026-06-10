cmake_minimum_required(VERSION 3.30)

project(RoughPy_JAX_CUDA VERSION 0.1.0 LANGUAGES C CXX CUDA)

include(GNUInstallDirs)

option(RPJ_CUDA_INPLACE_BUILD "Build the plugin into the local package directory" OFF)

set(RPJ_CUDA_PACKAGE_DIR "roughpy_jax_cuda_plugin" CACHE STRING
        "Python package directory for the CUDA plugin wheel")
set(RPJ_CUDA_MODULE_NAME "_rpy_jax_cuda")
set(RPJ_CUDA_VARIANT "" CACHE STRING
        "CUDA toolkit variant label recorded in the plugin metadata, e.g. 12.8 or 13")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

function(rpj_detect_supported_cuda_architectures out_var minimum_arch)
    if (NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "Cannot infer CUDA architectures without a configured CUDA compiler")
    endif ()

    execute_process(
            COMMAND "${CMAKE_CUDA_COMPILER}" --list-gpu-arch
            RESULT_VARIABLE arch_query_result
            OUTPUT_VARIABLE arch_query_output
            ERROR_VARIABLE arch_query_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if (NOT arch_query_result EQUAL 0)
        message(FATAL_ERROR
                "Failed to query supported CUDA architectures from ${CMAKE_CUDA_COMPILER}: ${arch_query_error}")
    endif ()

    string(REPLACE "\n" ";" arch_query_lines "${arch_query_output}")
    set(supported_architectures "")
    foreach (arch_entry IN LISTS arch_query_lines)
        string(STRIP "${arch_entry}" arch_entry)
        if (arch_entry MATCHES "^compute_([0-9]+)$")
            set(arch_number "${CMAKE_MATCH_1}")
            if (arch_number GREATER_EQUAL minimum_arch)
                list(APPEND supported_architectures "${arch_number}")
            endif ()
        endif ()
    endforeach ()

    if (NOT supported_architectures)
        message(FATAL_ERROR
                "CUDA compiler ${CMAKE_CUDA_COMPILER} does not report any supported architectures >= sm_${minimum_arch}")
    endif ()

    list(REMOVE_DUPLICATES supported_architectures)
    set(${out_var} "${supported_architectures}" PARENT_SCOPE)
endfunction()

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    rpj_detect_supported_cuda_architectures(CMAKE_CUDA_ARCHITECTURES 80)
    message(STATUS "Auto-selected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(Python_FIND_VIRTUAL_ENVIRONMENT FIRST)
find_package(
        Python 3.11 REQUIRED COMPONENTS Development.Module Interpreter ${SKBUILD_SABI_COMPONENT}
)
find_package(CUDAToolkit ${RPJ_CUDA_TOOLKIT_MAJOR} REQUIRED)
find_package(RoughPathPrimitives CONFIG REQUIRED COMPONENTS GPU)

if (NOT DEFINED RPJ_CUDA_TOOLKIT_MAJOR)
#    message(FATAL_ERROR "RPJ_CUDA_TOOLKIT_MAJOR must be provided")
    set(RPJ_CUDA_TOOLKIT_MAJOR ${CUDAToolkit_VERSION_MAJOR})
endif ()

if (RPJ_CUDA_VARIANT STREQUAL "")
    set(RPJ_CUDA_VARIANT "${RPJ_CUDA_TOOLKIT_MAJOR}")
endif ()

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
        src/cuda/dense_lie_pairing.cu
        src/cuda/dense_tensor_pairing.h
        src/cuda/dense_st_adj_mul.cu
        src/cuda/dense_st_adj_mul.h
        src/cuda/dense_st_fma.cu
        src/cuda/dense_st_fma.h
        src/cuda/dense_tensor_pairing.cu
        src/cuda/dense_tensor_pairing.h
        src/cuda/xla_headers.hpp
        src/cuda/scalars.hpp
        src/cuda/select_strategy.cuh
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

set(RPJ_CUDA_GENERATED_PACKAGE_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/${RPJ_CUDA_PACKAGE_DIR}")
file(MAKE_DIRECTORY "${RPJ_CUDA_GENERATED_PACKAGE_DIR}")
configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/package/roughpy_jax_cuda_plugin/__init__.py.in"
        "${RPJ_CUDA_GENERATED_PACKAGE_DIR}/__init__.py"
        @ONLY
)

install(
        TARGETS RoughPyJAXCudaPlugin
        LIBRARY DESTINATION ${RPJ_CUDA_PACKAGE_DIR}
        RUNTIME DESTINATION ${RPJ_CUDA_PACKAGE_DIR}
)

install(
        DIRECTORY "${RPJ_CUDA_GENERATED_PACKAGE_DIR}/"
        DESTINATION ${RPJ_CUDA_PACKAGE_DIR}
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.pyi"
        PATTERN "py.typed"
)
