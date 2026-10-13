cmake_minimum_required(VERSION 3.31)

if (SKBUILD AND DEFINED SKBUILD_PROJECT_VERSION)
    set(RPJ_PLUGIN_VERSION ${SKBUILD_PROJECT_VERSION})
else ()
    set(RPJ_PLUGIN_VERSION 1.0.0)
endif ()

project(RoughPy_JAX_CUDA VERSION "${RPJ_PLUGIN_VERSION}" LANGUAGES C CXX CUDA)

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

if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "all-major")
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

if (NOT DEFINED RPJ_REPO_ROOT)
    message(FATAL_ERROR "Set RPJ_REPO_ROOT to the RoughPy JAX repository root.")
endif ()
get_filename_component(RPJ_REPO_ROOT "${RPJ_REPO_ROOT}" ABSOLUTE
        BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

if (NOT IS_DIRECTORY "${RPJ_REPO_ROOT}")
    message(FATAL_ERROR "RPJ repository root does not exist: ${RPJ_REPO_ROOT}")
endif ()

set(RPJ_RPP_SOURCE_DIR "${RPJ_REPO_ROOT}/third_party/rough_path_primitives")
set(RPJ_COMMON_SOURCE_DIR "${RPJ_REPO_ROOT}/common")

if (NOT IS_DIRECTORY "${RPJ_COMMON_SOURCE_DIR}/include/xla_helpers")
    message(FATAL_ERROR "Shared XLA helpers are missing from ${RPJ_COMMON_SOURCE_DIR}/include")
endif ()

if (NOT IS_DIRECTORY "${RPJ_RPP_SOURCE_DIR}")
    message(FATAL_ERROR "RPP source directory does not exist: ${RPJ_RPP_SOURCE_DIR}")
endif ()

if (NOT EXISTS "${RPJ_RPP_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
        "No RPP CMakeLists.txt found in ${RPJ_RPP_SOURCE_DIR}. "
        "Check RPJ_REPO_ROOT and initialise the submodule with "
        "git submodule update --init --recursive.")
endif ()

set(RPP_ENABLE_CUDA ON CACHE BOOL "" FORCE)
set(RPP_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(RPP_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(RPP_ADD_STUB_TARGET OFF CACHE BOOL "" FORCE)
add_subdirectory("${RPJ_RPP_SOURCE_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/third_party/rough_path_primitives"
        EXCLUDE_FROM_ALL)

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
        src/cuda/dense_ft_adj_lmul.cu
        src/cuda/dense_ft_adj_lmul.h
        src/cuda/dense_ft_adj_rmul.cu
        src/cuda/dense_ft_adj_rmul.h
        src/cuda/dense_ft_antipode.cu
        src/cuda/dense_ft_antipode.h
        src/cuda/dense_ft_exp.cu
        src/cuda/dense_ft_exp.h
        src/cuda/dense_ft_fma.h
        src/cuda/dense_ft_fma.cu
        src/cuda/dense_ft_mul.cu
        src/cuda/dense_ft_mul.h
        src/cuda/dense_ft_fmexp.cu
        src/cuda/dense_ft_fmexp.h
        src/cuda/dense_ft_log.cu
        src/cuda/dense_ft_log.h
        src/cuda/dense_st_adj_mul.cu
        src/cuda/dense_st_adj_mul.h
        src/cuda/dense_st_fma.cu
        src/cuda/dense_st_fma.h
        src/cuda/dense_st_mul.cu
        src/cuda/dense_st_mul.h
        src/cuda/dense_tensor_pairing.cu
        src/cuda/dense_tensor_pairing.h
        src/module.c
)

target_link_libraries(RoughPyJAXCudaPlugin PRIVATE CUDA::cudart)



target_include_directories(
        RoughPyJAXCudaPlugin
        PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${RPJ_COMMON_SOURCE_DIR}/include"
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
