cmake_minimum_required(VERSION 3.30)


if (SKBUILD AND DEFINED SKBUILD_PROJECT_VERSION)
    string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" RPJ_VERSION
            "${SKBUILD_PROJECT_VERSION}")
else ()
    set(RPJ_VERSION 1.0.0)
endif ()

project(RoughPy_JAX VERSION "${RPJ_VERSION}" LANGUAGES C CXX)

include(GNUInstallDirs)

option(RPJ_INPLACE_BUILD "build the library in-place" ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set(Python_FIND_VIRTUAL_ENVIRONMENT FIRST)
find_package(Python 3.11 REQUIRED COMPONENTS
        Development.Module Interpreter ${SKBUILD_SABI_COMPONENT})


execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "from jax import ffi; print(ffi.include_dir())"
        RESULT_VARIABLE XLA_DIR_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR
        ERROR_VARIABLE XLA_DIR_ERROR
)
message(STATUS "XLA_DIR=${XLA_DIR}")
if (NOT XLA_DIR_RESULT EQUAL 0 OR XLA_DIR STREQUAL "")
    message(FATAL_ERROR "Failed to locate JAX FFI headers: ${XLA_DIR_ERROR}")
endif ()

add_library(XLA::FFI INTERFACE IMPORTED)
set_target_properties(XLA::FFI PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${XLA_DIR}")


set(RPP_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(RPP_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(RPP_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(RPP_ADD_STUB_TARGET OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/rough_path_primitives
        EXCLUDE_FROM_ALL)


if (DEFINED SKBUILD_SABI_VERSION)
    python_add_library(Roughpy_JAXModule MODULE WITH_SOABI USE_SABI ${SKBUILD_SABI_VERSION})
else ()
    python_add_library(Roughpy_JAXModule MODULE WITH_SOABI)
endif ()


set_target_properties(Roughpy_JAXModule PROPERTIES
        OUTPUT_NAME _rpy_jax_internals
)

if (RPJ_INPLACE_BUILD)
    message(STATUS "Library artifact will be placed in roughpy_jax directory")
    set_target_properties(Roughpy_JAXModule PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/roughpy_jax"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/roughpy_jax")
endif ()

target_link_libraries(Roughpy_JAXModule PRIVATE
        XLA::FFI
)

target_include_directories(Roughpy_JAXModule PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/common/include"
)


add_subdirectory(roughpy_jax)

target_link_libraries(Roughpy_JAXModule PRIVATE RPP::RPP)

set(RPJ_INSTALL_DIR roughpy_jax)

install(
        TARGETS Roughpy_JAXModule
        LIBRARY DESTINATION ${RPJ_INSTALL_DIR}
        RUNTIME DESTINATION ${RPJ_INSTALL_DIR}
)

install(
        DIRECTORY roughpy_jax
        DESTINATION .
        FILES_MATCHING
        PATTERN "*.py"
        PATTERN "*.pyi"
        PATTERN "py.typed"
        PATTERN "src/*" EXCLUDE
        PATTERN "tests/*" EXCLUDE
)
