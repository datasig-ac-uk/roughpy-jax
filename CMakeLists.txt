cmake_minimum_required(VERSION 3.30)


project(RoughPy_JAX VERSION 0.0.1 LANGUAGES C CXX)

include(GNUInstallDirs)


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
endif()

add_library(XLA::FFI INTERFACE IMPORTED)
set_target_properties(XLA::FFI PROPERTIES 
  INTERFACE_INCLUDE_DIRECTORIES "${XLA_DIR}")


execute_process(
  COMMAND "${Python_EXECUTABLE}"
          -c "import roughpy; print(roughpy.get_include())"
  RESULT_VARIABLE RPY_COMPUTE_DIR_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE RPY_COMPUTE_DIR
  ERROR_VARIABLE RPY_COMPUTE_DIR_ERROR
)
message(STATUS "RPC_DIR=${RPY_COMPUTE_DIR}")
if (NOT RPY_COMPUTE_DIR_RESULT EQUAL 0 OR RPY_COMPUTE_DIR STREQUAL "")
  message(FATAL_ERROR "Failed to locate RoughPy headers: ${RPY_COMPUTE_DIR_ERROR}")
endif()

add_library(RoughPy::Compute INTERFACE IMPORTED)
set_target_properties(RoughPy::Compute PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${RPY_COMPUTE_DIR}")



if (DEFINED SKBUILD_SABI_VERSION)
python_add_library(Roughpy_JAXModule MODULE WITH_SOABI USE_SABI ${SKBUILD_SABI_VERSION})
else()
python_add_library(Roughpy_JAXModule MODULE WITH_SOABI)
endif()



set_target_properties(Roughpy_JAXModule PROPERTIES
    OUTPUT_NAME _rpy_jax_internals
    )

target_link_libraries(Roughpy_JAXModule PRIVATE
    XLA::FFI
    RoughPy::Compute
)


add_subdirectory(roughpy_jax)


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
