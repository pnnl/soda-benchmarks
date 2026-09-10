cmake_minimum_required(VERSION 3.20.0)
project(soda-plugins LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  find_package(MLIR REQUIRED CONFIG)

  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

  set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
  set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
  set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else()
  # Build via external projects mechanism
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIR ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_GENERATED_INCLUDE_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  set(MLIR_INCLUDE_DIRS "${MLIR_INCLUDE_DIR};${MLIR_GENERATED_INCLUDE_DIR}")
endif()

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

set(SODAPLUGINS_SOURCE_DIR ${PROJECT_SOURCE_DIR})
set(SODAPLUGINS_BINARY_DIR ${PROJECT_BINARY_DIR})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${SODAPLUGINS_SOURCE_DIR}/include)
include_directories(${SODAPLUGINS_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Where Bambu/PandA is installed -- the directory holding settings.sh.
# Only the Bambu backend passes need it, and only at run time, so the plugin
# still builds without it; the tests that need it are then UNSUPPORTED.
set(SODAP_BAMBU_ROOT "" CACHE PATH "Bambu/PandA install root")
if(NOT SODAP_BAMBU_ROOT)
  message(WARNING
    "SODAP_BAMBU_ROOT is not set: the Bambu backend passes will have no usable "
    "defaults. Configure with -DSODAP_BAMBU_ROOT=/opt/panda")
else()
  message(STATUS "Using Bambu install in: ${SODAP_BAMBU_ROOT}")
endif()

# Those two paths become the defaults of the bambu backend pass options
set(SODAP_BAMBU_INCLUDE "${SODAP_BAMBU_ROOT}/include/panda")
set(SODAP_BAMBU_CLANGXX "${SODAP_BAMBU_ROOT}/compilers/clang-19/bin/clang++-19")

configure_file(
  ${SODAPLUGINS_SOURCE_DIR}/include/sodap/Config/Config.h.in
  ${SODAPLUGINS_BINARY_DIR}/include/sodap/Config.h
  @ONLY)

add_subdirectory(include)
add_subdirectory(lib)

# TODO: Python bindings are not yet implemented.
# if(MLIR_ENABLE_BINDINGS_PYTHON)
#   message(STATUS "Enabling Python API")
#   add_subdirectory(python)
# endif()
add_subdirectory(test)
add_subdirectory(soda-plugins)
