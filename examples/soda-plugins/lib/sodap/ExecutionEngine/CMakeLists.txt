set(LLVM_OPTIONAL_SOURCES
  EspRuntimeMock.cpp
)

add_mlir_library(mlir_mockesp_runner_utils
  SHARED
  EspRuntimeMock.cpp

  EXCLUDE_FROM_LIBMLIR
)
target_compile_definitions(mlir_mockesp_runner_utils PRIVATE mlir_esp_runner_utils_EXPORTS)
target_include_directories(mlir_mockesp_runner_utils PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)
