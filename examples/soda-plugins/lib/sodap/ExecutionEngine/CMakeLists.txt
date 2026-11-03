set(LLVM_OPTIONAL_SOURCES
  EspRuntime.cpp
  EspRuntimeMock.cpp
  DynamicOpCounterRuntime.cpp
  esp_prof.c
)

# esp_prof.c is plain C and lives in both ESP runtime libraries, so a kernel
# lowered with profile=true links against either. On the host it times with
# CLOCK_MONOTONIC; on the SoC, with mcycle.
add_mlir_library(mlir_mockesp_runner_utils
  SHARED
  EspRuntimeMock.cpp
  esp_prof.c

  EXCLUDE_FROM_LIBMLIR
)
target_compile_definitions(mlir_mockesp_runner_utils PRIVATE mlir_esp_runner_utils_EXPORTS)
target_include_directories(mlir_mockesp_runner_utils PRIVATE
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/include/sodap/ExecutionEngine
)

add_mlir_library(mlir_sodap_instr_runner_utils
  SHARED
  DynamicOpCounterRuntime.cpp

  EXCLUDE_FROM_LIBMLIR
)
target_compile_definitions(mlir_sodap_instr_runner_utils PRIVATE mlir_sodap_instr_runner_utils_EXPORTS)
target_include_directories(mlir_sodap_instr_runner_utils PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

# --- The real ESP runtime ----------------------------------------------------
# Off by default: EspRuntime.cpp includes <esp_accelerator.h> and <esp_probe.h>,
# which come from an ESP checkout rather than from this repository, so a plain
# build of soda-plugins cannot compile it. The usual way to build it is not here
# at all but inside an ESP tree, from the esp-app/ directory
# scripts/ll_to_riscv.sh stages; this target exists so that a host with the ESP
# headers can compile-check it in place.
option(SODAP_ENABLE_ESP_RUNTIME "Build the real ESP accelerator runtime" OFF)
set(SODAP_ESP_ROOT "" CACHE PATH
    "ESP checkout providing esp_accelerator.h and esp_probe.h")

if(SODAP_ENABLE_ESP_RUNTIME)
  if(NOT SODAP_ESP_ROOT)
    message(FATAL_ERROR
      "SODAP_ENABLE_ESP_RUNTIME=ON needs SODAP_ESP_ROOT pointing at an ESP "
      "checkout that provides esp_accelerator.h and esp_probe.h. Configure with "
      "-DSODAP_ESP_ROOT=/path/to/esp, or leave the option OFF and build the "
      "runtime from the esp-app/ directory scripts/ll_to_riscv.sh stages.")
  endif()

  add_mlir_library(mlir_esp_runner_utils
    SHARED
    EspRuntime.cpp
    esp_prof.c

    EXCLUDE_FROM_LIBMLIR
  )
  target_compile_definitions(mlir_esp_runner_utils PRIVATE mlir_esp_runner_utils_EXPORTS)
  target_include_directories(mlir_esp_runner_utils PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/sodap/ExecutionEngine
    ${SODAP_ESP_ROOT}/soft/common/drivers/common/include
    ${SODAP_ESP_ROOT}/soft/common/drivers/baremetal/include
  )
endif()
