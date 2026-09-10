// RUN: mlir-opt %s --split-input-file --verify-diagnostics \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp{marshal=ir})"
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp)" | FileCheck %s --check-prefix=RUNTIME

// Runtime marshalling still supports dynamic non-batch dimensions.
// RUNTIME-LABEL: func.func @dynamic_rows
// RUNTIME: call @esp_alloc_shared
// RUNTIME: call @esp_free_shared
// RUNTIME: return
func.func @dynamic_rows(%a: memref<1x?x8xf32>, %b: memref<1x8x4xf32>, %c: memref<1x?x4xf32>) {
  // expected-error @+1 {{marshal=ir needs static shapes}}
  linalg.batch_matmul ins(%a, %b : memref<1x?x8xf32>, memref<1x8x4xf32>) outs(%c : memref<1x?x4xf32>)
  return
}

// -----

// The restriction is per function, not just per block: a nested offload may
// otherwise try to allocate while the enclosing block's buffer is still live.
// RUNTIME-LABEL: func.func @nested_offload
// RUNTIME: call @esp_alloc_shared
// RUNTIME: call @esp_free_shared
// RUNTIME: scf.if
// RUNTIME: call @esp_alloc_shared
// RUNTIME: call @esp_free_shared
// RUNTIME: return
func.func @nested_offload(%a: memref<1x4x8xf32>, %b: memref<1x8x4xf32>, %c: memref<1x4x4xf32>, %cond: i1) {
  linalg.batch_matmul ins(%a, %b : memref<1x4x8xf32>, memref<1x8x4xf32>) outs(%c : memref<1x4x4xf32>)
  scf.if %cond {
    // expected-error @+1 {{marshal=ir supports at most one ESP offload per function}}
    linalg.batch_matmul ins(%a, %b : memref<1x4x8xf32>, memref<1x8x4xf32>) outs(%c : memref<1x4x4xf32>)
  }
  return
}
