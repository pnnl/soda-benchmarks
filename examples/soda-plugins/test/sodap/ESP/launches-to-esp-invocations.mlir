// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-launches-to-esp)" | FileCheck %s

// Two launches of one matmul tile (sodap-join-multiple-tiles): one arena,
// William's calls (ReusableGemmABI.h). The 128-byte header, then each buffer
// at 8 bytes an element: A 128, B 256, T 384, C 512, D 640. T stays in the
// arena; only the host's arguments are copied.

// CHECK-LABEL: func.func @forward_kernel(
// CHECK-SAME:    %[[A:.*]]: memref<4x4xf32>, %[[B:.*]]: memref<4x4xf32>, %[[C:.*]]: memref<4x4xf32>, %[[D:.*]]: memref<4x4xf32>)
// CHECK-NOT:     memref.alloc
// CHECK:         %[[CTX:.*]] = call @esp_gemm_open(%c768_i64, %c0_i64)
// CHECK:         call @esp_gemm_copy_in(%[[CTX]], %c128_i64
// CHECK:         call @esp_gemm_copy_in(%[[CTX]], %c256_i64
// CHECK:         call @esp_gemm_copy_in(%[[CTX]], %c512_i64
// CHECK:         call @esp_gemm_invoke(%[[CTX]], %c128_i64{{.*}}, %c256_i64{{.*}}, %c384_i64{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %c32_i64{{.*}}, %c32_i64{{.*}}, %c32_i64{{.*}}, %cst{{.*}}, %cst
// CHECK:         call @esp_gemm_invoke(%[[CTX]], %c384_i64{{.*}}, %c512_i64{{.*}}, %c640_i64
// CHECK:         call @esp_gemm_copy_out(%[[CTX]], %c640_i64
// CHECK:         call @esp_gemm_close(%[[CTX]])
// CHECK-NOT:     dataflow.launch
// CHECK-NOT:     module @forward_kernel
module {
  module @forward_kernel attributes {sodap.operator = "matmul", sodap.tile = 0 : i64} {
    func.func private @forward_kernel(memref<?x?xf32>, memref<?x?xf32>, index, index, index, memref<?x?xf32>)
  }
  func.func @forward(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>, %d: memref<4x4xf32>) {
    %c4 = arith.constant 4 : index
    %t = memref.alloc() : memref<4x4xf32>
    dataflow.launch @forward_kernel::@forward_kernel(%a, %b, %c4, %c4, %c4) outs(%t : memref<4x4xf32>) : (memref<4x4xf32>, memref<4x4xf32>, index, index, index) -> ()
    %r = memref.alloc() : memref<4x4xf32>
    dataflow.launch @forward_kernel::@forward_kernel(%t, %c, %c4, %c4, %c4) outs(%r : memref<4x4xf32>) : (memref<4x4xf32>, memref<4x4xf32>, index, index, index) -> ()
    memref.copy %r, %d : memref<4x4xf32> to memref<4x4xf32>
    return
  }
}
