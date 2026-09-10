// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp)" | \
// RUN: mlir-opt \
// RUN:   -convert-linalg-to-loops -convert-scf-to-cf \
// RUN:   --canonicalize --cse \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-math-to-llvm --convert-math-to-libm \
// RUN:   -arith-expand -memref-expand \
// RUN:   --convert-arith-to-llvm \
// RUN:   --convert-func-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner \
// RUN:   -e main -entry-point-result=void \
// RUN:   -shared-libs=%sodap_libs/libmlir_mockesp_runner_utils%shlibext \
// RUN:   -shared-libs=%llvm_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%llvm_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// Helper: allocate a 3D memref filled with a constant value.
func.func @alloc_3d_filled_f32(%d0: index, %d1: index, %d2: index, %f: f32)
    -> memref<?x?x?xf32> {
  %buf = memref.alloc(%d0, %d1, %d2) : memref<?x?x?xf32>
  linalg.fill ins(%f : f32) outs(%buf : memref<?x?x?xf32>)
  return %buf : memref<?x?x?xf32>
}

// MLIR runner utility for printing memrefs.
func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  // Dimensions: batch=1, M=4, K=8, N=4 (the supported hardware layout).
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %cst_a = arith.constant 1.0 : f32
  %cst_b = arith.constant 0.5 : f32
  %cst_c = arith.constant 0.0 : f32

  // Allocate A[1x4x8], B[1x8x4], C[1x4x4]
  %A_dyn = call @alloc_3d_filled_f32(%c1, %c4, %c8, %cst_a)
      : (index, index, index, f32) -> memref<?x?x?xf32>
  %B_dyn = call @alloc_3d_filled_f32(%c1, %c8, %c4, %cst_b)
      : (index, index, index, f32) -> memref<?x?x?xf32>
  %C_dyn = call @alloc_3d_filled_f32(%c1, %c4, %c4, %cst_c)
      : (index, index, index, f32) -> memref<?x?x?xf32>

  // Cast to static shapes for batch_matmul
  %A = memref.cast %A_dyn : memref<?x?x?xf32> to memref<1x4x8xf32>
  %B = memref.cast %B_dyn : memref<?x?x?xf32> to memref<1x8x4xf32>
  %C = memref.cast %C_dyn : memref<?x?x?xf32> to memref<1x4x4xf32>

  // Print input A
  %A_print = memref.cast %A : memref<1x4x8xf32> to memref<*xf32>
  call @printMemrefF32(%A_print) : (memref<*xf32>) -> ()

  // This will be replaced by the ESP pass with the full call sequence
  linalg.batch_matmul ins(%A, %B : memref<1x4x8xf32>, memref<1x8x4xf32>)
                      outs(%C : memref<1x4x4xf32>)

  // Print output C
  %C_print = memref.cast %C : memref<1x4x4xf32> to memref<*xf32>
  call @printMemrefF32(%C_print) : (memref<*xf32>) -> ()

  // Cleanup
  memref.dealloc %A_dyn : memref<?x?x?xf32>
  memref.dealloc %B_dyn : memref<?x?x?xf32>
  memref.dealloc %C_dyn : memref<?x?x?xf32>

  return
}

// The mock runtime prints each step. The shapes come from decoding the
// (i64 rank, void *descriptor) pair a memref<*xf32> argument arrives as, so
// they are what checks that the runtime reads the descriptor MLIR passes. The
// offsets, strides and register values are the layout the pass computed for
// N=4 padded to 8: I@0 (ld K=8), W@32 (ld Npad=8), B@96, O@104.
// CHECK: esp_alloc_shared
// CHECK-NEXT: total_bytes=544
// CHECK: esp_float2fixed_f32
// CHECK-NEXT: src: rank=3, shape=1x4x8, elements=32
// CHECK-NEXT: offset=0, ld=8
// CHECK: esp_float2fixed_f32
// CHECK-NEXT: src: rank=3, shape=1x8x4, elements=32
// CHECK-NEXT: offset=32, ld=8
// CHECK: esp_accel_write_reg
// CHECK-NEXT: offset=0x58, value=4
// CHECK: esp_accel_write_reg
// CHECK-NEXT: offset=0x54, value=8
// CHECK: esp_accel_write_reg
// CHECK-NEXT: offset=0x50, value=8
// CHECK: esp_accel_write_reg
// CHECK-NEXT: offset=0x4c, value=0
// CHECK: esp_accel_write_reg
// CHECK-NEXT: offset=0x48, value=32
// CHECK: esp_accel_write_reg
// CHECK-NEXT: offset=0x44, value=96
// CHECK: esp_accel_write_reg
// CHECK-NEXT: offset=0x40, value=104
// CHECK: esp_accel_start
// CHECK: esp_accel_wait
// CHECK: esp_fixed2float_f32
// CHECK-NEXT: dst: rank=3, shape=1x4x4, elements=16
// CHECK-NEXT: offset=104, ld=8
// CHECK: esp_free_shared
