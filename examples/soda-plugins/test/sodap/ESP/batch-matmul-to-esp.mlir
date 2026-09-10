// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp{vec-len=8},canonicalize)" | \
// RUN:   FileCheck %s
// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp{vec-len=8 profile=true},canonicalize)" | \
// RUN:   FileCheck %s --check-prefix=PROF
// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-linalg-batch-matmul-to-esp{vec-len=8 marshal=ir profile=true},canonicalize)" | \
// RUN:   FileCheck %s --check-prefix=IR

// Verify that all ESP runtime functions are declared.
// CHECK-DAG: func.func private @esp_alloc_shared(i64) -> i64
// CHECK-DAG: func.func private @esp_free_shared(i64)
// CHECK-DAG: func.func private @esp_float2fixed_f32(memref<*xf32>, i64, i64, i64)
// CHECK-DAG: func.func private @esp_fixed2float_f32(i64, i64, i64, memref<*xf32>)
// CHECK-DAG: func.func private @esp_accel_write_reg(i32, i32)
// CHECK-DAG: func.func private @esp_accel_start()
// CHECK-DAG: func.func private @esp_accel_wait()
// CHECK-NOT: func.func private @esp_prof_begin

// With profile=true the profiler is declared as well.
// PROF-DAG: func.func private @esp_prof_begin(i32)
// PROF-DAG: func.func private @esp_prof_end(i32)

// With marshal=ir the buffer is a pointer and there are no copy functions.
// IR-DAG: func.func private @esp_alloc_shared(i64) -> memref<i32>
// IR-DAG: func.func private @esp_free_shared(memref<i32>)
// IR-NOT: func.func private @esp_float2fixed_f32
// IR-NOT: func.func private @esp_fixed2float_f32

// -----

// N=4 is padded to Npad=8 (vec-len). Layout in elements:
//   I: M*K = 4*8 = 32       @ 0
//   W: K*Npad = 8*8 = 64    @ 32
//   B: Npad = 8             @ 96
//   O: M*Npad = 4*8 = 32    @ 104
//   total 136 elements = 544 bytes
//
// CHECK-LABEL: func.func @test_basic
// CHECK-NOT: linalg.batch_matmul
// CHECK-DAG: %[[A_UR:.*]] = memref.cast %arg0 : memref<1x4x8xf32> to memref<*xf32>
// CHECK-DAG: %[[B_UR:.*]] = memref.cast %arg1 : memref<1x8x4xf32> to memref<*xf32>
// CHECK-DAG: %[[C_UR:.*]] = memref.cast %arg2 : memref<1x4x4xf32> to memref<*xf32>
// Step 1: Allocate shared memory, sized for the padded layout
// CHECK: %[[MEM:.*]] = call @esp_alloc_shared(%c544_i64)
// Step 2: Copy inputs (float -> fixed-point). A rows are K=8 wide; B rows are
// stored Npad=8 apart.
// CHECK: call @esp_float2fixed_f32(%[[A_UR]], %[[MEM]], %c0_i64, %c8_i64)
// CHECK: call @esp_float2fixed_f32(%[[B_UR]], %[[MEM]], %c32_i64, %c8_i64)
// Step 3: Registers, one call each, with the padded outdim and the four bases
// CHECK: call @esp_accel_write_reg(%c88_i32, %c4_i32)
// CHECK: call @esp_accel_write_reg(%c84_i32, %c8_i32)
// CHECK: call @esp_accel_write_reg(%c80_i32, %c8_i32)
// CHECK: call @esp_accel_write_reg(%c76_i32, %c0_i32)
// CHECK: call @esp_accel_write_reg(%c72_i32, %c32_i32)
// CHECK: call @esp_accel_write_reg(%c68_i32, %c96_i32)
// CHECK: call @esp_accel_write_reg(%c64_i32, %c104_i32)
// Step 4/5: Start, wait
// CHECK: call @esp_accel_start()
// CHECK: call @esp_accel_wait()
// Step 6: Copy output (fixed-point -> float), reading rows Npad apart
// CHECK: call @esp_fixed2float_f32(%[[MEM]], %c104_i64, %c8_i64, %[[C_UR]])
// Step 7: Free shared memory
// CHECK: call @esp_free_shared(%[[MEM]])
// CHECK: return
//
// PROF-LABEL: func.func @test_basic
// PROF: call @esp_prof_begin(%c1_i32)
// PROF: call @esp_float2fixed_f32
// PROF: call @esp_float2fixed_f32
// PROF: call @esp_prof_end(%c1_i32)
// PROF: call @esp_prof_begin(%c2_i32)
// PROF: call @esp_accel_start()
// PROF: call @esp_accel_wait()
// PROF: call @esp_prof_end(%c2_i32)
// PROF: call @esp_prof_begin(%c3_i32)
// PROF: call @esp_fixed2float_f32
// PROF: call @esp_prof_end(%c3_i32)
// The single matmul of this block owns the epilogue: from here to the return.
// PROF: call @esp_free_shared
// PROF: call @esp_prof_begin(%c4_i32)
// PROF: call @esp_prof_end(%c4_i32)
// PROF: return
//
// With marshal=ir the same layout appears as strided views of the buffer, and
// the conversions as linalg.generic over them. W's view has rows Npad=8 apart
// and the padding columns are outside it; the free waits for the block's end.
// IR-LABEL: func.func @test_basic
// IR-NOT: linalg.batch_matmul
// IR: %[[MEM:.*]] = call @esp_alloc_shared(%c544_i64) : (i64) -> memref<i32>
// IR: %[[VI:.*]] = memref.reinterpret_cast %[[MEM]] to offset: [0], sizes: [1, 4, 8], strides: [32, 8, 1]
// IR: linalg.generic {{.*}} ins(%arg0 : memref<1x4x8xf32>) outs(%[[VI]] : memref<1x4x8xi32, strided<[32, 8, 1]>>)
// IR: arith.mulf
// IR: arith.fptosi
// IR: %[[VW:.*]] = memref.reinterpret_cast %[[MEM]] to offset: [32], sizes: [1, 8, 4], strides: [64, 8, 1]
// IR: linalg.generic {{.*}} ins(%arg1 : memref<1x8x4xf32>) outs(%[[VW]] : memref<1x8x4xi32, strided<[64, 8, 1], offset: 32>>)
// IR: call @esp_accel_write_reg(%c88_i32, %c4_i32)
// IR: call @esp_accel_write_reg(%c64_i32, %c104_i32)
// IR: call @esp_accel_start()
// IR: call @esp_accel_wait()
// IR: %[[VO:.*]] = memref.reinterpret_cast %[[MEM]] to offset: [104], sizes: [1, 4, 4], strides: [32, 8, 1]
// IR: linalg.generic {{.*}} ins(%[[VO]] : memref<1x4x4xi32, strided<[32, 8, 1], offset: 104>>) outs(%arg2 : memref<1x4x4xf32>)
// IR: arith.sitofp
// IR: arith.mulf
// IR-NOT: esp_float2fixed_f32
// IR-NOT: esp_fixed2float_f32
// IR: call @esp_prof_end(%c4_i32)
// IR: call @esp_free_shared(%[[MEM]]) : (memref<i32>) -> ()
// IR: return
func.func @test_basic(%A: memref<1x4x8xf32>, %B: memref<1x8x4xf32>,
                       %C: memref<1x4x4xf32>) -> memref<1x4x4xf32> {
  linalg.batch_matmul ins(%A, %B : memref<1x4x8xf32>, memref<1x8x4xf32>)
                      outs(%C : memref<1x4x4xf32>)
  return %C : memref<1x4x4xf32>
}

// Verify that two batch_matmul ops each get their own full call sequence,
// but the runtime functions are declared only once. N=16 is already a multiple
// of vec-len, so no padding: W is 32x16 @ 512, B 16 @ 1024, O 16x16 @ 1040.
// CHECK-LABEL: func.func @test_two_matmuls
// CHECK-NOT: linalg.batch_matmul
// CHECK: call @esp_alloc_shared
// CHECK: call @esp_float2fixed_f32
// CHECK: call @esp_float2fixed_f32
// CHECK-COUNT-7: call @esp_accel_write_reg
// CHECK: call @esp_accel_start
// CHECK: call @esp_accel_wait
// CHECK: call @esp_fixed2float_f32
// CHECK: call @esp_free_shared
// Second matmul:
// CHECK: call @esp_alloc_shared
// CHECK: call @esp_float2fixed_f32
// CHECK: call @esp_float2fixed_f32
// CHECK-COUNT-7: call @esp_accel_write_reg
// CHECK: call @esp_accel_start
// CHECK: call @esp_accel_wait
// CHECK: call @esp_fixed2float_f32
// CHECK: call @esp_free_shared
// CHECK: return
//
// Two matmuls in one block: no epilogue bracket, it would be ambiguous.
// PROF-LABEL: func.func @test_two_matmuls
// PROF-NOT: call @esp_prof_begin(%c4_i32)
// PROF: return
func.func @test_two_matmuls(%A: memref<1x16x32xf32>, %B: memref<1x32x16xf32>,
                             %C: memref<1x16x16xf32>, %D: memref<1x16x16xf32>)
    -> memref<1x16x16xf32> {
  linalg.batch_matmul ins(%A, %B : memref<1x16x32xf32>, memref<1x32x16xf32>)
                      outs(%C : memref<1x16x16xf32>)
  linalg.batch_matmul ins(%A, %B : memref<1x16x32xf32>, memref<1x32x16xf32>)
                      outs(%D : memref<1x16x16xf32>)
  return %D : memref<1x16x16xf32>
}
