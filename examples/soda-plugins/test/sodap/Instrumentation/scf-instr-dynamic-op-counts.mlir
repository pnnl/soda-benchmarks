// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-dynamic-op-counts-at-loop-bounds)" | \
// RUN:   FileCheck %s

module {
  // CHECK: func.func private @sodaInstrCollectOpCounts(i64, index, i64, i64, i64, i64)
  // CHECK-LABEL: func.func @instr_dynamic
  func.func @instr_dynamic(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    scf.for %iv = %c0 to %c4 step %c1 {
      // CHECK: %[[RUN_START:.*]] = arith.constant 1 : i64
      // CHECK: %[[LOOP_ID:.*]] = arith.constant 0 : index
      // CHECK: %[[ZERO:.*]] = arith.constant 0 : i64
      // CHECK: func.call @sodaInstrCollectOpCounts(%[[RUN_START]], %[[LOOP_ID]], %[[ZERO]], %[[ZERO]], %[[ZERO]], %[[ZERO]]) : (i64, index, i64, i64, i64, i64) -> ()
      %f0 = memref.load %arg0[%iv] : memref<4xf32>
      %f1 = memref.load %arg1[%iv] : memref<4xf32>
      %fm = arith.mulf %f0, %f1 : f32
      %fa = arith.addf %fm, %f0 : f32
      memref.store %fa, %arg0[%iv] : memref<4xf32>

      %i0 = memref.load %arg2[%iv] : memref<4xi32>
      %ia = arith.addi %i0, %i0 : i32
      %im = arith.muli %ia, %i0 : i32
      memref.store %im, %arg2[%iv] : memref<4xi32>

      // CHECK: %[[RUN_STOP:.*]] = arith.constant 0 : i64
      // CHECK: %[[LOADS:.*]] = arith.constant 3 : i64
      // CHECK: %[[STORES:.*]] = arith.constant 2 : i64
      // CHECK: %[[FP:.*]] = arith.constant 2 : i64
      // CHECK: %[[INT:.*]] = arith.constant 2 : i64
      // CHECK: func.call @sodaInstrCollectOpCounts(%[[RUN_STOP]], %[[LOOP_ID]], %[[LOADS]], %[[STORES]], %[[FP]], %[[INT]]) : (i64, index, i64, i64, i64, i64) -> ()
    }

    return
  }
}
