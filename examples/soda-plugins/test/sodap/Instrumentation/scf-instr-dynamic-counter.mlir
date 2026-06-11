// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-dynamic-counter{tracked-kinds=memref-load+memref-store+arith-int+arith-float})" | \
// RUN:   FileCheck %s

module {
  // CHECK: func.func private @sodaInstrDynamicCounter(i64, i64)
  // CHECK-LABEL: func.func @instr_dynamic_counter
  func.func @instr_dynamic_counter(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    scf.for %iv = %c0 to %c4 step %c1 {
      // CHECK: %[[LOAD_ID:.*]] = arith.constant 0 : i64
      // CHECK: %[[ONE0:.*]] = arith.constant 1 : i64
      // CHECK: func.call @sodaInstrDynamicCounter(%[[LOAD_ID]], %[[ONE0]]) : (i64, i64) -> ()
      %f0 = memref.load %arg0[%iv] : memref<4xf32>

      // CHECK: %[[LOAD_ID_2:.*]] = arith.constant 0 : i64
      // CHECK: %[[ONE1:.*]] = arith.constant 1 : i64
      // CHECK: func.call @sodaInstrDynamicCounter(%[[LOAD_ID_2]], %[[ONE1]]) : (i64, i64) -> ()
      %f1 = memref.load %arg1[%iv] : memref<4xf32>

      // CHECK: %[[FP_ID:.*]] = arith.constant 3 : i64
      // CHECK: %[[ONE2:.*]] = arith.constant 1 : i64
      // CHECK: func.call @sodaInstrDynamicCounter(%[[FP_ID]], %[[ONE2]]) : (i64, i64) -> ()
      %fm = arith.mulf %f0, %f1 : f32

      // CHECK: %[[FP_ID_2:.*]] = arith.constant 3 : i64
      // CHECK: %[[ONE3:.*]] = arith.constant 1 : i64
      // CHECK: func.call @sodaInstrDynamicCounter(%[[FP_ID_2]], %[[ONE3]]) : (i64, i64) -> ()
      %fa = arith.addf %fm, %f0 : f32

      // CHECK: %[[STORE_ID:.*]] = arith.constant 1 : i64
      // CHECK: %[[ONE4:.*]] = arith.constant 1 : i64
      // CHECK: func.call @sodaInstrDynamicCounter(%[[STORE_ID]], %[[ONE4]]) : (i64, i64) -> ()
      memref.store %fa, %arg0[%iv] : memref<4xf32>

      // CHECK: %[[LOAD_ID_3:.*]] = arith.constant 0 : i64
      // CHECK: %[[ONE5:.*]] = arith.constant 1 : i64
      // CHECK: func.call @sodaInstrDynamicCounter(%[[LOAD_ID_3]], %[[ONE5]]) : (i64, i64) -> ()
      %i0 = memref.load %arg2[%iv] : memref<4xi32>

      // CHECK: %[[INT_ID:.*]] = arith.constant 2 : i64
      // CHECK: %[[ONE6:.*]] = arith.constant 1 : i64
      // CHECK: func.call @sodaInstrDynamicCounter(%[[INT_ID]], %[[ONE6]]) : (i64, i64) -> ()
      %ia = arith.addi %i0, %i0 : i32

      // CHECK: %[[INT_ID_2:.*]] = arith.constant 2 : i64
      // CHECK: %[[ONE7:.*]] = arith.constant 1 : i64
      // CHECK: func.call @sodaInstrDynamicCounter(%[[INT_ID_2]], %[[ONE7]]) : (i64, i64) -> ()
      %im = arith.muli %ia, %i0 : i32

      // CHECK: %[[STORE_ID_2:.*]] = arith.constant 1 : i64
      // CHECK: %[[ONE8:.*]] = arith.constant 1 : i64
      // CHECK: func.call @sodaInstrDynamicCounter(%[[STORE_ID_2]], %[[ONE8]]) : (i64, i64) -> ()
      memref.store %im, %arg2[%iv] : memref<4xi32>
    }

    return
  }
}
