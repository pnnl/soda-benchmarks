// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-scf-for-assert-bounds)" |\
// RUN:   FileCheck %s 

module {
  // CHECK-LABEL: func.func @assert_for
  func.func @assert_for(%arg0: memref<8x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<8x8xf32>) -> memref<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c8 step %c1 {
      // CHECK: @sodaInstrAssertLessThen(%arg3, %c8) : (index, index) -> ()
      scf.for %arg4 = %c0 to %c8 step %c1 {
        // CHECK: @sodaInstrAssertLessThen(%arg4, %c8) : (index, index) -> ()
        scf.for %arg5 = %c0 to %c16 step %c1 {
          // CHECK: @sodaInstrAssertLessThen(%arg5, %c16) : (index, index) -> ()
          %0 = memref.load %arg0[%arg3, %arg5] : memref<8x16xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<16x8xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<8x8xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<8x8xf32>
        }
      }
    }
    return %arg2 : memref<8x8xf32>
  }
}
