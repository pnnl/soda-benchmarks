// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-hw-counters-at-loop-bounds)" |\
// RUN:   FileCheck %s 

module {
  // CHECK: func.func private @sodaInstrHWCounters(i1, index)
  // CHECK-LABEL: func.func @instr_for
  func.func @instr_for(%arg0: memref<8x16xf32>, %arg1: memref<16x8xf32>, %arg2: memref<8x8xf32>) -> memref<8x8xf32> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    scf.for %arg3 = %c0 to %c8 step %c1 {
      // CHECK: %[[START3:.*]] = arith.constant true
      // CHECK: %[[STOP3:.*]] = arith.constant false
      // CHECK: %[[LOC3:.*]] = arith.constant 2 : index
      // CHECK: func.call @sodaInstrHWCounters(%[[START3]], %[[LOC3]]) : (i1, index) -> ()
      scf.for %arg4 = %c0 to %c8 step %c1 {
        // CHECK: %[[START4:.*]] = arith.constant true
        // CHECK: %[[STOP4:.*]] = arith.constant false
        // CHECK: %[[LOC4:.*]] = arith.constant 1 : index
        // CHECK: func.call @sodaInstrHWCounters(%[[START4]], %[[LOC4]]) : (i1, index) -> ()
        scf.for %arg5 = %c0 to %c16 step %c1 {
          // CHECK: %[[START5:.*]] = arith.constant true
          // CHECK: %[[STOP5:.*]] = arith.constant false
          // CHECK: %[[LOC5:.*]] = arith.constant 0 : index
          // CHECK: func.call @sodaInstrHWCounters(%[[START5]], %[[LOC5]]) : (i1, index) -> ()
          %0 = memref.load %arg0[%arg3, %arg5] : memref<8x16xf32>
          %1 = memref.load %arg1[%arg5, %arg4] : memref<16x8xf32>
          %2 = memref.load %arg2[%arg3, %arg4] : memref<8x8xf32>
          %3 = arith.mulf %0, %1 : f32
          %4 = arith.addf %2, %3 : f32
          memref.store %4, %arg2[%arg3, %arg4] : memref<8x8xf32>
          // CHECK: func.call @sodaInstrHWCounters(%[[STOP5]], %[[LOC5]]) : (i1, index) -> ()
        }
        // CHECK: func.call @sodaInstrHWCounters(%[[STOP4]], %[[LOC4]]) : (i1, index) -> ()
      }
      // CHECK: func.call @sodaInstrHWCounters(%[[STOP3]], %[[LOC3]]) : (i1, index) -> ()
    }
    return %arg2 : memref<8x8xf32>
  }
}
