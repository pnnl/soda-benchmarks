// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-hw-counters-at-loop-bounds{report-at-end=true})" |\
// RUN:   FileCheck %s

module {
  // CHECK: func.func private @sodaInstrHWCounters(i1, index)
  // CHECK-LABEL: func.func @instr_for
  func.func @instr_for(%arg0: memref<8xf32>) -> memref<8xf32> {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    scf.for %arg1 = %c0 to %c8 step %c1 {
      // CHECK: %[[START:.*]] = arith.constant true
      // CHECK: %[[STOP:.*]] = arith.constant false
      // CHECK: %[[LOC:.*]] = arith.constant 0 : index
      // CHECK: func.call @sodaInstrHWCounters(%[[START]], %[[LOC]]) : (i1, index) -> ()
      %0 = memref.load %arg0[%arg1] : memref<8xf32>
      memref.store %0, %arg0[%arg1] : memref<8xf32>
      // CHECK: func.call @sodaInstrHWCounters(%[[STOP]], %[[LOC]]) : (i1, index) -> ()
    }
    // A single finalize call is inserted right before the return, using the
    // report sentinel location (-1, all bits set) recognized by the hardware
    // IP as "print every counter's final value once".
    // CHECK: %[[REPORT_START:.*]] = arith.constant true
    // CHECK: %[[REPORT_LOC:.*]] = arith.constant -1 : index
    // CHECK: @sodaInstrHWCounters(%[[REPORT_START]], %[[REPORT_LOC]]) : (i1, index) -> ()
    return %arg0 : memref<8xf32>
  }
}
