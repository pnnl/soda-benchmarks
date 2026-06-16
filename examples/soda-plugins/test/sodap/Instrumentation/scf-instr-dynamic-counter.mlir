// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-dynamic-counter{tracked-kinds=memref-load+memref-store+arith-int+arith-float})" | \
// RUN:   FileCheck %s

module {
  func.func @instr_dynamic_counter(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index

    scf.for %iv = %c0 to %c4 step %c1 {
      %f0 = memref.load %arg0[%iv] : memref<4xf32>
      %f1 = memref.load %arg1[%iv] : memref<4xf32>
      %fm = arith.mulf %f0, %f1 : f32
      %fa = arith.addf %fm, %f0 : f32
      memref.store %fa, %arg0[%iv] : memref<4xf32>
      %i0 = memref.load %arg2[%iv] : memref<4xi32>
      %ia = arith.addi %i0, %i0 : i32
      %im = arith.muli %ia, %i0 : i32
      memref.store %im, %arg2[%iv] : memref<4xi32>
    }

    return
  }
}
// CHECK-DAG: func.func private @sodaInstrDynamicCounter(i64, i64)
// CHECK-DAG: func.func private @sodaInstrDynamicCounterStartGroup(i64)
// CHECK-DAG: func.func private @sodaInstrDynamicCounterFlush(i64)

// CHECK-LABEL: func.func @instr_dynamic_counter(
  // CHECK: %[[C0I64_START:.*]] = arith.constant 0 : i64
  // CHECK: call @sodaInstrDynamicCounterStartGroup(%[[C0I64_START]]) : (i64) -> ()

  // In the loop, for each instrumentation point:
  // CHECK: scf.for

  // load arg0 -> kind 0, count 1
  // CHECK: %[[C0I64_0:.*]] = arith.constant 0 : i64
  // CHECK: %[[C1I64_0:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C0I64_0]], %[[C1I64_0]]) : (i64, i64) -> ()
  // CHECK: memref.load {{.*}} {soda.dynamic_counter.instrumented}

  // load arg1 -> kind 0, count 1
  // CHECK: %[[C0I64_1:.*]] = arith.constant 0 : i64
  // CHECK: %[[C1I64_1:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C0I64_1]], %[[C1I64_1]]) : (i64, i64) -> ()
  // CHECK: memref.load {{.*}} {soda.dynamic_counter.instrumented}

  // mulf -> kind 3
  // CHECK: %[[C3I64_0:.*]] = arith.constant 3 : i64
  // CHECK: %[[C1I64_2:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C3I64_0]], %[[C1I64_2]]) : (i64, i64) -> ()
  // CHECK: arith.mulf {{.*}} {soda.dynamic_counter.instrumented}

  // addf -> kind 3
  // CHECK: %[[C3I64_1:.*]] = arith.constant 3 : i64
  // CHECK: %[[C1I64_3:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C3I64_1]], %[[C1I64_3]]) : (i64, i64) -> ()
  // CHECK: arith.addf {{.*}} {soda.dynamic_counter.instrumented}

  // store f32 -> kind 1
  // CHECK: %[[C1I64_4:.*]] = arith.constant 1 : i64
  // CHECK: %[[C1I64_5:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C1I64_4]], %[[C1I64_5]]) : (i64, i64) -> ()
  // CHECK: memref.store {{.*}} {soda.dynamic_counter.instrumented}

  // load i32 -> kind 0
  // CHECK: %[[C0I64_2:.*]] = arith.constant 0 : i64
  // CHECK: %[[C1I64_6:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C0I64_2]], %[[C1I64_6]]) : (i64, i64) -> ()
  // CHECK: memref.load {{.*}} {soda.dynamic_counter.instrumented}

  // addi -> kind 2
  // CHECK: %[[C2I64_0:.*]] = arith.constant 2 : i64
  // CHECK: %[[C1I64_7:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C2I64_0]], %[[C1I64_7]]) : (i64, i64) -> ()
  // CHECK: arith.addi {{.*}} {soda.dynamic_counter.instrumented}

  // muli -> kind 2
  // CHECK: %[[C2I64_1:.*]] = arith.constant 2 : i64
  // CHECK: %[[C1I64_8:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C2I64_1]], %[[C1I64_8]]) : (i64, i64) -> ()
  // CHECK: arith.muli {{.*}} {soda.dynamic_counter.instrumented}

  // store i32 -> kind 1
  // CHECK: %[[C1I64_9:.*]] = arith.constant 1 : i64
  // CHECK: %[[C1I64_10:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C1I64_9]], %[[C1I64_10]]) : (i64, i64) -> ()
  // CHECK: memref.store {{.*}} {soda.dynamic_counter.instrumented}

  // flush at end
  // CHECK: %[[C0I64_FLUSH:.*]] = arith.constant 0 : i64
  // CHECK: call @sodaInstrDynamicCounterFlush(%[[C0I64_FLUSH]]) : (i64) -> ()