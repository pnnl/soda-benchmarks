// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-dynamic-counter{tracked-kinds=memref-load+memref-store})" | \
// RUN:   FileCheck %s

module {
  func.func @instr_dynamic(%arg0: memref<4xf32>, %arg1: memref<4xf32>, %arg2: memref<4xi32>) {
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
// CHECK-DAG: func.func private @sodaInstrDynamicCounterSetGroupFunctionName(i64, i64)
// CHECK-DAG: func.func private @sodaInstrDynamicCounterFlush(i64)

// CHECK-LABEL: func.func @instr_dynamic(
  // CHECK: call @sodaInstrDynamicCounterSetGroupFunctionName({{.*}}) : (i64, i64) -> ()
  // CHECK: call @sodaInstrDynamicCounterStartGroup({{.*}}) : (i64) -> ()

  // CHECK: scf.for

  // load arg0 -> kind 0 (memref-load), delta 1
  // CHECK: %[[C0I64_0:.*]] = arith.constant 0 : i64
  // CHECK: %[[C1I64_0:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C0I64_0]], %[[C1I64_0]]) : (i64, i64) -> ()
  // CHECK: memref.load {{.*}} {soda.dynamic_counter.instrumented}

  // load arg1 -> kind 0 (memref-load), delta 1
  // CHECK: %[[C0I64_1:.*]] = arith.constant 0 : i64
  // CHECK: %[[C1I64_1:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C0I64_1]], %[[C1I64_1]]) : (i64, i64) -> ()
  // CHECK: memref.load {{.*}} {soda.dynamic_counter.instrumented}

  // arith.mulf and arith.addf are NOT instrumented (no arith tracking)
  // CHECK-NOT: arith.mulf {{.*}} {soda.dynamic_counter.instrumented}
  // CHECK-NOT: arith.addf {{.*}} {soda.dynamic_counter.instrumented}

  // store f32 -> kind 1 (memref-store), delta 1
  // CHECK: %[[C1I64_2:.*]] = arith.constant 1 : i64
  // CHECK: %[[C1I64_3:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C1I64_2]], %[[C1I64_3]]) : (i64, i64) -> ()
  // CHECK: memref.store {{.*}} {soda.dynamic_counter.instrumented}

  // load i32 -> kind 0 (memref-load), delta 1
  // CHECK: %[[C0I64_2:.*]] = arith.constant 0 : i64
  // CHECK: %[[C1I64_4:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C0I64_2]], %[[C1I64_4]]) : (i64, i64) -> ()
  // CHECK: memref.load {{.*}} {soda.dynamic_counter.instrumented}

  // arith.addi and arith.muli are NOT instrumented (no arith tracking)
  // CHECK-NOT: arith.addi {{.*}} {soda.dynamic_counter.instrumented}
  // CHECK-NOT: arith.muli {{.*}} {soda.dynamic_counter.instrumented}

  // store i32 -> kind 1 (memref-store), delta 1
  // CHECK: %[[C1I64_5:.*]] = arith.constant 1 : i64
  // CHECK: %[[C1I64_6:.*]] = arith.constant 1 : i64
  // CHECK: func.call @sodaInstrDynamicCounter(%[[C1I64_5]], %[[C1I64_6]]) : (i64, i64) -> ()
  // CHECK: memref.store {{.*}} {soda.dynamic_counter.instrumented}

  // CHECK: call @sodaInstrDynamicCounterFlush({{.*}}) : (i64) -> ()
