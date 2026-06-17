// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-dynamic-counter{tracked-kinds=memref-load+memref-store+arith-int+arith-float})" | \
// RUN: mlir-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(convert-linalg-to-loops,lower-affine,convert-scf-to-cf,convert-arith-to-llvm),convert-vector-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" | \
// RUN: mlir-cpu-runner \
// RUN:   -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%sodap_libs/libmlir_sodap_instr_runner_utils%shlibext \
// RUN:   -shared-libs=%llvm_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%llvm_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @func_a_store() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %f1 = arith.constant 1.0 : f32

  %A = memref.alloc() : memref<1xf32>

  scf.for %i = %c0 to %c4 step %c1 {
    memref.store %f1, %A[%c0] : memref<1xf32>
  }

  memref.dealloc %A : memref<1xf32>
  return
}

func.func @func_b_float() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32

  %A = memref.alloc() : memref<1xf32>
  %B = memref.alloc() : memref<1xf32>

  memref.store %f1, %A[%c0] : memref<1xf32>
  memref.store %f2, %B[%c0] : memref<1xf32>

  scf.for %i = %c0 to %c4 step %c1 {
    %a = memref.load %A[%c0] : memref<1xf32>
    %b = memref.load %B[%c0] : memref<1xf32>
    %m = arith.mulf %a, %b : f32
    %s = arith.addf %m, %a : f32
    memref.store %s, %A[%c0] : memref<1xf32>
  }

  memref.dealloc %A : memref<1xf32>
  memref.dealloc %B : memref<1xf32>
  return
}

func.func @func_c_int() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32

  %I = memref.alloc() : memref<1xi32>

  // Keep input values defined without introducing extra top-level loops.
  memref.store %i1, %I[%c0] : memref<1xi32>

  scf.for %i = %c0 to %c4 step %c1 {
    %v = memref.load %I[%c0] : memref<1xi32>
    %a = arith.addi %v, %i2 : i32
    %m = arith.muli %a, %i2 : i32
    memref.store %m, %I[%c0] : memref<1xi32>
  }

  memref.dealloc %I : memref<1xi32>
  return
}

func.func @main() {
  call @func_a_store() : () -> ()
  call @func_b_float() : () -> ()
  call @func_c_int() : () -> ()
  return
}

// Each function has exactly one top-level loop group (group id 0), and the
// runtime should include the parent function name in the group header.

// CHECK: --- Loop Group 0 Function func_a_store ---
// CHECK-NEXT: SODA_DYNAMIC_COUNTER name=memref.store{{[[:space:]]+}}count=4

// CHECK: --- Loop Group 0 Function func_b_float ---
// CHECK-DAG: SODA_DYNAMIC_COUNTER name=memref.load{{[[:space:]]+}}count=8
// CHECK-DAG: SODA_DYNAMIC_COUNTER name=memref.store{{[[:space:]]+}}count=4
// CHECK-DAG: SODA_DYNAMIC_COUNTER name=arith.float{{[[:space:]]+}}count=8

// CHECK: --- Loop Group 0 Function func_c_int ---
// CHECK-DAG: SODA_DYNAMIC_COUNTER name=memref.load{{[[:space:]]+}}count=4
// CHECK-DAG: SODA_DYNAMIC_COUNTER name=memref.store{{[[:space:]]+}}count=4
// CHECK-DAG: SODA_DYNAMIC_COUNTER name=arith.int{{[[:space:]]+}}count=8
