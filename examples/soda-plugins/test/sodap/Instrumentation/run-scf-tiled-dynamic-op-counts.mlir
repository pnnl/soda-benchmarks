// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-dynamic-op-counts-at-loop-bounds)" | \
// RUN: mlir-opt \
// RUN:   --pass-pipeline="builtin.module(func.func(convert-linalg-to-loops,lower-affine,convert-scf-to-cf,convert-arith-to-llvm),convert-vector-to-llvm,finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" | \
// RUN: mlir-cpu-runner \
// RUN:   -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%sodap_libs/libmlir_sodap_instr_runner_utils%shlibext \
// RUN:   -shared-libs=%llvm_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%llvm_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func.func @main() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index

  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32

  %mfA = memref.alloc() : memref<4xf32>
  %mfB = memref.alloc() : memref<4xf32>
  %miA = memref.alloc() : memref<4xi32>

  // Initialize.
  scf.for %i = %c0 to %c4 step %c1 {
    memref.store %f1, %mfA[%i] : memref<4xf32>
    memref.store %f2, %mfB[%i] : memref<4xf32>
    memref.store %i1, %miA[%i] : memref<4xi32>
  }

  // Tiled loop: outer tiles of size 2, inner iterates within a tile.
  scf.for %ii = %c0 to %c4 step %c2 {
    scf.for %j = %c0 to %c2 step %c1 {
      %i = arith.addi %ii, %j : index

      %a = memref.load %mfA[%i] : memref<4xf32>
      %b = memref.load %mfB[%i] : memref<4xf32>
      %fm = arith.mulf %a, %b : f32
      %fa = arith.addf %fm, %a : f32
      memref.store %fa, %mfA[%i] : memref<4xf32>

      %iv = memref.load %miA[%i] : memref<4xi32>
      %ia = arith.addi %iv, %i2 : i32
      %im = arith.muli %ia, %i2 : i32
      memref.store %im, %miA[%i] : memref<4xi32>
    }
    memref.store %f1, %mfA[%ii] : memref<4xf32>
    // Note: the store above is outside the inner loop, so it doesn't contribute to the counts for the inner loop.
  }

  memref.dealloc %mfA : memref<4xf32>
  memref.dealloc %mfB : memref<4xf32>
  memref.dealloc %miA : memref<4xi32>
  return
}

// Counting the *tiled* nest:
// Inner body per iteration: 3 loads (A,B,iA), 2 stores (A,iA), 2 fp ops, 2 int ops.
// Inner loop runs 2 iters per tile, outer loop runs 2 tiles => 4 total iterations.
// => loads=12 stores=8 fp=8 int=8.
//
// Depending on your pass, you may see counters for both loops (outer and inner).
// The line below matches the loop that corresponds to the inner work-carrying loop.
// CHECK: SODA_COUNTER loop={{[0-9]+}} loads=12 stores=8 fp=8 int=8