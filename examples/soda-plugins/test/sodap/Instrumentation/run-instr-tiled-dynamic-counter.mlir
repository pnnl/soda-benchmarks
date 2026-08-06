// RUN: mlir-opt %s \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(soda-instr-dynamic-counter{tracked-kinds=all})" | \
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
  %c1 = arith.constant 1 : index

  %c32 = arith.constant 32 : index
  %c16   = arith.constant 16 : index

  //Tile sizes
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index

  %f1 = arith.constant 1.0 : f32
  %f2 = arith.constant 2.0 : f32
  %i1 = arith.constant 1 : i32
  %i2 = arith.constant 2 : i32

  // 32x16 matrices.
  %A  = memref.alloc() : memref<32x16xf32>
  %B  = memref.alloc() : memref<32x16xf32>
  %IA = memref.alloc() : memref<32x16xi32>

  // Initialize A,B,IA.
  scf.for %i = %c0 to %c32 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      memref.store %f1, %A[%i, %j] : memref<32x16xf32>
      memref.store %f2, %B[%i, %j] : memref<32x16xf32>
      memref.store %i1, %IA[%i, %j] : memref<32x16xi32>
    }
  }

  // Tiled compute: tile sizes 16x16.
  // Per element: A = A*B + A; IA = (IA+2)*2.
  scf.for %ii = %c0 to %c32 step %c16 {
    scf.for %jj = %c0 to %c16 step %c16 {
      scf.for %i0 = %c0 to %c16 step %c1 {
        %i = arith.addi %ii, %i0 : index
        scf.for %j0 = %c0 to %c16 step %c1 {
          %j = arith.addi %jj, %j0 : index

          %a  = memref.load %A[%i, %j]  : memref<32x16xf32>
          %b  = memref.load %B[%i, %j]  : memref<32x16xf32>
          %fm = arith.mulf %a, %b : f32
          %fa = arith.addf %fm, %a : f32
          memref.store %fa, %A[%i, %j] : memref<32x16xf32>

          %iv = memref.load %IA[%i, %j] : memref<32x16xi32>
          %ia = arith.addi %iv, %i2 : i32
          %im = arith.muli %ia, %i2 : i32
          memref.store %im, %IA[%i, %j] : memref<32x16xi32>
        }
      }
    }
  }
  memref.dealloc %A  : memref<32x16xf32>
  memref.dealloc %B  : memref<32x16xf32>
  memref.dealloc %IA : memref<32x16xi32>
  return
}

// CHECK: --- Dynamic Counter Totals By Function (SDCF=SODA_DYNAMIC_COUNTER_FUNCTION) ---
// CHECK-DAG: SDCF function=main      name=memref.load        count=1536
// CHECK-DAG: SDCF function=main      name=memref.store       count=2560
// CHECK-DAG: SDCF function=main      name=arith.float        count=1024

// Expected dynamic totals ignoring "scf" and any extra index arith counted by your pass:
// N = 32*16 = 512
// init stores: 3N = 1536
// compute loads: 3N = 1536
// compute stores: 2N = 1024
// total stores: 5N = 2560
// arith.float: 2N = 1024
// arith.int: 2N = 1024
//
// CHECK: SODA_DYNAMIC_COUNTER name=memref.load count=1536
// CHECK: SODA_DYNAMIC_COUNTER name=memref.store count=2560
// CHECK: SODA_DYNAMIC_COUNTER name=arith.int count=1024
// CHECK: SODA_DYNAMIC_COUNTER name=arith.float count=1024
// CHECK: SODA_DYNAMIC_COUNTER name=scf

