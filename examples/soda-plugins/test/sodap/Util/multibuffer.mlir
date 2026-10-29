// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter --canonicalize | FileCheck %s

// matmul tiled 4x4x4 with promoted tiles. The slot counts the tiles of the
// whole nest: (i/4)*12 + (j/4)*3 + k/4 for A and B, (i/4)*4 + j/4 for C, mod 2
// (the i terms are even and fold away). With upstream multibuffer A and B
// would use (k/4) mod 2, which gives slot 0 to both k = 8 and the next k = 0.

// CHECK-DAG: #[[CSLOT:.*]] = affine_map<(d0) -> ((d0 floordiv 4) mod 2)>
// CHECK-DAG: #[[ABSLOT:.*]] = affine_map<(d0, d1) -> ((d1 floordiv 4 + (d0 floordiv 4) * 3) mod 2)>
// CHECK-LABEL: func.func @matmul
// CHECK: %[[BC:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK: %[[BA:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK: %[[BB:.*]] = memref.alloc() : memref<2x4x4xf32>
// CHECK: scf.for %[[I:.*]] =
// CHECK:   scf.for %[[J:.*]] =
// CHECK:     %[[SC:.*]] = affine.apply #[[CSLOT]](%[[J]])
// CHECK:     %[[VC:.*]] = memref.subview %[[BC]][%[[SC]], 0, 0] [1, 4, 4] [1, 1, 1]
// CHECK:     linalg.fill {{.*}} outs(%[[VC]] :
// CHECK:     scf.for %[[K:.*]] =
// CHECK:       %[[SB:.*]] = affine.apply #[[ABSLOT]](%[[J]], %[[K]])
// CHECK-NEXT:  %[[VB:.*]] = memref.subview %[[BB]][%[[SB]], 0, 0] [1, 4, 4] [1, 1, 1]
// CHECK-NEXT:  %[[SA:.*]] = affine.apply #[[ABSLOT]](%[[J]], %[[K]])
// CHECK-NEXT:  %[[VA:.*]] = memref.subview %[[BA]][%[[SA]], 0, 0] [1, 4, 4] [1, 1, 1]
// CHECK:       memref.copy %{{.*}}, %[[VA]]
// CHECK:       linalg.matmul ins(%[[VA]], %[[VB]] : {{.*}}) outs(%[[VC]] :
// CHECK:     memref.copy %[[VC]], %{{.*}}
func.func @matmul(%A: memref<8x12xf32>, %B: memref<12x16xf32>, %C: memref<8x16xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %c12 = arith.constant 12 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  %bC = memref.alloc() : memref<4x4xf32>
  %bA = memref.alloc() : memref<4x4xf32>
  %bB = memref.alloc() : memref<4x4xf32>
  scf.for %i = %c0 to %c8 step %c4 {
    scf.for %j = %c0 to %c16 step %c4 {
      %Ct = memref.subview %C[%i, %j] [4, 4] [1, 1] : memref<8x16xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
      linalg.fill ins(%zero : f32) outs(%bC : memref<4x4xf32>)
      scf.for %k = %c0 to %c12 step %c4 {
        %At = memref.subview %A[%i, %k] [4, 4] [1, 1] : memref<8x12xf32> to memref<4x4xf32, strided<[12, 1], offset: ?>>
        %Bt = memref.subview %B[%k, %j] [4, 4] [1, 1] : memref<12x16xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
        memref.copy %At, %bA : memref<4x4xf32, strided<[12, 1], offset: ?>> to memref<4x4xf32>
        memref.copy %Bt, %bB : memref<4x4xf32, strided<[16, 1], offset: ?>> to memref<4x4xf32>
        linalg.matmul ins(%bA, %bB : memref<4x4xf32>, memref<4x4xf32>) outs(%bC : memref<4x4xf32>)
      }
      memref.copy %bC, %Ct : memref<4x4xf32> to memref<4x4xf32, strided<[16, 1], offset: ?>>
    }
  }
  return
}

// A scalar copied in once before the nest has nothing to alternate: its
// buffer stays as it is; the tile buffer next to it is doubled.
// CHECK-LABEL: func.func @scalar
// CHECK-DAG: %[[BS:.*]] = memref.alloc() : memref<f32>
// CHECK-DAG: memref.alloc() : memref<2x4xf32>
// CHECK: memref.copy %{{.*}}, %[[BS]] : memref<f32> to memref<f32>
// CHECK: scf.for
// CHECK: linalg.generic {{.*}} ins(%[[BS]] : memref<f32>)
func.func @scalar(%alpha: memref<f32>, %X: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %bS = memref.alloc() : memref<f32>
  %bX = memref.alloc() : memref<4xf32>
  memref.copy %alpha, %bS : memref<f32> to memref<f32>
  scf.for %i = %c0 to %c8 step %c4 {
    %Xt = memref.subview %X[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %Xt, %bX : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> ()>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
        ins(%bS : memref<f32>) outs(%bX : memref<4xf32>) {
    ^bb0(%a: f32, %x: f32):
      %0 = arith.mulf %a, %x : f32
      linalg.yield %0 : f32
    }
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %root : (!transform.any_op) -> !transform.op<"memref.alloc">
    %mb = transform.sodap.multibuffer %allocs {factor = 2 : i64} : (!transform.op<"memref.alloc">) -> !transform.any_op
    transform.yield
  }
}
