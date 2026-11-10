// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-load-store))" -split-input-file -verify-diagnostics | FileCheck %s

memref.global "private" constant @mask : memref<4xf32> = dense<1.0>

// Two compute tasks, as bufferization leaves them: they read an argument and
// a constant straight from memory, and the result is copied into the
// out-param after the dispatch. Afterwards a load task copies what each
// compute task reads into buffers of its own (%a twice, one per task), and
// the store copies the yielded buffer into the out-param, in the dispatch.
// CHECK-LABEL: func.func @trmm_like
// CHECK-SAME:    (%[[A:.*]]: memref<4xf32>, %[[OUT:.*]]: memref<4xf32>)
// CHECK:         %[[MASK:.*]] = memref.get_global @mask
// CHECK:         %[[R:.*]] = dataflow.dispatch
// CHECK:           %[[A1:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[M1:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           %[[A2:.*]] = memref.alloc() : memref<4xf32>
// CHECK:           dataflow.task attributes {load} {
// CHECK-NEXT:        memref.copy %[[A]], %[[A1]]
// CHECK-NEXT:        memref.copy %[[MASK]], %[[M1]]
// CHECK-NEXT:        memref.copy %[[A]], %[[A2]]
// CHECK:           dataflow.task {
// CHECK:             linalg.mul ins(%[[A1]], %[[M1]] : {{.*}}) outs(%[[T:.*]] : {{.*}})
// CHECK:           dataflow.task {
// CHECK:             linalg.add ins(%[[T]], %[[A2]] : {{.*}}) outs(%[[O:.*]] : {{.*}})
// CHECK:           dataflow.task attributes {store} {
// CHECK-NEXT:        memref.copy %[[O]], %[[OUT]]
// CHECK:           dataflow.yield %[[O]]
// CHECK-NOT:     memref.copy
// CHECK:         return
func.func @trmm_like(%a: memref<4xf32>, %out: memref<4xf32>) {
  %mask = memref.get_global @mask : memref<4xf32>
  %r = dataflow.dispatch : memref<4xf32> {
    %t = memref.alloc() : memref<4xf32>
    dataflow.task {
      linalg.mul ins(%a, %mask : memref<4xf32>, memref<4xf32>) outs(%t : memref<4xf32>)
      dataflow.yield
    }
    %o = memref.alloc() : memref<4xf32>
    dataflow.task {
      linalg.add ins(%t, %a : memref<4xf32>, memref<4xf32>) outs(%o : memref<4xf32>)
      dataflow.yield
    }
    dataflow.yield %o : memref<4xf32>
  }
  memref.copy %r, %out : memref<4xf32> to memref<4xf32>
  return
}

// -----

// One output only, for now.
// expected-error @below {{expected one output: more aren't supported yet}}
func.func @two_outputs(%a: memref<4xf32>, %out0: memref<4xf32>, %out1: memref<4xf32>) {
  %r:2 = dataflow.dispatch : memref<4xf32>, memref<4xf32> {
    %o0 = memref.alloc() : memref<4xf32>
    %o1 = memref.alloc() : memref<4xf32>
    dataflow.task {
      linalg.copy ins(%a : memref<4xf32>) outs(%o0 : memref<4xf32>)
      linalg.copy ins(%a : memref<4xf32>) outs(%o1 : memref<4xf32>)
      dataflow.yield
    }
    dataflow.yield %o0, %o1 : memref<4xf32>, memref<4xf32>
  }
  memref.copy %r#0, %out0 : memref<4xf32> to memref<4xf32>
  memref.copy %r#1, %out1 : memref<4xf32> to memref<4xf32>
  return
}

// -----

// A compute task writing an argument would write a load buffer instead, and
// the write would be lost: only the store writes memory.
func.func @compute_writes_memory(%a: memref<8xf32>, %v: f32) {
  dataflow.dispatch {
    // expected-error @below {{expected only the store to write memory}}
    dataflow.task {
      %s = memref.subview %a[0] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1]>>
      // expected-note @below {{write}}
      linalg.fill ins(%v : f32) outs(%s : memref<4xf32, strided<[1]>>)
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// A tiled writer accumulating into the result inside k, over (i, j): the
// store copies a tile per (i, j), its writer's prefix, once the tile is final.
// CHECK-LABEL: func.func @tiled_store
// CHECK-SAME:    (%[[A:.*]]: memref<8x8xf32>, %[[OUT:.*]]: memref<8x8xf32>)
// CHECK:         dataflow.dispatch
// CHECK:           memref.copy %[[A]]
// CHECK:           %[[O:.*]] = memref.alloc() : memref<8x8xf32>
// CHECK-NEXT:      dataflow.task {
// CHECK:             linalg.add
// CHECK:           dataflow.task attributes {store} {
// CHECK-NEXT:        %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[C8:.*]] = arith.constant 8 : index
// CHECK-NEXT:        %[[C4:.*]] = arith.constant 4 : index
// CHECK-NEXT:        scf.for %[[I:.*]] = %[[C0]] to %[[C8]] step %[[C4]] {
// CHECK:               scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK-NEXT:            %[[S:.*]] = memref.subview %[[O]][%[[I]], %[[J]]] [4, 4] [1, 1]
// CHECK-NEXT:            %[[T:.*]] = memref.subview %[[OUT]][%[[I]], %[[J]]] [4, 4] [1, 1]
// CHECK-NEXT:            memref.copy %[[S]], %[[T]]
func.func @tiled_store(%a: memref<8x8xf32>, %out: memref<8x8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %r = dataflow.dispatch : memref<8x8xf32> {
    %o = memref.alloc() : memref<8x8xf32>
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        scf.for %j = %c0 to %c8 step %c4 {
          scf.for %k = %c0 to %c8 step %c4 {
            %x = memref.subview %a[%i, %k] [4, 4] [1, 1] : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1], offset: ?>>
            %y = memref.subview %o[%i, %j] [4, 4] [1, 1] : memref<8x8xf32> to memref<4x4xf32, strided<[8, 1], offset: ?>>
            linalg.add ins(%x, %y : memref<4x4xf32, strided<[8, 1], offset: ?>>, memref<4x4xf32, strided<[8, 1], offset: ?>>) outs(%y : memref<4x4xf32, strided<[8, 1], offset: ?>>)
          }
        }
      }
      dataflow.yield
    }
    dataflow.yield %o : memref<8x8xf32>
  }
  memref.copy %r, %out : memref<8x8xf32> to memref<8x8xf32>
  return
}
