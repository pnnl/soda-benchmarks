// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-schedule))" -split-input-file -verify-diagnostics | FileCheck %s

// trmm, default partition: A_lower, transpose, matmul, epilogue, store. One
// nest, the store's: each tile read from another task is preceded by that
// task's nest for the tile, its tile loops pinned to it. Per tile (i, j) of
// the result: alpha for the epilogue, then per k the matmul's operands: the
// A_lower tile (k, i) behind the transpose's (i, k), and B (k, j). Every task
// keeps its ops of the nest: A_lower and the transpose run per (i, j, k).
// CHECK-LABEL: func.func @trmm_default
// CHECK-SAME:    (%[[ALPHA:.*]]: memref<f32>, %[[A:.*]]: memref<60x60xf32>, %[[B:.*]]: memref<60x80xf32>,
// CHECK:         %[[MASK:.*]] = memref.get_global @mask
// CHECK:         %[[EYE:.*]] = memref.get_global @eye
// CHECK:         dataflow.dispatch
// CHECK:           %[[LA:.*]] = memref.alloc() : memref<60x60xf32>
// CHECK:           %[[LMASK:.*]] = memref.alloc() : memref<60x60xf32>
// CHECK:           %[[LEYE:.*]] = memref.alloc() : memref<60x60xf32>
// CHECK:           %[[LB:.*]] = memref.alloc() : memref<60x80xf32>
// CHECK:           %[[LALPHA:.*]] = memref.alloc() : memref<f32>
// CHECK:           dataflow.task {
// CHECK-NEXT:        scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT:          scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT:            memref.copy %[[ALPHA]], %[[LALPHA]]
// CHECK-NEXT:            %[[COL:.*]] = memref.subview %[[LB]][0, %[[J]]] [60, 10] [1, 1]
// CHECK-NEXT:            scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT:              %[[TB:.*]] = memref.subview %[[COL]][%[[K]], 0] [10, 10] [1, 1]
// CHECK-NEXT:              %[[TA:.*]] = memref.subview %[[LA]][%[[K]], %[[I]]] [10, 10] [1, 1]
// CHECK-NEXT:              %[[TMASK:.*]] = memref.subview %[[LMASK]][%[[K]], %[[I]]] [10, 10] [1, 1]
// CHECK-NEXT:              %[[TEYE:.*]] = memref.subview %[[LEYE]][%[[K]], %[[I]]] [10, 10] [1, 1]
// CHECK-NEXT:              %[[SA:.*]] = memref.subview %[[A]][%[[K]], %[[I]]] [10, 10] [1, 1]
// CHECK-NEXT:              memref.copy %[[SA]], %[[TA]]
// CHECK-NEXT:              %[[SMASK:.*]] = memref.subview %[[MASK]][%[[K]], %[[I]]] [10, 10] [1, 1]
// CHECK-NEXT:              memref.copy %[[SMASK]], %[[TMASK]]
// CHECK-NEXT:              %[[SEYE:.*]] = memref.subview %[[EYE]][%[[K]], %[[I]]] [10, 10] [1, 1]
// CHECK-NEXT:              memref.copy %[[SEYE]], %[[TEYE]]
// CHECK-NEXT:              %[[SCOL:.*]] = memref.subview %[[B]][0, %[[J]]] [60, 10] [1, 1]
// CHECK-NEXT:              %[[SB:.*]] = memref.subview %[[SCOL]][%[[K]], 0] [10, 10] [1, 1]
// CHECK-NEXT:              memref.copy %[[SB]], %[[TB]]
// CHECK-NEXT:            }
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK:           dataflow.task {
// CHECK-NEXT:        scf.for %[[I1:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK-NEXT:            scf.for %[[K1:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT:              memref.subview %[[LA]][%[[K1]], %[[I1]]] [10, 10] [1, 1]
// CHECK:                   linalg.generic
// CHECK:           dataflow.task {
// CHECK-NEXT:        scf.for %[[I2:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT:          scf.for %{{.*}} = %{{.*}} to %{{.*}} step
// CHECK-NEXT:            scf.for %[[K2:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT:              memref.subview %{{.*}}[%[[K2]], %[[I2]]] [10, 10] [1, 1]
// CHECK-NEXT:              memref.subview %{{.*}}[%[[I2]], %[[K2]]] [10, 10] [1, 1]
// CHECK-NEXT:              linalg.transpose
// CHECK:           dataflow.task {
// CHECK-NEXT:        scf.for
// CHECK-NEXT:          scf.for
// CHECK:                 linalg.fill
// CHECK-NEXT:            scf.for
// CHECK:                   linalg.matmul
// CHECK:           dataflow.task {
// CHECK-NEXT:        scf.for
// CHECK-NEXT:          scf.for
// CHECK:                 linalg.generic
// CHECK:           dataflow.task {
// CHECK-NEXT:        scf.for
// CHECK-NEXT:          scf.for
// CHECK-NEXT:            memref.subview
// CHECK-NEXT:            memref.subview
// CHECK-NEXT:            memref.copy
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
memref.global "private" constant @mask : memref<60x60xf32> = dense<1.0>
memref.global "private" constant @eye : memref<60x60xf32> = dense<0.0>
func.func @trmm_default(%arg0: memref<f32>, %arg1: memref<60x60xf32>, %arg2: memref<60x80xf32>, %arg3: memref<60x80xf32>) {
  %c80 = arith.constant 80 : index
  %c10 = arith.constant 10 : index
  %c60 = arith.constant 60 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @mask : memref<60x60xf32>
  %1 = memref.get_global @eye : memref<60x60xf32>
  dataflow.dispatch {
    %alloc = memref.alloc() : memref<60x60xf32>
    %alloc_0 = memref.alloc() : memref<60x60xf32>
    %alloc_1 = memref.alloc() : memref<60x60xf32>
    %alloc_2 = memref.alloc() : memref<60x80xf32>
    %alloc_3 = memref.alloc() : memref<f32>
    dataflow.task {
      memref.copy %arg1, %alloc : memref<60x60xf32> to memref<60x60xf32>
      memref.copy %0, %alloc_0 : memref<60x60xf32> to memref<60x60xf32>
      memref.copy %1, %alloc_1 : memref<60x60xf32> to memref<60x60xf32>
      memref.copy %arg2, %alloc_2 : memref<60x80xf32> to memref<60x80xf32>
      memref.copy %arg0, %alloc_3 : memref<f32> to memref<f32>
      dataflow.yield
    }
    %alloc_4 = memref.alloc() : memref<60x60xf32>
    dataflow.task {
      scf.for %arg4 = %c0 to %c60 step %c10 {
        scf.for %arg5 = %c0 to %c60 step %c10 {
          %subview = memref.subview %alloc[%arg4, %arg5] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          %subview_8 = memref.subview %alloc_0[%arg4, %arg5] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          %subview_9 = memref.subview %alloc_1[%arg4, %arg5] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          %subview_10 = memref.subview %alloc_4[%arg4, %arg5] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_8, %subview_9 : memref<10x10xf32, strided<[60, 1], offset: ?>>, memref<10x10xf32, strided<[60, 1], offset: ?>>, memref<10x10xf32, strided<[60, 1], offset: ?>>) outs(%subview_10 : memref<10x10xf32, strided<[60, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_11: f32, %in_12: f32, %out: f32):
            %3 = arith.mulf %in, %in_11 : f32
            %4 = arith.addf %3, %in_12 : f32
            linalg.yield %4 : f32
          }
        }
      }
      dataflow.yield
    }
    %alloc_5 = memref.alloc() : memref<60x60xf32>
    dataflow.task {
      scf.for %arg4 = %c0 to %c60 step %c10 {
        scf.for %arg5 = %c0 to %c60 step %c10 {
          %subview = memref.subview %alloc_4[%arg5, %arg4] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          %subview_8 = memref.subview %alloc_5[%arg4, %arg5] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          linalg.transpose ins(%subview : memref<10x10xf32, strided<[60, 1], offset: ?>>) outs(%subview_8 : memref<10x10xf32, strided<[60, 1], offset: ?>>) permutation = [1, 0]
        }
      }
      dataflow.yield
    }
    %alloc_6 = memref.alloc() : memref<60x80xf32>
    dataflow.task {
      scf.for %arg4 = %c0 to %c60 step %c10 {
        scf.for %arg5 = %c0 to %c80 step %c10 {
          %subview = memref.subview %alloc_5[%arg4, 0] [10, 60] [1, 1] : memref<60x60xf32> to memref<10x60xf32, strided<[60, 1], offset: ?>>
          %subview_8 = memref.subview %alloc_2[0, %arg5] [60, 10] [1, 1] : memref<60x80xf32> to memref<60x10xf32, strided<[80, 1], offset: ?>>
          %subview_9 = memref.subview %alloc_6[%arg4, %arg5] [10, 10] [1, 1] : memref<60x80xf32> to memref<10x10xf32, strided<[80, 1], offset: ?>>
          linalg.fill ins(%cst : f32) outs(%subview_9 : memref<10x10xf32, strided<[80, 1], offset: ?>>)
          scf.for %arg6 = %c0 to %c60 step %c10 {
            %subview_10 = memref.subview %subview[0, %arg6] [10, 10] [1, 1] : memref<10x60xf32, strided<[60, 1], offset: ?>> to memref<10x10xf32, strided<[60, 1], offset: ?>>
            %subview_11 = memref.subview %subview_8[%arg6, 0] [10, 10] [1, 1] : memref<60x10xf32, strided<[80, 1], offset: ?>> to memref<10x10xf32, strided<[80, 1], offset: ?>>
            linalg.matmul ins(%subview_10, %subview_11 : memref<10x10xf32, strided<[60, 1], offset: ?>>, memref<10x10xf32, strided<[80, 1], offset: ?>>) outs(%subview_9 : memref<10x10xf32, strided<[80, 1], offset: ?>>)
          }
        }
      }
      dataflow.yield
    }
    %alloc_7 = memref.alloc() : memref<60x80xf32>
    dataflow.task {
      scf.for %arg4 = %c0 to %c60 step %c10 {
        scf.for %arg5 = %c0 to %c80 step %c10 {
          %subview = memref.subview %alloc_6[%arg4, %arg5] [10, 10] [1, 1] : memref<60x80xf32> to memref<10x10xf32, strided<[80, 1], offset: ?>>
          %subview_8 = memref.subview %alloc_7[%arg4, %arg5] [10, 10] [1, 1] : memref<60x80xf32> to memref<10x10xf32, strided<[80, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_3, %subview : memref<f32>, memref<10x10xf32, strided<[80, 1], offset: ?>>) outs(%subview_8 : memref<10x10xf32, strided<[80, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_9: f32, %out: f32):
            %3 = arith.mulf %in, %in_9 : f32
            linalg.yield %3 : f32
          }
        }
      }
      dataflow.yield
    }
    dataflow.task {
      scf.for %arg4 = %c0 to %c60 step %c10 {
        scf.for %arg5 = %c0 to %c80 step %c10 {
          %subview = memref.subview %alloc_7[%arg4, %arg5] [10, 10] [1, 1] : memref<60x80xf32> to memref<10x10xf32, strided<[80, 1], offset: ?>>
          %subview_1 = memref.subview %arg3[%arg4, %arg5] [10, 10] [1, 1] : memref<60x80xf32> to memref<10x10xf32, strided<[80, 1], offset: ?>>
          memref.copy %subview, %subview_1 : memref<10x10xf32, strided<[80, 1], offset: ?>> to memref<10x10xf32, strided<[80, 1], offset: ?>>
        }
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// trmm, plan partition: the transpose in the task of its producer, the
// epilogue in the matmul's. The same loads per k; alpha comes after the k
// loop, where the epilogue reads it. The matmul task's accumulator stays at
// its start.
// CHECK-LABEL: func.func @trmm_plan
// CHECK-SAME:    (%[[ALPHA:.*]]: memref<f32>, %[[A:.*]]: memref<60x60xf32>, %[[B:.*]]: memref<60x80xf32>,
// CHECK:           %[[LA:.*]] = memref.alloc() : memref<60x60xf32>
// CHECK:           %[[LALPHA:.*]] = memref.alloc() : memref<f32>
// CHECK:           dataflow.task {
// CHECK-NEXT:        scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT:          scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step
// CHECK-NEXT:            memref.subview
// CHECK-NEXT:            scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step
// CHECK:                   %[[SA:.*]] = memref.subview %[[A]][%[[K]], %[[I]]] [10, 10] [1, 1]
// CHECK-NEXT:              memref.copy %[[SA]], %{{.*}}
// CHECK-COUNT-3:           memref.copy
// CHECK-NEXT:            }
// CHECK-NEXT:            memref.copy %[[ALPHA]], %[[LALPHA]]
// CHECK-NEXT:          }
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK:           dataflow.task {
// CHECK-NEXT:        scf.for
// CHECK-NEXT:          scf.for
// CHECK-NEXT:            scf.for
// CHECK:                   linalg.transpose
// CHECK:           dataflow.task {
// CHECK-NEXT:        memref.alloc() : memref<60x80xf32>
// CHECK-NEXT:        scf.for
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> ()>
memref.global "private" constant @mask : memref<60x60xf32> = dense<1.0>
memref.global "private" constant @eye : memref<60x60xf32> = dense<0.0>
func.func @trmm_plan(%arg0: memref<f32>, %arg1: memref<60x60xf32>, %arg2: memref<60x80xf32>, %arg3: memref<60x80xf32>) {
  %c80 = arith.constant 80 : index
  %c10 = arith.constant 10 : index
  %c60 = arith.constant 60 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.get_global @mask : memref<60x60xf32>
  %1 = memref.get_global @eye : memref<60x60xf32>
  dataflow.dispatch {
    %alloc = memref.alloc() : memref<60x60xf32>
    %alloc_0 = memref.alloc() : memref<60x60xf32>
    %alloc_1 = memref.alloc() : memref<60x60xf32>
    %alloc_2 = memref.alloc() : memref<60x80xf32>
    %alloc_3 = memref.alloc() : memref<f32>
    dataflow.task {
      memref.copy %arg1, %alloc : memref<60x60xf32> to memref<60x60xf32>
      memref.copy %0, %alloc_0 : memref<60x60xf32> to memref<60x60xf32>
      memref.copy %1, %alloc_1 : memref<60x60xf32> to memref<60x60xf32>
      memref.copy %arg2, %alloc_2 : memref<60x80xf32> to memref<60x80xf32>
      memref.copy %arg0, %alloc_3 : memref<f32> to memref<f32>
      dataflow.yield
    }
    %alloc_5 = memref.alloc() : memref<60x60xf32>
    dataflow.task {
      scf.for %arg4 = %c0 to %c60 step %c10 {
        scf.for %arg5 = %c0 to %c60 step %c10 {
          %subview = memref.subview %alloc[%arg5, %arg4] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          %subview_8 = memref.subview %alloc_0[%arg5, %arg4] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          %subview_9 = memref.subview %alloc_1[%arg5, %arg4] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          %tmp = memref.alloc() : memref<10x10xf32>
          linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%subview, %subview_8, %subview_9 : memref<10x10xf32, strided<[60, 1], offset: ?>>, memref<10x10xf32, strided<[60, 1], offset: ?>>, memref<10x10xf32, strided<[60, 1], offset: ?>>) outs(%tmp : memref<10x10xf32>) {
          ^bb0(%in: f32, %in_11: f32, %in_12: f32, %out: f32):
            %3 = arith.mulf %in, %in_11 : f32
            %4 = arith.addf %3, %in_12 : f32
            linalg.yield %4 : f32
          }
          %subview_10 = memref.subview %alloc_5[%arg4, %arg5] [10, 10] [1, 1] : memref<60x60xf32> to memref<10x10xf32, strided<[60, 1], offset: ?>>
          linalg.transpose ins(%tmp : memref<10x10xf32>) outs(%subview_10 : memref<10x10xf32, strided<[60, 1], offset: ?>>) permutation = [1, 0]
        }
      }
      dataflow.yield
    }
    %alloc_7 = memref.alloc() : memref<60x80xf32>
    dataflow.task {
      %acc = memref.alloc() : memref<60x80xf32>
      scf.for %arg4 = %c0 to %c60 step %c10 {
        scf.for %arg5 = %c0 to %c80 step %c10 {
          %subview = memref.subview %alloc_5[%arg4, 0] [10, 60] [1, 1] : memref<60x60xf32> to memref<10x60xf32, strided<[60, 1], offset: ?>>
          %subview_8 = memref.subview %alloc_2[0, %arg5] [60, 10] [1, 1] : memref<60x80xf32> to memref<60x10xf32, strided<[80, 1], offset: ?>>
          %subview_9 = memref.subview %acc[%arg4, %arg5] [10, 10] [1, 1] : memref<60x80xf32> to memref<10x10xf32, strided<[80, 1], offset: ?>>
          linalg.fill ins(%cst : f32) outs(%subview_9 : memref<10x10xf32, strided<[80, 1], offset: ?>>)
          scf.for %arg6 = %c0 to %c60 step %c10 {
            %subview_10 = memref.subview %subview[0, %arg6] [10, 10] [1, 1] : memref<10x60xf32, strided<[60, 1], offset: ?>> to memref<10x10xf32, strided<[60, 1], offset: ?>>
            %subview_11 = memref.subview %subview_8[%arg6, 0] [10, 10] [1, 1] : memref<60x10xf32, strided<[80, 1], offset: ?>> to memref<10x10xf32, strided<[80, 1], offset: ?>>
            linalg.matmul ins(%subview_10, %subview_11 : memref<10x10xf32, strided<[60, 1], offset: ?>>, memref<10x10xf32, strided<[80, 1], offset: ?>>) outs(%subview_9 : memref<10x10xf32, strided<[80, 1], offset: ?>>)
          }
          %subview_12 = memref.subview %alloc_7[%arg4, %arg5] [10, 10] [1, 1] : memref<60x80xf32> to memref<10x10xf32, strided<[80, 1], offset: ?>>
          linalg.generic {indexing_maps = [#map1, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_3, %subview_9 : memref<f32>, memref<10x10xf32, strided<[80, 1], offset: ?>>) outs(%subview_12 : memref<10x10xf32, strided<[80, 1], offset: ?>>) {
          ^bb0(%in: f32, %in_9: f32, %out: f32):
            %3 = arith.mulf %in, %in_9 : f32
            linalg.yield %3 : f32
          }
        }
      }
      dataflow.yield
    }
    dataflow.task {
      scf.for %arg4 = %c0 to %c60 step %c10 {
        scf.for %arg5 = %c0 to %c80 step %c10 {
          %subview = memref.subview %alloc_7[%arg4, %arg5] [10, 10] [1, 1] : memref<60x80xf32> to memref<10x10xf32, strided<[80, 1], offset: ?>>
          %subview_1 = memref.subview %arg3[%arg4, %arg5] [10, 10] [1, 1] : memref<60x80xf32> to memref<10x10xf32, strided<[80, 1], offset: ?>>
          memref.copy %subview, %subview_1 : memref<10x10xf32, strided<[80, 1], offset: ?>> to memref<10x10xf32, strided<[80, 1], offset: ?>>
        }
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// A compute task whose result nothing reads: the store doesn't depend on it.
func.func @dead_task(%a: memref<4xf32>, %out: memref<4xf32>) {
  dataflow.dispatch {
    %in = memref.alloc() : memref<4xf32>
    dataflow.task {
      memref.copy %a, %in : memref<4xf32> to memref<4xf32>
      dataflow.yield
    }
    %dead = memref.alloc() : memref<4xf32>
    // expected-error @below {{expected a compute task the store depends on}}
    dataflow.task {
      linalg.copy ins(%in : memref<4xf32>) outs(%dead : memref<4xf32>)
      dataflow.yield
    }
    %o = memref.alloc() : memref<4xf32>
    dataflow.task {
      linalg.copy ins(%in : memref<4xf32>) outs(%o : memref<4xf32>)
      dataflow.yield
    }
    dataflow.task {
      memref.copy %o, %out : memref<4xf32> to memref<4xf32>
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// A compute task read by two tasks (gemver's A_hat, read by the transpose and
// by a matvec): its tiles, made for each read, can't go to both.
func.func @two_readers(%a: memref<4xf32>, %out: memref<4xf32>) {
  dataflow.dispatch {
    %in = memref.alloc() : memref<4xf32>
    dataflow.task {
      memref.copy %a, %in : memref<4xf32> to memref<4xf32>
      dataflow.yield
    }
    %o = memref.alloc() : memref<4xf32>
    // expected-error @below {{expected a compute task read by one task, got 2}}
    dataflow.task {
      linalg.copy ins(%in : memref<4xf32>) outs(%o : memref<4xf32>)
      dataflow.yield
    }
    %p = memref.alloc() : memref<4xf32>
    dataflow.task {
      linalg.copy ins(%o : memref<4xf32>) outs(%p : memref<4xf32>)
      dataflow.yield
    }
    %q = memref.alloc() : memref<4xf32>
    dataflow.task {
      linalg.add ins(%o, %p : memref<4xf32>, memref<4xf32>) outs(%q : memref<4xf32>)
      dataflow.yield
    }
    dataflow.task {
      memref.copy %q, %out : memref<4xf32> to memref<4xf32>
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
