// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-merge-tasks))" -split-input-file -verify-diagnostics | FileCheck %s

// What sodap-dataflow-load-store leaves: load, two compute tasks, store. One
// task afterwards, the last, running them in order, with the allocs of the
// dispatch at its start (Bambu takes an alloca of the top for a channel).
// CHECK-LABEL: func.func @load_compute_store
// CHECK-SAME:    (%[[A:.*]]: memref<4xf32>, %[[OUT:.*]]: memref<4xf32>)
// CHECK:         dataflow.dispatch {
// CHECK-NEXT:      dataflow.task {
// CHECK-NEXT:        %[[IN:.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT:        %[[T:.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT:        %[[O:.*]] = memref.alloc() : memref<4xf32>
// CHECK-NEXT:        memref.copy %[[A]], %[[IN]]
// CHECK-NEXT:        linalg.copy ins(%[[IN]] : {{.*}}) outs(%[[T]] : {{.*}})
// CHECK-NEXT:        linalg.add ins(%[[T]], %[[IN]] : {{.*}}) outs(%[[O]] : {{.*}})
// CHECK-NEXT:        memref.copy %[[O]], %[[OUT]]
// CHECK-NEXT:      }
// CHECK-NEXT:    }
func.func @load_compute_store(%a: memref<4xf32>, %out: memref<4xf32>) {
  dataflow.dispatch {
    %in = memref.alloc() : memref<4xf32>
    dataflow.task attributes {load} {
      memref.copy %a, %in : memref<4xf32> to memref<4xf32>
      dataflow.yield
    }
    %t = memref.alloc() : memref<4xf32>
    dataflow.task attributes {compute} {
      linalg.copy ins(%in : memref<4xf32>) outs(%t : memref<4xf32>)
      dataflow.yield
    }
    %o = memref.alloc() : memref<4xf32>
    dataflow.task attributes {compute} {
      linalg.add ins(%t, %in : memref<4xf32>, memref<4xf32>) outs(%o : memref<4xf32>)
      dataflow.yield
    }
    dataflow.task attributes {store} {
      memref.copy %o, %out : memref<4xf32> to memref<4xf32>
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// A lone task is left as it is.
// CHECK-LABEL: func.func @one_task
// CHECK:         dataflow.task attributes {store} {
// CHECK-NEXT:      memref.copy
func.func @one_task(%a: memref<4xf32>, %out: memref<4xf32>) {
  dataflow.dispatch {
    dataflow.task attributes {store} {
      memref.copy %a, %out : memref<4xf32> to memref<4xf32>
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
