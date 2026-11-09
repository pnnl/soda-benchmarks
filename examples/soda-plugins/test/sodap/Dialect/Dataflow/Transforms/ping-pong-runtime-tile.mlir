// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-ping-pong))" -split-input-file -verify-diagnostics | FileCheck %s

// One token per runtime-sized tail tile, not per scalar element. The producer
// publishes after the entire tile; the consumer releases after its last load.
// CHECK-LABEL: func.func @runtime_tiles
// CHECK: memref.alloc() : memref<2x4xf32>
// CHECK: dataflow.task
// CHECK: scf.for
// CHECK: scf.execute_region
// CHECK-NOT: dataflow.stream_write
// CHECK: scf.yield
// CHECK: dataflow.stream_write
// CHECK: dataflow.task
// CHECK: scf.for
// CHECK: dataflow.stream_peek
// CHECK: scf.execute_region
// CHECK-NOT: dataflow.stream_read
// CHECK: scf.yield
// CHECK: dataflow.stream_read
func.func @runtime_tiles(%n: index, %input: memref<?xf32>, %output: memref<?xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  dataflow.dispatch {
    %tile = memref.alloc() : memref<4xf32>
    dataflow.task {
      scf.for %start = %c0 to %n step %c4 {
        %left = arith.subi %n, %start : index
        %count = arith.minui %left, %c4 : index
        scf.execute_region {
          scf.for %i = %c0 to %count step %c1 {
            %offset = arith.addi %start, %i : index
            %value = memref.load %input[%offset] : memref<?xf32>
            memref.store %value, %tile[%i] : memref<4xf32>
          }
          scf.yield
        } {sodap.dataflow_tile}
      }
      dataflow.yield
    }
    dataflow.task {
      scf.for %start = %c0 to %n step %c4 {
        %left = arith.subi %n, %start : index
        %count = arith.minui %left, %c4 : index
        scf.execute_region {
          scf.for %i = %c0 to %count step %c1 {
            %value = memref.load %tile[%i] : memref<4xf32>
            %offset = arith.addi %start, %i : index
            memref.store %value, %output[%offset] : memref<?xf32>
          }
          scf.yield
        } {sodap.dataflow_tile}
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

func.func @mixed_scopes(%output: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : f32
  dataflow.dispatch {
    %tile = memref.alloc() : memref<4xf32>
    // expected-error @below {{all accesses to a shared buffer must belong to one explicit tile scope}}
    dataflow.task {
      scf.execute_region {
        memref.store %zero, %tile[%c0] : memref<4xf32>
        scf.yield
      } {sodap.dataflow_tile}
      memref.store %zero, %tile[%c0] : memref<4xf32>
      dataflow.yield
    }
    dataflow.task {
      memref.copy %tile, %output : memref<4xf32> to memref<4xf32>
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
