// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-dataflow-ping-pong))" -split-input-file -verify-diagnostics | FileCheck %s

// load -> accumulate -> store, a tile of 4 per i. Each shared buffer becomes
// two slots of 4 and a depth-1 token stream. The load writes the slot then
// sends it; the compute, which accumulates into its tile, works on a private
// one, copied in the slot before the token; the readers peek before their
// access and read after.
// CHECK-LABEL: func.func @chain
// CHECK-SAME:    (%[[IN:.*]]: memref<8xf32>, %[[OUT:.*]]: memref<8xf32>)
// CHECK:         dataflow.dispatch {
// CHECK-DAG:       %[[T0:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>
// CHECK-DAG:       %[[T1:.*]] = dataflow.stream {depth = 1 : i32} : <i8, 1>
// CHECK:           %[[BIN:.*]] = memref.alloc() : memref<2x4xf32>
// Load.
// CHECK:           dataflow.task {
// CHECK-NEXT:        %[[VIN:.*]] = memref.alloca() : memref<i8>
// CHECK:             scf.for %[[I:.*]] =
// CHECK-NEXT:          %[[SRC:.*]] = memref.subview %[[IN]][%[[I]]] [4] [1]
// CHECK-NEXT:          %[[S:.*]] = memref.load %[[VIN]][] : memref<i8>
// CHECK-NEXT:          %[[X:.*]] = arith.index_cast %[[S]] : i8 to index
// CHECK-NEXT:          %[[SLOT:.*]] = memref.subview %[[BIN]][%[[X]], 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-NEXT:          memref.copy %[[SRC]], %[[SLOT]]
// CHECK-NEXT:          dataflow.stream_write %[[TIN:.*]], %[[S]]
// CHECK:               arith.xori %[[S]]
// CHECK-NEXT:          memref.store %{{.*}}, %[[VIN]][] : memref<i8>
// CHECK:           %[[BACC:.*]] = memref.alloc() : memref<2x4xf32>
// Compute.
// CHECK:           dataflow.task {
// CHECK-NEXT:        %[[VACC:.*]] = memref.alloca() : memref<i8>
// CHECK:             %[[PRIV:.*]] = memref.alloc() : memref<4xf32>
// CHECK:             scf.for
// CHECK-NEXT:          linalg.fill {{.*}} outs(%[[PRIV]] :
// CHECK-NEXT:          %[[P:.*]] = dataflow.stream_peek %[[TIN]]
// CHECK-NEXT:          arith.index_cast %[[P]]
// CHECK-NEXT:          %[[RIN:.*]] = memref.subview %[[BIN]]
// CHECK-NEXT:          linalg.add ins(%[[RIN]], %[[PRIV]] : {{.*}}) outs(%[[PRIV]] :
// CHECK-NEXT:          %[[SA:.*]] = memref.load %[[VACC]][] : memref<i8>
// CHECK-NEXT:          arith.index_cast %[[SA]]
// CHECK-NEXT:          %[[WACC:.*]] = memref.subview %[[BACC]]
// CHECK-NEXT:          linalg.copy ins(%[[PRIV]] : {{.*}}) outs(%[[WACC]] :
// CHECK-NEXT:          dataflow.stream_write %[[TACC:.*]], %[[SA]]
// CHECK:               memref.store %{{.*}}, %[[VACC]][] : memref<i8>
// CHECK-NEXT:          dataflow.stream_read %[[TIN]]
// Store.
// CHECK:           dataflow.task {
// CHECK-NOT:         memref.alloca
// CHECK:             scf.for %[[J:.*]] =
// CHECK-NEXT:          %[[DST:.*]] = memref.subview %[[OUT]][%[[J]]] [4] [1]
// CHECK-NEXT:          %[[Q:.*]] = dataflow.stream_peek %[[TACC]]
// CHECK-NEXT:          arith.index_cast %[[Q]]
// CHECK-NEXT:          %[[RACC:.*]] = memref.subview %[[BACC]]
// CHECK-NEXT:          memref.copy %[[RACC]], %[[DST]]
// CHECK-NEXT:          dataflow.stream_read %[[TACC]]
func.func @chain(%a: memref<8xf32>, %out: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %zero = arith.constant 0.0 : f32
  dataflow.dispatch {
    %in = memref.alloc() : memref<8xf32>
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        %s = memref.subview %a[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        %t = memref.subview %in[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        memref.copy %s, %t : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
      }
      dataflow.yield
    }
    %acc = memref.alloc() : memref<8xf32>
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        %x = memref.subview %in[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        %y = memref.subview %acc[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        linalg.fill ins(%zero : f32) outs(%y : memref<4xf32, strided<[1], offset: ?>>)
        linalg.add ins(%x, %y : memref<4xf32, strided<[1], offset: ?>>, memref<4xf32, strided<[1], offset: ?>>) outs(%y : memref<4xf32, strided<[1], offset: ?>>)
      }
      dataflow.yield
    }
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        %y = memref.subview %acc[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        %o = memref.subview %out[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        memref.copy %y, %o : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}

// -----

// The reader reads each tile twice (a reuse loop), the load writes it once:
// the counts differ, run sodap-dataflow-schedule first.
func.func @reuse(%a: memref<8xf32>, %out: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  dataflow.dispatch {
    %in = memref.alloc() : memref<8xf32>
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        %s = memref.subview %a[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        %t = memref.subview %in[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
        memref.copy %s, %t : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
      }
      dataflow.yield
    }
    // expected-error @below {{expected to read as many tiles of a shared buffer as its writer writes, got 4 and 2}}
    dataflow.task {
      scf.for %i = %c0 to %c8 step %c4 {
        scf.for %k = %c0 to %c2 step %c1 {
          %x = memref.subview %in[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
          %o = memref.subview %out[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
          memref.copy %x, %o : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
        }
      }
      dataflow.yield
    }
    dataflow.yield
  }
  return
}
