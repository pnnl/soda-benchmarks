// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-convert-memrefs-to-streams))" -split-input-file | FileCheck %s
// The simplest channel: one band fills the buffer, the next drains it.
// CHECK-LABEL: func.func @producer_consumer
func.func @producer_consumer(%in: memref<8xf32>, %out: memref<8xf32>) {
  // CHECK-NOT: memref.alloc
  // CHECK: %[[C:.*]] = dataflow.stream {depth = 8 : i32} : <f32, 8>
  %buf = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    // CHECK: dataflow.stream_write %[[C]], %{{.*}} : <f32, 8>, f32
    affine.store %v, %buf[%i] : memref<8xf32>
  }
  affine.for %i = 0 to 8 {
    // CHECK: dataflow.stream_read %[[C]] : (!dataflow.stream<f32, 8>) -> f32
    %v = affine.load %buf[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  return
}

// -----

// Depth is the flattened element count, so a 2-D buffer becomes a 32-deep
// channel and both sides keep their row-major traversal.
// CHECK-LABEL: func.func @two_dimensional
func.func @two_dimensional(%in: memref<4x8xf32>, %out: memref<4x8xf32>) {
  // CHECK: dataflow.stream {depth = 32 : i32} : <f32, 32>
  %buf = memref.alloc() : memref<4x8xf32>
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      %v = affine.load %in[%i, %j] : memref<4x8xf32>
      affine.store %v, %buf[%i, %j] : memref<4x8xf32>
    }
  }
  affine.for %i = 0 to 4 {
    affine.for %j = 0 to 8 {
      %v = affine.load %buf[%i, %j] : memref<4x8xf32>
      affine.store %v, %out[%i, %j] : memref<4x8xf32>
    }
  }
  return
}

// -----

// Rewriting in place means an affine.if guarding the producer keeps guarding
// the channel write. This is the shape the conversion has to preserve for a
// reduction that only emits its result on the last iteration.
// CHECK-LABEL: func.func @guarded_producer
#set = affine_set<(d0) : (d0 - 7 == 0)>
func.func @guarded_producer(%in: memref<8xf32>, %out: memref<8xf32>) {
  %buf = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %in[%i] : memref<8xf32>
      // CHECK: affine.if
      affine.if #set(%j) {
        // CHECK-NEXT: dataflow.stream_write
        affine.store %v, %buf[%i] : memref<8xf32>
      }
    }
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %buf[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  return
}

// -----

// A matching dealloc goes away with the buffer it freed.
// CHECK-LABEL: func.func @dealloc_removed
func.func @dealloc_removed(%in: memref<8xf32>, %out: memref<8xf32>) {
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  %buf = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %buf[%i] : memref<8xf32>
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %buf[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  memref.dealloc %buf : memref<8xf32>
  return
}
