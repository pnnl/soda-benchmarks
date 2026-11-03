// Buffers that must survive the conversion untouched. Converting any of these
// would change program meaning, so the pass has to decline rather than guess.
//
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-convert-memrefs-to-streams))" -split-input-file | FileCheck %s

// Two consumers: a FIFO is drained by exactly one reader, so this buffer would
// need duplicating first.
// CHECK-LABEL: func.func @two_consumers
// CHECK: memref.alloc
// CHECK-NOT: dataflow.stream
func.func @two_consumers(%in: memref<8xf32>, %a: memref<8xf32>, %b: memref<8xf32>) {
  %buf = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %buf[%i] : memref<8xf32>
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %buf[%i] : memref<8xf32>
    affine.store %v, %a[%i] : memref<8xf32>
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %buf[%i] : memref<8xf32>
    affine.store %v, %b[%i] : memref<8xf32>
  }
  return
}

// -----

// Producer and consumer in the same band: this is a value carried inside one
// stage, not a channel between two.
// CHECK-LABEL: func.func @same_loop_band
// CHECK: memref.alloc
// CHECK-NOT: dataflow.stream
func.func @same_loop_band(%in: memref<8xf32>, %out: memref<8xf32>) {
  %buf = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %buf[%i] : memref<8xf32>
    %w = affine.load %buf[%i] : memref<8xf32>
    affine.store %w, %out[%i] : memref<8xf32>
  }
  return
}

// -----

// Transposed consumer: the reader wants element (j,i) while the writer
// produced (i,j), so no FIFO ordering can serve both.
// CHECK-LABEL: func.func @transposed_access
// CHECK: memref.alloc
// CHECK-NOT: dataflow.stream
func.func @transposed_access(%in: memref<8x8xf32>, %out: memref<8x8xf32>) {
  %buf = memref.alloc() : memref<8x8xf32>
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %in[%i, %j] : memref<8x8xf32>
      affine.store %v, %buf[%i, %j] : memref<8x8xf32>
    }
  }
  affine.for %i = 0 to 8 {
    affine.for %j = 0 to 8 {
      %v = affine.load %buf[%j, %i] : memref<8x8xf32>
      affine.store %v, %out[%i, %j] : memref<8x8xf32>
    }
  }
  return
}

// -----

// A non-affine user is not analysable, so the buffer stays put rather than
// having that access silently dropped.
// CHECK-LABEL: func.func @non_affine_user
// CHECK: memref.alloc
// CHECK-NOT: dataflow.stream
func.func @non_affine_user(%in: memref<8xf32>, %out: memref<8xf32>, %idx: index) {
  %buf = memref.alloc() : memref<8xf32>
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<8xf32>
    affine.store %v, %buf[%i] : memref<8xf32>
  }
  // The result is consumed, so this access cannot be dead-code eliminated
  // away before the pattern sees it.
  %w = memref.load %buf[%idx] : memref<8xf32>
  memref.store %w, %out[%idx] : memref<8xf32>
  affine.for %i = 0 to 8 {
    %v = affine.load %buf[%i] : memref<8xf32>
    affine.store %v, %out[%i] : memref<8xf32>
  }
  return
}

// -----

// Without a static shape there is no depth to give the channel.
// CHECK-LABEL: func.func @dynamic_shape
// CHECK: memref.alloc
// CHECK-NOT: dataflow.stream
func.func @dynamic_shape(%in: memref<?xf32>, %out: memref<?xf32>, %n: index) {
  %buf = memref.alloc(%n) : memref<?xf32>
  affine.for %i = 0 to 8 {
    %v = affine.load %in[%i] : memref<?xf32>
    affine.store %v, %buf[%i] : memref<?xf32>
  }
  affine.for %i = 0 to 8 {
    %v = affine.load %buf[%i] : memref<?xf32>
    affine.store %v, %out[%i] : memref<?xf32>
  }
  return
}
