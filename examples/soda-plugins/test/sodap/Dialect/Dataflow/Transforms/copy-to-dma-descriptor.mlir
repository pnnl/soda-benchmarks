// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-copy-to-dma{top-func=top dma-word-stride=2},canonicalize)" | FileCheck %s

// A tile with its shapes as arguments (sodap-join-multiple-tiles): the load
// task reads the descriptor (16 slots, 32 words at address 0) and forwards it
// to the store; M becomes field 2, and a 2x4 tile of A at row i starts at
// A + (i + r) * lda (fields 5 and 8), of C at C + (i + r) * ldc (7 and 10).

// CHECK-LABEL: func.func @top(
// CHECK-SAME:    %[[RC:.*]]: !dataflow.stream<i64, 4>, %[[RD:.*]]: !dataflow.stream<i32, 1>, %[[WC:.*]]: !dataflow.stream<i64, 4>, %[[WD:.*]]: !dataflow.stream<i32, 1>)
// CHECK:         dataflow.dispatch {
// CHECK:           %[[DESC:.*]] = dataflow.stream {depth = 16 : i32} : <i32, 16>
// CHECK:           dataflow.task attributes {load} {
// CHECK:             %[[LOCAL:.*]] = memref.alloca() : memref<16xi32>
// CHECK:             dataflow.stream_write %[[RC]], %c137438953472_i64
// CHECK:             scf.for %[[F:.*]] = %c0 to %c16 step %c1 {
// CHECK-NEXT:          %[[W:.*]] = dataflow.stream_read %[[RD]]
// CHECK-NEXT:          dataflow.stream_read %[[RD]]
// CHECK-NEXT:          dataflow.stream_write %[[DESC]], %[[W]]
// CHECK-NEXT:          memref.store %[[W]], %[[LOCAL]][%[[F]]]
// CHECK:             %[[MW:.*]] = memref.load %[[LOCAL]][%c2]
// CHECK-NEXT:        %[[M:.*]] = arith.index_castui %[[MW]]
// CHECK-NEXT:        %[[AW:.*]] = memref.load %[[LOCAL]][%c5]
// CHECK-NEXT:        %[[A:.*]] = arith.index_castui %[[AW]]
// CHECK-NEXT:        %[[LW:.*]] = memref.load %[[LOCAL]][%c8]
// CHECK-NEXT:        %[[LDA:.*]] = arith.index_castui %[[LW]]
// CHECK-NEXT:        scf.for %[[I:.*]] = %c0 to %[[M]] step %c2 {
// CHECK-NEXT:          scf.for %[[R:.*]] = %c0 to %c2 step %c1 {
// CHECK-NEXT:            %[[ROW:.*]] = arith.addi %[[I]], %[[R]]
// CHECK-NEXT:            %[[OFF:.*]] = arith.muli %[[ROW]], %[[LDA]]
// CHECK-NEXT:            %[[ADDR:.*]] = arith.addi %[[A]], %[[OFF]]
// CHECK:           dataflow.task attributes {store} {
// CHECK:             scf.for {{.*}} {
// CHECK-NEXT:          dataflow.stream_read %[[DESC]]
func.func @top(%a: memref<?x?xf32> {sodap.descriptor_field = 5 : i64, sodap.descriptor_ld = 8 : i64},
               %m: index {sodap.descriptor_field = 2 : i64},
               %c: memref<?x?xf32> {sodap.descriptor_field = 7 : i64, sodap.descriptor_ld = 10 : i64}) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  dataflow.dispatch {
    %buf = memref.alloc() : memref<2x4xf32>
    dataflow.task attributes {load} {
      scf.for %i = %c0 to %m step %c2 {
        %t = memref.subview %a[%i, 0] [2, 4] [1, 1] : memref<?x?xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
        memref.copy %t, %buf : memref<2x4xf32, strided<[?, 1], offset: ?>> to memref<2x4xf32>
      }
    }
    dataflow.task attributes {store} {
      scf.for %i = %c0 to %m step %c2 {
        %t = memref.subview %c[%i, 0] [2, 4] [1, 1] : memref<?x?xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
        memref.copy %buf, %t : memref<2x4xf32> to memref<2x4xf32, strided<[?, 1], offset: ?>>
      }
    }
  }
  return
}
