// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter --verify-diagnostics

// bX is read by two compute stages. The producer switches the slot after
// writing the token, so a multibuffered buffer can only have one consumer.
func.func @two_consumers(%X: memref<8xf32>) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %z = arith.constant 0 : i8
  %one = arith.constant 1 : i8
  %two = arith.constant 2 : i8
  %bX = memref.alloc() : memref<2x4xf32>
  %bY = memref.alloc() : memref<4xf32>
  %bZ = memref.alloc() : memref<4xf32>
  %var = memref.alloca() : memref<i8>
  memref.store %z, %var[] : memref<i8>
  // expected-note @below {{nest}}
  scf.for %i = %c0 to %c8 step %c4 {
    %s = memref.load %var[] : memref<i8>
    %idx = arith.index_cast %s : i8 to index
    %vX = memref.subview %bX[%idx, 0] [1, 4] [1, 1] : memref<2x4xf32> to memref<4xf32, strided<[1], offset: ?>>
    %Xt = memref.subview %X[%i] [4] [1] : memref<8xf32> to memref<4xf32, strided<[1], offset: ?>>
    memref.copy %Xt, %vX : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32, strided<[1], offset: ?>>
    linalg.copy ins(%vX : memref<4xf32, strided<[1], offset: ?>>) outs(%bY : memref<4xf32>)
    linalg.copy ins(%vX : memref<4xf32, strided<[1], offset: ?>>) outs(%bZ : memref<4xf32>)
    %a = arith.addi %s, %one : i8
    %n = arith.remui %a, %two : i8
    memref.store %n, %var[] : memref<i8>
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.op<"scf.for">
    // expected-error @below {{a multibuffered buffer must have exactly one consumer}}
    %dispatch = transform.sodap.split_stages %loops : (!transform.op<"scf.for">) -> !transform.any_op
    transform.yield
  }
}
