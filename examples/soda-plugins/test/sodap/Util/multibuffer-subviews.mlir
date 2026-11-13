// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter --canonicalize | FileCheck %s --check-prefix=IR
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter --test-transform-dialect-erase-schedule --convert-linalg-to-loops --lower-affine --convert-scf-to-cf --expand-strided-metadata --lower-affine --finalize-memref-to-llvm --convert-arith-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | mlir-cpu-runner -e main -entry-point-result=i32 | FileCheck %s --check-prefix=RESULT
// RESULT: 210
// IR-LABEL: func.func @kernel
// IR: memref.alloc() : memref<2x4xf32>
// IR: memref.subview {{.*}} [1, 4] [1, 1]
// IR: memref.subview {{.*}}[1] [2] [1] : memref<4xf32, strided<[1], offset: ?>> to memref<2xf32, strided<[1], offset: ?>>
// A static-offset slice becomes dynamically offset when its source alternates
// banks. A nested slice must be rebased as well. Three iterations reuse bank 0.
memref.global "private" @result : memref<3x2xf32> = dense<0.0>
func.func @kernel(%out: memref<3x2xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %b = memref.alloc() : memref<4xf32>
  scf.for %i = %c0 to %c3 step %c1 {
    %n = arith.index_cast %i : index to i32
    %f = arith.sitofp %n : i32 to f32
    linalg.fill ins(%f : f32) outs(%b : memref<4xf32>)
    %s = memref.subview %b[1] [2] [1] : memref<4xf32> to memref<2xf32, strided<[1], offset: 1>>
    %t = memref.subview %s[0] [2] [1] : memref<2xf32, strided<[1], offset: 1>> to memref<2xf32, strided<[1], offset: 1>>
    %o = memref.subview %out[%i, 0] [1, 2] [1, 1] : memref<3x2xf32> to memref<2xf32, strided<[1], offset: ?>>
    memref.copy %t, %o : memref<2xf32, strided<[1], offset: 1>> to memref<2xf32, strided<[1], offset: ?>>
  }
  return
}
func.func @main() -> i32 {
  %out = memref.get_global @result : memref<3x2xf32>
  call @kernel(%out) : (memref<3x2xf32>) -> ()
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %ten = arith.constant 10.0 : f32
  %hundred = arith.constant 100.0 : f32
  %a = memref.load %out[%c0, %c1] : memref<3x2xf32>
  %b = memref.load %out[%c1, %c1] : memref<3x2xf32>
  %c = memref.load %out[%c2, %c1] : memref<3x2xf32>
  %b10 = arith.mulf %b, %ten : f32
  %c100 = arith.mulf %c, %hundred : f32
  %ab = arith.addf %a, %b10 : f32
  %sum = arith.addf %ab, %c100 : f32
  %result = arith.fptosi %sum : f32 to i32
  return %result : i32
}
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %root : (!transform.any_op) -> !transform.op<"memref.alloc">
    %mb = transform.sodap.multibuffer %allocs {factor = 2 : i64} : (!transform.op<"memref.alloc">) -> !transform.any_op
    transform.yield
  }
}
