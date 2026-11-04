// RUN: c++ -std=c++17 -shared -fPIC -I %S/../../../include %S/Inputs/test_kernel_runtime.cpp -o %t.so
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-kernel-to-esp-accelerator{profile=true},sodap-profile-kernel,sodap-profile-kernel,finalize-memref-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void -shared-libs=%t.so | FileCheck %s
// RUN: c++ -std=c++17 -shared -fPIC -DSLOT_STRIDE=2 -I %S/../../../include %S/Inputs/test_kernel_runtime.cpp -o %t.padded.so
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-kernel-to-esp-accelerator{profile=true dma-word-stride=2},sodap-profile-kernel,finalize-memref-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)" | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void -shared-libs=%t.padded.so | FileCheck %s
// CHECK: kernel inout and profiling PASS

memref.global "private" @values : memref<4xf32> = dense<[5.0, 6.0, 7.0, 8.0]>
func.func private @check_result(memref<*xf32>)
func.func @forward_kernel(%a: memref<4xf32>) {
  %c0 = arith.constant 0 : index
  %one = arith.constant 1.0 : f32
  %v = memref.load %a[%c0] : memref<4xf32>
  %next = arith.addf %v, %one : f32
  memref.store %next, %a[%c0] : memref<4xf32>
  return
}
func.func @main() {
  %a = memref.get_global @values : memref<4xf32>
  func.call @forward_kernel(%a) : (memref<4xf32>) -> ()
  func.call @forward_kernel(%a) : (memref<4xf32>) -> ()
  %u = memref.cast %a : memref<4xf32> to memref<*xf32>
  func.call @check_result(%u) : (memref<*xf32>) -> ()
  return
}
