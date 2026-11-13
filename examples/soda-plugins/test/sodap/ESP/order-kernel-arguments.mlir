// RUN: mlir-opt %s --allow-unregistered-dialect --split-input-file --verify-diagnostics \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline="builtin.module(sodap-order-kernel-arguments)" | FileCheck %s

// The outlining gave the kernel the order of first use (b, c, a): it gets
// the caller's (a, b, c), operands, arguments and types alike. The soda
// dialect isn't in the plugin, so its ops are unregistered here.
// CHECK-LABEL: func.func @forward
// CHECK-SAME:    (%[[A:.*]]: memref<f32>, %[[B:.*]]: memref<2xf32>, %[[C:.*]]: memref<4xf32>)
// CHECK:         "soda.launch_func"(%[[A]], %[[B]], %[[C]])
// CHECK-SAME:      (memref<f32>, memref<2xf32>, memref<4xf32>) -> ()
// CHECK:         "soda.func"()
// CHECK-NEXT:    ^bb0(%[[X:.*]]: memref<f32>, %[[Y:.*]]: memref<2xf32>, %[[Z:.*]]: memref<4xf32>):
// CHECK-NEXT:      "test.use"(%[[Y]], %[[Z]], %[[X]])
// CHECK:         function_type = (memref<f32>, memref<2xf32>, memref<4xf32>) -> ()
func.func @forward(%a: memref<f32>, %b: memref<2xf32>, %c: memref<4xf32>) {
  "soda.launch_func"(%b, %c, %a) {kernel = @forward_kernel::@forward_kernel, operandSegmentSizes = array<i32: 0, 3>} : (memref<2xf32>, memref<4xf32>, memref<f32>) -> ()
  return
}
"soda.module"() ({
  "soda.func"() ({
  ^bb0(%y: memref<2xf32>, %z: memref<4xf32>, %x: memref<f32>):
    "test.use"(%y, %z, %x) : (memref<2xf32>, memref<4xf32>, memref<f32>) -> ()
    "soda.return"() : () -> ()
  }) {function_type = (memref<2xf32>, memref<4xf32>, memref<f32>) -> (), sym_name = "forward_kernel"} : () -> ()
  "soda.module_end"() : () -> ()
}) {sym_name = "forward_kernel"} : () -> ()

// -----

func.func @not_an_argument(%a: memref<f32>) {
  %b = memref.alloc() : memref<f32>
  // expected-error @below {{expected distinct arguments of the caller as operands}}
  "soda.launch_func"(%b, %a) {kernel = @k::@k, operandSegmentSizes = array<i32: 0, 2>} : (memref<f32>, memref<f32>) -> ()
  return
}
"soda.module"() ({
  "soda.func"() ({
  ^bb0(%x: memref<f32>, %y: memref<f32>):
    "soda.return"() : () -> ()
  }) {function_type = (memref<f32>, memref<f32>) -> (), sym_name = "k"} : () -> ()
  "soda.module_end"() : () -> ()
}) {sym_name = "k"} : () -> ()
