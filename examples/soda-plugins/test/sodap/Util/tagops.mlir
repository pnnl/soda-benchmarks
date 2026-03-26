// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-tagops{anchor-op=func.call prefix=my_id})" | FileCheck %s
// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter | FileCheck %s

func.func private @foo()
func.func private @bar()
func.func private @baz()

// CHECK-LABEL: func @test
func.func @test() {
  // CHECK: call @foo() {uid = "my_id_0"}
  call @foo() : () -> ()
  // CHECK: call @bar() {uid = "my_id_1"}
  call @bar() : () -> ()
  // CHECK: call @baz() {uid = "my_id_2"}
  call @baz() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %calls = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.sodap.tag_ops %calls, "my_id" : !transform.any_op
    transform.yield
  }
}
