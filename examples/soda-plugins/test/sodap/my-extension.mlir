// RUN: mlir-opt --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter %s | FileCheck %s

func.func private @orig()
func.func private @updated()

// CHECK-LABEL: func @test
func.func @test() {
  // CHECK: call @updated
  call @orig() : () -> ()
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op) {
    %call = transform.structured.match ops{["func.call"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    // CHECK: transform.my.change_call_target %{{.*}}, "updated" : !transform.any_op
    transform.my.change_call_target %call, "updated" : !transform.any_op
    transform.yield
  }
}