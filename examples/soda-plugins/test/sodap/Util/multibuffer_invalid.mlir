// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter --verify-diagnostics

func.func @factor3() {
  %b = memref.alloc() : memref<4xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %root : (!transform.any_op) -> !transform.op<"memref.alloc">
    // expected-error @below {{only factor 2 is supported}}
    %mb = transform.sodap.multibuffer %allocs {factor = 3 : i64} : (!transform.op<"memref.alloc">) -> !transform.any_op
    transform.yield
  }
}
