// The top function is a property of the design, so the pass has no default.
// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-infer-top-interface)" 2>&1 | FileCheck --check-prefix=MISSING %s
// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-infer-top-interface{top-func=nope})" 2>&1 | FileCheck --check-prefix=NOTFOUND %s

// MISSING: option 'top-func' is required
// NOTFOUND: top function 'nope' not found

func.func @top(%s: !dataflow.stream<i32, 4>) {
  return
}
