// Without top-func, the top is the function named after the module (a tile of
// sodap-tosa-outline-kernel); an unnamed module has none.
// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-infer-top-interface)" 2>&1 | FileCheck --check-prefix=MISSING %s
// RUN: not mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-infer-top-interface{top-func=nope})" 2>&1 | FileCheck --check-prefix=NOTFOUND %s
// RUN: mlir-opt %S/Inputs/named-top.mlir --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-dataflow-infer-top-interface)" | FileCheck --check-prefix=NAMED %s

// MISSING: top function '' not found
// NOTFOUND: top function 'nope' not found

// NAMED: module @top
// NAMED: func.func @top(%{{.*}}: !dataflow.stream<i32, 4>)

func.func @top(%s: !dataflow.stream<i32, 4>) {
  return
}
