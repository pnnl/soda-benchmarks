// RUN: mlir-opt -allow-unregistered-dialect -mlir-elide-elementsattrs-if-larger=2 --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(soda-view-op-graph)" %s -o %t 2>&1 | FileCheck -check-prefix=DFG %s

// DFG-LABEL: digraph G {
// DFG:   subgraph {{.*}} {
// DFG:     subgraph {{.*}}
// DFG:       label = "func.func{{.*}}test_tosa
// DFG:       subgraph {{.*}} {
// DFG:         v[[ARG0:.*]] [label = "arg0"
// DFG:         v[[TOSA_ADD:.*]] [{{.*}}label = "tosa.add
// DFG:         v[[TOSA_MATMUL:.*]] [{{.*}}label = "tosa.matmul
// DFG:       }
// DFG:   v[[ARG0]] -> v[[TOSA_ADD]] [{{.*}}weight = "64"{{.*}}]
// DFG:   v[[ARG0]] -> v[[TOSA_ADD]] [{{.*}}weight = "64"{{.*}}]
// DFG:   v[[TOSA_ADD]] -> v[[TOSA_MATMUL]] [{{.*}}weight = "64"{{.*}}]
// DFG:   v[[ARG0]] -> v[[TOSA_MATMUL]] [{{.*}}weight = "24"{{.*}}]

module {
  func.func @test_tosa(%arg0: tensor<4x4xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x4xf32> {
    // 4x4xf32 = 64 bytes (4*4*4 = 64)
    // 2x3xf32 = 24 bytes (2*3*4 = 24)
    %0 = tosa.add %arg0, %arg0 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    // 4x4xf32 = 64 bytes
    %1 = tosa.matmul %0, %arg1 : tensor<4x4xf32>, tensor<2x3xf32> -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}

