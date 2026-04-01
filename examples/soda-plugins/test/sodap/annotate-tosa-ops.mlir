// RUN: mlir-opt -allow-unregistered-dialect %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(sodap-annotate-tosa-ops)" | FileCheck %s

module {
  func.func @test_tosa_add(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    // CHECK: tosa.add
    // CHECK-SAME: numArithmeticOpsEstimative = {{[0-9]+}} : i64
    // CHECK-SAME: numMemoryOpsEstimative = {{[0-9]+}} : i64
    // CHECK-SAME: numArithmeticOpsInKernel = {{[0-9]+}} : i64
    // CHECK-SAME: numMemoryOpsInKernel = {{[0-9]+}} : i64
    %0 = tosa.add %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }

  func.func @test_tosa_matmul(%arg0: tensor<2x3xf32>, %arg1: tensor<3x4xf32>) -> tensor<2x4xf32> {
    // CHECK: tosa.matmul
    // CHECK-SAME: numArithmeticOpsEstimative = {{[0-9]+}} : i64
    // CHECK-SAME: numMemoryOpsEstimative = {{[0-9]+}} : i64
    // CHECK-SAME: numArithmeticOpsInKernel = {{[0-9]+}} : i64
    // CHECK-SAME: numMemoryOpsInKernel = {{[0-9]+}} : i64
    %0 = tosa.matmul %arg0, %arg1 : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
    return %0 : tensor<2x4xf32>
  }
}

