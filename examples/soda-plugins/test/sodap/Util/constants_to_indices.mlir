// RUN: mlir-opt %s --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(func.func(sodap-linalg-constants-to-indices))" | FileCheck %s

#id = affine_map<(d0, d1) -> (d0, d1)>
#tr = affine_map<(d0, d1) -> (d1, d0)>

// trmm: A * tril(ones) + eye. tril is d >= 1 ? 0 : 1, eye is d == 0 ? 1 : 0.
// CHECK-LABEL: func @trmm
// CHECK-NOT:     arith.constant dense
// CHECK:         linalg.generic {{.*}} ins(%{{.*}} : tensor<3x3xf32>)
// CHECK:         ^bb0(%[[A:.*]]: f32, %{{.*}}: f32):
// CHECK-DAG:       %[[I:.*]] = linalg.index 0
// CHECK-DAG:       %[[J:.*]] = linalg.index 1
// CHECK-DAG:       %[[D:.*]] = arith.subi %[[J]], %[[I]]
// CHECK-DAG:       arith.cmpi eq, %[[D]], %c0
// CHECK-DAG:       arith.cmpi sge, %{{.*}}, %c1
// CHECK:           arith.mulf %[[A]]
func.func @trmm(%a: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %tril = arith.constant dense<[[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]> : tensor<3x3xf32>
  %eye = arith.constant dense<[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]> : tensor<3x3xf32>
  %e = tensor.empty() : tensor<3x3xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%a, %tril, %eye : tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) outs(%e : tensor<3x3xf32>) {
  ^bb0(%x: f32, %m: f32, %i: f32, %o: f32):
    %1 = arith.mulf %x, %m : f32
    %2 = arith.addf %1, %i : f32
    linalg.yield %2 : f32
  } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// symm: A * mask + Aᵀ * (1 - mask), and a non-square triu(1) read transposed:
// element (d1, d0), so i is d1 and j is d0.
// CHECK-LABEL: func @transposed
// CHECK-NOT:     arith.constant dense
// CHECK:           %[[I:.*]] = linalg.index 1
// CHECK:           %[[J:.*]] = linalg.index 0
// CHECK:           %[[D:.*]] = arith.subi %[[J]], %[[I]]
// CHECK:           %[[C:.*]] = arith.cmpi sge, %[[D]], %c1
// CHECK:           arith.select %[[C]], %cst{{.*}}, %cst
func.func @transposed(%a: tensor<3x2xf32>) -> tensor<3x2xf32> {
  %triu = arith.constant dense<[[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]> : tensor<2x3xf32>
  %e = tensor.empty() : tensor<3x2xf32>
  %0 = linalg.generic {indexing_maps = [#id, #tr, #id], iterator_types = ["parallel", "parallel"]}
      ins(%a, %triu : tensor<3x2xf32>, tensor<2x3xf32>) outs(%e : tensor<3x2xf32>) {
  ^bb0(%x: f32, %m: f32, %o: f32):
    %1 = arith.mulf %x, %m : f32
    linalg.yield %1 : f32
  } -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// Not one value per diagonal (weights), or a diagonal band: left alone.
// CHECK-LABEL: func @left_alone
// CHECK:         %[[W:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 2.000000e+00
// CHECK:         %[[B:.*]] = arith.constant dense<{{\[\[}}1.000000e+00, 1.000000e+00, 0.000000e+00
// CHECK:         linalg.generic {{.*}} ins(%{{.*}}, %[[W]], %[[B]] :
// CHECK-NOT:     linalg.index
func.func @left_alone(%a: tensor<3x3xf32>) -> tensor<3x3xf32> {
  %w = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]> : tensor<3x3xf32>
  %band = arith.constant dense<[[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]> : tensor<3x3xf32>
  %e = tensor.empty() : tensor<3x3xf32>
  %0 = linalg.generic {indexing_maps = [#id, #id, #id, #id], iterator_types = ["parallel", "parallel"]}
      ins(%a, %w, %band : tensor<3x3xf32>, tensor<3x3xf32>, tensor<3x3xf32>) outs(%e : tensor<3x3xf32>) {
  ^bb0(%x: f32, %m: f32, %n: f32, %o: f32):
    %1 = arith.mulf %x, %m : f32
    %2 = arith.addf %1, %n : f32
    linalg.yield %2 : f32
  } -> tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}
