#map = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "SimpleMLP"} {
  func.func @forward(%arg0: tensor<1x8x16xf32>, %arg1: tensor<1x16x12xf32>) -> tensor<1x4x6xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<1x8x12xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<1x8x12xf32>) -> tensor<1x8x12xf32>
    %2 = linalg.batch_matmul ins(%arg0, %arg1 : tensor<1x8x16xf32>, tensor<1x16x12xf32>) outs(%1 : tensor<1x8x12xf32>) -> tensor<1x8x12xf32>
    %expanded = tensor.expand_shape %2 [[0], [1], [2, 3]] output_shape [1, 8, 12, 1] : tensor<1x8x12xf32> into tensor<1x8x12x1xf32>
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %3 = tensor.empty() : tensor<1x4x6x1xf32>
    %4 = linalg.fill ins(%cst_0 : f32) outs(%3 : tensor<1x4x6x1xf32>) -> tensor<1x4x6x1xf32>
    %5 = tensor.empty() : tensor<2x2xf32>
    %6 = linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%expanded, %5 : tensor<1x8x12x1xf32>, tensor<2x2xf32>) outs(%4 : tensor<1x4x6x1xf32>) -> tensor<1x4x6x1xf32>
    %collapsed = tensor.collapse_shape %6 [[0], [1], [2, 3]] : tensor<1x4x6x1xf32> into tensor<1x4x6xf32>
    %expanded_1 = tensor.expand_shape %collapsed [[0, 1], [2], [3]] output_shape [1, 1, 4, 6] : tensor<1x4x6xf32> into tensor<1x1x4x6xf32>
    %7 = tensor.empty() : tensor<1x1x4x6xf32>
    %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_1 : tensor<1x1x4x6xf32>) outs(%7 : tensor<1x1x4x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      %cst_3 = arith.constant 0.000000e+00 : f32
      %cst_4 = arith.constant 3.40282347E+38 : f32
      %9 = arith.minimumf %in, %cst_4 : f32
      %10 = arith.maximumf %9, %cst_3 : f32
      linalg.yield %10 : f32
    } -> tensor<1x1x4x6xf32>
    %collapsed_2 = tensor.collapse_shape %8 [[0, 1], [2], [3]] : tensor<1x1x4x6xf32> into tensor<1x4x6xf32>
    return %collapsed_2 : tensor<1x4x6xf32>
  }
}

