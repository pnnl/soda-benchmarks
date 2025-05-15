module attributes {torch.debug_module_name = "SimpleMLP"} {
  func.func @forward(%arg0: tensor<1x8x16xf32>, %arg1: tensor<1x16x12xf32>) -> tensor<1x4x6xf32> {
    %0 = tosa.matmul %arg0, %arg1 : (tensor<1x8x16xf32>, tensor<1x16x12xf32>) -> tensor<1x8x12xf32>
    %1 = tosa.reshape %0 {new_shape = array<i64: 1, 8, 12, 1>} : (tensor<1x8x12xf32>) -> tensor<1x8x12x1xf32>
    %2 = tosa.max_pool2d %1 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x8x12x1xf32>) -> tensor<1x4x6x1xf32>
    %3 = tosa.reshape %2 {new_shape = array<i64: 1, 1, 4, 6>} : (tensor<1x4x6x1xf32>) -> tensor<1x1x4x6xf32>
    %4 = tosa.clamp %3 {max_fp = 3.40282347E+38 : f32, max_int = 2147483647 : i64, min_fp = 0.000000e+00 : f32, min_int = 0 : i64} : (tensor<1x1x4x6xf32>) -> tensor<1x1x4x6xf32>
    %5 = tosa.reshape %4 {new_shape = array<i64: 1, 4, 6>} : (tensor<1x1x4x6xf32>) -> tensor<1x4x6xf32>
    return %5 : tensor<1x4x6xf32>
  }
}
