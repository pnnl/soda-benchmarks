#map = affine_map<(d0, d1, d2, d3) -> (0, 0, d2, d3)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module attributes {torch.debug_module_name = "SimpleMLP"} {
  func.func @forward(%arg0: memref<1x8x16xf32, strided<[?, ?, ?], offset: ?>>, %arg1: memref<1x16x12xf32, strided<[?, ?, ?], offset: ?>>, %arg2: memref<1x4x6xf32>) {
    %cst = arith.constant 3.40282347E+38 : f32
    %cst_0 = arith.constant -3.40282347E+38 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x8x12xf32>
    linalg.fill ins(%cst_1 : f32) outs(%alloc : memref<1x8x12xf32>)
    linalg.batch_matmul ins(%arg0, %arg1 : memref<1x8x16xf32, strided<[?, ?, ?], offset: ?>>, memref<1x16x12xf32, strided<[?, ?, ?], offset: ?>>) outs(%alloc : memref<1x8x12xf32>)
    %expand_shape = memref.expand_shape %alloc [[0], [1], [2, 3]] output_shape [1, 8, 12, 1] : memref<1x8x12xf32> into memref<1x8x12x1xf32>
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x4x6x1xf32>
    linalg.fill ins(%cst_0 : f32) outs(%alloc_2 : memref<1x4x6x1xf32>)
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<2x2xf32>
    linalg.pooling_nhwc_max {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins(%expand_shape, %alloc_3 : memref<1x8x12x1xf32>, memref<2x2xf32>) outs(%alloc_2 : memref<1x4x6x1xf32>)
    %collapse_shape = memref.collapse_shape %alloc_2 [[0], [1], [2, 3]] : memref<1x4x6x1xf32> into memref<1x4x6xf32>
    %expand_shape_4 = memref.expand_shape %collapse_shape [[0, 1], [2], [3]] output_shape [1, 1, 4, 6] : memref<1x4x6xf32> into memref<1x1x4x6xf32>
    %alloc_5 = memref.alloc() {alignment = 64 : i64} : memref<1x1x4x6xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expand_shape_4 : memref<1x1x4x6xf32>) outs(%alloc_5 : memref<1x1x4x6xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.minimumf %in, %cst : f32
      %1 = arith.maximumf %0, %cst_1 : f32
      linalg.yield %1 : f32
    }
    %collapse_shape_6 = memref.collapse_shape %alloc_5 [[0, 1], [2], [3]] : memref<1x1x4x6xf32> into memref<1x4x6xf32>
    memref.copy %collapse_shape_6, %arg2 : memref<1x4x6xf32> to memref<1x4x6xf32>
    return
  }
}

