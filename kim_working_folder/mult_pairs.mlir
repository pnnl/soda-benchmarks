#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d2, d1)>
#map5 = affine_map<(d0, d1) -> (d0)>
#map6 = affine_map<(d0, d1) -> (d1)>
#map7 = affine_map<(d0, d1, d2) -> (d0)>
#map8 = affine_map<(d0, d1, d2) -> (d1)>
#map9 = affine_map<(d0, d1, d2) -> (d2)>
module {
  func.func @ops_1d(%arg0: memref<8xf32>, %arg1: memref<8xf32>, %arg2: memref<8xf32>, %arg3: memref<8xf32, strided<[2], offset: 1>>, %arg4: memref<8xf32, strided<[2], offset: 1>>) {
    linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<8xf32>, memref<8xf32>) outs(%arg2 : memref<8xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg0 : memref<8xf32>) outs(%arg2 : memref<8xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.negf %in : f32
      linalg.yield %0 : f32
    }
    linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg3 : memref<8xf32, strided<[2], offset: 1>>) outs(%arg4 : memref<8xf32, strided<[2], offset: 1>>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.negf %in : f32
      linalg.yield %0 : f32
    }
    return
  }
  func.func @ops_2d(%arg0: memref<4x6xf32>, %arg1: memref<4x6xf32>, %arg2: memref<4x6xf32>, %arg3: memref<4x6xf32, strided<[8, 1]>>, %arg4: memref<4x6xf32, strided<[8, 1]>>, %arg5: memref<4x6xf32, strided<[16, 2], offset: 3>>, %arg6: memref<4x6xf32, strided<[16, 2], offset: 3>>) {
    linalg.generic {indexing_maps = [#map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x6xf32>, memref<4x6xf32>) outs(%arg2 : memref<4x6xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<4x6xf32>, memref<4x6xf32>) outs(%arg2 : memref<4x6xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg3 : memref<4x6xf32, strided<[8, 1]>>) outs(%arg4 : memref<4x6xf32, strided<[8, 1]>>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.negf %in : f32
      linalg.yield %0 : f32
    }
    linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg5 : memref<4x6xf32, strided<[16, 2], offset: 3>>) outs(%arg6 : memref<4x6xf32, strided<[16, 2], offset: 3>>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.negf %in : f32
      linalg.yield %0 : f32
    }
    return
  }
  func.func @ops_3d(%arg0: memref<2x3x4xf32>, %arg1: memref<2x3x4xf32>, %arg2: memref<2x3x4xf32>, %arg3: memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>, %arg4: memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>, %arg5: memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>) {
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x3x4xf32>, memref<2x3x4xf32>) outs(%arg2 : memref<2x3x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %arg1 : memref<2x3x4xf32>, memref<2x3x4xf32>) outs(%arg2 : memref<2x3x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    linalg.generic {indexing_maps = [#map3, #map3, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg3, %arg4 : memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>, memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>) outs(%arg5 : memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    return
  }
  func.func private @format_address_pair(i64, i64) -> i64
  func.func private @gen_addr_BD24D677EBA71AAF(%arg0: i64, %arg1: memref<8xf32>) -> i64 {
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg1 : memref<8xf32> -> memref<f32>, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = affine.apply #map(%0)
    %2 = arith.muli %1, %strides : index
    %3 = arith.addi %offset, %2 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<8xf32> -> index
    %c4 = arith.constant 4 : index
    %4 = arith.muli %3, %c4 : index
    %5 = arith.addi %intptr, %4 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.index_cast %4 : index to i64
    %8 = call @format_address_pair(%6, %7) : (i64, i64) -> i64
    return %8 : i64
  }
  func.func private @gen_addr_3C62F6BF06123A3A(%arg0: i64, %arg1: memref<8xf32, strided<[2], offset: 1>>) -> i64 {
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg1 : memref<8xf32, strided<[2], offset: 1>> -> memref<f32>, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = affine.apply #map(%0)
    %2 = arith.muli %1, %strides : index
    %3 = arith.addi %offset, %2 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<8xf32, strided<[2], offset: 1>> -> index
    %c4 = arith.constant 4 : index
    %4 = arith.muli %3, %c4 : index
    %5 = arith.addi %intptr, %4 : index
    %6 = arith.index_cast %5 : index to i64
    %7 = arith.index_cast %4 : index to i64
    %8 = call @format_address_pair(%6, %7) : (i64, i64) -> i64
    return %8 : i64
  }
  func.func private @gen_addr_3F5FA80A2FA68935(%arg0: i64, %arg1: memref<4x6xf32>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#1 : index
    %2 = arith.divui %0, %sizes#1 : index
    %3 = affine.apply #map5(%2, %1)
    %4 = affine.apply #map6(%2, %1)
    %5 = arith.muli %3, %strides#0 : index
    %6 = arith.addi %offset, %5 : index
    %7 = arith.muli %4, %strides#1 : index
    %8 = arith.addi %6, %7 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4x6xf32> -> index
    %c4 = arith.constant 4 : index
    %9 = arith.muli %8, %c4 : index
    %10 = arith.addi %intptr, %9 : index
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.index_cast %9 : index to i64
    %13 = call @format_address_pair(%11, %12) : (i64, i64) -> i64
    return %13 : i64
  }
  func.func private @gen_addr_B4CE78009D95094C(%arg0: i64, %arg1: memref<4x6xf32>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#0 : index
    %2 = arith.divui %0, %sizes#0 : index
    %3 = affine.apply #map6(%2, %1)
    %4 = affine.apply #map5(%2, %1)
    %5 = arith.muli %3, %strides#0 : index
    %6 = arith.addi %offset, %5 : index
    %7 = arith.muli %4, %strides#1 : index
    %8 = arith.addi %6, %7 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4x6xf32> -> index
    %c4 = arith.constant 4 : index
    %9 = arith.muli %8, %c4 : index
    %10 = arith.addi %intptr, %9 : index
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.index_cast %9 : index to i64
    %13 = call @format_address_pair(%11, %12) : (i64, i64) -> i64
    return %13 : i64
  }
  func.func private @gen_addr_15787FD38E08381C(%arg0: i64, %arg1: memref<4x6xf32, strided<[8, 1]>>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<4x6xf32, strided<[8, 1]>> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#1 : index
    %2 = arith.divui %0, %sizes#1 : index
    %3 = affine.apply #map5(%2, %1)
    %4 = affine.apply #map6(%2, %1)
    %5 = arith.muli %3, %strides#0 : index
    %6 = arith.addi %offset, %5 : index
    %7 = arith.muli %4, %strides#1 : index
    %8 = arith.addi %6, %7 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4x6xf32, strided<[8, 1]>> -> index
    %c4 = arith.constant 4 : index
    %9 = arith.muli %8, %c4 : index
    %10 = arith.addi %intptr, %9 : index
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.index_cast %9 : index to i64
    %13 = call @format_address_pair(%11, %12) : (i64, i64) -> i64
    return %13 : i64
  }
  func.func private @gen_addr_CA28281C32BBACB9(%arg0: i64, %arg1: memref<4x6xf32, strided<[16, 2], offset: 3>>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<4x6xf32, strided<[16, 2], offset: 3>> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#1 : index
    %2 = arith.divui %0, %sizes#1 : index
    %3 = affine.apply #map5(%2, %1)
    %4 = affine.apply #map6(%2, %1)
    %5 = arith.muli %3, %strides#0 : index
    %6 = arith.addi %offset, %5 : index
    %7 = arith.muli %4, %strides#1 : index
    %8 = arith.addi %6, %7 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4x6xf32, strided<[16, 2], offset: 3>> -> index
    %c4 = arith.constant 4 : index
    %9 = arith.muli %8, %c4 : index
    %10 = arith.addi %intptr, %9 : index
    %11 = arith.index_cast %10 : index to i64
    %12 = arith.index_cast %9 : index to i64
    %13 = call @format_address_pair(%11, %12) : (i64, i64) -> i64
    return %13 : i64
  }
  func.func private @gen_addr_77D7486F1DFE5E10(%arg0: i64, %arg1: memref<2x3x4xf32>) -> i64 {
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg1 : memref<2x3x4xf32> -> memref<f32>, index, index, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#2 : index
    %2 = arith.divui %0, %sizes#2 : index
    %3 = arith.remui %2, %sizes#1 : index
    %4 = arith.divui %2, %sizes#1 : index
    %5 = affine.apply #map7(%4, %3, %1)
    %6 = affine.apply #map8(%4, %3, %1)
    %7 = affine.apply #map9(%4, %3, %1)
    %8 = arith.muli %5, %strides#0 : index
    %9 = arith.addi %offset, %8 : index
    %10 = arith.muli %6, %strides#1 : index
    %11 = arith.addi %9, %10 : index
    %12 = arith.muli %7, %strides#2 : index
    %13 = arith.addi %11, %12 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<2x3x4xf32> -> index
    %c4 = arith.constant 4 : index
    %14 = arith.muli %13, %c4 : index
    %15 = arith.addi %intptr, %14 : index
    %16 = arith.index_cast %15 : index to i64
    %17 = arith.index_cast %14 : index to i64
    %18 = call @format_address_pair(%16, %17) : (i64, i64) -> i64
    return %18 : i64
  }
  func.func private @gen_addr_356660CF787F0086(%arg0: i64, %arg1: memref<2x3x4xf32>) -> i64 {
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg1 : memref<2x3x4xf32> -> memref<f32>, index, index, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#1 : index
    %2 = arith.divui %0, %sizes#1 : index
    %3 = arith.remui %2, %sizes#2 : index
    %4 = arith.divui %2, %sizes#2 : index
    %5 = affine.apply #map7(%4, %3, %1)
    %6 = affine.apply #map9(%4, %3, %1)
    %7 = affine.apply #map8(%4, %3, %1)
    %8 = arith.muli %5, %strides#0 : index
    %9 = arith.addi %offset, %8 : index
    %10 = arith.muli %6, %strides#1 : index
    %11 = arith.addi %9, %10 : index
    %12 = arith.muli %7, %strides#2 : index
    %13 = arith.addi %11, %12 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<2x3x4xf32> -> index
    %c4 = arith.constant 4 : index
    %14 = arith.muli %13, %c4 : index
    %15 = arith.addi %intptr, %14 : index
    %16 = arith.index_cast %15 : index to i64
    %17 = arith.index_cast %14 : index to i64
    %18 = call @format_address_pair(%16, %17) : (i64, i64) -> i64
    return %18 : i64
  }
  func.func private @gen_addr_A031EAD10AD123D0(%arg0: i64, %arg1: memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>) -> i64 {
    %base_buffer, %offset, %sizes:3, %strides:3 = memref.extract_strided_metadata %arg1 : memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>> -> memref<f32>, index, index, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#2 : index
    %2 = arith.divui %0, %sizes#2 : index
    %3 = arith.remui %2, %sizes#1 : index
    %4 = arith.divui %2, %sizes#1 : index
    %5 = affine.apply #map7(%4, %3, %1)
    %6 = affine.apply #map8(%4, %3, %1)
    %7 = affine.apply #map9(%4, %3, %1)
    %8 = arith.muli %5, %strides#0 : index
    %9 = arith.addi %offset, %8 : index
    %10 = arith.muli %6, %strides#1 : index
    %11 = arith.addi %9, %10 : index
    %12 = arith.muli %7, %strides#2 : index
    %13 = arith.addi %11, %12 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>> -> index
    %c4 = arith.constant 4 : index
    %14 = arith.muli %13, %c4 : index
    %15 = arith.addi %intptr, %14 : index
    %16 = arith.index_cast %15 : index to i64
    %17 = arith.index_cast %14 : index to i64
    %18 = call @format_address_pair(%16, %17) : (i64, i64) -> i64
    return %18 : i64
  }
}

