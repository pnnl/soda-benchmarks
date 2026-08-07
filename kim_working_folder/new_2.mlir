#map = affine_map<(d0, d1, d2) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1) -> (d1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#map5 = affine_map<(d0, d1, d2) -> (d1, d0)>
#map6 = affine_map<(d0, d1, d2) -> (d0)>
#map7 = affine_map<(d0, d1, d2) -> (d1)>
#map8 = affine_map<(d0, d1, d2) -> (d2)>
#map9 = affine_map<(d0, d1) -> (d0)>
module {
  func.func @matmul(%arg0: memref<4x8xf32>, %arg1: memref<8x6xf32>, %arg2: memref<4x6xf32>, %arg3: memref<4x8xf32, strided<[16, 1]>>, %arg4: memref<8x6xf32, strided<[8, 1]>>, %arg5: memref<4x6xf32, strided<[8, 1]>>) {
    call @gen_trace_BBBFA08CC1B9C093(%arg0, %arg1, %arg2) : (memref<4x8xf32>, memref<8x6xf32>, memref<4x6xf32>) -> ()
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : memref<4x8xf32>, memref<8x6xf32>) outs(%arg2 : memref<4x6xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %0, %out : f32
      linalg.yield %1 : f32
    }
    call @gen_trace_6D00D03429886E50(%arg3, %arg4, %arg5) : (memref<4x8xf32, strided<[16, 1]>>, memref<8x6xf32, strided<[8, 1]>>, memref<4x6xf32, strided<[8, 1]>>) -> ()
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg3, %arg4 : memref<4x8xf32, strided<[16, 1]>>, memref<8x6xf32, strided<[8, 1]>>) outs(%arg5 : memref<4x6xf32, strided<[8, 1]>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %0, %out : f32
      linalg.yield %1 : f32
    }
    return
  }
  func.func @edge_cases(%arg0: memref<6xf32>, %arg1: memref<4x6xf32>, %arg2: memref<4x6xf32>, %arg3: memref<8x4xf32>, %arg4: memref<8x6xf32>, %arg5: memref<4x6xf32>) {
    call @gen_trace_54384F8228B68719(%arg0, %arg1, %arg2) : (memref<6xf32>, memref<4x6xf32>, memref<4x6xf32>) -> ()
    linalg.generic {indexing_maps = [#map3, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<6xf32>, memref<4x6xf32>) outs(%arg2 : memref<4x6xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
    call @gen_trace_8F9610077BE2B9C9(%arg3, %arg4, %arg5) : (memref<8x4xf32>, memref<8x6xf32>, memref<4x6xf32>) -> ()
    linalg.generic {indexing_maps = [#map5, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg3, %arg4 : memref<8x4xf32>, memref<8x6xf32>) outs(%arg5 : memref<4x6xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %0, %out : f32
      linalg.yield %1 : f32
    }
    return
  }
  func.func private @format_address_pair(i64, i64) -> i64
  func.func private @gen_addr_920B00BBA9FACB8B(%arg0: i64, %arg1: memref<4x8xf32>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<4x8xf32> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %c1 : index
    %2 = arith.divui %0, %c1 : index
    %3 = arith.remui %2, %sizes#1 : index
    %4 = arith.divui %2, %sizes#1 : index
    %5 = affine.apply #map6(%4, %3, %1)
    %6 = affine.apply #map7(%4, %3, %1)
    %7 = arith.muli %5, %strides#0 : index
    %8 = arith.addi %offset, %7 : index
    %9 = arith.muli %6, %strides#1 : index
    %10 = arith.addi %8, %9 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4x8xf32> -> index
    %c4 = arith.constant 4 : index
    %11 = arith.muli %10, %c4 : index
    %12 = arith.addi %intptr, %11 : index
    %13 = arith.index_cast %12 : index to i64
    %14 = arith.index_cast %11 : index to i64
    %15 = call @format_address_pair(%13, %14) : (i64, i64) -> i64
    return %15 : i64
  }
  func.func private @gen_addr_F270B5B28D0A4C21(%arg0: i64, %arg1: memref<8x6xf32>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<8x6xf32> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#1 : index
    %2 = arith.divui %0, %sizes#1 : index
    %3 = arith.remui %2, %sizes#0 : index
    %4 = arith.divui %2, %sizes#0 : index
    %5 = affine.apply #map7(%4, %3, %1)
    %6 = affine.apply #map8(%4, %3, %1)
    %7 = arith.muli %5, %strides#0 : index
    %8 = arith.addi %offset, %7 : index
    %9 = arith.muli %6, %strides#1 : index
    %10 = arith.addi %8, %9 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<8x6xf32> -> index
    %c4 = arith.constant 4 : index
    %11 = arith.muli %10, %c4 : index
    %12 = arith.addi %intptr, %11 : index
    %13 = arith.index_cast %12 : index to i64
    %14 = arith.index_cast %11 : index to i64
    %15 = call @format_address_pair(%13, %14) : (i64, i64) -> i64
    return %15 : i64
  }
  func.func private @gen_addr_3AC88FC0BAF54F4F(%arg0: i64, %arg1: memref<4x6xf32>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#1 : index
    %2 = arith.divui %0, %sizes#1 : index
    %3 = arith.remui %2, %c1 : index
    %4 = arith.divui %2, %c1 : index
    %5 = affine.apply #map6(%4, %3, %1)
    %6 = affine.apply #map8(%4, %3, %1)
    %7 = arith.muli %5, %strides#0 : index
    %8 = arith.addi %offset, %7 : index
    %9 = arith.muli %6, %strides#1 : index
    %10 = arith.addi %8, %9 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4x6xf32> -> index
    %c4 = arith.constant 4 : index
    %11 = arith.muli %10, %c4 : index
    %12 = arith.addi %intptr, %11 : index
    %13 = arith.index_cast %12 : index to i64
    %14 = arith.index_cast %11 : index to i64
    %15 = call @format_address_pair(%13, %14) : (i64, i64) -> i64
    return %15 : i64
  }
  func.func private @gen_addr_2A59C30F3FF4E8BE(%arg0: i64, %arg1: memref<4x8xf32, strided<[16, 1]>>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<4x8xf32, strided<[16, 1]>> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %c1 : index
    %2 = arith.divui %0, %c1 : index
    %3 = arith.remui %2, %sizes#1 : index
    %4 = arith.divui %2, %sizes#1 : index
    %5 = affine.apply #map6(%4, %3, %1)
    %6 = affine.apply #map7(%4, %3, %1)
    %7 = arith.muli %5, %strides#0 : index
    %8 = arith.addi %offset, %7 : index
    %9 = arith.muli %6, %strides#1 : index
    %10 = arith.addi %8, %9 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4x8xf32, strided<[16, 1]>> -> index
    %c4 = arith.constant 4 : index
    %11 = arith.muli %10, %c4 : index
    %12 = arith.addi %intptr, %11 : index
    %13 = arith.index_cast %12 : index to i64
    %14 = arith.index_cast %11 : index to i64
    %15 = call @format_address_pair(%13, %14) : (i64, i64) -> i64
    return %15 : i64
  }
  func.func private @gen_addr_4A401C3FFBD877E9(%arg0: i64, %arg1: memref<8x6xf32, strided<[8, 1]>>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<8x6xf32, strided<[8, 1]>> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#1 : index
    %2 = arith.divui %0, %sizes#1 : index
    %3 = arith.remui %2, %sizes#0 : index
    %4 = arith.divui %2, %sizes#0 : index
    %5 = affine.apply #map7(%4, %3, %1)
    %6 = affine.apply #map8(%4, %3, %1)
    %7 = arith.muli %5, %strides#0 : index
    %8 = arith.addi %offset, %7 : index
    %9 = arith.muli %6, %strides#1 : index
    %10 = arith.addi %8, %9 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<8x6xf32, strided<[8, 1]>> -> index
    %c4 = arith.constant 4 : index
    %11 = arith.muli %10, %c4 : index
    %12 = arith.addi %intptr, %11 : index
    %13 = arith.index_cast %12 : index to i64
    %14 = arith.index_cast %11 : index to i64
    %15 = call @format_address_pair(%13, %14) : (i64, i64) -> i64
    return %15 : i64
  }
  func.func private @gen_addr_22D50E118B951A4E(%arg0: i64, %arg1: memref<4x6xf32, strided<[8, 1]>>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<4x6xf32, strided<[8, 1]>> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#1 : index
    %2 = arith.divui %0, %sizes#1 : index
    %3 = arith.remui %2, %c1 : index
    %4 = arith.divui %2, %c1 : index
    %5 = affine.apply #map6(%4, %3, %1)
    %6 = affine.apply #map8(%4, %3, %1)
    %7 = arith.muli %5, %strides#0 : index
    %8 = arith.addi %offset, %7 : index
    %9 = arith.muli %6, %strides#1 : index
    %10 = arith.addi %8, %9 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<4x6xf32, strided<[8, 1]>> -> index
    %c4 = arith.constant 4 : index
    %11 = arith.muli %10, %c4 : index
    %12 = arith.addi %intptr, %11 : index
    %13 = arith.index_cast %12 : index to i64
    %14 = arith.index_cast %11 : index to i64
    %15 = call @format_address_pair(%13, %14) : (i64, i64) -> i64
    return %15 : i64
  }
  func.func private @gen_addr_9C210D0E4AC5E793(%arg0: i64, %arg1: memref<6xf32>) -> i64 {
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg1 : memref<6xf32> -> memref<f32>, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes : index
    %2 = arith.divui %0, %sizes : index
    %3 = affine.apply #map3(%2, %1)
    %4 = arith.muli %3, %strides : index
    %5 = arith.addi %offset, %4 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<6xf32> -> index
    %c4 = arith.constant 4 : index
    %6 = arith.muli %5, %c4 : index
    %7 = arith.addi %intptr, %6 : index
    %8 = arith.index_cast %7 : index to i64
    %9 = arith.index_cast %6 : index to i64
    %10 = call @format_address_pair(%8, %9) : (i64, i64) -> i64
    return %10 : i64
  }
  func.func private @gen_addr_3F5FA80A2FA68935(%arg0: i64, %arg1: memref<4x6xf32>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %sizes#1 : index
    %2 = arith.divui %0, %sizes#1 : index
    %3 = affine.apply #map9(%2, %1)
    %4 = affine.apply #map3(%2, %1)
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
  func.func private @gen_addr_49E1FA1CB4998AE4(%arg0: i64, %arg1: memref<8x4xf32>) -> i64 {
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg1 : memref<8x4xf32> -> memref<f32>, index, index, index, index, index
    %c1 = arith.constant 1 : index
    %0 = arith.index_cast %arg0 : i64 to index
    %1 = arith.remui %0, %c1 : index
    %2 = arith.divui %0, %c1 : index
    %3 = arith.remui %2, %sizes#0 : index
    %4 = arith.divui %2, %sizes#0 : index
    %5 = affine.apply #map7(%4, %3, %1)
    %6 = affine.apply #map6(%4, %3, %1)
    %7 = arith.muli %5, %strides#0 : index
    %8 = arith.addi %offset, %7 : index
    %9 = arith.muli %6, %strides#1 : index
    %10 = arith.addi %8, %9 : index
    %intptr = memref.extract_aligned_pointer_as_index %arg1 : memref<8x4xf32> -> index
    %c4 = arith.constant 4 : index
    %11 = arith.muli %10, %c4 : index
    %12 = arith.addi %intptr, %11 : index
    %13 = arith.index_cast %12 : index to i64
    %14 = arith.index_cast %11 : index to i64
    %15 = call @format_address_pair(%13, %14) : (i64, i64) -> i64
    return %15 : i64
  }
  func.func private @gen_trace_BBBFA08CC1B9C093(%arg0: memref<4x8xf32>, %arg1: memref<8x6xf32>, %arg2: memref<4x6xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<4x8xf32> -> memref<f32>, index, index, index, index, index
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %0 = arith.muli %c1_1, %sizes#0 : index
    %1 = arith.muli %0, %sizes#1 : index
    %2 = arith.muli %1, %c1_0 : index
    %3 = arith.cmpi ugt, %2, %c0 : index
    %4 = arith.select %3, %2, %c0 : index
    %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg1 : memref<8x6xf32> -> memref<f32>, index, index, index, index, index
    %c1_6 = arith.constant 1 : index
    %c1_7 = arith.constant 1 : index
    %5 = arith.muli %c1_7, %c1_6 : index
    %6 = arith.muli %5, %sizes_4#0 : index
    %7 = arith.muli %6, %sizes_4#1 : index
    %8 = arith.cmpi ugt, %7, %4 : index
    %9 = arith.select %8, %7, %4 : index
    %base_buffer_8, %offset_9, %sizes_10:2, %strides_11:2 = memref.extract_strided_metadata %arg2 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1_12 = arith.constant 1 : index
    %c1_13 = arith.constant 1 : index
    %10 = arith.muli %c1_13, %sizes_10#0 : index
    %11 = arith.muli %10, %c1_12 : index
    %12 = arith.muli %11, %sizes_10#1 : index
    %13 = arith.cmpi ugt, %12, %9 : index
    %14 = arith.select %13, %12, %9 : index
    %15 = arith.muli %14, %c3 : index
    scf.for %arg3 = %c0 to %15 step %c1 {
      %16 = arith.remui %arg3, %c3 : index
      %17 = arith.divui %arg3, %c3 : index
      %c0_14 = arith.constant 0 : index
      %18 = arith.cmpi eq, %16, %c0_14 : index
      %19 = arith.cmpi ult, %17, %2 : index
      %20 = arith.andi %18, %19 : i1
      scf.if %20 {
        %27 = arith.index_cast %17 : index to i64
        %28 = func.call @gen_addr_920B00BBA9FACB8B(%27, %arg0) : (i64, memref<4x8xf32>) -> i64
      }
      %c1_15 = arith.constant 1 : index
      %21 = arith.cmpi eq, %16, %c1_15 : index
      %22 = arith.cmpi ult, %17, %7 : index
      %23 = arith.andi %21, %22 : i1
      scf.if %23 {
        %27 = arith.index_cast %17 : index to i64
        %28 = func.call @gen_addr_F270B5B28D0A4C21(%27, %arg1) : (i64, memref<8x6xf32>) -> i64
      }
      %c2 = arith.constant 2 : index
      %24 = arith.cmpi eq, %16, %c2 : index
      %25 = arith.cmpi ult, %17, %12 : index
      %26 = arith.andi %24, %25 : i1
      scf.if %26 {
        %27 = arith.index_cast %17 : index to i64
        %28 = func.call @gen_addr_3AC88FC0BAF54F4F(%27, %arg2) : (i64, memref<4x6xf32>) -> i64
      }
    }
    return
  }
  func.func private @gen_trace_6D00D03429886E50(%arg0: memref<4x8xf32, strided<[16, 1]>>, %arg1: memref<8x6xf32, strided<[8, 1]>>, %arg2: memref<4x6xf32, strided<[8, 1]>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<4x8xf32, strided<[16, 1]>> -> memref<f32>, index, index, index, index, index
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %0 = arith.muli %c1_1, %sizes#0 : index
    %1 = arith.muli %0, %sizes#1 : index
    %2 = arith.muli %1, %c1_0 : index
    %3 = arith.cmpi ugt, %2, %c0 : index
    %4 = arith.select %3, %2, %c0 : index
    %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg1 : memref<8x6xf32, strided<[8, 1]>> -> memref<f32>, index, index, index, index, index
    %c1_6 = arith.constant 1 : index
    %c1_7 = arith.constant 1 : index
    %5 = arith.muli %c1_7, %c1_6 : index
    %6 = arith.muli %5, %sizes_4#0 : index
    %7 = arith.muli %6, %sizes_4#1 : index
    %8 = arith.cmpi ugt, %7, %4 : index
    %9 = arith.select %8, %7, %4 : index
    %base_buffer_8, %offset_9, %sizes_10:2, %strides_11:2 = memref.extract_strided_metadata %arg2 : memref<4x6xf32, strided<[8, 1]>> -> memref<f32>, index, index, index, index, index
    %c1_12 = arith.constant 1 : index
    %c1_13 = arith.constant 1 : index
    %10 = arith.muli %c1_13, %sizes_10#0 : index
    %11 = arith.muli %10, %c1_12 : index
    %12 = arith.muli %11, %sizes_10#1 : index
    %13 = arith.cmpi ugt, %12, %9 : index
    %14 = arith.select %13, %12, %9 : index
    %15 = arith.muli %14, %c3 : index
    scf.for %arg3 = %c0 to %15 step %c1 {
      %16 = arith.remui %arg3, %c3 : index
      %17 = arith.divui %arg3, %c3 : index
      %c0_14 = arith.constant 0 : index
      %18 = arith.cmpi eq, %16, %c0_14 : index
      %19 = arith.cmpi ult, %17, %2 : index
      %20 = arith.andi %18, %19 : i1
      scf.if %20 {
        %27 = arith.index_cast %17 : index to i64
        %28 = func.call @gen_addr_2A59C30F3FF4E8BE(%27, %arg0) : (i64, memref<4x8xf32, strided<[16, 1]>>) -> i64
      }
      %c1_15 = arith.constant 1 : index
      %21 = arith.cmpi eq, %16, %c1_15 : index
      %22 = arith.cmpi ult, %17, %7 : index
      %23 = arith.andi %21, %22 : i1
      scf.if %23 {
        %27 = arith.index_cast %17 : index to i64
        %28 = func.call @gen_addr_4A401C3FFBD877E9(%27, %arg1) : (i64, memref<8x6xf32, strided<[8, 1]>>) -> i64
      }
      %c2 = arith.constant 2 : index
      %24 = arith.cmpi eq, %16, %c2 : index
      %25 = arith.cmpi ult, %17, %12 : index
      %26 = arith.andi %24, %25 : i1
      scf.if %26 {
        %27 = arith.index_cast %17 : index to i64
        %28 = func.call @gen_addr_22D50E118B951A4E(%27, %arg2) : (i64, memref<4x6xf32, strided<[8, 1]>>) -> i64
      }
    }
    return
  }
  func.func private @gen_trace_54384F8228B68719(%arg0: memref<6xf32>, %arg1: memref<4x6xf32>, %arg2: memref<4x6xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<6xf32> -> memref<f32>, index, index, index
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %0 = arith.muli %c1_1, %c1_0 : index
    %1 = arith.muli %0, %sizes : index
    %2 = arith.cmpi ugt, %1, %c0 : index
    %3 = arith.select %2, %1, %c0 : index
    %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg1 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1_6 = arith.constant 1 : index
    %c1_7 = arith.constant 1 : index
    %4 = arith.muli %c1_7, %sizes_4#0 : index
    %5 = arith.muli %4, %sizes_4#1 : index
    %6 = arith.cmpi ugt, %5, %3 : index
    %7 = arith.select %6, %5, %3 : index
    %base_buffer_8, %offset_9, %sizes_10:2, %strides_11:2 = memref.extract_strided_metadata %arg2 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1_12 = arith.constant 1 : index
    %c1_13 = arith.constant 1 : index
    %8 = arith.muli %c1_13, %sizes_10#0 : index
    %9 = arith.muli %8, %sizes_10#1 : index
    %10 = arith.cmpi ugt, %9, %7 : index
    %11 = arith.select %10, %9, %7 : index
    %12 = arith.muli %11, %c3 : index
    scf.for %arg3 = %c0 to %12 step %c1 {
      %13 = arith.remui %arg3, %c3 : index
      %14 = arith.divui %arg3, %c3 : index
      %c0_14 = arith.constant 0 : index
      %15 = arith.cmpi eq, %13, %c0_14 : index
      %16 = arith.cmpi ult, %14, %1 : index
      %17 = arith.andi %15, %16 : i1
      scf.if %17 {
        %24 = arith.index_cast %14 : index to i64
        %25 = func.call @gen_addr_9C210D0E4AC5E793(%24, %arg0) : (i64, memref<6xf32>) -> i64
      }
      %c1_15 = arith.constant 1 : index
      %18 = arith.cmpi eq, %13, %c1_15 : index
      %19 = arith.cmpi ult, %14, %5 : index
      %20 = arith.andi %18, %19 : i1
      scf.if %20 {
        %24 = arith.index_cast %14 : index to i64
        %25 = func.call @gen_addr_3F5FA80A2FA68935(%24, %arg1) : (i64, memref<4x6xf32>) -> i64
      }
      %c2 = arith.constant 2 : index
      %21 = arith.cmpi eq, %13, %c2 : index
      %22 = arith.cmpi ult, %14, %9 : index
      %23 = arith.andi %21, %22 : i1
      scf.if %23 {
        %24 = arith.index_cast %14 : index to i64
        %25 = func.call @gen_addr_3F5FA80A2FA68935(%24, %arg2) : (i64, memref<4x6xf32>) -> i64
      }
    }
    return
  }
  func.func private @gen_trace_8F9610077BE2B9C9(%arg0: memref<8x4xf32>, %arg1: memref<8x6xf32>, %arg2: memref<4x6xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<8x4xf32> -> memref<f32>, index, index, index, index, index
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    %0 = arith.muli %c1_1, %sizes#1 : index
    %1 = arith.muli %0, %sizes#0 : index
    %2 = arith.muli %1, %c1_0 : index
    %3 = arith.cmpi ugt, %2, %c0 : index
    %4 = arith.select %3, %2, %c0 : index
    %base_buffer_2, %offset_3, %sizes_4:2, %strides_5:2 = memref.extract_strided_metadata %arg1 : memref<8x6xf32> -> memref<f32>, index, index, index, index, index
    %c1_6 = arith.constant 1 : index
    %c1_7 = arith.constant 1 : index
    %5 = arith.muli %c1_7, %c1_6 : index
    %6 = arith.muli %5, %sizes_4#0 : index
    %7 = arith.muli %6, %sizes_4#1 : index
    %8 = arith.cmpi ugt, %7, %4 : index
    %9 = arith.select %8, %7, %4 : index
    %base_buffer_8, %offset_9, %sizes_10:2, %strides_11:2 = memref.extract_strided_metadata %arg2 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1_12 = arith.constant 1 : index
    %c1_13 = arith.constant 1 : index
    %10 = arith.muli %c1_13, %sizes_10#0 : index
    %11 = arith.muli %10, %c1_12 : index
    %12 = arith.muli %11, %sizes_10#1 : index
    %13 = arith.cmpi ugt, %12, %9 : index
    %14 = arith.select %13, %12, %9 : index
    %15 = arith.muli %14, %c3 : index
    scf.for %arg3 = %c0 to %15 step %c1 {
      %16 = arith.remui %arg3, %c3 : index
      %17 = arith.divui %arg3, %c3 : index
      %c0_14 = arith.constant 0 : index
      %18 = arith.cmpi eq, %16, %c0_14 : index
      %19 = arith.cmpi ult, %17, %2 : index
      %20 = arith.andi %18, %19 : i1
      scf.if %20 {
        %27 = arith.index_cast %17 : index to i64
        %28 = func.call @gen_addr_49E1FA1CB4998AE4(%27, %arg0) : (i64, memref<8x4xf32>) -> i64
      }
      %c1_15 = arith.constant 1 : index
      %21 = arith.cmpi eq, %16, %c1_15 : index
      %22 = arith.cmpi ult, %17, %7 : index
      %23 = arith.andi %21, %22 : i1
      scf.if %23 {
        %27 = arith.index_cast %17 : index to i64
        %28 = func.call @gen_addr_F270B5B28D0A4C21(%27, %arg1) : (i64, memref<8x6xf32>) -> i64
      }
      %c2 = arith.constant 2 : index
      %24 = arith.cmpi eq, %16, %c2 : index
      %25 = arith.cmpi ult, %17, %12 : index
      %26 = arith.andi %24, %25 : i1
      scf.if %26 {
        %27 = arith.index_cast %17 : index to i64
        %28 = func.call @gen_addr_3AC88FC0BAF54F4F(%27, %arg2) : (i64, memref<4x6xf32>) -> i64
      }
    }
    return
  }

  func.func @main() {
    %m_4x8 = memref.alloc() : memref<4x8xf32>
    %m_8x6 = memref.alloc() : memref<8x6xf32>
    %m_4x6 = memref.alloc() : memref<4x6xf32>

    // Storage sizes:
    // 4x8 with strides [16, 1] requires 3*16 + 8 = 56 elements.
    // 8x6 with strides [8, 1] requires 7*8 + 6 = 62 elements.
    // 4x6 with strides [8, 1] requires 3*8 + 6 = 30 elements.
    %base_56 = memref.alloc() : memref<56xf32>
    %base_62 = memref.alloc() : memref<62xf32>
    %base_30 = memref.alloc() : memref<30xf32>

    %m_4x8_s = memref.reinterpret_cast %base_56
      to offset: [0], sizes: [4, 8], strides: [16, 1]
      : memref<56xf32> to memref<4x8xf32, strided<[16, 1]>>

    %m_8x6_s = memref.reinterpret_cast %base_62
      to offset: [0], sizes: [8, 6], strides: [8, 1]
      : memref<62xf32> to memref<8x6xf32, strided<[8, 1]>>

    %m_4x6_s = memref.reinterpret_cast %base_30
      to offset: [0], sizes: [4, 6], strides: [8, 1]
      : memref<30xf32> to memref<4x6xf32, strided<[8, 1]>>

    func.call @matmul(
      %m_4x8,
      %m_8x6,
      %m_4x6,
      %m_4x8_s,
      %m_8x6_s,
      %m_4x6_s
    ) : (
      memref<4x8xf32>,
      memref<8x6xf32>,
      memref<4x6xf32>,
      memref<4x8xf32, strided<[16, 1]>>,
      memref<8x6xf32, strided<[8, 1]>>,
      memref<4x6xf32, strided<[8, 1]>>
    ) -> ()

    memref.dealloc %m_4x8 : memref<4x8xf32>
    memref.dealloc %m_8x6 : memref<8x6xf32>
    memref.dealloc %m_4x6 : memref<4x6xf32>
    memref.dealloc %base_56 : memref<56xf32>
    memref.dealloc %base_62 : memref<62xf32>
    memref.dealloc %base_30 : memref<30xf32>

    return
  }
}

