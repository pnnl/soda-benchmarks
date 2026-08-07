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
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<4x8xf32> -> memref<f32>, index, index, index, index, index
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg1 : memref<8x6xf32> -> memref<f32>, index, index, index, index, index
    %base_buffer_4, %offset_5, %sizes_6:2, %strides_7:2 = memref.extract_strided_metadata %arg2 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1_8 = arith.constant 1 : index
    %0 = arith.muli %c1_8, %sizes#0 : index
    %1 = arith.muli %0, %sizes_2#1 : index
    %c1_9 = arith.constant 1 : index
    %2 = arith.muli %c1_9, %sizes#1 : index
    %c2 = arith.constant 2 : index
    %c1_10 = arith.constant 1 : index
    %3 = arith.muli %2, %c2 : index
    %4 = arith.addi %3, %c1_10 : index
    %5 = arith.muli %1, %4 : index
    scf.for %arg3 = %c0 to %5 step %c1 {
      %6 = arith.divui %arg3, %4 : index
      %7 = arith.remui %arg3, %4 : index
      %8 = arith.remui %6, %sizes_2#1 : index
      %9 = arith.divui %6, %sizes_2#1 : index
      %10 = arith.cmpi ult, %7, %3 : index
      scf.if %10 {
        %11 = arith.remui %7, %c2 : index
        %12 = arith.divui %7, %c2 : index
        %c0_11 = arith.constant 0 : index
        %13 = arith.cmpi eq, %11, %c0_11 : index
        scf.if %13 {
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %15 = arith.muli %c0_13, %sizes#0 : index
          %16 = arith.addi %15, %9 : index
          %17 = arith.muli %16, %sizes#1 : index
          %18 = arith.addi %17, %12 : index
          %19 = arith.muli %18, %c1_14 : index
          %20 = arith.addi %19, %c0_13 : index
          %21 = arith.index_cast %20 : index to i64
          %22 = func.call @gen_addr_920B00BBA9FACB8B(%21, %arg0) : (i64, memref<4x8xf32>) -> i64
        }
        %c1_12 = arith.constant 1 : index
        %14 = arith.cmpi eq, %11, %c1_12 : index
        scf.if %14 {
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %15 = arith.muli %c0_13, %c1_14 : index
          %16 = arith.addi %15, %c0_13 : index
          %17 = arith.muli %16, %sizes#1 : index
          %18 = arith.addi %17, %12 : index
          %19 = arith.muli %18, %sizes_2#1 : index
          %20 = arith.addi %19, %8 : index
          %21 = arith.index_cast %20 : index to i64
          %22 = func.call @gen_addr_F270B5B28D0A4C21(%21, %arg1) : (i64, memref<8x6xf32>) -> i64
        }
      } else {
        %11 = arith.subi %7, %3 : index
        %c0_11 = arith.constant 0 : index
        %12 = arith.cmpi eq, %11, %c0_11 : index
        scf.if %12 {
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          %13 = arith.muli %c0_12, %sizes#0 : index
          %14 = arith.addi %13, %9 : index
          %15 = arith.muli %14, %c1_13 : index
          %16 = arith.addi %15, %c0_12 : index
          %17 = arith.muli %16, %sizes_2#1 : index
          %18 = arith.addi %17, %8 : index
          %19 = arith.index_cast %18 : index to i64
          %20 = func.call @gen_addr_3AC88FC0BAF54F4F(%19, %arg2) : (i64, memref<4x6xf32>) -> i64
        }
      }
    }
    return
  }
  func.func private @gen_trace_6D00D03429886E50(%arg0: memref<4x8xf32, strided<[16, 1]>>, %arg1: memref<8x6xf32, strided<[8, 1]>>, %arg2: memref<4x6xf32, strided<[8, 1]>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<4x8xf32, strided<[16, 1]>> -> memref<f32>, index, index, index, index, index
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg1 : memref<8x6xf32, strided<[8, 1]>> -> memref<f32>, index, index, index, index, index
    %base_buffer_4, %offset_5, %sizes_6:2, %strides_7:2 = memref.extract_strided_metadata %arg2 : memref<4x6xf32, strided<[8, 1]>> -> memref<f32>, index, index, index, index, index
    %c1_8 = arith.constant 1 : index
    %0 = arith.muli %c1_8, %sizes#0 : index
    %1 = arith.muli %0, %sizes_2#1 : index
    %c1_9 = arith.constant 1 : index
    %2 = arith.muli %c1_9, %sizes#1 : index
    %c2 = arith.constant 2 : index
    %c1_10 = arith.constant 1 : index
    %3 = arith.muli %2, %c2 : index
    %4 = arith.addi %3, %c1_10 : index
    %5 = arith.muli %1, %4 : index
    scf.for %arg3 = %c0 to %5 step %c1 {
      %6 = arith.divui %arg3, %4 : index
      %7 = arith.remui %arg3, %4 : index
      %8 = arith.remui %6, %sizes_2#1 : index
      %9 = arith.divui %6, %sizes_2#1 : index
      %10 = arith.cmpi ult, %7, %3 : index
      scf.if %10 {
        %11 = arith.remui %7, %c2 : index
        %12 = arith.divui %7, %c2 : index
        %c0_11 = arith.constant 0 : index
        %13 = arith.cmpi eq, %11, %c0_11 : index
        scf.if %13 {
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %15 = arith.muli %c0_13, %sizes#0 : index
          %16 = arith.addi %15, %9 : index
          %17 = arith.muli %16, %sizes#1 : index
          %18 = arith.addi %17, %12 : index
          %19 = arith.muli %18, %c1_14 : index
          %20 = arith.addi %19, %c0_13 : index
          %21 = arith.index_cast %20 : index to i64
          %22 = func.call @gen_addr_2A59C30F3FF4E8BE(%21, %arg0) : (i64, memref<4x8xf32, strided<[16, 1]>>) -> i64
        }
        %c1_12 = arith.constant 1 : index
        %14 = arith.cmpi eq, %11, %c1_12 : index
        scf.if %14 {
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %15 = arith.muli %c0_13, %c1_14 : index
          %16 = arith.addi %15, %c0_13 : index
          %17 = arith.muli %16, %sizes#1 : index
          %18 = arith.addi %17, %12 : index
          %19 = arith.muli %18, %sizes_2#1 : index
          %20 = arith.addi %19, %8 : index
          %21 = arith.index_cast %20 : index to i64
          %22 = func.call @gen_addr_4A401C3FFBD877E9(%21, %arg1) : (i64, memref<8x6xf32, strided<[8, 1]>>) -> i64
        }
      } else {
        %11 = arith.subi %7, %3 : index
        %c0_11 = arith.constant 0 : index
        %12 = arith.cmpi eq, %11, %c0_11 : index
        scf.if %12 {
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          %13 = arith.muli %c0_12, %sizes#0 : index
          %14 = arith.addi %13, %9 : index
          %15 = arith.muli %14, %c1_13 : index
          %16 = arith.addi %15, %c0_12 : index
          %17 = arith.muli %16, %sizes_2#1 : index
          %18 = arith.addi %17, %8 : index
          %19 = arith.index_cast %18 : index to i64
          %20 = func.call @gen_addr_22D50E118B951A4E(%19, %arg2) : (i64, memref<4x6xf32, strided<[8, 1]>>) -> i64
        }
      }
    }
    return
  }
  func.func private @gen_trace_54384F8228B68719(%arg0: memref<6xf32>, %arg1: memref<4x6xf32>, %arg2: memref<4x6xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %arg0 : memref<6xf32> -> memref<f32>, index, index, index
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg1 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %base_buffer_4, %offset_5, %sizes_6:2, %strides_7:2 = memref.extract_strided_metadata %arg2 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1_8 = arith.constant 1 : index
    %0 = arith.muli %c1_8, %sizes_2#0 : index
    %1 = arith.muli %0, %sizes : index
    %c1_9 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1_10 = arith.constant 1 : index
    %2 = arith.muli %c1_9, %c2 : index
    %3 = arith.addi %2, %c1_10 : index
    %4 = arith.muli %1, %3 : index
    scf.for %arg3 = %c0 to %4 step %c1 {
      %5 = arith.divui %arg3, %3 : index
      %6 = arith.remui %arg3, %3 : index
      %7 = arith.remui %5, %sizes : index
      %8 = arith.divui %5, %sizes : index
      %9 = arith.cmpi ult, %6, %2 : index
      scf.if %9 {
        %10 = arith.remui %6, %c2 : index
        %11 = arith.divui %6, %c2 : index
        %c0_11 = arith.constant 0 : index
        %12 = arith.cmpi eq, %10, %c0_11 : index
        scf.if %12 {
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %14 = arith.muli %c0_13, %c1_14 : index
          %15 = arith.addi %14, %c0_13 : index
          %16 = arith.muli %15, %sizes : index
          %17 = arith.addi %16, %7 : index
          %18 = arith.index_cast %17 : index to i64
          %19 = func.call @gen_addr_9C210D0E4AC5E793(%18, %arg0) : (i64, memref<6xf32>) -> i64
        }
        %c1_12 = arith.constant 1 : index
        %13 = arith.cmpi eq, %10, %c1_12 : index
        scf.if %13 {
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %14 = arith.muli %c0_13, %sizes_2#0 : index
          %15 = arith.addi %14, %8 : index
          %16 = arith.muli %15, %sizes : index
          %17 = arith.addi %16, %7 : index
          %18 = arith.index_cast %17 : index to i64
          %19 = func.call @gen_addr_3F5FA80A2FA68935(%18, %arg1) : (i64, memref<4x6xf32>) -> i64
        }
      } else {
        %10 = arith.subi %6, %2 : index
        %c0_11 = arith.constant 0 : index
        %11 = arith.cmpi eq, %10, %c0_11 : index
        scf.if %11 {
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          %12 = arith.muli %c0_12, %sizes_2#0 : index
          %13 = arith.addi %12, %8 : index
          %14 = arith.muli %13, %sizes : index
          %15 = arith.addi %14, %7 : index
          %16 = arith.index_cast %15 : index to i64
          %17 = func.call @gen_addr_3F5FA80A2FA68935(%16, %arg2) : (i64, memref<4x6xf32>) -> i64
        }
      }
    }
    return
  }
  func.func private @gen_trace_8F9610077BE2B9C9(%arg0: memref<8x4xf32>, %arg1: memref<8x6xf32>, %arg2: memref<4x6xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg0 : memref<8x4xf32> -> memref<f32>, index, index, index, index, index
    %base_buffer_0, %offset_1, %sizes_2:2, %strides_3:2 = memref.extract_strided_metadata %arg1 : memref<8x6xf32> -> memref<f32>, index, index, index, index, index
    %base_buffer_4, %offset_5, %sizes_6:2, %strides_7:2 = memref.extract_strided_metadata %arg2 : memref<4x6xf32> -> memref<f32>, index, index, index, index, index
    %c1_8 = arith.constant 1 : index
    %0 = arith.muli %c1_8, %sizes#1 : index
    %1 = arith.muli %0, %sizes_2#1 : index
    %c1_9 = arith.constant 1 : index
    %2 = arith.muli %c1_9, %sizes#0 : index
    %c2 = arith.constant 2 : index
    %c1_10 = arith.constant 1 : index
    %3 = arith.muli %2, %c2 : index
    %4 = arith.addi %3, %c1_10 : index
    %5 = arith.muli %1, %4 : index
    scf.for %arg3 = %c0 to %5 step %c1 {
      %6 = arith.divui %arg3, %4 : index
      %7 = arith.remui %arg3, %4 : index
      %8 = arith.remui %6, %sizes_2#1 : index
      %9 = arith.divui %6, %sizes_2#1 : index
      %10 = arith.cmpi ult, %7, %3 : index
      scf.if %10 {
        %11 = arith.remui %7, %c2 : index
        %12 = arith.divui %7, %c2 : index
        %c0_11 = arith.constant 0 : index
        %13 = arith.cmpi eq, %11, %c0_11 : index
        scf.if %13 {
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %15 = arith.muli %c0_13, %sizes#1 : index
          %16 = arith.addi %15, %9 : index
          %17 = arith.muli %16, %sizes#0 : index
          %18 = arith.addi %17, %12 : index
          %19 = arith.muli %18, %c1_14 : index
          %20 = arith.addi %19, %c0_13 : index
          %21 = arith.index_cast %20 : index to i64
          %22 = func.call @gen_addr_49E1FA1CB4998AE4(%21, %arg0) : (i64, memref<8x4xf32>) -> i64
        }
        %c1_12 = arith.constant 1 : index
        %14 = arith.cmpi eq, %11, %c1_12 : index
        scf.if %14 {
          %c0_13 = arith.constant 0 : index
          %c1_14 = arith.constant 1 : index
          %15 = arith.muli %c0_13, %c1_14 : index
          %16 = arith.addi %15, %c0_13 : index
          %17 = arith.muli %16, %sizes#0 : index
          %18 = arith.addi %17, %12 : index
          %19 = arith.muli %18, %sizes_2#1 : index
          %20 = arith.addi %19, %8 : index
          %21 = arith.index_cast %20 : index to i64
          %22 = func.call @gen_addr_F270B5B28D0A4C21(%21, %arg1) : (i64, memref<8x6xf32>) -> i64
        }
      } else {
        %11 = arith.subi %7, %3 : index
        %c0_11 = arith.constant 0 : index
        %12 = arith.cmpi eq, %11, %c0_11 : index
        scf.if %12 {
          %c0_12 = arith.constant 0 : index
          %c1_13 = arith.constant 1 : index
          %13 = arith.muli %c0_12, %sizes#1 : index
          %14 = arith.addi %13, %9 : index
          %15 = arith.muli %14, %c1_13 : index
          %16 = arith.addi %15, %c0_12 : index
          %17 = arith.muli %16, %sizes_2#1 : index
          %18 = arith.addi %17, %8 : index
          %19 = arith.index_cast %18 : index to i64
          %20 = func.call @gen_addr_3AC88FC0BAF54F4F(%19, %arg2) : (i64, memref<4x6xf32>) -> i64
        }
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

