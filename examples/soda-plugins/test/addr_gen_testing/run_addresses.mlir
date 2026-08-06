// RUN: mlir-opt %s -mlir-disable-threading \
// RUN:   -convert-linalg-to-affine-loops \
// RUN:   -expand-strided-metadata \
// RUN:   -lower-affine \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-vector-to-llvm \
// RUN:   --convert-math-to-llvm \
// RUN:   --convert-math-to-libm \
// RUN:   -arith-expand \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-arith-to-llvm \
// RUN:   -convert-func-to-llvm='use-bare-ptr-memref-call-conv=1' \
// RUN:   -convert-cf-to-llvm \
// RUN:   -reconcile-unrealized-casts \
// RUN:   -symbol-dce \
// RUN: | mlir-cpu-runner \
// RUN:   -O0 -e main -entry-point-result=void \
// RUN:   -shared-libs=%sodap_libs/libmlir_sodap_instr_runner_utils%shlibext \
// RUN:   -shared-libs=%llvm_lib_dir/libmlir_runner_utils%shlibext \
// RUN:   -shared-libs=%llvm_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN: | sed -n 's/.*base0=//p' \
// RUN: | FileCheck %s --check-prefix=BASE0

// BASE0-LABEL: 0x0
// 4x8 dense: contiguous +4 byte offsets through 0x7c.
// BASE0-NEXT: 0x4
// BASE0-NEXT: 0x8
// BASE0-NEXT: 0xc
// BASE0-NEXT: 0x10
// BASE0-NEXT: 0x14
// BASE0-NEXT: 0x18
// BASE0-NEXT: 0x1c
// BASE0-NEXT: 0x20
// BASE0-NEXT: 0x24
// BASE0-NEXT: 0x28
// BASE0-NEXT: 0x2c
// BASE0-NEXT: 0x30
// BASE0-NEXT: 0x34
// BASE0-NEXT: 0x38
// BASE0-NEXT: 0x3c
// BASE0-NEXT: 0x40
// BASE0-NEXT: 0x44
// BASE0-NEXT: 0x48
// BASE0-NEXT: 0x4c
// BASE0-NEXT: 0x50
// BASE0-NEXT: 0x54
// BASE0-NEXT: 0x58
// BASE0-NEXT: 0x5c
// BASE0-NEXT: 0x60
// BASE0-NEXT: 0x64
// BASE0-NEXT: 0x68
// BASE0-NEXT: 0x6c
// BASE0-NEXT: 0x70
// BASE0-NEXT: 0x74
// BASE0-NEXT: 0x78
// BASE0-NEXT: 0x7c

// 8x6 dense: contiguous +4 through 0xbc.
// BASE0-NEXT: 0x0
// BASE0-NEXT: 0x4
// BASE0-NEXT: 0x8
// BASE0-NEXT: 0xc
// BASE0: 0xb0
// BASE0-NEXT: 0xb4
// BASE0-NEXT: 0xb8
// BASE0-NEXT: 0xbc

// 4x6 dense: contiguous +4 through 0x5c.
// BASE0-NEXT: 0x0
// BASE0-NEXT: 0x4
// BASE0-NEXT: 0x8
// BASE0-NEXT: 0xc
// BASE0: 0x50
// BASE0-NEXT: 0x54
// BASE0-NEXT: 0x58
// BASE0-NEXT: 0x5c

// 4x8 strided<[16,1]>: row jumps by 0x40 bytes.
// BASE0-NEXT: 0x0
// BASE0-NEXT: 0x4
// BASE0-NEXT: 0x8
// BASE0-NEXT: 0xc
// BASE0-NEXT: 0x10
// BASE0-NEXT: 0x14
// BASE0-NEXT: 0x18
// BASE0-NEXT: 0x1c
// BASE0-NEXT: 0x40
// BASE0: 0x5c
// BASE0-NEXT: 0x80
// BASE0: 0x9c
// BASE0-NEXT: 0xc0
// BASE0: 0xdc

// 8x6 strided<[8,1]>: row jumps by 0x20 bytes.
// BASE0-NEXT: 0x0
// BASE0-NEXT: 0x4
// BASE0-NEXT: 0x8
// BASE0-NEXT: 0xc
// BASE0-NEXT: 0x10
// BASE0-NEXT: 0x14
// BASE0-NEXT: 0x20
// BASE0: 0x34
// BASE0-NEXT: 0x40
// BASE0: 0x54
// BASE0-NEXT: 0x60
// BASE0: 0x74
// BASE0-NEXT: 0x80
// BASE0: 0x94
// BASE0-NEXT: 0xa0
// BASE0: 0xb4
// BASE0-NEXT: 0xc0
// BASE0: 0xd4
// BASE0-NEXT: 0xe0
// BASE0-NEXT: 0xe4
// BASE0-NEXT: 0xe8
// BASE0-NEXT: 0xec
// BASE0-NEXT: 0xf0
// BASE0-NEXT: 0xf4

// 4x6 strided<[8,1]>: row jumps by 0x20 bytes.
// BASE0-NEXT: 0x0
// BASE0-NEXT: 0x4
// BASE0-NEXT: 0x8
// BASE0-NEXT: 0xc
// BASE0-NEXT: 0x10
// BASE0-NEXT: 0x14
// BASE0-NEXT: 0x20
// BASE0: 0x34
// BASE0-NEXT: 0x40
// BASE0: 0x54
// BASE0-NEXT: 0x60
// BASE0-NEXT: 0x64
// BASE0-NEXT: 0x68
// BASE0-NEXT: 0x6c
// BASE0-NEXT: 0x70
// BASE0-NEXT: 0x74

// 6x1 dense.
// BASE0-NEXT: 0x0
// BASE0-NEXT: 0x4
// BASE0-NEXT: 0x8
// BASE0-NEXT: 0xc
// BASE0-NEXT: 0x10
// BASE0-NEXT: 0x14

// 4x6 dense again.
// BASE0-NEXT: 0x0
// BASE0-NEXT: 0x4
// BASE0-NEXT: 0x8
// BASE0-NEXT: 0xc
// BASE0: 0x50
// BASE0-NEXT: 0x54
// BASE0-NEXT: 0x58
// BASE0-NEXT: 0x5c

// 8x4 with permuted access: column-major walk over row-major buffer.
// BASE0-NEXT: 0x0
// BASE0-NEXT: 0x10
// BASE0-NEXT: 0x20
// BASE0-NEXT: 0x30
// BASE0-NEXT: 0x40
// BASE0-NEXT: 0x50
// BASE0-NEXT: 0x60
// BASE0-NEXT: 0x70
// BASE0-NEXT: 0x4
// BASE0-NEXT: 0x14
// BASE0-NEXT: 0x24
// BASE0-NEXT: 0x34
// BASE0-NEXT: 0x44
// BASE0-NEXT: 0x54
// BASE0-NEXT: 0x64
// BASE0-NEXT: 0x74
// BASE0-NEXT: 0x8
// BASE0-NEXT: 0x18
// BASE0-NEXT: 0x28
// BASE0-NEXT: 0x38
// BASE0-NEXT: 0x48
// BASE0-NEXT: 0x58
// BASE0-NEXT: 0x68
// BASE0-NEXT: 0x78
// BASE0-NEXT: 0xc
// BASE0-NEXT: 0x1c
// BASE0-NEXT: 0x2c
// BASE0-NEXT: 0x3c
// BASE0-NEXT: 0x4c
// BASE0-NEXT: 0x5c
// BASE0-NEXT: 0x6c
// BASE0-NEXT: 0x7c

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
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg0, %arg1 : memref<4x8xf32>, memref<8x6xf32>) outs(%arg2 : memref<4x6xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %0, %out : f32
      linalg.yield %1 : f32
    }
    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "reduction", "parallel"]} ins(%arg3, %arg4 : memref<4x8xf32, strided<[16, 1]>>, memref<8x6xf32, strided<[8, 1]>>) outs(%arg5 : memref<4x6xf32, strided<[8, 1]>>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %0, %out : f32
      linalg.yield %1 : f32
    }
    return
  }
  func.func @edge_cases(%arg0: memref<6xf32>, %arg1: memref<4x6xf32>, %arg2: memref<4x6xf32>, %arg3: memref<8x4xf32>, %arg4: memref<8x6xf32>, %arg5: memref<4x6xf32>) {
    linalg.generic {indexing_maps = [#map3, #map4, #map4], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<6xf32>, memref<4x6xf32>) outs(%arg2 : memref<4x6xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.addf %in, %in_0 : f32
      linalg.yield %0 : f32
    }
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
    %15 = arith.index_cast %11 : index to i64
    %14 = call @format_address_pair(%13, %15) : (i64, i64) -> i64
    return %14 : i64
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
    %15 = arith.index_cast %11 : index to i64
    %14 = call @format_address_pair(%13, %15) : (i64, i64) -> i64
    return %14 : i64
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
    %15 = arith.index_cast %11 : index to i64
    %14 = call @format_address_pair(%13, %15) : (i64, i64) -> i64
    return %14 : i64
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
    %15 = arith.index_cast %11 : index to i64
    %14 = call @format_address_pair(%13, %15) : (i64, i64) -> i64
    return %14 : i64
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
    %15 = arith.index_cast %11 : index to i64
    %14 = call @format_address_pair(%13, %15) : (i64, i64) -> i64
    return %14 : i64
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
    %15 = arith.index_cast %11 : index to i64
    %14 = call @format_address_pair(%13, %15) : (i64, i64) -> i64
    return %14 : i64
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
    %10 = arith.index_cast %6 : index to i64
    %9 = call @format_address_pair(%8, %10) : (i64, i64) -> i64
    return %9 : i64
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
    %13 = arith.index_cast %9 : index to i64
    %12 = call @format_address_pair(%11, %13) : (i64, i64) -> i64
    return %12 : i64
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
    %15 = arith.index_cast %11 : index to i64
    %14 = call @format_address_pair(%13, %15) : (i64, i64) -> i64
    return %14 : i64
  }
  func.func @main() {
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    %c24 = arith.constant 24 : index
    %c32 = arith.constant 32 : index
    %c48 = arith.constant 48 : index

    %m_4x8 = memref.alloc() : memref<4x8xf32>
    %m_8x6 = memref.alloc() : memref<8x6xf32>
    %m_4x6 = memref.alloc() : memref<4x6xf32>
    %m_6 = memref.alloc() : memref<6xf32>
    %m_8x4 = memref.alloc() : memref<8x4xf32>

    %base_56 = memref.alloc() : memref<56xf32>
    %base_62 = memref.alloc() : memref<62xf32>
    %base_30 = memref.alloc() : memref<30xf32>

    %m_4x8_s = memref.reinterpret_cast %base_56 to offset: [0], sizes: [4, 8], strides: [16, 1] : memref<56xf32> to memref<4x8xf32, strided<[16, 1]>>
    %m_8x6_s = memref.reinterpret_cast %base_62 to offset: [0], sizes: [8, 6], strides: [8, 1] : memref<62xf32> to memref<8x6xf32, strided<[8, 1]>>
    %m_4x6_s = memref.reinterpret_cast %base_30 to offset: [0], sizes: [4, 6], strides: [8, 1] : memref<30xf32> to memref<4x6xf32, strided<[8, 1]>>

    scf.for %i = %c0 to %c32 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %addr = func.call @gen_addr_920B00BBA9FACB8B(%i64, %m_4x8) : (i64, memref<4x8xf32>) -> i64
    }
    scf.for %i = %c0 to %c48 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %addr = func.call @gen_addr_F270B5B28D0A4C21(%i64, %m_8x6) : (i64, memref<8x6xf32>) -> i64
    }
    scf.for %i = %c0 to %c24 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %addr = func.call @gen_addr_3AC88FC0BAF54F4F(%i64, %m_4x6) : (i64, memref<4x6xf32>) -> i64
    }
    scf.for %i = %c0 to %c32 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %addr = func.call @gen_addr_2A59C30F3FF4E8BE(%i64, %m_4x8_s) : (i64, memref<4x8xf32, strided<[16, 1]>>) -> i64
    }
    scf.for %i = %c0 to %c48 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %addr = func.call @gen_addr_4A401C3FFBD877E9(%i64, %m_8x6_s) : (i64, memref<8x6xf32, strided<[8, 1]>>) -> i64
    }
    scf.for %i = %c0 to %c24 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %addr = func.call @gen_addr_22D50E118B951A4E(%i64, %m_4x6_s) : (i64, memref<4x6xf32, strided<[8, 1]>>) -> i64
    }
    scf.for %i = %c0 to %c6 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %addr = func.call @gen_addr_9C210D0E4AC5E793(%i64, %m_6) : (i64, memref<6xf32>) -> i64
    }
    scf.for %i = %c0 to %c24 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %addr = func.call @gen_addr_3F5FA80A2FA68935(%i64, %m_4x6) : (i64, memref<4x6xf32>) -> i64
    }
    scf.for %i = %c0 to %c32 step %c1 {
      %i64 = arith.index_cast %i : index to i64
      %addr = func.call @gen_addr_49E1FA1CB4998AE4(%i64, %m_8x4) : (i64, memref<8x4xf32>) -> i64
    }

    memref.dealloc %m_4x8 : memref<4x8xf32>
    memref.dealloc %m_8x6 : memref<8x6xf32>
    memref.dealloc %m_4x6 : memref<4x6xf32>
    memref.dealloc %m_6 : memref<6xf32>
    memref.dealloc %m_8x4 : memref<8x4xf32>
    memref.dealloc %base_56 : memref<56xf32>
    memref.dealloc %base_62 : memref<62xf32>
    memref.dealloc %base_30 : memref<30xf32>

    return
  }
}

