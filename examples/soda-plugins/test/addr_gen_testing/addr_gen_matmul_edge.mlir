// RUN: mlir-opt %s -mlir-disable-threading --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline='builtin.module(gen-addr-function-pass)' | FileCheck %s

// 2D maps (needed for broadcast edge case)
#map2d           = affine_map<(d0, d1) -> (d0, d1)>
// Broadcast: 2-D loop space, only the column index reaches the 1-D memref.
#map2d_bcast_row = affine_map<(d0, d1) -> (d1)>

// Matmul maps: C[m][n] += A[m][k] * B[k][n]
// Loop order: d0=m (parallel), d1=k (reduction), d2=n (parallel).
#map_mm_a   = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_mm_b   = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_mm_c   = affine_map<(d0, d1, d2) -> (d0, d2)>
// Transposed-A matmul: A stored as (K x M), accessed as A[k][m].
#map_mm_a_t = affine_map<(d0, d1, d2) -> (d1, d0)>

module {
  // -----------------------------------------------------------------------
  // Matrix multiply: C[m][n] += A[m][k] * B[k][n]
  // Loop order: d0=m (parallel), d1=k (reduction), d2=n (parallel).
  //   A : (d0,d1,d2) -> (d0,d1)   memref<4x8xf32>
  //   B : (d0,d1,d2) -> (d1,d2)   memref<8x6xf32>
  //   C : (d0,d1,d2) -> (d0,d2)   memref<4x6xf32>
  // -----------------------------------------------------------------------
  func.func @matmul(
      %a:   memref<4x8xf32>,
      %b:   memref<8x6xf32>,
      %c:   memref<4x6xf32>,
      %a_s: memref<4x8xf32, strided<[16, 1], offset: 0>>,
      %b_s: memref<8x6xf32, strided<[8,  1], offset: 0>>,
      %c_s: memref<4x6xf32, strided<[8,  1], offset: 0>>) {

    // Standard matmul (plain memrefs).
    linalg.generic {
      indexing_maps = [#map_mm_a, #map_mm_b, #map_mm_c],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%a, %b : memref<4x8xf32>, memref<8x6xf32>)
      outs(%c : memref<4x6xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %mul = arith.mulf %x, %y : f32
      %add = arith.addf %mul, %acc : f32
      linalg.yield %add : f32
    }

    // Strided matmul – same logical shapes, non-trivial layouts.
    linalg.generic {
      indexing_maps = [#map_mm_a, #map_mm_b, #map_mm_c],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%a_s, %b_s : memref<4x8xf32, strided<[16, 1], offset: 0>>,
                       memref<8x6xf32, strided<[8,  1], offset: 0>>)
      outs(%c_s : memref<4x6xf32, strided<[8, 1], offset: 0>>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %mul = arith.mulf %x, %y : f32
      %add = arith.addf %mul, %acc : f32
      linalg.yield %add : f32
    }

    return
  }

  // -----------------------------------------------------------------------
  // Edge cases
  // -----------------------------------------------------------------------
  func.func @edge_cases(
      %vec:    memref<6xf32>,      // 1-D row vector for broadcast
      %mat:    memref<4x6xf32>,
      %out_bc: memref<4x6xf32>,
      %a_t:    memref<8x4xf32>,    // transposed A: stored (K x M)
      %b_mm:   memref<8x6xf32>,
      %c_mm:   memref<4x6xf32>) {

    // Broadcast: row-vector %vec (1-D) added to every row of %mat (2-D).
    // %vec uses map (d0,d1)->d1; the row index d0 is dropped so each
    // column value is broadcast across all rows.
    linalg.generic {
      indexing_maps = [#map2d_bcast_row, #map2d, #map2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%vec, %mat : memref<6xf32>, memref<4x6xf32>)
      outs(%out_bc : memref<4x6xf32>) {
    ^bb0(%v: f32, %m: f32, %o: f32):
      %sum = arith.addf %v, %m : f32
      linalg.yield %sum : f32
    }

    // Transposed-A matmul: A_t stored (K x M), accessed as A[m][k] via
    // map (d0,d1,d2)->(d1,d0).  B and C use the standard matmul maps.
    // Iterator types: d0=m (parallel), d1=k (reduction), d2=n (parallel).
    linalg.generic {
      indexing_maps = [#map_mm_a_t, #map_mm_b, #map_mm_c],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%a_t, %b_mm : memref<8x4xf32>, memref<8x6xf32>)
      outs(%c_mm : memref<4x6xf32>) {
    ^bb0(%x: f32, %y: f32, %acc: f32):
      %mul = arith.mulf %x, %y : f32
      %add = arith.addf %mul, %acc : f32
      linalg.yield %add : f32
    }

    return
  }
}


// We should generate 9 addr helpers
//   matmul plain:   memref<4x8xf32>  (d0,d1,d2)->(d0,d1)    (1)
//   matmul plain:   memref<8x6xf32>  (d0,d1,d2)->(d1,d2)    (1, dedup: @edge_cases reuses same type+map)
//   matmul plain:   memref<4x6xf32>  (d0,d1,d2)->(d0,d2)    (1, dedup: @edge_cases reuses same type+map)
//   matmul strided: memref<4x8xf32, strided<[16, 1]>>        (1)
//   matmul strided: memref<8x6xf32, strided<[8, 1]>>         (1)
//   matmul strided: memref<4x6xf32, strided<[8, 1]>>         (1)
//   broadcast:      memref<6xf32>    (d0,d1)->d1             (1)
//   2-D:       memref<4x6xf32>  (d0,d1)->(d0,d1)       (1, distinct from 3-D map above)
//   transposed-A:   memref<8x4xf32>  (d0,d1,d2)->(d1,d0)    (1)
//
//
// 1. memref<4x8xf32>                    A  (d0,d1,d2)->(d0,d1)
// CHECK: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<4x8xf32>) -> i64 {
// 2. memref<8x6xf32>                    B  (d0,d1,d2)->(d1,d2)  ← @matmul+@edge_cases dedup → 1
// CHECK: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<8x6xf32>) -> i64 {
// 3. memref<4x6xf32>                    C  (d0,d1,d2)->(d0,d2)
// CHECK: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<4x6xf32>) -> i64 {
// 4. memref<4x8xf32, strided<[16, 1]>>  strided A
// CHECK: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<4x8xf32, strided<[16, 1]>>) -> i64 {
// 5. memref<8x6xf32, strided<[8, 1]>>   strided B
// CHECK: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<8x6xf32, strided<[8, 1]>>) -> i64 {
// 6. memref<4x6xf32, strided<[8, 1]>>   strided C
// CHECK: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<4x6xf32, strided<[8, 1]>>) -> i64 {
// 7. memref<6xf32>                      broadcast vec  (d0,d1)->d1
// CHECK: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<6xf32>) -> i64 {
// 8. memref<4x6xf32>                    C  (d0,d1)->(d0,d1) ← distinct map → different hash from #3
// CHECK: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<4x6xf32>) -> i64 {
// 9. memref<8x4xf32>                    transposed A  (d0,d1,d2)->(d1,d0)
// CHECK: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<8x4xf32>) -> i64 {
// No further addr helpers should be emitted.
// CHECK-NOT: func.func private @gen_addr_
