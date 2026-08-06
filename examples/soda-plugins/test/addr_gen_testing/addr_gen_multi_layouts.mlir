// RUN: mlir-opt %s -mlir-disable-threading --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline='builtin.module(gen-addr-function-pass)' | FileCheck %s

// 1D maps
#map1d = affine_map<(d0) -> (d0)>

// 2D maps
#map2d          = affine_map<(d0, d1) -> (d0, d1)>
#map2d_permuted = affine_map<(d0, d1) -> (d1, d0)>

// 3D maps
#map3d          = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map3d_permuted = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

module {
  // -----------------------------------------------------------------------
  // 1-D operations
  // Tests: plain memref, strided memref, and two ops that share the same
  // plain memref type → GenAddrFunctionPass must emit exactly one addr
  // helper for memref<8xf32> (dedup check).
  // -----------------------------------------------------------------------
  func.func @ops_1d(
      %a:     memref<8xf32>,
      %b:     memref<8xf32>,
      %out:   memref<8xf32>,
      %a_s:   memref<8xf32, strided<[2], offset: 1>>,
      %out_s: memref<8xf32, strided<[2], offset: 1>>) {

    // 1-D elementwise add (plain memref).
    linalg.generic {
      indexing_maps = [#map1d, #map1d, #map1d],
      iterator_types = ["parallel"]
    } ins(%a, %b : memref<8xf32>, memref<8xf32>)
      outs(%out : memref<8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }

    // 1-D negate – same memref<8xf32> type as above.
    // Dedup test: should reuse the already-generated addr function, not add a
    // second one.
    linalg.generic {
      indexing_maps = [#map1d, #map1d],
      iterator_types = ["parallel"]
    } ins(%a : memref<8xf32>)
      outs(%out : memref<8xf32>) {
    ^bb0(%x: f32, %o: f32):
      %neg = arith.negf %x : f32
      linalg.yield %neg : f32
    }

    // 1-D negate (strided with non-zero offset) – distinct memref type.
    linalg.generic {
      indexing_maps = [#map1d, #map1d],
      iterator_types = ["parallel"]
    } ins(%a_s : memref<8xf32, strided<[2], offset: 1>>)
      outs(%out_s : memref<8xf32, strided<[2], offset: 1>>) {
    ^bb0(%x: f32, %o: f32):
      %neg = arith.negf %x : f32
      linalg.yield %neg : f32
    }

    return
  }

  // -----------------------------------------------------------------------
  // 2-D operations
  // Tests: identity map, permuted map on the same plain memref type (dedup),
  // strided with zero offset, strided with non-zero offset.
  // -----------------------------------------------------------------------
  func.func @ops_2d(
      %a0:   memref<4x6xf32>,
      %b0:   memref<4x6xf32>,
      %out0: memref<4x6xf32>,

      %a1:   memref<4x6xf32, strided<[8, 1], offset: 0>>,
      %out1: memref<4x6xf32, strided<[8, 1], offset: 0>>,

      %a2:   memref<4x6xf32, strided<[16, 2], offset: 3>>,
      %out2: memref<4x6xf32, strided<[16, 2], offset: 3>>) {

    // 2-D identity add.
    linalg.generic {
      indexing_maps = [#map2d, #map2d, #map2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %b0 : memref<4x6xf32>, memref<4x6xf32>)
      outs(%out0 : memref<4x6xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }

    // 2-D transposed (permuted indexing) – same memref<4x6xf32> type.
    // Dedup test: must NOT emit a second addr function for memref<4x6xf32>.
    linalg.generic {
      indexing_maps = [#map2d_permuted, #map2d_permuted, #map2d_permuted],
      iterator_types = ["parallel", "parallel"]
    } ins(%a0, %b0 : memref<4x6xf32>, memref<4x6xf32>)
      outs(%out0 : memref<4x6xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }

    // 2-D strided, stride > 1, zero base offset.
    linalg.generic {
      indexing_maps = [#map2d, #map2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%a1 : memref<4x6xf32, strided<[8, 1], offset: 0>>)
      outs(%out1 : memref<4x6xf32, strided<[8, 1], offset: 0>>) {
    ^bb0(%x: f32, %o: f32):
      %neg = arith.negf %x : f32
      linalg.yield %neg : f32
    }

    // 2-D strided, non-unit strides + non-zero base offset.
    linalg.generic {
      indexing_maps = [#map2d, #map2d],
      iterator_types = ["parallel", "parallel"]
    } ins(%a2 : memref<4x6xf32, strided<[16, 2], offset: 3>>)
      outs(%out2 : memref<4x6xf32, strided<[16, 2], offset: 3>>) {
    ^bb0(%x: f32, %o: f32):
      %neg = arith.negf %x : f32
      linalg.yield %neg : f32
    }

    return
  }

  // -----------------------------------------------------------------------
  // 3-D operations
  // Tests: identity, last-two-dims permuted, strided with non-trivial offset.
  // -----------------------------------------------------------------------
  func.func @ops_3d(
      %a0:   memref<2x3x4xf32>,
      %b0:   memref<2x3x4xf32>,
      %out0: memref<2x3x4xf32>,

      %a1:   memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>,
      %b1:   memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>,
      %out1: memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>) {

    // 3-D identity add.
    linalg.generic {
      indexing_maps = [#map3d, #map3d, #map3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a0, %b0 : memref<2x3x4xf32>, memref<2x3x4xf32>)
      outs(%out0 : memref<2x3x4xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }

    // 3-D permuted: swap last two dimensions (d1 <-> d2).
    linalg.generic {
      indexing_maps = [#map3d_permuted, #map3d_permuted, #map3d_permuted],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a0, %b0 : memref<2x3x4xf32>, memref<2x3x4xf32>)
      outs(%out0 : memref<2x3x4xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }

    // 3-D strided: non-trivial strides + non-zero base offset.
    linalg.generic {
      indexing_maps = [#map3d, #map3d, #map3d],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%a1, %b1 : memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>,
                     memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>)
      outs(%out1 : memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    }

    return
  }
}


// We should generate 9 addr helpers
//   1-D plain:   memref<8xf32> (1, dedup: add + negate share same type+map)
//   1-D strided: memref<8xf32, strided<[2], offset: 1>>  (1)
//   2-D plain:   memref<4x6xf32> (2: identity map + permuted map)
//   2-D strided: memref<4x6xf32, strided<[8, 1]>> (1)
//   2-D strided: memref<4x6xf32, strided<[16, 2], offset: 3>> (1)
//   3-D plain:   memref<2x3x4xf32> (2: identity map + permuted map)
//   3-D strided: memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>> (1)
//
// 1. memref<8xf32>  — 1-D add AND negate share it (dedup → 1 function)
// CHECK:         func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<8xf32>) -> i64 {
// 2. memref<8xf32, strided<[2], offset: 1>>
// CHECK:         func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<8xf32, strided<[2], offset: 1>>) -> i64 {
// 3+4. memref<4x6xf32> plain × 2  (identity map + permuted map → different hashes)
// CHECK-COUNT-2: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<4x6xf32>) -> i64 {
// 5. memref<4x6xf32, strided<[8, 1]>>
// CHECK:         func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<4x6xf32, strided<[8, 1]>>) -> i64 {
// 6. memref<4x6xf32, strided<[16, 2], offset: 3>>
// CHECK:         func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<4x6xf32, strided<[16, 2], offset: 3>>) -> i64 {
// 7+8. memref<2x3x4xf32> plain × 2  (identity map + permuted map → different hashes)
// CHECK-COUNT-2: func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<2x3x4xf32>) -> i64 {
// 9. memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>
// CHECK:         func.func private @gen_addr_{{[0-9A-F]+}}({{.*}}: i64, {{.*}}: memref<2x3x4xf32, strided<[24, 8, 2], offset: 1>>) -> i64 {
// No further addr helpers should be emitted.
// CHECK-NOT:     func.func private @gen_addr_
