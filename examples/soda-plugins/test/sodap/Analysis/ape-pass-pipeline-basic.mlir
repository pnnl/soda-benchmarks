// RUN: mlir-opt %s -mlir-disable-threading \
// RUN:   --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:   --pass-pipeline='builtin.module(func.func(soda-linalg-ape-analysis,soda-affine-ape-insertion))' \
// RUN: | FileCheck %s
// This test checks the APE pass for a simple matmul kernel. The pass should insert calls to the APE runtime functions that will be used to generate the addresses for the matmul operands and result. The test checks that the correct calls are inserted and that the correct attributes are set on the calls. Linalg statements will generate pre_issue_APE_requests and issue_APE_requests. The pre_issue_APE_requests are generated before linalg statements, and the issue_APE_requests are generated within the affine stages and after the linalg statements. Attributes can be incomplete if they are not fully supported yet. We are moving away from this method of trace generation, at which point this test will be restructured to reflect the new method of trace generation. 
// TODO KIM: update this test to use the new pass features
#map_a = affine_map<(d0, d1, d2) -> (d0, d1)>
#map_b = affine_map<(d0, d1, d2) -> (d1, d2)>
#map_c = affine_map<(d0, d1, d2) -> (d0, d2)>

module {
  func.func @kernel(%A: memref<4x8xf32>, %B: memref<8x6xf32>, %C: memref<4x6xf32>) {
    linalg.generic {
      indexing_maps = [#map_a, #map_b, #map_c],
      iterator_types = ["parallel", "reduction", "parallel"]
    } ins(%A, %B : memref<4x8xf32>, memref<8x6xf32>)
      outs(%C : memref<4x6xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %mul = arith.mulf %a, %b : f32
      %sum = arith.addf %mul, %c : f32
      linalg.yield %sum : f32
    }
    return
  }
}

// CHECK: func.func private @ape_incomplete()
// CHECK: func.func private @pre_issue_APE_request()
// CHECK: func.func private @issue_APE_request(index, i32, i32, i1, i64, i64, i64, index, index)

// CHECK-LABEL: func.func @kernel(
// CHECK-SAME: ape.tensor_info
// CHECK: call @ape_incomplete() : () -> ()
// CHECK: call @pre_issue_APE_request() {ape.pre_issue_info = {op_kind = "incomplete"
// CHECK: linalg.generic
// CHECK-SAME: ape.tensor_info
