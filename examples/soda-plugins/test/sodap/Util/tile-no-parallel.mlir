// RUN: mlir-opt %s --verify-diagnostics --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --transform-interpreter -o /dev/null

#id = affine_map<(d0, d1) -> (d0, d1)>
#a = affine_map<(d0, d1, d2) -> (d0, d2)>
#b = affine_map<(d0, d1, d2) -> (d2, d1)>
#c = affine_map<(d0, d1, d2) -> (d0, d1)>

// gemm in one task: the epilogue (PP) produces the yielded value. i and j are
// tiled with the fill and the contraction (PPR) fused in; only the contraction
// gets a k loop. The nest writes the edge, the alloc_tensor in the dispatch.
func.func @epilogue(%A: tensor<8x12xf32>, %B: tensor<12x16xf32>, %C: tensor<8x16xf32>) -> tensor<8x16xf32> {
  %zero = arith.constant 0.0 : f32
  %r = dataflow.dispatch : tensor<8x16xf32> {
    %edge = bufferization.alloc_tensor() : tensor<8x16xf32>
    %t = dataflow.task : tensor<8x16xf32> {
      %empty = tensor.empty() : tensor<8x16xf32>
      %fill = linalg.fill ins(%zero : f32) outs(%empty : tensor<8x16xf32>) -> tensor<8x16xf32>
      %mm = linalg.generic {indexing_maps = [#a, #b, #c], iterator_types = ["parallel", "parallel", "reduction"]}
          ins(%A, %B : tensor<8x12xf32>, tensor<12x16xf32>) outs(%fill : tensor<8x16xf32>) {
      ^bb0(%x: f32, %y: f32, %acc: f32):
        %0 = arith.mulf %x, %y : f32
        %1 = arith.addf %acc, %0 : f32
        linalg.yield %1 : f32
      } -> tensor<8x16xf32>
      %out = linalg.generic {indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]}
          ins(%mm, %C : tensor<8x16xf32>, tensor<8x16xf32>) outs(%edge : tensor<8x16xf32>) {
      ^bb0(%x: f32, %y: f32, %o: f32):
        %0 = arith.addf %x, %y : f32
        linalg.yield %0 : f32
      } -> tensor<8x16xf32>
      dataflow.yield %out : tensor<8x16xf32>
    }
    dataflow.yield %t : tensor<8x16xf32>
  }
  return %r : tensor<8x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%root: !transform.any_op {transform.readonly}) {
    %tasks = transform.structured.match ops{["dataflow.task"]} in %root : (!transform.any_op) -> !transform.any_op
    // expected-remark @below {{dma_cycles = }}
    // expected-remark @below {{tile_sizes = [[0, 0]]}}
    // expected-error @below {{expected at least one tiled parallel dimension}}
    transform.sodap.tile_and_fuse %tasks {tile_sizes = [[0, 0]]} : !transform.any_op
    transform.yield
  }
}

