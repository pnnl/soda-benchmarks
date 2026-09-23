module @transforms attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      // Match the function containing the operations.
      %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      
      %linalg_ops = transform.collect_matching @match_linalg in %func_op : (!transform.any_op) -> !transform.any_op
      transform.sodap.tag_ops %linalg_ops, "tag" : !transform.any_op

      %matmul1 = transform.structured.match attributes{uid = "tag_1"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %tiled_op1, %loops1:3 = transform.structured.tile_using_for %matmul1 tile_sizes [1, 40, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      // Apply the lowering to affine loops pass.

      %func_op2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %lowered = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %func_op2 : (!transform.any_op) -> !transform.any_op
      
      %4 = transform.apply_registered_pass "affine-data-copy-generate" to %lowered {options = "generate-dma=false fast-mem-space=0"} : (!transform.any_op) -> !transform.any_op
      %5 = transform.apply_registered_pass "erase-buffer-deallocation" to %4 : (!transform.any_op) -> !transform.any_op
      %2 = transform.apply_registered_pass "promote-buffers-to-stack" to %5 {options = "max-rank-of-allocated-memref=4 max-alloc-size-in-bytes=4096"} : (!transform.any_op) -> !transform.any_op
      
      %affine_for_ops = transform.collect_matching @match_affinefors in %2 : (!transform.any_op) -> !transform.any_op
      transform.sodap.tag_ops %affine_for_ops, "loop" : !transform.any_op

      %loop5 = transform.structured.match attributes{uid = "loop_5"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop5 : !transform.any_op
      %loop6 = transform.structured.match attributes{uid = "loop_6"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop6 : !transform.any_op
      %loop7 = transform.structured.match attributes{uid = "loop_7"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop7 : !transform.any_op
      %loop8 = transform.structured.match attributes{uid = "loop_8"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop8 : !transform.any_op
      
      %loop0_unroll = transform.structured.match attributes{uid = "loop_0"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop0_unroll {factor = 2} : !transform.any_op
      %loop3_unroll = transform.structured.match attributes{uid = "loop_3"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop3_unroll {factor = 2} : !transform.any_op
      

      // Apply the scalrep pass.
      %sroa = transform.apply_registered_pass "affine-scalrep" to %2 : (!transform.any_op) -> !transform.any_op
      
      transform.yield
    }
    transform.named_sequence @match_affinefors(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.match.operation_name %arg0 ["affine.for"] : !transform.any_op
      transform.yield %arg0 : !transform.any_op
    }

    transform.named_sequence @match_linalg(%arg0: !transform.any_op {transform.readonly}) -> !transform.any_op {
      transform.match.operation_name %arg0 ["linalg.fill", "linalg.generic", "linalg.batch_matmul"] : !transform.any_op
      transform.yield %arg0 : !transform.any_op
    }
  }
