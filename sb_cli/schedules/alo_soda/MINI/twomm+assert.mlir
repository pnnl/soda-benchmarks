module @transforms attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      // Match the function containing the operations.
      %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      
      %linalg_ops = transform.collect_matching @match_linalg in %func_op : (!transform.any_op) -> !transform.any_op
      transform.sodap.tag_ops %linalg_ops, "tag" : !transform.any_op

      %fill1 = transform.structured.match attributes{uid = "tag_0"} in %func_op : (!transform.any_op) -> !transform.any_op
      // %tiled_op0, %loops0:2 = transform.structured.tile_using_for %fill1 tile_sizes [1, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      %batch_matmul1 = transform.structured.match attributes{uid = "tag_1"} in %func_op : (!transform.any_op) -> !transform.any_op
      %tiled_op1, %loops1:3 = transform.structured.tile_using_for %batch_matmul1 tile_sizes [1, 16, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

      %generic2 = transform.structured.match attributes{uid = "tag_2"} in %func_op : (!transform.any_op) -> !transform.any_op
      %tiled_op2, %loops2:2 = transform.structured.tile_using_for %generic2 tile_sizes [2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      %generic3 = transform.structured.match attributes{uid = "tag_3"} in %func_op : (!transform.any_op) -> !transform.any_op
      %tiled_op3, %loops3:2 = transform.structured.tile_using_for %generic3 tile_sizes [2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      %fill2 = transform.structured.match attributes{uid = "tag_4"} in %func_op : (!transform.any_op) -> !transform.any_op
      // %tiled_op4, %loops4:2 = transform.structured.tile_using_for %fill2 tile_sizes [2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

      %batch_matmul5 = transform.structured.match attributes{uid = "tag_5"} in %func_op : (!transform.any_op) -> !transform.any_op
      %tiled_op5, %loops5:3 = transform.structured.tile_using_for %batch_matmul5 tile_sizes [1, 16, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

      %generic6 = transform.structured.match attributes{uid = "tag_6"} in %func_op : (!transform.any_op) -> !transform.any_op
      %tiled_op6, %loops6:2 = transform.structured.tile_using_for %generic6 tile_sizes [2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      // Apply the lowering to affine loops pass.

      %func_op2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %lowered = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %func_op2 : (!transform.any_op) -> !transform.any_op
      
      %adcg = transform.apply_registered_pass "affine-data-copy-generate" to %lowered {options = "generate-dma=false fast-mem-space=0"} : (!transform.any_op) -> !transform.any_op
      %ebd = transform.apply_registered_pass "erase-buffer-deallocation" to %adcg : (!transform.any_op) -> !transform.any_op
      %2 = transform.apply_registered_pass "promote-buffers-to-stack" to %ebd {options = "max-rank-of-allocated-memref=4 max-alloc-size-in-bytes=4096"} : (!transform.any_op) -> !transform.any_op

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
      
      %loop9 = transform.structured.match attributes{uid = "loop_9"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop9 : !transform.any_op
      
      %loop10 = transform.structured.match attributes{uid = "loop_10"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop10 : !transform.any_op
      
      %loop11 = transform.structured.match attributes{uid = "loop_11"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop11 : !transform.any_op
      
      %loop12 = transform.structured.match attributes{uid = "loop_12"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop12 : !transform.any_op
      
      %loop18 = transform.structured.match attributes{uid = "loop_18"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop18 : !transform.any_op
      
      %loop19 = transform.structured.match attributes{uid = "loop_19"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop19 : !transform.any_op
      
      %loop20 = transform.structured.match attributes{uid = "loop_20"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop20 : !transform.any_op
      
      %loop21 = transform.structured.match attributes{uid = "loop_21"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop21 : !transform.any_op
      
      %loop22 = transform.structured.match attributes{uid = "loop_22"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop22 : !transform.any_op
      
      %loop23 = transform.structured.match attributes{uid = "loop_23"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop23 : !transform.any_op

      %loop0 = transform.structured.match attributes{uid = "loop_0"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop0 {factor = 2} : !transform.any_op
      
      %loop3 = transform.structured.match attributes{uid = "loop_3"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop3 {factor = 2} : !transform.any_op
      
      %loop13 = transform.structured.match attributes{uid = "loop_13"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop13 {factor = 2} : !transform.any_op
      
      %loop16 = transform.structured.match attributes{uid = "loop_16"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop16 {factor = 2} : !transform.any_op
      
      

      // // Apply the scalrep pass.
      %sroa = transform.apply_registered_pass "affine-scalrep" to %2 : (!transform.any_op) -> !transform.any_op
      
      // --- instrumentation: assert ------------------------------------------------
      // Runs last, on the scf.for tile loops tile_using_for produced
      // above. Those survive the whole schedule: the affine lowering
      // only rewrites linalg ops, and the unrolls target affine.for.
      // That matters because the SODA instrumentation passes walk
      // scf::ForOp only -- see
      // examples/soda-plugins/lib/sodap/InstrPasses.cpp.
      //
      // It also has to be last: the pass declares a func.func private
      // for the IP call, and any match ops{["func.func"]} running
      // afterwards would bind two funcs, which makes
      // apply_registered_pass set its results twice and abort.
      %instr_func = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %instr_done = transform.apply_registered_pass "soda-instr-scf-for-assert-bounds" to %instr_func : (!transform.any_op) -> !transform.any_op
      
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
