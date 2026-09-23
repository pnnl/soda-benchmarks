module @transforms attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      // Match the function containing the operations.
      %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      
      %linalg_ops = transform.collect_matching @match_linalg in %func_op : (!transform.any_op) -> !transform.any_op
      transform.sodap.tag_ops %linalg_ops, "linalg_tag" : !transform.any_op

      %matmul1 = transform.structured.match attributes{uid = "linalg_tag_1"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %matmul2 = transform.structured.match attributes{uid = "linalg_tag_3"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %matmul3 = transform.structured.match attributes{uid = "linalg_tag_5"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %tiled_op1, %loops1:3 = transform.structured.tile_using_for %matmul1 tile_sizes [1, 16, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_op2, %loops2:3 = transform.structured.tile_using_for %matmul2 tile_sizes [1, 18, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_op3, %loops3:3 = transform.structured.tile_using_for %matmul3 tile_sizes [1, 16, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      // Apply the lowering to affine loops pass.

      %fill1 = transform.structured.match attributes{uid = "linalg_tag_0"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %fill2 = transform.structured.match attributes{uid = "linalg_tag_2"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %fill3 = transform.structured.match attributes{uid = "linalg_tag_4"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %func_op2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %lowered = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %func_op2 : (!transform.any_op) -> !transform.any_op
      
      // %21 = transform.apply_registered_pass "affine-loop-tile" to %lowered {options = "tile-size=2"}: (!transform.any_op) -> !transform.any_op 
      %4 = transform.apply_registered_pass "affine-data-copy-generate" to %lowered {options = "generate-dma=false fast-mem-space=0"} : (!transform.any_op) -> !transform.any_op
      %5 = transform.apply_registered_pass "erase-buffer-deallocation" to %4 : (!transform.any_op) -> !transform.any_op
      %2 = transform.apply_registered_pass "promote-buffers-to-stack" to %5 {options = "max-rank-of-allocated-memref=4 max-alloc-size-in-bytes=4096"} : (!transform.any_op) -> !transform.any_op
      // Match and unroll affine.for ops by a factor of 1.
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
      
      %loop14 = transform.structured.match attributes{uid = "loop_14"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop14 : !transform.any_op
      
      %loop15 = transform.structured.match attributes{uid = "loop_15"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop15 : !transform.any_op
      
      %loop16 = transform.structured.match attributes{uid = "loop_16"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop16 : !transform.any_op
      
      %loop17 = transform.structured.match attributes{uid = "loop_17"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop17 : !transform.any_op
      
      %loop23 = transform.structured.match attributes{uid = "loop_23"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop23 : !transform.any_op
      
      %loop24 = transform.structured.match attributes{uid = "loop_24"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop24 : !transform.any_op
      
      %loop25 = transform.structured.match attributes{uid = "loop_25"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop25 : !transform.any_op
      
      %loop26 = transform.structured.match attributes{uid = "loop_26"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop26 : !transform.any_op
      
      // Unroll by 2 loops: 0,3,9,12,18,21
      %uloop0 = transform.structured.match attributes{uid = "loop_0"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %uloop0 {factor = 2} : !transform.any_op
      
      %uloop3 = transform.structured.match attributes{uid = "loop_3"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %uloop3 {factor = 2} : !transform.any_op
      
      %uloop9 = transform.structured.match attributes{uid = "loop_9"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %uloop9 {factor = 2} : !transform.any_op
      
      %uloop12 = transform.structured.match attributes{uid = "loop_12"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %uloop12 {factor = 2} : !transform.any_op
      
      %uloop18 = transform.structured.match attributes{uid = "loop_18"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %uloop18 {factor = 2} : !transform.any_op
      
      %uloop21 = transform.structured.match attributes{uid = "loop_21"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %uloop21 {factor = 2} : !transform.any_op
      

      // Apply the scalrep pass.
      %sroa = transform.apply_registered_pass "affine-scalrep" to %2 : (!transform.any_op) -> !transform.any_op
      
      // --- instrumentation: hw-counters -------------------------------------------
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
      %instr_done = transform.apply_registered_pass "soda-instr-hw-counters-at-loop-bounds" to %instr_func : (!transform.any_op) -> !transform.any_op
      
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
