module @transforms attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
      // Match the function containing the operations.
      %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      
      %linalg_ops = transform.collect_matching @match_linalg in %func_op : (!transform.any_op) -> !transform.any_op
      transform.sodap.tag_ops %linalg_ops, "tag" : !transform.any_op

      //generics: 0,1,2,3,6,7,8,11,12
      //matmul: 5,10
      //fill: 4, 9
      %matmul1 = transform.structured.match attributes{uid = "tag_5"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %matmul2 = transform.structured.match attributes{uid = "tag_10"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %generic1 = transform.structured.match attributes{uid = "tag_0"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %generic2 = transform.structured.match attributes{uid = "tag_1"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %generic3 = transform.structured.match attributes{uid = "tag_2"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %generic4 = transform.structured.match attributes{uid = "tag_3"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %generic5 = transform.structured.match attributes{uid = "tag_6"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %generic6 = transform.structured.match attributes{uid = "tag_7"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %generic7 = transform.structured.match attributes{uid = "tag_8"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %generic8 = transform.structured.match attributes{uid = "tag_11"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %generic9 = transform.structured.match attributes{uid = "tag_12"} in %arg0 : (!transform.any_op) -> !transform.any_op
      %tiled_op1, %loops1:3 = transform.structured.tile_using_for %matmul1 tile_sizes [1, 10, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_op2, %loops2:3 = transform.structured.tile_using_for %matmul2 tile_sizes [1, 10, 1] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_op3, %loops3:2 = transform.structured.tile_using_for %generic1 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_op4, %loops4:2 = transform.structured.tile_using_for %generic2 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_op5, %loops5:2 = transform.structured.tile_using_for %generic3 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_op6, %loops6:2 = transform.structured.tile_using_for %generic4 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
      %tiled_op7, %loops7:1 = transform.structured.tile_using_for %generic5 tile_sizes [10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op8, %loops8:1 = transform.structured.tile_using_for %generic6 tile_sizes [10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op9, %loops9:1 = transform.structured.tile_using_for %generic7 tile_sizes [10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op10, %loops10:1 = transform.structured.tile_using_for %generic8 tile_sizes [10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
      %tiled_op11, %loops11:1 = transform.structured.tile_using_for %generic9 tile_sizes [10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

      %func_op2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
      %lowered = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %func_op2 : (!transform.any_op) -> !transform.any_op
      
      %4 = transform.apply_registered_pass "affine-data-copy-generate" to %lowered {options = "generate-dma=false fast-mem-space=0"} : (!transform.any_op) -> !transform.any_op
      %5 = transform.apply_registered_pass "erase-buffer-deallocation" to %4 : (!transform.any_op) -> !transform.any_op
      %2 = transform.apply_registered_pass "promote-buffers-to-stack" to %5 {options = "max-rank-of-allocated-memref=4 max-alloc-size-in-bytes=4096"} : (!transform.any_op) -> !transform.any_op

      %affine_for_ops = transform.collect_matching @match_affinefors in %2 : (!transform.any_op) -> !transform.any_op
      transform.sodap.tag_ops %affine_for_ops, "loop" : !transform.any_op
      //full: 0,1,2,3,4,5,6,7,18,22,23,24,29,33,34
      %loop0 = transform.structured.match attributes{uid = "loop_0"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop0 : !transform.any_op
      %loop1 = transform.structured.match attributes{uid = "loop_1"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop1 : !transform.any_op
      %loop2 = transform.structured.match attributes{uid = "loop_2"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop2 : !transform.any_op
      %loop3 = transform.structured.match attributes{uid = "loop_3"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop3 : !transform.any_op
      %loop4 = transform.structured.match attributes{uid = "loop_4"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop4 : !transform.any_op
      %loop5 = transform.structured.match attributes{uid = "loop_5"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop5 : !transform.any_op
      %loop6 = transform.structured.match attributes{uid = "loop_6"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop6 : !transform.any_op
      %loop7 = transform.structured.match attributes{uid = "loop_7"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop7 : !transform.any_op
      %loop18 = transform.structured.match attributes{uid = "loop_18"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop18 : !transform.any_op
      %loop22 = transform.structured.match attributes{uid = "loop_22"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop22 : !transform.any_op
      %loop23 = transform.structured.match attributes{uid = "loop_23"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop23 : !transform.any_op
      %loop24 = transform.structured.match attributes{uid = "loop_24"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop24 : !transform.any_op
      %loop29 = transform.structured.match attributes{uid = "loop_29"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop29 : !transform.any_op
      %loop33 = transform.structured.match attributes{uid = "loop_33"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop33 : !transform.any_op
      %loop34 = transform.structured.match attributes{uid = "loop_34"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.fullunroll %loop34 : !transform.any_op
      //by 2: 8,10,12,17,28
      %loop8 = transform.structured.match attributes{uid = "loop_8"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop8 {factor = 2} : !transform.any_op
      %loop10 = transform.structured.match attributes{uid = "loop_10"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop10 {factor = 2} : !transform.any_op
      %loop12 = transform.structured.match attributes{uid = "loop_12"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop12 {factor = 2} : !transform.any_op
      %loop17 = transform.structured.match attributes{uid = "loop_17"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop17 {factor = 2} : !transform.any_op
      %loop28 = transform.structured.match attributes{uid = "loop_28"} in %2 : (!transform.any_op) -> !transform.any_op
      transform.loop.unroll %loop28 {factor = 2} : !transform.any_op
      // Apply the scalrep pass.
      %sroa = transform.apply_registered_pass "affine-scalrep" to %2 : (!transform.any_op) -> !transform.any_op
      
      // --- instrumentation: change-location ---------------------------------------
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
      %instr_done = transform.apply_registered_pass "soda-instr-change-location" to %instr_func : (!transform.any_op) -> !transform.any_op
      
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
