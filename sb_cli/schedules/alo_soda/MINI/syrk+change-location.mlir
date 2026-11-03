module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // Match the convolution operation
    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
     
    %linalg_ops = transform.collect_matching @match_linalg in %func_op : (!transform.any_op) -> !transform.any_op
    transform.sodap.tag_ops %linalg_ops, "tag" : !transform.any_op
    // matmuls: 0
    // generics: 1,2,3
    %generic1 = transform.structured.match attributes{uid = "tag_1"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %generic2 = transform.structured.match attributes{uid = "tag_2"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %generic3 = transform.structured.match attributes{uid = "tag_3"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %matmul1 = transform.structured.match attributes{uid = "tag_0"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op1, %loops1:2 = transform.structured.tile_using_for %generic1 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op2, %loops2:2 = transform.structured.tile_using_for %generic2 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op3, %loops3:2 = transform.structured.tile_using_for %generic3 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op5, %loops5:3 = transform.structured.tile_using_for %matmul1 tile_sizes [1,10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %func_op2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %lowered = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %func_op2 : (!transform.any_op) -> !transform.any_op
    
    %4 = transform.apply_registered_pass "affine-data-copy-generate" to %lowered {options = "generate-dma=false fast-mem-space=0"} : (!transform.any_op) -> !transform.any_op
    %5 = transform.apply_registered_pass "erase-buffer-deallocation" to %4 : (!transform.any_op) -> !transform.any_op
    %2 = transform.apply_registered_pass "promote-buffers-to-stack" to %5 {options = "max-rank-of-allocated-memref=4 max-alloc-size-in-bytes=4096"} : (!transform.any_op) -> !transform.any_op

    %affine_for_ops = transform.collect_matching @match_affinefors in %2 : (!transform.any_op) -> !transform.any_op
    transform.sodap.tag_ops %affine_for_ops, "loop" : !transform.any_op
    //full 11,12,15,16,17,18,19,20
    %loop11 = transform.structured.match attributes{uid = "loop_11"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop11 : !transform.any_op
    %loop12 = transform.structured.match attributes{uid = "loop_12"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop12 : !transform.any_op
    %loop15 = transform.structured.match attributes{uid = "loop_15"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop15 : !transform.any_op
    %loop16 = transform.structured.match attributes{uid = "loop_16"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop16 : !transform.any_op
    %loop17 = transform.structured.match attributes{uid = "loop_17"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop17 : !transform.any_op
    %loop18 = transform.structured.match attributes{uid = "loop_18"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop18 : !transform.any_op
    %loop19 = transform.structured.match attributes{uid = "loop_19"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop19 : !transform.any_op
    %loop20 = transform.structured.match attributes{uid = "loop_20"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop20 : !transform.any_op
    //by 2: 0,2,4,6,9
    %loop0 = transform.structured.match attributes{uid = "loop_0"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop0 {factor = 2} : !transform.any_op
    %loop2 = transform.structured.match attributes{uid = "loop_2"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop2 {factor = 2} : !transform.any_op
    %loop4 = transform.structured.match attributes{uid = "loop_4"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop4 {factor = 2} : !transform.any_op
    %loop6 = transform.structured.match attributes{uid = "loop_6"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop6 {factor = 2} : !transform.any_op
    %loop9 = transform.structured.match attributes{uid = "loop_9"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop9 {factor = 2} : !transform.any_op
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
    transform.match.operation_name %arg0 ["linalg.matmul", "linalg.generic", "linalg.batch_matmul"] : !transform.any_op
    transform.yield %arg0 : !transform.any_op
  }
}
