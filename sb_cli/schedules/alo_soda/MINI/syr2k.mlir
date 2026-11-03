module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    // Match the convolution operation
    %func_op = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
     
    %linalg_ops = transform.collect_matching @match_linalg in %func_op : (!transform.any_op) -> !transform.any_op
    transform.sodap.tag_ops %linalg_ops, "tag" : !transform.any_op
    // matmuls: 0,1
    // generics: 2,3,4,5
    %generic1 = transform.structured.match attributes{uid = "tag_2"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %generic2 = transform.structured.match attributes{uid = "tag_3"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %generic3 = transform.structured.match attributes{uid = "tag_4"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %generic4 = transform.structured.match attributes{uid = "tag_5"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %matmul1 = transform.structured.match attributes{uid = "tag_0"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %matmul2 = transform.structured.match attributes{uid = "tag_1"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %tiled_op1, %loops1:2 = transform.structured.tile_using_for %generic1 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op2, %loops2:2 = transform.structured.tile_using_for %generic2 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op3, %loops3:2 = transform.structured.tile_using_for %generic3 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op4, %loops4:2 = transform.structured.tile_using_for %generic4 tile_sizes [10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op5, %loops5:3 = transform.structured.tile_using_for %matmul1 tile_sizes [1,10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %tiled_op6, %loops6:3 = transform.structured.tile_using_for %matmul2 tile_sizes [1,10,10] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %func_op2 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %lowered = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %func_op2 : (!transform.any_op) -> !transform.any_op
    
    %4 = transform.apply_registered_pass "affine-data-copy-generate" to %lowered {options = "generate-dma=false fast-mem-space=0"} : (!transform.any_op) -> !transform.any_op
    %5 = transform.apply_registered_pass "erase-buffer-deallocation" to %4 : (!transform.any_op) -> !transform.any_op
    %2 = transform.apply_registered_pass "promote-buffers-to-stack" to %5 {options = "max-rank-of-allocated-memref=4 max-alloc-size-in-bytes=4096"} : (!transform.any_op) -> !transform.any_op

    %affine_for_ops = transform.collect_matching @match_affinefors in %2 : (!transform.any_op) -> !transform.any_op
    transform.sodap.tag_ops %affine_for_ops, "loop" : !transform.any_op

    //full 11,12,26,27,30,31,32,33,34,35,36,37
    %loop11 = transform.structured.match attributes{uid = "loop_11"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop11 : !transform.any_op
    %loop12 = transform.structured.match attributes{uid = "loop_12"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop12 : !transform.any_op
    %loop26 = transform.structured.match attributes{uid = "loop_26"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop26 : !transform.any_op
    %loop27 = transform.structured.match attributes{uid = "loop_27"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop27 : !transform.any_op
    %loop30 = transform.structured.match attributes{uid = "loop_30"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop30 : !transform.any_op
    %loop31 = transform.structured.match attributes{uid = "loop_31"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop31 : !transform.any_op
    %loop32 = transform.structured.match attributes{uid = "loop_32"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop32 : !transform.any_op
    %loop33 = transform.structured.match attributes{uid = "loop_33"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop33 : !transform.any_op
    %loop34 = transform.structured.match attributes{uid = "loop_34"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop34 : !transform.any_op
    %loop35 = transform.structured.match attributes{uid = "loop_35"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop35 : !transform.any_op
    %loop36 = transform.structured.match attributes{uid = "loop_36"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop36 : !transform.any_op
    %loop37 = transform.structured.match attributes{uid = "loop_37"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.fullunroll %loop37 : !transform.any_op
    //by 2: 0,2,4,6,9,15,17,19,21,24
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
    %loop15 = transform.structured.match attributes{uid = "loop_15"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop15 {factor = 2} : !transform.any_op
    %loop17 = transform.structured.match attributes{uid = "loop_17"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop17 {factor = 2} : !transform.any_op
    %loop19 = transform.structured.match attributes{uid = "loop_19"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop19 {factor = 2} : !transform.any_op
    %loop21 = transform.structured.match attributes{uid = "loop_21"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop21 {factor = 2} : !transform.any_op
    %loop24 = transform.structured.match attributes{uid = "loop_24"} in %2 : (!transform.any_op) -> !transform.any_op
    transform.loop.unroll %loop24 {factor = 2} : !transform.any_op
    %sroa = transform.apply_registered_pass "affine-scalrep" to %2 : (!transform.any_op) -> !transform.any_op
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
