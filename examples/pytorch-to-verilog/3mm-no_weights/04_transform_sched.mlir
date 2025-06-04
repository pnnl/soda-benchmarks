module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {

    %all = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op

    %matmul = transform.collect_matching @match_matmul in %root : (!transform.any_op) -> !transform.any_op

    %matmul_tile = transform.collect_matching @match_tile_matmul in %root : (!transform.any_op) -> !transform.any_op


    transform.include @tiling failures(propagate) (%matmul_tile) : (!transform.any_op) -> ()

    transform.yield
  }

  transform.named_sequence @tiling (%entry: !transform.any_op {transform.consumed}){
    %L, %loops:3 = transform.structured.tile_using_for %entry tile_sizes [1, 2, 2] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %1 = transform.get_parent_op %L {op_name = "scf.for"} : (!transform.any_op) -> !transform.op<"scf.for">
    %2 = transform.get_parent_op %L {op_name = "scf.for", nth_parent=2} : (!transform.any_op) -> !transform.op<"scf.for">
    %3 = transform.get_parent_op %L {op_name = "scf.for", nth_parent=3} : (!transform.any_op) -> !transform.op<"scf.for">
    transform.loop.unroll %1 { factor = 1} : !transform.op<"scf.for"> 
    transform.loop.unroll %2 { factor = 1} : !transform.op<"scf.for"> 
    transform.loop.unroll %3 { factor = 1} : !transform.op<"scf.for"> 
    transform.yield  
  }

  transform.named_sequence @match_tile_matmul(
    %candidate: !transform.any_op {transform.readonly}) -> !transform.any_op {
  // Match a structured linear algebra operation.
  transform.match.structured %candidate : !transform.any_op {
  ^bb0(%c: !transform.any_op):
    transform.match.operation_name %candidate ["linalg.batch_matmul"] : !transform.any_op
  }
  transform.yield %candidate : !transform.any_op
}

  transform.named_sequence @match_matmul(
      %entry: !transform.any_op {transform.readonly}) -> !transform.any_op {
    transform.match.operation_name %entry ["linalg.batch_matmul"] : !transform.any_op
    transform.yield %entry : !transform.any_op
  }

}