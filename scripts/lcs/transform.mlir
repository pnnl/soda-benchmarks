module attributes {transform.with_named_sequence} {
  // for i, j (TxT): C tile = fill; for k (T): C tile += A tile * B tile. T is
  // the largest divisor of the loop ranges up to 10: 4 on the LCS sizes.
  transform.named_sequence @tile(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %func {max_tile_size = 10 : i64} : !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.fold_tensor_empty
    } : !transform.any_op
    transform.yield
  }

  transform.named_sequence @promote(%root: !transform.any_op {transform.readonly}) {
    %ops = transform.structured.match interface{LinalgOp} in %root : (!transform.any_op) -> !transform.any_op
    %promoted = transform.sodap.promote %ops : (!transform.any_op) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @multibuffer(%root: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %root : (!transform.any_op) -> !transform.op<"memref.alloc">
    // The slot is a variable that is never reset, so the ping-pong alternates
    // across loop boundaries too (upstream restarts with the innermost loop).
    %mb = transform.sodap.multibuffer %allocs {factor = 2 : i64} : (!transform.op<"memref.alloc">) -> !transform.any_op
    transform.yield
  }

  transform.named_sequence @split(%root: !transform.any_op {transform.readonly}) {
    %loops = transform.structured.match ops{["scf.for"]} in %root : (!transform.any_op) -> !transform.op<"scf.for">
    %dispatch = transform.sodap.split_stages %loops : (!transform.op<"scf.for">) -> !transform.any_op
    transform.yield
  }
}
