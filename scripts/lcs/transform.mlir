// Schedules for scripts/tosa_to_lcs.sh. The new flow runs one of the
// partitions, @tasks (default) or @tasks_trmm (LCS_ENTRY). @tile, @promote,
// @multibuffer and @split are the old flow's (step 7 of
// docs/LinalgToLCSPlan.md drops them).
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

  // The default partition: ops chained through their inits share a task (a
  // fill with the op it initializes), every other linalg op gets its own.
  transform.named_sequence @tasks(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %dispatch = transform.sodap.create_tasks %func : (!transform.any_op) -> !transform.any_op
    transform.include @tile_tasks failures(propagate) (%func) : (!transform.any_op) -> ()
    transform.yield
  }

  // trmm as in docs/LinalgToLCSPlan.md: the transpose with its producer
  // (A_lower), the matmul with its consumer (the epilogue). Meant for one
  // linalg.transpose and one linalg.matmul: without them the handles are
  // empty and this is the default partition.
  transform.named_sequence @tasks_trmm(%root: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %root : (!transform.any_op) -> !transform.any_op
    %transpose = transform.structured.match ops{["linalg.transpose"]} in %func : (!transform.any_op) -> !transform.any_op
    %producer = transform.get_producer_of_operand %transpose[0] : (!transform.any_op) -> !transform.any_op
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %func : (!transform.any_op) -> !transform.any_op
    %consumer = transform.get_consumers_of_result %matmul[0] : (!transform.any_op) -> !transform.any_op
    %compute1 = transform.merge_handles %producer, %transpose : !transform.any_op
    %compute2 = transform.merge_handles %matmul, %consumer : !transform.any_op
    %dispatch = transform.sodap.create_tasks %func groups(%compute1, %compute2)
        : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.include @tile_tasks failures(propagate) (%func) : (!transform.any_op) -> ()
    transform.yield
  }

  // Each task gets its own nest, its sizes from the cost model (64 KiB of
  // buffers; a DMA request 64 cycles, a word 1). LCS_TILE_SIZES and
  // LCS_PLM_BYTES, LCS_DMA_REQUEST_CYCLES, LCS_DMA_WORD_CYCLES in
  // tosa_to_lcs.sh replace these attributes.
  // fold_tensor_empty shrinks the private tensor.empty to tiles; its pattern
  // driver also removes the untiled ops left dead by the fusion.
  transform.named_sequence @tile_tasks(%func: !transform.any_op {transform.readonly}) {
    %tasks = transform.structured.match ops{["dataflow.task"]} in %func : (!transform.any_op) -> !transform.any_op
    transform.sodap.tile_and_fuse %tasks {plm_bytes = 65536 : i64, dma_request_cycles = 64 : i64, dma_word_cycles = 1 : i64} : !transform.any_op
    transform.apply_patterns to %func {
      transform.apply_patterns.tensor.fold_tensor_empty
    } : !transform.any_op
    transform.yield
  }
}
