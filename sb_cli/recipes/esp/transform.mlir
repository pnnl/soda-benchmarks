// transform.mlir — offload linalg.batch_matmul to the ESP accelerator.
//
// Applies `sodap-linalg-batch-matmul-to-esp` from SODAPlugin.so, which replaces
// each linalg.batch_matmul with the seven-call ESP runtime sequence
// (alloc_shared / float2fixed x2 / accel_cfg_regs / accel_start / accel_wait /
// fixed2float / free_shared). See sb_cli/recipes/esp/README.md.
module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {

    // The pass is anchored on builtin.module, not func.func, so match the module
    // rather than the function the instrumentation recipes reach for. By the time
    // the interpreter runs, -soda-outline-bambu-code has already nested the
    // outlined kernel in its own module, and the match finds both.
    %module = transform.structured.match ops{["builtin.module"]} in %root
        : (!transform.any_op) -> !transform.any_op

    transform.apply_registered_pass "sodap-linalg-batch-matmul-to-esp" to %module
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
