module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {

    %module = transform.structured.match ops{["builtin.module"]} in %root : (!transform.any_op) -> !transform.any_op

    // Swap every fixed-size `linalg.dot` reduction for a call to the
    // `sodaVectorDot` HW module, instead of merely instrumenting it.
    transform.apply_registered_pass "sodap-swap-op-to-hw" to %module : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
