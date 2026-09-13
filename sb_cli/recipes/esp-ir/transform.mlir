// transform.mlir — offload linalg.batch_matmul to the ESP accelerator, with
// the float<->token conversion generated as IR and fused with its consumer.
//
// Same pass as the `esp` recipe, with marshal=ir: the conversions become
// linalg.generic over strided views of the shared buffer instead of runtime
// calls, and the buffer itself is a memref<i32>. Because they are ordinary
// loops, the affine pipeline below fuses the unpack with the alpha/beta
// epilogue that consumes it -- one pass over the output instead of four, and
// no intermediate buffers. See README.md here and docs/ESPBackend.md.
//
// Needs a soda-opt that registers fold-memref-alias-ops and
// affine-loop-normalize (mlir-opt has both; soda-opt registers passes by hand).
module @transforms attributes { transform.with_named_sequence } {
  transform.named_sequence @__transform_main(
      %root: !transform.any_op {transform.readonly}) {

    // The pass is anchored on builtin.module, not func.func, so match the module
    // rather than the function the instrumentation recipes reach for. By the time
    // the interpreter runs, -soda-outline-bambu-code has already nested the
    // outlined kernel in its own module, and the match finds both.
    %module = transform.structured.match ops{["builtin.module"]} in %root
        : (!transform.any_op) -> !transform.any_op

    %esp = transform.apply_registered_pass "sodap-linalg-batch-matmul-to-esp" to %module
        { options = "vec-len=8 profile=true marshal=ir" }
        : (!transform.any_op) -> !transform.any_op

    // Everything is loops now. Fold the reshapes into the accesses so fusion
    // sees the same memref on both sides, fuse, then replace the 1x1 buffers
    // fusion leaves behind with SSA values. (Matched inside the pass's result:
    // apply_registered_pass consumes its handle, and %root with it.)
    %func = transform.structured.match ops{["func.func"]} in %esp
        : (!transform.any_op) -> !transform.any_op
    %f1 = transform.apply_registered_pass "convert-linalg-to-affine-loops" to %func
        : (!transform.any_op) -> !transform.any_op
    %f2 = transform.apply_registered_pass "fold-memref-alias-ops" to %f1
        : (!transform.any_op) -> !transform.any_op
    %f3 = transform.apply_registered_pass "affine-loop-fusion" to %f2
        : (!transform.any_op) -> !transform.any_op
    %f3a = transform.apply_registered_pass "canonicalize" to %f3
        : (!transform.any_op) -> !transform.any_op
    %f3b = transform.apply_registered_pass "cse" to %f3a
        : (!transform.any_op) -> !transform.any_op
    %f4 = transform.apply_registered_pass "affine-scalrep" to %f3b
        : (!transform.any_op) -> !transform.any_op
    %f5 = transform.apply_registered_pass "affine-loop-invariant-code-motion" to %f4
        : (!transform.any_op) -> !transform.any_op
    %f6 = transform.apply_registered_pass "affine-loop-normalize" to %f5
        : (!transform.any_op) -> !transform.any_op
    %f7 = transform.apply_registered_pass "canonicalize" to %f6
        : (!transform.any_op) -> !transform.any_op
    %f8 = transform.apply_registered_pass "cse" to %f7
        : (!transform.any_op) -> !transform.any_op

    transform.yield
  }
}
