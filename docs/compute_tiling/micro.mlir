// Second (register) tiling of the matmul compute task: @MR@ x @NR@ blocks of C,
// k outermost inside the block, the block fully unrolled. micro_tiling.sh
// replaces @MR@ and @NR@; they must divide the memory tile (20 x 70 for gemm).
module attributes {transform.with_named_sequence} {
  transform.named_sequence @micro(%root: !transform.any_op {transform.readonly}) {
    %mm = transform.structured.match ops{["linalg.matmul"]} in %root
      : (!transform.any_op) -> !transform.any_op
    %t, %li, %lj = transform.structured.tile_using_for %mm tile_sizes [@MR@, @NR@, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %g = transform.structured.generalize %t : (!transform.any_op) -> !transform.any_op
    // (i, j, k) -> (k, i, j): k becomes the outer loop of the block.
    %x = transform.structured.interchange %g iterator_interchange = [2, 0, 1]
      : (!transform.any_op) -> !transform.any_op
    %loops = transform.structured.convert_to_loops %x
      : (!transform.any_op) -> !transform.op<"scf.for">
    %lk, %lii, %ljj = transform.split_handle %loops
      : (!transform.op<"scf.for">) -> (!transform.op<"scf.for">, !transform.op<"scf.for">, !transform.op<"scf.for">)
    transform.loop.unroll %ljj {factor = @NR@} : !transform.op<"scf.for">
    transform.loop.unroll %lii {factor = @MR@} : !transform.op<"scf.for">
    transform.yield
  }
}
