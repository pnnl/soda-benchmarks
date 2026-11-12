//===-- SODAPTransforms.td - SODAP transform dialect ops ---*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines SODAP transform dialect extension operations.
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_TRANSFORMS
#define SODAP_TRANSFORMS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def TagOpsOp : Op<Transform_Dialect, "sodap.tag_ops",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Tags payload operations with an incrementing unique ID";
  let description = [{
    For each payload operation associated with the handle, sets a `uid` string
    attribute of the form `<prefix>_<N>`, where `<prefix>` is provided as an
    attribute and `<N>` is an ever-increasing counter starting from 0.

    The counter increments in the order the payload operations appear in the
    handle's association list.

    Only reads the operand. Modifies the payload.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    StrAttr:$prefix);

  let results = (outs);

  let assemblyFormat = "$target `,` $prefix attr-dict `:` type($target)";
}

def TileAndFuseOp : Op<Transform_Dialect, "sodap.tile_and_fuse",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Tiles the nest of each result of a task, by iterator type";
  let description = [{
    Takes tasks on tensors (`dataflow.task`, from
    `transform.sodap.create_tasks`), each tiled on its own. For each linalg
    op producing a value the task yields:

    - tiles its parallel dims by the tile and fuses its producers, as
      `transform.structured.fuse` does, so a fill lands in the parallel loops.
      Only producers in the task are fused: the task's inputs (function
      arguments, constants, other tasks' results) and the edge it writes
      (a `bufferization.alloc_tensor` in the dispatch) are not tileable ops;
    - then, in that nest, tiles the reduction dims of every linalg op having
      some by the tile, as `transform.structured.tile_using_for` does, so
      the accumulator stays across the reduction loop.

    The tile sizes are given by exactly one of:

    - `max_tile_size`, automatic: one tile T for every loop, the largest
      divisor of the GCD of the loop ranges of every linalg op in the task's
      function that is at most `max_tile_size`: every loop tiles evenly, so
      every tile is static, and every task of the function gets the same
      tile. Loop ranges 60, 70, 80 and `max_tile_size = 10` give 10; 8, 12,
      16 give 4.
    - `tile_sizes`, explicit: one list per yielded op, in the order of the
      targets and then of their yields, each with one size per loop of that
      op in its iteration order, parallel and reduction alike; 0 leaves the
      loop untiled. Every other size must divide its loop range, so every
      tile is static. gemm's fill + matmul, then its epilogue:
      `[[10, 10, 10], [10, 10]]`. Any other op with reductions in a nest
      (a fused producer) is an error: its sizes have no place in the list.
    - `plm_bytes`, a cost model: the sizes with the fewest DMA cycles whose
      buffers fit in `plm_bytes` bytes. It models the tasks as
      `sodap-dataflow-load-store` and `sodap-dataflow-copy-to-dma` will cut
      them: every tile of an argument is loaded, and every tile of a result
      stored, at the innermost loop around its reader or writer, as a request
      per row (`dma_request_cycles`) then one word at a time
      (`dma_word_cycles`). Every buffer between tasks counts twice
      (ping-pong); a buffer inside a compute task counts once. Compute is
      left out: the same iterations run whatever the tiles. Each size divides
      its range, every yielded op keeps a tiled parallel loop, and loops whose
      tiles meet in a buffer between two tasks get one size. Every
      combination of divisors is tried.

    The parallel sizes tile the nest, the reduction sizes the reduction:
    `[4, 4, 4]` on parallel-parallel-reduction tiles by `[4, 4, 0]` then
    `[0, 0, 4]`. Every form reports the sizes used as a remark,
    `tile_sizes = [...]`, which given back as `tile_sizes` tiles the same,
    and the model's prediction for them as another,
    `dma_cycles = ..., plm_bytes = ...`.

    #### Return modes

    Returns nothing, and only reads the handle. Fails silenceably, without
    changing the IR, if not exactly one of `max_tile_size`, `tile_sizes` and
    `plm_bytes` is set, no sizes fit in `plm_bytes`, a target is not a `dataflow.task` in a `func.func`, a yielded value
    is not produced by a linalg op with a parallel dim, a loop range in the
    function is dynamic, or `tile_sizes` doesn't fit the yielded ops. Fails
    definitely if tiling fails, or if `tile_sizes` is given and a fused op
    has reductions.
  }];

  let arguments = (ins
    TransformHandleTypeInterface:$target,
    OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$max_tile_size,
    OptionalAttr<ArrayAttr>:$tile_sizes,
    OptionalAttr<ConfinedAttr<I64Attr, [IntPositive]>>:$plm_bytes,
    DefaultValuedAttr<I64Attr, "64">:$dma_request_cycles,
    DefaultValuedAttr<I64Attr, "1">:$dma_word_cycles);

  let results = (outs);

  let assemblyFormat = "$target attr-dict `:` type($target)";
}

// Not PromoteOp: that class is transform.structured.promote, and a second one
// would share its TypeID and never get registered.
def CreateTasksOp : Op<Transform_Dialect, "sodap.create_tasks",
    [DeclareOpInterfaceMethods<TransformOpInterface>,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Groups the linalg ops of a function into compute tasks";
  let description = [{
    Takes functions on tensors, before tiling, holding only linalg ops,
    `tensor.empty` and constants. Wraps their linalg ops in one
    `dataflow.dispatch` with one `dataflow.task` per group, just before the
    `func.return`:

    - Ops chained through their inits share a group, so a fill lands with the
      op it initializes. With no `groups`, this is the partition.
    - Each `groups` handle puts its ops, and the ops chained to them, in one
      group. A handle's ops in another function are ignored.

    The tasks follow the order of their dependencies, in program order when
    there is a choice. Each task yields its results used outside it: those
    are the edges. The init each edge is written into, through the chain of
    inits in the task, must be a `tensor.empty`: it becomes a
    `bufferization.alloc_tensor` in the dispatch, right before the task, so
    the edge is allocated there. The other `tensor.empty` inits get a copy in
    the task, and stay private to it. The dispatch yields the returned
    values.

    Constants and function arguments stay outside the dispatch: they are the
    memory edges, for the load tasks.

    #### Return modes

    Returns the dispatch ops. Fails silenceably, without changing the IR, if
    a target is not a `func.func` with one block, holds another kind of op
    or no linalg op, a grouped op is not one of its linalg ops, the groups
    depend on each other, or an edge isn't written into a `tensor.empty`.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target,
                       Variadic<TransformHandleTypeInterface>:$groups);

  let results = (outs TransformHandleTypeInterface:$dispatch);

  let assemblyFormat = [{
    $target (`groups` `(` $groups^ `)`)? attr-dict `:`
    functional-type(operands, results)
  }];
}

def PromoteTilesOp : Op<Transform_Dialect, "sodap.promote",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Copies the tiles of the function arguments into local buffers";
  let description = [{
    Like `transform.structured.promote`, gives the operands of the targeted
    linalg ops local buffers, with these differences:

    - The operands are chosen, not listed: every operand that is a
      `memref.subview` of a function argument, of an op inside an `scf.for`.
    - The buffer is a typed `memref.alloc` placed before the outermost
      enclosing loop, not an `i8` buffer behind a `memref.view` inside it,
      which is what `transform.sodap.multibuffer` needs.
    - The tile is copied in (`memref.copy`) right after its subview when its
      first user reads it, and copied out after its last writer. Both copies
      sit in the block of the subview, so a tile used by an inner loop is
      copied around that loop, not at every iteration of it.
    - Targeted ops using the same subview share one buffer.

    A scalar argument (`memref<f32>`), which cannot be tiled, is a tile of
    its own: a buffer per nest, copied in once, right before the outermost
    loop. Whole static tensor arguments are also copied once when all their
    uses (including subviews) are known reads. Written tensors or arguments
    with unknown users are left untouched. This fallback uses a full tensor
    buffer; it does not choose a smaller tile or guarantee a hardware fit.

    #### Return modes

    Returns the targeted ops. Fails silenceably, without changing the IR, if
    a target is not a linalg op, a tile has a dynamic shape or a scalar is
    written.
  }];

  let arguments = (ins TransformHandleTypeInterface:$target);

  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

def MultiBufferOp : Op<Transform_Dialect, "sodap.multibuffer",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Multibuffers an allocation over the whole loop nest";
  let description = [{
    Like `transform.memref.multibuffer`, expands the allocation by `factor`
    along a new leading dimension and makes its users, which must be linalg
    ops or `memref.copy` inside an `scf.for`, access one slot of it. Nested
    `memref.subview` users are supported: their types are recomputed for the
    selected bank, and the common loop encloses the views and their users.

    The difference is the slot index. Upstream uses the iteration of the
    innermost loop alone, which restarts every time that loop does. Here the
    slot is an `i8` variable, a `memref<i8>` next to the allocation, loaded at
    the start of the innermost loop holding every user and switched at the end
    of its body:

        var = (var + 1) % factor

    so consecutive tiles of the nest always use different slots. Only a
    `factor` of 2 is supported for now.

    There is no analysis: nothing may be carried in the buffer from one
    iteration to the next.

    An allocation filled by a `memref.copy` outside the loops, once before
    the nest (a scalar, see `transform.sodap.promote`), has nothing to
    alternate: it is left as it is, and not returned.

    #### Return modes

    Returns the new allocations. Fails silenceably if `factor` is not 2, or if
    an allocation has a dynamic shape or a user of another kind or outside a
    loop.
  }];

  let arguments =
      (ins Transform_ConcreteOpType<"memref.alloc">:$target,
           ConfinedAttr<I64Attr, [IntPositive]>:$factor);

  let results = (outs TransformHandleTypeInterface:$transformed);

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

def SplitStagesOp : Op<Transform_Dialect, "sodap.split_stages",
    [FunctionalStyleTransformOpTrait, MemoryEffectsOpInterface,
     DeclareOpInterfaceMethods<TransformOpInterface>]> {
  let summary = "Splits a tile loop nest into load, compute and store tasks";
  let description = [{
    Takes loops of nests prepared by `transform.sodap.promote` and
    `transform.sodap.multibuffer` (any of their loops: each nest is handled
    once, through its outermost loop) and replaces each nest with a
    `dataflow.dispatch` holding one `dataflow.task` per stage:

    - load: the `memref.copy`s from a tile of a function argument, and the
      copies of a whole one right before the nest (a scalar,
      `transform.sodap.promote`), which go before the nest in the task,
    - one compute task per group of linalg ops writing the same buffer (a
      fill lands with the op it initializes),
    - store: the `memref.copy`s into a tile of a function argument.

    Every task gets a clone of the whole nest without the other stages' ops;
    loops and index computations it does not need are removed.

    A buffer accessed by more than one stage must be written by exactly one,
    have one consumer, and be multibuffered: used only through one
    `memref.subview` whose first offset is the slot variable. Its consumer
    gets a token, a `!dataflow.stream<i8, 1>`, in the loop body where the slot
    is loaded: the producer, which owns the variable, writes the slot before
    the first loop after its last write of the buffer in that body, or at the
    end of the body if there is none (not always at the end: a load at an outer
    level would wait for the inner loops after it, which wait for the
    consumer), and switches the variable right after. The consumer peeks the
    token at the start, indexes the buffer with it and reads it at the end, so
    a slot is refilled only once its previous reader has released it.

    A buffer loaded before the nest is not multibuffered: it has one token
    per consumer for the whole task, which carries 0. Load writes it after the copies,
    before the nest; a consumer peeks it before its loops and reads it after
    them.

    A producer that also reads the buffer, like a matmul accumulating its
    tile, works on a private buffer instead, copied into the slot
    (`linalg.copy`) at the end of that loop body, and signals after it: Bambu
    requires every stage connected to a ping-pong buffer to be either a
    writer or a reader of it.

    #### Return modes

    Returns the dispatch ops. Fails silenceably, without changing the IR, if
    the nest has another kind of side-effecting op, a copy that is not a
    load or a store, or a shared buffer that breaks the rules above.
  }];

  let arguments = (ins Transform_ConcreteOpType<"scf.for">:$target);

  let results = (outs TransformHandleTypeInterface:$dispatch);

  let assemblyFormat =
    "$target attr-dict `:` functional-type(operands, results)";
}

#endif // SODAP_TRANSFORMS
