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

// Not PromoteOp: that class is transform.structured.promote, and a second one
// would share its TypeID and never get registered.
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
    loop. Other arguments used whole inside the loop are left untouched.

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
    ops or `memref.copy` inside an `scf.for`, access one slot of it.

    The difference is the slot index. Upstream uses the iteration of the
    innermost loop alone, which restarts every time that loop does. Here the
    slot is the position of the iteration in the whole nest, counting all the
    `scf.for`s around the users that do not contain the allocation:

        for i, for j, for k: slot = (i' * Tj * Tk + j' * Tk + k') mod factor

    with `x' = (x - lb) / step` and `T` the trip counts, so consecutive tiles
    of the nest always use different slots. This needs static loop bounds.

    There is no analysis: nothing may be carried in the buffer from one
    iteration to the next.

    An allocation filled by a `memref.copy` outside the loops, once before
    the nest (a scalar, see `transform.sodap.promote`), has nothing to
    alternate: it is left as it is, and not returned.

    #### Return modes

    Returns the new allocations. Fails silenceably if an allocation has a
    dynamic shape, a user of another kind or outside a loop, or a loop with
    dynamic bounds.
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
    and be multibuffered: used only through one `memref.subview` whose first
    offset is the slot. Each consumer of it gets a token, a
    `!dataflow.stream<i8, 1>`, in the loop body where the slot is computed:
    the producer writes the slot before the first loop after its last write
    of the buffer in that body, or at the end of the body if there is none
    (not always at the end: a load at an outer level would wait for the inner
    loops after it, which wait for the consumer), the consumer peeks it at the
    start and reads it at the end, so a slot is refilled only once its
    previous reader has released it.

    A buffer loaded before the nest is not multibuffered: it has one token
    for the whole task, which carries 0. Load writes it after the copies,
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
