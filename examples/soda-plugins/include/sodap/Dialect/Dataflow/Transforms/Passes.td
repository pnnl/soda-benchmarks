//===- Passes.td - Dataflow dialect transforms -------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_TRANSFORMS_PASSES_TD
#define SODAP_DIALECT_DATAFLOW_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LoadStore :
    Pass<"sodap-dataflow-load-store", "::mlir::func::FuncOp"> {
  let summary = "Add the load and the store task of a dispatch";
  let description = [{
    Works on memrefs, after `buffer-results-to-out-params`, on a function
    holding one `dataflow.dispatch` of compute tasks. Afterwards nothing but
    allocations and constants is left outside the dispatch, which runs
    load → compute tasks → store:

    - Load: every function argument or constant (`memref.get_global`) a
      compute task reads gets a buffer of its own in the dispatch, which the
      task reads instead, and one task at the start of the dispatch copies
      them all in (`memref.copy`). An input read by two compute tasks is
      copied into a buffer for each.
    - Store: the `memref.copy` of a dispatch result into an out-param, which
      `buffer-results-to-out-params` adds after the dispatch, moves into it,
      as a task at its end copying the buffer the dispatch yields. The
      dispatch result is left unused, for canonicalize to drop. The store
      copies a tile per iteration of its writer's prefix: the writer's outer
      loops whose induction variable is the offset of its access along a dim
      of the buffer, with the step as size (`Prefix.h`). Each tile goes out as
      soon as the writer is done with it. With an empty prefix, it copies the
      whole buffer.

    The load buffers are whole: `sodap-dataflow-schedule` tiles the loads.

    Example:

    ```mlir
    %r = dataflow.dispatch : memref<4xf32> {
      %o = memref.alloc() : memref<4xf32>
      dataflow.task {
        linalg.copy ins(%arg0 : memref<4xf32>) outs(%o : memref<4xf32>)
      }
      dataflow.yield %o : memref<4xf32>
    }
    memref.copy %r, %arg1 : memref<4xf32> to memref<4xf32>
    // becomes
    %r = dataflow.dispatch : memref<4xf32> {
      %in = memref.alloc() : memref<4xf32>
      dataflow.task {
        memref.copy %arg0, %in : memref<4xf32> to memref<4xf32>
      }
      %o = memref.alloc() : memref<4xf32>
      dataflow.task {
        linalg.copy ins(%in : memref<4xf32>) outs(%o : memref<4xf32>)
      }
      dataflow.task {
        memref.copy %o, %arg1 : memref<4xf32> to memref<4xf32>
      }
      dataflow.yield %o : memref<4xf32>
    }
    ```

    Fails on more than one dispatch, on more than one out-param (not
    supported yet), and on a compute task that writes a function argument or
    a constant, or reads one with a dynamic shape or another layout.
  }];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def Schedule :
    Pass<"sodap-dataflow-schedule", "::mlir::func::FuncOp"> {
  let summary = "Give the tasks of a dispatch one loop nest, built on demand";
  let description = [{
    Works on the dispatches `sodap-dataflow-load-store` leaves: a first task
    copying whole arguments and constants into load buffers, the compute
    tasks, and a last task, the store. Every task gets the same loop nest,
    built on demand, and keeps its own ops of it:

    - Start from the nest of the store, the last task.
    - Where a task reads a tile of a buffer another task writes, put the
      writer's nest, with the loops stepping through the written tiles
      pinned to the tile read (or narrowed to it, if it holds several), and
      go on into it the same way.
    - Where a task reads a load buffer, the load task copies the tile in from
      the argument or constant, through the same views. A scalar is copied at
      each read.

    The load task keeps the copies, each compute task its ops, the store its
    copies; the loops and views left with none of a task's ops are dropped,
    and a task's top-level allocations stay at its start. This is the
    always-load schedule: a tile read again (under a loop that doesn't index
    it) is made again, its loads issued again. On every buffer shared by two
    tasks, the writer and the reader then walk the same tiles in the same
    order: `sodap-dataflow-ping-pong` can synchronise them with a depth-1
    token per tile.

    With `reuse-buffer-budget` greater than zero, regular static intermediate
    edges may instead retain a rectangular slab across repeated reads. The
    common writer/reader loop prefix determines the slab dimensions; no kernel
    names, matrix sizes, or compute tile constants are used by the analysis.
    The producer runs before the consumer's reuse loops. Ping-pong lowering
    publishes the whole slab and holds its token until every use completes.
    The budget covers two slots plus a conservative private-accumulator
    allowance for each selected edge. It is a reuse-storage budget, not a
    bound on all accelerator resources. Selection follows allocation order;
    this is not a global cost model. Unproven and over-budget edges retain
    the always-load schedule. Dynamic/ragged footprints, fan-out, off-chip
    materialization, and loop-order search are not added by this option.

    On trmm (matmul of the transpose of A_lower, then `alpha *`, then the
    store), per tile of the result, the loads are:

    ```
    for i: for j:
      load alpha
      for k:
        load the A, lower_mask, eye tiles of (k, i)   // A_lower behind A_lower^T (i, k)
        load B[k, j]
    ```

    and A_lower, its transpose and the matmul run under the same i, j, k.

    Fails on a compute task read by more than one task (each tile it makes
    goes to one read), on a compute task the store doesn't depend on, on a
    read of another task's buffer that isn't a tile of it through static,
    unit-stride subviews, or covers part of a written tile.
  }];
  let options = [Option<"reuseBufferBudget", "reuse-buffer-budget", "uint64_t",
      "0", "Byte budget for inferred intermediate reuse buffers (two slots plus accumulator); zero preserves the always-load schedule">,
      Option<"serial", "serial", "bool", "false",
      "Emit the same demand-driven schedule as one sequential LCS task with single-copy tiles/slabs; preserve reuse selection">];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def PingPong :
    Pass<"sodap-dataflow-ping-pong", "::mlir::func::FuncOp"> {
  let summary = "Synchronise the tasks of a dispatch with ping-pong buffers";
  let description = [{
    Works on the dispatches `sodap-dataflow-schedule` leaves: tasks sharing
    buffers a tile at a time, the writer and the readers of each walking the
    same tiles in the same order. Each such buffer becomes a ping-pong of its
    tile, `memref<2 x tile>`, with a `!dataflow.stream<i8, 1>` per reader
    carrying the slot. In a task, the tile's level is the innermost block
    holding all the task's accesses to the buffer, run once per tile:

    A scalar implementation can mark a zero-result scf.execute_region with
    `sodap.dataflow_tile`. All accesses to a shared buffer in a task must then
    occur in that same region. The token brackets the entire region, including
    dynamic tail loops, rather than individual scalar loads/stores. The caller
    must generate matching writer/reader traversals for runtime trip counts.
    Run this pass before canonicalization inlines these explicit tile scopes.

    - The writer keeps a slot variable. It writes the tile in the slot, then,
      after its last access of the level, writes the slot to every reader's
      stream and switches to the other slot. A writer that also reads the
      buffer (an accumulation) works on a private tile, copied in the slot
      before the tokens: Bambu wants each task on a ping-pong buffer to be a
      writer or a reader, not both.
    - A reader peeks the slot before its first access of the level and reads
      the stream after its last.

    Depth 1 is enough: the writer's push of the next token blocks until the
    reader frees the previous slot, so it never writes a slot in use.

    Fails on a buffer shared by tasks written by none or several of them, on
    a task using it in tiles of another shape, or on a reader reading a
    number of tiles other than the writer writes (when the loops have
    constant trip counts).
  }];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def OutstandingLoad :
    Pass<"sodap-dataflow-outstanding-load", "::mlir::func::FuncOp"> {
  let summary = "Issue the read requests of the load task from a task of their own";
  let description = [{
    Works on the top `sodap-dataflow-copy-to-dma` leaves, whose first argument
    is the read-request stream of the DMA. The load task (the `dataflow.task`
    with the `load` attribute) sends a request per row and then reads the row,
    so a request waits for the data of the previous one. A new task, with the
    `outstanding_load` attribute, takes the requests through a depth-one
    descriptor stream supplied by the receiver. The DMA command payload
    depends on a blocking descriptor read; it cannot issue speculatively
    while the receiver is blocked publishing a previous tile. The existing
    tile memories receive the payload; no new payload buffer is allocated.

    This conservative admission policy keeps one row transaction at a time.
    The pass name is retained for pipeline compatibility; unrestricted
    read-ahead is unsafe with a serialized read/write DMA socket.

    Run `canonicalize` afterwards: it drops the loops left empty in the new
    task and the request arithmetic left dead in the request task.

    Example:

    ```mlir
    dataflow.task attributes {load} {
      scf.for %r = %c0 to %c4 step %c1 {
        %req = arith.ori %len, ... : i64
        dataflow.stream_write %rq, %req : <i64, 4>, i64
        scf.for %e = %c0 to %c4 step %c1 {
          %w = dataflow.stream_read %rd : (!dataflow.stream<i32, 1>) -> i32
          memref.store ...
        }
      }
    }
    // becomes, after canonicalize
    %admitted = dataflow.stream {depth = 1 : i32} : <i64, 1>
    dataflow.task attributes {outstanding_load} {
      scf.for %r = %c0 to %c4 step %c1 {
        %req = dataflow.stream_read %admitted : (!dataflow.stream<i64, 1>) -> i64
        dataflow.stream_write %rq, %req {sodap.dma_ordered} : <i64, 4>, i64
      }
    }
    dataflow.task attributes {load} {
      scf.for %r = %c0 to %c4 step %c1 {
        %req = arith.ori %len, ... : i64
        dataflow.stream_write %admitted, %req {sodap.dma_ordered} : <i64, 1>, i64
        scf.for %e = %c0 to %c4 step %c1 {
          %w = dataflow.stream_read %rd : (!dataflow.stream<i32, 1>) -> i32
          memref.store ...
        }
      }
    }
    ```

    Does nothing on a function with no load task. Fails on more than one
    load task, or on a first argument that isn't a stream only written to.
  }];
  let dependentDialects = ["::mlir::sodap::dataflow::DataflowDialect"];
}

def MergeTasks :
    Pass<"sodap-dataflow-merge-tasks", "::mlir::func::FuncOp"> {
  let summary = "Merge the tasks of a dispatch into one, run in order";
  let description = [{
    The sequential flow (no dataflow, no double buffering) of
    `scripts/tosa_to_lcs.sh` with `LCS_SEQUENTIAL=true`: run after
    `sodap-dataflow-load-store`, it puts the bodies of every task of a
    dispatch, in order, into its last task (the store): load everything,
    compute everything, store everything, on the whole buffers. One task is
    one node, so Bambu runs it alone, with nothing to synchronise. The last
    task is the target since every alloc and constant of the dispatch is
    defined before it. The allocs of the dispatch used only in the task move
    into it, local to its node: Bambu takes an alloca of the dataflow top for
    a ping-pong channel between two modules. Role attributes (`load`,
    `compute`, `store`) are dropped.

    Example:

    ```mlir
    %in = memref.alloc() : memref<4xf32>
    dataflow.task attributes {load} { memref.copy %arg0, %in ... }
    %o = memref.alloc() : memref<4xf32>
    dataflow.task attributes {compute} { linalg.copy ins(%in ...) outs(%o ...) }
    dataflow.task attributes {store} { memref.copy %o, %arg1 ... }
    // becomes
    dataflow.task {
      %in = memref.alloc() : memref<4xf32>
      %o = memref.alloc() : memref<4xf32>
      memref.copy %arg0, %in ...
      linalg.copy ins(%in ...) outs(%o ...)
      memref.copy %o, %arg1 ...
    }
    ```

    Fails on a task with results.
  }];
  let dependentDialects = ["::mlir::sodap::dataflow::DataflowDialect"];
}

def TasksToNodes :
    Pass<"sodap-dataflow-tasks-to-nodes", "::mlir::func::FuncOp"> {
  let summary = "Lower dataflow tasks to nodes with explicit ports";
  let description = [{
    Replace every `dataflow.task` with a `dataflow.node` whose inputs, outputs,
    and parameters are explicit.

    Liveness analysis over the task body determines the ports: a livein of
    memref, stream, or tensor type becomes an output if the task writes it and
    an input otherwise, while every other livein becomes a parameter. Because a
    node is `IsolatedFromAbove`, each port arrives as a block argument, which is
    what lets a node later be emitted as a standalone HLS sub-function.
  }];
  let dependentDialects = ["::mlir::sodap::dataflow::DataflowDialect"];
}

def NodesToFunc :
    Pass<"sodap-dataflow-nodes-to-func", "::mlir::ModuleOp"> {
  let summary = "Lower dataflow nodes to functions and calls";
  let description = [{
    Outline each `dataflow.node` into its own `func.func` and replace it with a
    `func.call`, then unwrap the enclosing `dataflow.dispatch`.

    This flattens the dataflow hierarchy into the plain call-graph form that
    HLS backends consume, and is deliberately kept out of the default pipeline
    so that the structural dataflow operations survive when they are what the
    consumer wants to see.
  }];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::func::FuncDialect"
  ];
}

def InferTopInterface :
    Pass<"sodap-dataflow-infer-top-interface", "::mlir::ModuleOp"> {
  let summary = "Turn the external streams of the top into axis ports";
  let description = [{
    Works on the call-graph form `sodap-dataflow-nodes-to-func` produces. A
    stream argument of the top is part of the design's external interface
    when some stage actually reads or writes it; those are retyped from
    `!dataflow.stream<T, D>` to `!dataflow.stream<T, D, axis>`, in the top and
    in every function the stream is passed to, so the calls still type-check.

    A stream nobody touches is left a fifo, and so is every channel declared
    inside the design. A stream that is both read and written is rejected:
    an AXI4-Stream port carries data in one direction only. So is a function
    that receives an axis port at one call site and a fifo at another, since
    one signature cannot be both.

    Example:

    ```mlir
    func.func @top_node0(%req: !dataflow.stream<i64, 8>, ...)
    func.func @top(%req: !dataflow.stream<i64, 8>, ...) {
      call @top_node0(%req, ...) : ...
    }
    // after sodap-dataflow-infer-top-interface{top-func=top}, if top_node0
    // writes %req:
    func.func @top_node0(%req: !dataflow.stream<i64, 8, axis>, ...)
    func.func @top(%req: !dataflow.stream<i64, 8, axis>, ...)
    ```
  }];
  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"",
           "Top function of the design (default: the module's name)">
  ];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::func::FuncDialect"
  ];
}

def CopyToDMA :
    Pass<"sodap-dataflow-copy-to-dma", "::mlir::ModuleOp"> {
  let summary = "Turn the copies of the top's tiles into DMA transfers";
  let description = [{
    The memref arguments of the top become one DMA address space, laid out
    one after the other in argument order, in words. They are replaced by the
    four streams of a DMA engine, appended to the top's arguments:

        read requests  !dataflow.stream<i64, 4>
        read data      !dataflow.stream<i32, 1>
        write requests !dataflow.stream<i64, 4>
        write data     !dataflow.stream<i32, 1>

    Every `memref.copy` from a tile (`memref.subview`) of such an argument
    becomes, for each row of the tile along its innermost dimension, a read
    request `(length << 32) | address` and `length` reads of the data, stored
    into the copy's target. A copy into a tile is the same with writes. The
    data is 32-bit: `f32` goes through the channel as its bits.

    Every use of a memref argument must be such a tile, with static sizes and
    unit strides, copied from or into a local buffer, and the arguments must
    have static shapes, the identity layout, and 32-bit elements.

    Run it before `sodap-dataflow-tasks-to-nodes`, which then makes the
    streams ports of the load and store nodes, and let
    `sodap-dataflow-infer-top-interface` turn them into axis ports.
    dma-word-stride=2 stores each value in the low word of a 64-bit slot:
    read padding is discarded and write padding is zero. Host layout must match.
  }];
  let options = [
    Option<"topFunc", "top-func", "std::string", /*default=*/"",
           "Top function of the design (default: the module's name)">,
    Option<"wordStride", "dma-word-stride", "unsigned", /*default=*/"1",
           "32-bit words per shared-memory element; must match host lowering">
  ];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::affine::AffineDialect",
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect"
  ];
}

def TilesToLLVM :
    Pass<"sodap-tiles-to-llvm", "::mlir::ModuleOp"> {
  let summary = "Write each tile's LLVM IR, leave its kernel declared";
  let description = [{
    The analogue of `gpu-module-to-binary`, once each tile module (a nested
    module with `sodap.tile`) is in the LLVM dialect: translates it to LLVM IR
    in `<sodap.tile_dir>/acc.ll` (`mlir-translate` would only see the outer
    module), then empties it, but for its kernel, the function named after
    it, left as a declaration so the host's launches still resolve. Bambu
    runs on the files, outside.
  }];
}

def ReportLCSCost : Pass<"sodap-dataflow-report-cost", "::mlir::ModuleOp"> {
  let summary = "Report static DMA work and allocated storage after LCS scheduling";
  let options = [
    Option<"file", "file", "std::string", "\"schedule_cost.json\"", "JSON output" >,
    Option<"requestCycles", "request-cycles", "int64_t", "64", "Cost per row request">,
    Option<"wordCycles", "word-cycles", "int64_t", "1", "Cost per transferred 32-bit word">,
    Option<"wordStride", "dma-word-stride", "int64_t", "1", "Shared-memory words per element">,
    Option<"budget", "plm-bytes", "int64_t", "0", "Total logical storage cap, zero reports only">
  ];
}

#endif // SODAP_DIALECT_DATAFLOW_TRANSFORMS_PASSES_TD
