//===- SODAPPsss.td - SODAP dialect passes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_PASS
#define SODAP_PASS

include "mlir/Pass/PassBase.td"

def SODAPSwitchBarFoo: Pass<"sodap-switch-bar-foo", "::mlir::ModuleOp"> {
  let summary = "Switches the name of a FuncOp named `bar` to `foo` and folds.";
  let description = [{
    Switches the name of a FuncOp named `bar` to `foo` and folds.
    ```
    func.func @bar() {
      return
    }
    // Gets transformed to:
    func.func @foo() {
      return
    }
    ```
  }];
}

def LinalgRankReduceContractions
    : Pass<"sodap-linalg-rank-reduce-contractions"> {
  let summary = "Drop the unit batch dim of linalg contractions";
  let description = [{
    Runs upstream's `linalg::populateContractionOpRankReducingPatterns`, which
    has no pass of its own in LLVM 19: a contraction with a unit batch or a
    unit outer dim becomes the next smaller named op, between
    `collapse_shape`/`expand_shape` of its operands. For example
    `linalg.batch_matmul` 1×M×K becomes `linalg.matmul`, and `linalg.matmul`
    with a unit M becomes `linalg.vecmat`. The op stays named, so
    `linalg-fold-unit-extent-dims` and `linalg-fuse-elementwise-ops` leave it
    alone.
  }];
  let dependentDialects = [
    "::mlir::linalg::LinalgDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def LinalgConstantsToIndices
    : Pass<"sodap-linalg-constants-to-indices"> {
  let summary = "Compute tril/triu/eye-like constant inputs of linalg.generic "
                "from the loop indices";
  let description = [{
    A rank-2 dense constant whose element (i, j) depends only on the diagonal
    d = j - i, as a step (`d >= k ? c1 : c0`, tril and triu) or a single
    diagonal (`d == k ? c1 : c0`, eye), is dropped from the `ins` of every
    `linalg.generic` that reads it through a permutation. The body computes it
    instead, from `linalg.index`, `arith.cmpi` and `arith.select`, so the
    constant needs no buffer and no load. Other constants are left alone.

    This is upstream's `FoldScalarOrSplatConstant` (run by
    `linalg-fuse-elementwise-ops`) for diagonal-structured constants: TOSA
    has no tril, triu or eye, so PyTorch models pass them as constants.
    ```mlir
    %eye = arith.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf32>
    %0 = linalg.generic ... ins(%a, %eye) outs(%e) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    }
    // becomes
    %0 = linalg.generic ... ins(%a) outs(%e) {
    ^bb0(%x: f32, %o: f32):
      %i = linalg.index 0 : index
      %j = linalg.index 1 : index
      %d = arith.subi %j, %i : index
      %c = arith.cmpi eq, %d, %c0 : index
      %y = arith.select %c, %one, %zero : f32
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    }
    ```
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::linalg::LinalgDialect"
  ];
}

def ViewOpGraph : Pass<"soda-view-op-graph"> {
  let summary = "Print Graphviz visualization of an operation";
  let description = [{
    This pass prints a Graphviz graph of a module.

    - Operations are represented as nodes;
    - Uses (data flow) as edges;
    - Control flow as dashed edges;
    - Regions/blocks as subgraphs.

    By default, only data flow edges are printed.

    Note: See https://www.graphviz.org/doc/info/lang.html for more information
    about the Graphviz DOT language.
  }];
  let options = [
    Option<"maxLabelLen", "max-label-len", "unsigned",
            /*default=*/"20", "Limit attribute/type length to number of chars">,
    Option<"printAttrs", "print-attrs", "bool",
           /*default=*/"true", "Print attributes of operations">,
    Option<"printControlFlowEdges", "print-control-flow-edges", "bool",
           /*default=*/"false", "Print control flow edges">,
    Option<"printDataFlowEdges", "print-data-flow-edges", "bool",
           /*default=*/"true", "Print data flow edges">,
    Option<"printResultTypes", "print-result-types", "bool",
            /*default=*/"true", "Print result types of operations">
  ];
  let constructor = "mlir::sodap::createPrintOpGraphPass()";
}

def InstrBounds : Pass<"soda-instr-scf-for-assert-bounds"> {
  let summary = "Insert assertions to check bounds of SCF For loops";
  let description = [{
    This pass inserts assertion calls that will be mapped to SODA assertion library.
    It instruments all SCF For loops with a call to `@soda_assert_bounds(<bound>, <expected>)`
    where `<bound>` is the loop induction variable and `<expected>` is the expected
    upper bound of the loop. The pass assumes that the expected upper bound is
    statically known and can be determined from the loop bounds.
  }];
}

def TagOps : Pass<"sodap-tagops", "::mlir::ModuleOp"> {
  let summary = "Tag all instances of a specified anchor operation with a unique ID";
  let description = [{
    This pass walks the module and finds all operations matching the specified
    anchor operation name. Each matching operation is tagged with a `uid`
    string attribute of the form `<prefix>_<N>`, where `<prefix>` is a
    user-provided string and `<N>` is an ever-increasing counter assigned
    in visitation order.

    Example with `anchor-op=func.call prefix=my_id`:
    ```
    func.call @foo() : () -> ()
    func.call @bar() : () -> ()
    // Gets transformed to:
    func.call @foo() {uid = "my_id_0"} : () -> ()
    func.call @bar() {uid = "my_id_1"} : () -> ()
    ```
  }];
  let options = [
    Option<"anchorOp", "anchor-op", "std::string",
           /*default=*/"\"\"", "The operation name to match (e.g. func.call)">,
    Option<"prefix", "prefix", "std::string",
           /*default=*/"\"uid\"", "The prefix string for the uid tag">
  ];
}

def InstrHWCounters : Pass<"soda-instr-hw-counters-at-loop-bounds"> {
  let summary = "Insert hardware counters instrumentation";
  let description = [{
    This pass instruments loop basic blocks with hardware counter calls mapped to the SODA hardware counter library.
    At the start of each loop body, it inserts `@soda_instr_hw_counter(%run=true, %loc=id)` to start the counter,
    where `%run` is a boolean to enable instrumentation and `%loc` is a unique identifier for the counter.
    At the end of the loop body, it inserts `@soda_instr_hw_counter(%run=false, %loc=id)` to stop the counter.

    When `report-at-end` is enabled, a single extra
    `@sodaInstrHWCounters(%run=true, %loc=<report-sentinel>)` call is inserted
    right before each `func.return` in the instrumented function. The
    hardware IP recognizes the sentinel location (all bits set) as a request
    to print every counter's final value once, instead of relying on the
    default per-stop `$display` (which floods the simulation log with one
    line per loop iteration/stop).
  }];
  let options = [
    Option<"reportAtEnd", "report-at-end", "bool",
           /*default=*/"false",
           "Emit a single finalize call before each return to print all "
           "counters once at the end of simulation">
  ];
}

def InstrChangeLocation : Pass<"soda-instr-change-location"> {
  let summary = "Insert change-location calls for a single active HW counter";
  let description = [{
    This pass instruments loop bodies with a single `@sodaInstrChangeLocation(%loc=id)`
    call mapped to a single-active-counter SODA hardware IP. Unlike
    `soda-instr-hw-counters-at-loop-bounds`, which starts/stops many concurrent
    counters, this pass only ever switches which single running counter is
    active: at the start of each loop body it emits one call that changes the
    tracked location, instead of a pair of start/stop calls. Location `0`
    counts cycles from reset (i.e. the outermost/global scope); higher
    locations are used as the pass descends into nested loops.

    A single finalize `@sodaInstrChangeLocation(<report-sentinel>)` call
    (sentinel location with all bits set) is also inserted right before each
    `func.return`, asking the hardware IP to print every location's final
    count once. Because only one counter is ever active, this design never
    needs a per-stop `$display`.
  }];
}

def InstrDynamicOpCounts : Pass<"soda-instr-dynamic-op-counts-at-loop-bounds"> {
  let summary = "Collect dynamic operation counts at SCF loop boundaries";
  let description = [{
    Instruments each `scf.for` loop with calls to `@sodaInstrCollectOpCounts`.
    The generated calls collect dynamic counts at runtime for:

    - `memref.load`
    - `memref.store`
    - `arith.addf`, `arith.mulf` for f32/f64
    - `arith.addi`, `arith.muli` for i32/i64

    A start marker is inserted at the beginning of each loop body and a stop
    marker with per-iteration deltas is inserted before `scf.yield`.
  }];
}

def InstrDynamicCounter : Pass<"soda-instr-dynamic-counter"> {
  let summary = "Insert fully dynamic per-op counters with configurable filters";
  let description = [{
    Instruments operations directly with calls to `@sodaInstrDynamicCounter`.
    Unlike loop-boundary counting, this pass increments counters at each dynamic
    execution of a tracked operation.

    Use `tracked-kinds` as a `+`-separated list (or comma-separated) to
    choose what to count.
    Supported values:

    - `all`
    - `memref-load`
    - `memref-store`
    - `memref` (enables both load/store)
    - `arith-int`
    - `arith-float`
    - `arith` (enables both int/float)
    - `scf`
    - `affine`
  }];
  let options = [
    Option<"trackedKinds", "tracked-kinds", "std::string",
           /*default=*/"\"all\"",
          "Counter kinds to track, separated by '+' or ',' (e.g. memref-load+memref-store+arith-float)">
  ];
}

def SwapOpToHW : Pass<"sodap-swap-op-to-hw", "::mlir::ModuleOp"> {
  let summary = "Swap a linalg.dot reduction for a call to a HW dot-product module";
  let description = [{
    This pass matches `linalg.dot` operations -- a fixed-size vector dot
    product/reduction over two 1-D memrefs into a scalar memref -- and
    replaces each match with a single call to
    `@sodaVectorDot(%A, %B, %len, %out)`, mirroring a Bambu memory-master
    hardware IP (modeled on the `IP_integration_example` `module1`/`module2`
    memory-master pattern). This demonstrates the instrumentation flow
    adding *performance* hardware, not just observability
    (assertions/counters).

    Matcher: any `linalg.dot` op whose two 1-D memref inputs have the same
    statically-known length and whose output is a rank-0 (scalar) memref.
    Dot products with dynamic operand shapes are left untouched, since the
    HW module is modeled as a fixed-size vector engine.
  }];
}

def ProfileKernel : Pass<"sodap-profile-kernel", "::mlir::ModuleOp"> {
  let summary = "Measure a complete kernel call with the ESP region profiler";
  let description = [{
    Bracket the named function with region 7 (ESP_PROF_KERNEL). This measures
    the CPU kernel or the complete offload including allocation and transfers.
    Region 0 remains available to the application's outer timer.
  }];
  let options = [
    Option<"kernel", "kernel", "std::string", /*default=*/"\"forward_kernel\"",
           "Function to instrument">
  ];
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::func::FuncDialect"];
}

def KernelToESPAccelerator : Pass<"sodap-kernel-to-esp-accelerator", "::mlir::ModuleOp"> {
  let summary = "Replace a kernel's body with one run of its ESP accelerator";
  let description = [{
    For the CPU side of a kernel whose whole body is the accelerator, as Bambu
    builds it from the same function. The body becomes:

    1. `esp_alloc_shared` -- allocate the shared buffer
    2. `esp_copy_to_shared_f32` -- preserve every argument's initial contents
    3. `esp_accel_start`, `esp_accel_wait`
    4. `esp_copy_from_shared_f32` -- once per argument the body writes
    5. `esp_free_shared`

    The arguments lie one after the other in argument order, in words, which is
    the layout `sodap-dataflow-copy-to-dma` compiles into the accelerator. Run
    both on the same function, so the argument order is the same. The data is
    copied as its bits: the accelerator reads `f32` from the DMA channel as is.

    Arguments must be f32 memrefs with static shapes and the identity layout.
    Copy-in is conservative: a written argument may be read before writing or
    only partially overwritten. Removing these copies requires a full-overwrite
    proof. With profile=true, measure allocation, packing, execution and unpacking.
  }];
  let options = [
    Option<"kernel", "kernel", "std::string", /*default=*/"\"forward_kernel\"",
           "Function to replace">,
    Option<"profile", "profile", "bool", /*default=*/"false",
           "Measure allocation, packing, execution and unpacking">,
    Option<"dmaWordStride", "dma-word-stride", "unsigned", /*default=*/"1",
           "32-bit words per shared-memory element (1 dense, 2 padded for DMA64)">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::memref::MemRefDialect"
  ];
}

def OrderKernelArguments : Pass<"sodap-order-kernel-arguments", "::mlir::ModuleOp"> {
  let summary = "Give an outlined kernel its caller's argument order";
  let description = [{
    Run in soda-opt, right after `-soda-outline-bambu-code`. The outlining
    orders the kernel's arguments by their first use in the body, so two
    pipelines outlining the same function (the accelerator and the CPU
    baseline) can give the kernel two orders, while the test data is built for
    one (`argument_order.txt`). This pass permutes the operands of each
    `soda.launch_func`, and the arguments of the `soda.func` it launches, into
    the order of the caller's arguments: `argument_order.txt` is then
    `0 1 2 ...` whatever the body.

    Example:

    ```mlir
    soda.launch_func @k::@k args(%arg1 : memref<f32>, %arg0 : memref<4xf32>)
    soda.func @k(%arg0: memref<f32>, %arg1: memref<4xf32>) kernel { ... }
    // becomes
    soda.launch_func @k::@k args(%arg0 : memref<4xf32>, %arg1 : memref<f32>)
    soda.func @k(%arg0: memref<4xf32>, %arg1: memref<f32>) kernel { ... }
    ```

    Fails on a launch whose operands aren't distinct arguments of its caller,
    or with async dependencies. The soda ops are matched by name: the plugin
    doesn't link the soda dialect.
  }];
}

def LinalgBatchMatmulToESP : Pass<"sodap-linalg-batch-matmul-to-esp", "::mlir::ModuleOp"> {
  let summary = "Replace linalg.batch_matmul with ESP accelerator runtime calls";
  let description = [{
    This pass replaces `linalg.batch_matmul` operations with a sequence of calls
    to ESP accelerator runtime wrapper functions that mirror the C-level
    invocation flow:

    1. `esp_alloc_shared` — allocate shared memory buffer
    2. `esp_float2fixed_f32` — convert each input memref to fixed-point in shared mem
    3. `esp_accel_write_reg` — one call per accelerator register (offset, value)
    4. `esp_accel_start` — launch the accelerator
    5. `esp_accel_wait` — busy-wait for completion
    6. `esp_fixed2float_f32` — convert fixed-point output back to float
    7. `esp_free_shared` — free shared memory buffer

    The pass owns the shared-memory layout. It pads the output dimension to a
    multiple of `vec-len`, gives the bias its own region, and passes each
    operand's leading dimension to the copy routines, so the runtime needs no
    knowledge of the accelerator: it copies with the stride it is told and
    writes the registers it is told.

    With `marshal=ir`, steps 2 and 6 are not runtime calls but `linalg.generic`
    ops over strided views (`memref.reinterpret_cast`) of the shared buffer,
    which `esp_alloc_shared` then returns as a `memref<i32>`. The conversion
    becomes ordinary IR that later transformations can fuse and optimize, and
    the runtime no longer contains the token format.

    With `profile`, the sequence is bracketed by `esp_prof_begin`/`esp_prof_end`
    calls around the pack, accelerator and unpack phases, and -- when the block
    holds a single matmul -- the rest of the block as the epilogue.

    Operands must be rank-3 f32 memrefs with a statically known batch size of 1.
    `marshal=ir` additionally requires static shapes and permits at most one
    offload per function: its shared buffer stays live until the enclosing
    block terminator. Use `marshal=runtime` for sequential offloads. Reentrant
    offloads through calls while a shared buffer is live remain unsupported;
    this pass does not perform interprocedural lifetime analysis.
  }];
  let options = [
    Option<"vecLen", "vec-len", "unsigned",
           /*default=*/"8", "Accelerator vector length; outdim is padded to a multiple of it">,
    Option<"profile", "profile", "bool",
           /*default=*/"false", "Emit esp_prof_begin/end around pack, accel, unpack and epilogue">,
    Option<"marshal", "marshal", "std::string",
           /*default=*/"\"runtime\"",
           "Where the float<->token conversion runs: 'runtime' (calls into the ESP runtime) "
           "or 'ir' (linalg.generic over views of the shared buffer)">
  ];
}

#endif // SODAP_PASS
