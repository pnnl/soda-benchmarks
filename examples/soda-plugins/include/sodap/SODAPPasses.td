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

    With `profile`, the sequence is bracketed by `esp_prof_begin`/`esp_prof_end`
    calls around the pack, accelerator and unpack phases.
  }];
  let options = [
    Option<"vecLen", "vec-len", "unsigned",
           /*default=*/"8", "Accelerator vector length; outdim is padded to a multiple of it">,
    Option<"profile", "profile", "bool",
           /*default=*/"false", "Emit esp_prof_begin/end around pack, accel and unpack">
  ];
}

#endif // SODAP_PASS
