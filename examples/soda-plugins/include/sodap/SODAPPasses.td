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
    3. `esp_accel_cfg_regs` — configure accelerator registers with dimensions/offsets
    4. `esp_accel_start` — launch the accelerator
    5. `esp_accel_wait` — busy-wait for completion
    6. `esp_fixed2float_f32` — convert fixed-point output back to float
    7. `esp_free_shared` — free shared memory buffer
  }];
}

#endif // SODAP_PASS
