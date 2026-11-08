//===- Passes.td - SODAP passes on TOSA ---------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// In SODATosaPlugin, not SODAPlugin: they need the TOSA dialect, which
// mlir-opt has and soda-opt doesn't, and the plugins bind their MLIR symbols
// to the tool that loads them.
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_TOSA_PASSES
#define SODAP_TOSA_PASSES

include "mlir/Pass/PassBase.td"

def TosaSinkTransposes : Pass<"sodap-tosa-sink-transposes"> {
  let summary = "Move tosa.transpose below the elementwise ops reading it, "
                "and cancel the pairs that meet";
  let description = [{
    torch-mlir wraps every TOSA convolution in transposes, from PyTorch's NCHW
    to TOSA's NHWC and back, and the elementwise ops between two convolutions
    (ReLU, bias, residual adds) keep the transposes apart, so `canonicalize`
    can't cancel them. This pass moves a transpose below an elementwise TOSA
    op (one with `ResultsBroadcastableShape`) when every operand of the op is
    either transposed by the same permutation or a constant, which gets the
    inverse transpose (folded by `tosa-layerwise-constant-fold`). With
    tosa.transpose's canonicalization in the same pattern set, two inverse
    transposes that meet cancel:
    ```mlir
    %a = tosa.transpose %conv, %to_nchw
    %r = tosa.clamp %a
    %b = tosa.transpose %r, %to_nhwc
    // becomes
    %b = tosa.clamp %conv
    ```
    A transpose whose input is a constant counts as a constant, so a
    self-inverse permutation can't make the pattern loop.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def TosaHoistTransposes
    : Pass<"sodap-tosa-hoist-transposes", "::mlir::ModuleOp"> {
  let summary = "Move the transposes at the boundary of private functions "
                "into their callers";
  let description = [{
    For each private function only called by `func.call`, so that it takes
    and returns data in the layout it computes in:

    - an argument read only by `tosa.transpose`s with one constant
      permutation takes the transposed type instead, and every caller
      transposes the operand;
    - a result returned straight from a `tosa.transpose` is returned before
      it, and every caller transposes the result.

    In the caller, the inverse transposes between two calls then meet and
    `canonicalize` cancels them, and the transposes of constant weights fold
    (`tosa-layerwise-constant-fold`). For `compile_modular.py`'s blocks
    (weights as arguments, `@main` calling them), each block then works in
    NHWC, with weights in TOSA's layout, and transposes nothing. Run
    `sodap-tosa-sink-transposes` first, so a transpose ahead of the last
    elementwise ops reaches the `return`.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

#endif // SODAP_TOSA_PASSES
