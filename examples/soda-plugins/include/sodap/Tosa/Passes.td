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

def TosaRaiseTransposes : Pass<"sodap-tosa-raise-transposes"> {
  let summary = "Move tosa.transpose above elementwise ops with constant "
                "operands, toward the arguments";
  let description = [{
    `transpose(op(x, c...))` becomes `op(transpose(x), transpose(c)...)` when
    `op` is an elementwise TOSA op (`ResultsBroadcastableShape`) read only by
    the transpose and every operand but at most one is a `tosa.const`. The
    transposes of constants are left for `sodap-tosa-fold-constant-transposes`.
    With tosa.transpose's canonicalization in the same set, transposes that
    meet compose. trmm's masked matrix, `T(add(mul(A, mask), C))`, becomes
    `add(mul(T(A), T(mask)), T(C))`: the transpose reaches the argument,
    which `sodap-tosa-outline-kernel` leaves to the host.

    Only transposes and the ops created here are rewritten or folded: a
    function without a transpose is left as it is.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def TosaFoldConstantTransposes : Pass<"sodap-tosa-fold-constant-transposes"> {
  let summary = "Fold the transposes of constants into the constants";
  let description = [{
    Each `tosa.transpose` of a `tosa.const` (its input, not its permutation)
    becomes a constant holding the transposed data. The transpose must be the
    constant's only use; otherwise the pass fails, since folding would keep
    both layouts of the data.
  }];
  let dependentDialects = ["::mlir::tosa::TosaDialect"];
}

def TosaOutlineKernel : Pass<"sodap-tosa-outline-kernel", "::mlir::ModuleOp"> {
  let summary = "Move the kernel into a tile module, leave the host its "
                "transposes and a launch";
  let description = [{
    The module's one public function, `@forward`, splits in two:

    - the tile: `module @forward_kernel attributes {sodap.tile = 0,
      sodap.tile_dir = "tiles/forward_kernel"}` holding `@forward_kernel`,
      the whole computation;
    - the host: `@forward`, with the same signature, holding the transposes
      of its arguments and of its results, which the CPU computes, and a
      `dataflow.launch` of the kernel.

    The kernel's arguments are, for each argument of `@forward` in order, the
    argument if the computation reads it, then each of its transposes. Its
    results are `@forward`'s, before their transposes.

    Any other transpose (one between two ops) fails the pass: the
    accelerator must not transpose. Run `sodap-tosa-raise-transposes` and
    `sodap-tosa-fold-constant-transposes` first.
  }];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::tosa::TosaDialect"
  ];
}

def TosaOutlineOperators : Pass<"sodap-tosa-outline-operators", "::mlir::ModuleOp"> {
  let summary = "Move each operator into a tile module of its own";
  let description = [{
    The per-operator alternative to `sodap-tosa-outline-kernel`. Each
    `tosa.matmul`, in program order, becomes the only op of
    `module @matmul_<i> attributes {sodap.tile = i, sodap.tile_dir =
    "tiles/matmul_<i>"}` holding `@matmul_<i>`, and is replaced by a
    `dataflow.launch` of it. Everything else (reshapes, transposes,
    elementwise ops) stays in its function, the host, run by the CPU.
  }];
  let dependentDialects = [
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::tosa::TosaDialect"
  ];
}

def JoinMultipleTiles : Pass<"sodap-join-multiple-tiles", "::mlir::ModuleOp"> {
  let summary = "One tile per operator, its shapes as arguments";
  let description = [{
    After `sodap-tosa-outline-operators`: the tile modules with the same
    `sodap.operator` become one tile, `module @forward_kernel`, whose function
    takes every dimension as an `index` argument and its tensors with dynamic
    shapes. Each `dataflow.launch` of the old tiles launches it, with its
    dimensions as constants. One accelerator, several invocations.

    Each argument and result carries `sodap.descriptor_field`, its field in
    the descriptor the accelerator reads at every launch
    (`ReusableGemmABI.h`), and the module `sodap.invocations`, the dimensions
    of every launch, for the tile sizes' cost model.

    Only `matmul` today: `(A: MxK, B: KxN, M, K, N) -> C: MxN`.
  }];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::sodap::dataflow::DataflowDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::tosa::TosaDialect"
  ];
}

#endif // SODAP_TOSA_PASSES
