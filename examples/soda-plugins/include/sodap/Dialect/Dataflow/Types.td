//===- Types.td - Dataflow dialect types -------------------*- tablegen -*-===//
//
// Copyright 2020-2021 The ScaleHLS Authors
// Copyright 2024 Suhail Basalama
// Copyright 2026, Battelle Memorial Institute
//
//===----------------------------------------------------------------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Portions of this file were modified by Suhail Basalama in 2024 under the 
// MIT License. The full MIT text can be found in the root NOTICE file.
//
//===----------------------------------------------------------------------===//

#ifndef SODAP_DIALECT_DATAFLOW_TYPES_TD
#define SODAP_DIALECT_DATAFLOW_TYPES_TD

def StreamType : DataflowType<"Stream"> {
  let summary = "A stream type";
  let description = [{
    Represents a stream of values that can be transferred between HLS stages,
    i.e. the MLIR analogue of an `hls::stream`. The depth parameter is the
    channel's FIFO depth.

    Example:

    ```mlir
    !dataflow.stream<f32, 1024>
    ```
  }];
  let mnemonic = "stream";

  let parameters = (ins "mlir::Type":$elementType, "unsigned":$depth);
  let assemblyFormat = "`<` qualified($elementType) `,` $depth `>`";
}

#endif // SODAP_DIALECT_DATAFLOW_TYPES_TD
