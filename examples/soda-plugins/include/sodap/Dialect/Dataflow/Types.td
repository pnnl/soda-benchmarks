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

    The optional interface says how the stream is connected:

    - `fifo` (the default, not printed): a channel between two stages.
    - `axis`: an AXI4-Stream port of the design, read or written by something
      outside it, such as the DMA engine of an accelerator platform. Only an
      argument can be one; a `dataflow.stream` op declares a channel inside
      the design and is always a fifo. The depth is kept because the stage
      the port is passed to still sees it as a FIFO.

    Example:

    ```mlir
    !dataflow.stream<f32, 1024>
    !dataflow.stream<i64, 8, axis>
    ```
  }];
  let mnemonic = "stream";

  let parameters = (ins "mlir::Type":$elementType, "unsigned":$depth,
      DefaultValuedParameter<"::mlir::sodap::dataflow::StreamInterface",
                             "::mlir::sodap::dataflow::StreamInterface::FIFO",
                             "how the stream is connected">:$interface);
  let assemblyFormat =
      "`<` qualified($elementType) `,` $depth (`,` $interface^)? `>`";

  let skipDefaultBuilders = 1;
  let builders = [
    TypeBuilder<(ins "mlir::Type":$elementType, "unsigned":$depth,
        CArg<"::mlir::sodap::dataflow::StreamInterface",
             "::mlir::sodap::dataflow::StreamInterface::FIFO">:$interface), [{
      return $_get($_ctxt, elementType, depth, interface);
    }]>
  ];

  let extraClassDeclaration = [{
    /// Whether this stream is an AXI4-Stream port of the design.
    bool isAxis() const {
      return getInterface() == ::mlir::sodap::dataflow::StreamInterface::AXIS;
    }

    /// The same stream connected through `interface`.
    StreamType withInterface(
        ::mlir::sodap::dataflow::StreamInterface interface) const {
      return StreamType::get(getContext(), getElementType(), getDepth(),
                             interface);
    }
  }];
}

#endif // SODAP_DIALECT_DATAFLOW_TYPES_TD
