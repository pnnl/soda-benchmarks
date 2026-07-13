This requires compilation of the mlir-opt plugin tool.

# Vector dot-product HW module

This example wires the `sodap-swap-op-to-hw` pass and the `sodaVectorDot`
hardware IP. Unlike every other example under `instrumentation/` (which
*observes* the kernel via assertions/counters), this recipe *replaces* a
matched software operation with a call to hardware that accelerates it,
per the instrumentation TODO list item:

> Example of swapping an operation for a HW module that accelerates it -
> like a vector/dot product module in a matrix multiplication example

## What the pass does

`sodap-swap-op-to-hw` matches any `linalg.dot` op whose two 1-D memref
inputs have the same statically-known length and whose output is a rank-0
(scalar) memref, and replaces it in place with:

```mlir
call @sodaVectorDot(%a, %b, %len, %out) : (memref<Nxf32>, memref<Nxf32>, index, memref<f32>) -> ()
```

Dot products with dynamically-shaped operands are left untouched, since the
HW module below is modeled as a fixed-size vector engine.

## The `sodaVectorDot` IP

`IPs/sodaVectorDot.v` is modeled on `../IP_integration_example/module1.v`'s
memory-master pattern: it drives the shared RAM bus directly to read both
operand vectors, and reuses the `__builtin_memstore` block (exactly like
`module1`/`module2`) to write the reduced scalar result back to memory.

The accumulate datapath is a simplified integer multiply-accumulate
standing in for a real floating-point MAC core -- swapping it in would be
the natural next step for a production IP; the surrounding
handshake/memory-master/FSM structure would stay the same.

**Validation status:** the pass is covered by a lit test
(`test/sodap/Instrumentation/linalg-dot-swap-to-hw.mlir`, run via
`check-sodap`) and the `transform.mlir` schedule here has been confirmed to
correctly invoke the pass end-to-end via `--transform-interpreter`. The
Verilog IP has been lint-checked with `verilator --lint-only` (it uses the
Bambu-provided `__builtin_memstore` blackbox, exactly like `module1.v`, so
it cannot be resolved as a standalone top-level module without a stub).
It has **not** been run through Bambu HLS/Verilator end-to-end.
