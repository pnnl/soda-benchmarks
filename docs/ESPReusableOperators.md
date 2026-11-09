# Reusable GEMM invocation mechanism

Status: the opt-in mechanism is implemented in this review branch, based on
Tommaso's `feature/esp-lcs` at `33266a9`.
The opt-in path generates multiple host calls and one parameterized GEMM core.
The broader contract below also records requirements for subsequent optimization.
See [the validation instructions](../examples/esp-validation/reusable-gemm/README.md)
for the implemented subset, commands, and evidence.

Multiple GEMM operations in an algorithm must be able to invoke one physical
accelerator with different dimensions and operand locations. Parameter
isolation defines the computation and its invocation interface. Implementation
tuning and assignment to physical devices are separate compiler decisions.
The first execution policy assigns all compatible GEMMs to one instance and
executes them sequentially in dependency order.

## Objects represented by the compiler

| Object | Meaning | Example |
| --- | --- | --- |
| Operator contract | Computation, parameter meanings, numerical behavior, external layout, and valid invocation domain | `gemm_f32_v1` |
| Invocation | One occurrence of that operation, with its operands and parameters | `gemm_0`, `gemm_1` |
| Implementation | Generated hardware satisfying an operator contract, with its own schedule and resources | `gemm_impl_0` |
| Instance | One physical device containing an implementation | `gemm_device_0` |
| Binding | Assignment of an invocation to an instance | Both calls use `gemm_device_0` |

Ten invocations do not imply ten implementations or ten devices. Two physical
instances can contain identical implementations. Alternatively, two differently
tuned implementations can satisfy the same operator contract.

## Parameter isolation and the GEMM contract

The initial family computes `C = alpha * (A * B) + beta * C`, with float32
operands, positive dimensions, and row-major views. A is `M x K`, B is `K x N`,
and C is `M x N`. Transposition and batching require an explicit extension or
an explicit conversion before this interface; they are not inferred from shape
alone. C may be updated in place, but its storage must not overlap A or B.

The logical invocation descriptor contains:

| Field | Meaning |
| --- | --- |
| `A`, `B`, `C` | Views in an allocated shared arena; each has an arena identity, a byte offset, and a row stride in bytes |
| `M`, `N`, `K` | Matrix dimensions, independent of hardware tile dimensions |
| `alpha`, `beta` | Float32 coefficients with the behavior defined by the operator contract |

The instance handle belongs to the binding, not to the mathematical operator.
Tile sizes, unroll factors, bank counts, and buffer counts are implementation
settings and never appear in this descriptor.

For `beta = 0`, the contract overwrites C without reading its previous
contents. Plain matrix multiplication can use `alpha = 1, beta = 0`.
An invocation reads A and B, reads C when beta is nonzero, and writes C; these
effects must remain visible to dependency analysis and buffer lifetime analysis.
Lowering an accumulating or scaled expression must check its actual semantics;
in particular, skipping a floating-point read and multiply by zero is not
automatically equivalent for all inputs. Unsupported coefficient fusion stays
outside the offload. The numerical profile must fix accumulation precision,
permitted reduction reordering and FMA contraction, and exceptional-value
behavior before implementations are compared. Tuning cannot change that
profile or relax validation tolerances.

Parameter isolation runs while the structured operation and operand dimensions
are available, before hardware tiling. The current pipeline first outlines a
container function while retaining its individual structured matmuls. It derives
dimension relationships from the operation, gathers operand views and semantic
scalars, and produces an invocation of the common operator. Logical views are
assigned shared-arena offsets by subsequent memory planning. Recognition is
initially limited to supported matrix-multiplication patterns; arbitrary code
is not automatically generalized.

Known shapes remain constants at call sites and are also retained as workload
information for tuning. Runtime-dependent frontend shapes will require an
extension to isolation; the runtime and hardware already take dimensions as
descriptor values with checks in the runtime. Equal numeric extents in one example do not prove that two dimensions
must remain equal in the reusable operator.

## Orthogonality to optimization

Every implementation of one contract must accept the same declared invocation
domain and preserve the same descriptor meanings and external storage format.
The contract includes dimension and address limits, layout and alignment rules,
aliasing restrictions, and numerical semantics. These limits are established
from the target interface and desired workload before schedule exploration.

Changing a tile size must not require a new parameter list, new host packing,
or a new matrix-divisibility restriction. Partial tiles must remain supported
through correct bounds handling or an internal fallback. An optimized path
with stronger assumptions needs a correct path for the rest of the domain.
An implementation that changes the public domain is a different contract and
cannot silently replace an existing implementation.

The optimizer can use all invocation shapes, frequencies, and dependencies to
choose tiling, banking, unrolling, and buffering under resource constraints.
Its output implements the parameterized computation; it must not substitute
the shape of one call into the reusable hardware entry point. Parameter
isolation has no dependency on the selected transformation schedule.

The initial ESP storage format uses the existing DMA64 convention: one
float32 value in the low 32 bits of each eight-byte slot, with a zero padding
word. Offsets and row strides in the descriptor are bytes; hardware DMA request
fields use 32-bit words. Generated configuration and address lowering perform
this conversion with range and alignment checks; the existing DMA adapter then
converts word requests to ESP beats. This format is identical across candidates;
internal tile layouts and memory banking remain private to each implementation.
A later dense format needs an explicit storage-contract change or adapter.

## Binding and repeated execution

The following is conceptual notation, not registered MLIR operations or an
existing runtime API. An implementation can initially represent these objects
using functions, calls, and attributes rather than introducing a new dialect.

```text
operator       gemm_f32_v1
implementation gemm_impl_0 implements gemm_f32_v1 using schedule_0
instance       gemm_device_0 contains gemm_impl_0

invocation g0: H = gemm(X, W1)    # X: 7x513, W1: 513x65, H: 7x65
invocation g1: Y = gemm(H, W2)    # W2: 65x19, Y: 7x19
dependency: g0 produces the H consumed by g1

binding g0 -> gemm_device_0
binding g1 -> gemm_device_0
```

The two operations use different dimensions and weights, but the same device
and synthesized hardware. H has one shared-memory location and a layout that
both invocations accept. Its lifetime extends through completion of g1. Both
descriptors use `alpha = 1, beta = 0`.

The generated host follows this lifecycle:

```text
device = open_instance(gemm_device_0)
arena = allocate_shared_arena(X, W1, H, W2, Y)
attach_arena(device, arena)              # establish the ESP DMA mapping
copy_inputs_to_arena(X, W1, W2)

configure(device, g0_descriptor)         # M=7, N=65, K=513; C points to H
start(device)
wait_complete(device)

configure(device, g1_descriptor)         # M=7, N=19, K=65; A points to H
start(device)
wait_complete(device)

copy_output_from_arena(Y)
detach_arena(device, arena)
free_shared_arena(arena)
close_instance(device)
```

Each instance has an explicit handle and at most one invocation in flight in
the first implementation. Its state progresses from idle to configured to
running to completed and back to idle after acknowledgment. Configuration is
latched for the entire invocation. No register update or new launch is allowed
while it is running, and each call reinitializes accumulators and task state.

Completion must include the DMA visibility required by the next dependent
invocation; issuing the final store request alone is insufficient. The runtime
must apply the selected ESP coherence and fence rules when the CPU or another
device accesses the arena. A failed invocation is reported and prevents its
dependent calls from launching. Buffers and mappings stay live until their
last use completes. H is neither unpacked nor repacked between these calls.

## Checks and generated artifacts

Before binding, the compiler checks operation semantics, types, dimensions,
layouts, and aliasing against the common contract. Before launch, any facts
not proven statically are checked, including allocation bounds and overflow
in address and transfer-size calculations. Unsupported invocations produce a
diagnostic in the first implementation; no implicit CPU fallback is assumed.

One versioned interface definition must generate or validate both the host
configuration encoding and the accelerator configuration decoding. It records
field widths, units, the external layout, and the selected register or command
encoding. The exact physical encoding is an implementation task, not part of
parameter discovery. Host and hardware artifacts carry the same interface
version. An implementation identifier distinguishes schedules; an instance
binding identifies the actual device rather than merely its device type.

The compiler keeps the invocation graph, implementation descriptions, and
bindings as separately inspectable artifacts. Changing a binding or a schedule
does not rerun a different parameter-extraction policy. The initial binding
policy is explicit assignment to one instance; automatic allocation of one,
two, or ten devices is a subsequent optimization.

## Initial implementation

| Component | Implementation |
| --- | --- |
| [Frontend pipeline](../scripts/tosa_to_reusable_gemm.sh) | Bufferize and preserve individual matmuls; generate host and hardware separately |
| [Host isolation](../examples/soda-plugins/lib/sodap/Conversion/LinalgToESPRuntime/ReusableGemmToESP.cpp) | Emit an explicit instance handle, shared arena lifetime, and one attributed call per matmul; no tile options |
| [Hardware generation](../examples/soda-plugins/lib/sodap/Conversion/LinalgToESPRuntime/GenerateReusableGemm.cpp) | Generate runtime loop bounds and DMA addresses, fixed local tiles, and bounds handling for tails |
| [Runtime](../examples/soda-plugins/lib/sodap/ExecutionEngine/ReusableGemmRuntime.cpp) | Validate descriptors, encode configuration, launch and acknowledge sequential invocations on one instance |
| [Shared ABI](../examples/soda-plugins/include/sodap/ExecutionEngine/ReusableGemmABI.h) | Define one versioned header for the compiler and runtime |
| [ESP adapter](../examples/esp-validation/generated-gemm/sqr_bambu_basic_dma64.sv) | Carry the header and matrix traffic through the existing four streams; no extra socket registers |

The descriptor occupies the first 128 bytes of the shared arena. Hardware reads
and latches it at every launch. Fields use the low word of each DMA64 slot;
addresses and strides on the wire are in 32-bit words. Host API offsets and
strides are bytes. Configuration therefore uses the existing DMA transport.

Isolation currently supports a straight-line graph of static, contiguous f32
matmuls with zero-initialized, distinct outputs, including batch-one forms
produced by the Torch frontend. It emits `alpha=1, beta=0`. The hardware/runtime
also accept row strides and general coefficients, exercised by a separate
ten-call runtime/RTL suite. Dynamic frontend shapes, arbitrary coefficient
fusion, transposition, batching beyond one, concurrent launches, and automatic
device assignment remain future work.

The baseline uses sequential DMA and compute with static local tile buffers.
It establishes reusability; pipelining, double buffering, banking and unrolling
remain independent optimization work. No existing fixed-shape LCS schedule is
silently applied to this new path.

The arena limit is one GiB, further limited by platform memory and page-table
capacity. Matrix offsets/strides are aligned to eight bytes and C must not
overlap either input. ESP's current baremetal allocator does not reclaim freed
storage; close releases logical ownership, while repeated opens still consume
that allocator's pool. All calls within this graph share one allocation.

The implementation emits f32 multiply followed by f32 addition in increasing K
order, without fast-math/FMA flags. The recorded finite-input tests use the
same `0.001` absolute tolerance across schedules. Exceptional values, subnormals,
and a full IEEE compatibility profile require additional validation before
claiming that broader numerical domain.

The older fixed-point batch-matmul lowering demonstrates dimension extraction
and register writes, but its device contract is not the proposed float32 GEMM
contract. Existing fixed-shape board validation does not validate this mechanism.

## Acceptance criteria

1. One generated hardware implementation runs several GEMM shapes on the same
   instance, including dimensions smaller than tiles and non-divisible M, N,
   and K. Calls alternate shapes and addresses without a device reset between
   invocations; the chosen suite exercises tails for each tested schedule.
2. The dependent two-GEMM example above produces the reference result with H
   retained in shared memory and both calls using the same instance. A sequence
   of ten invocations also reuses that instance without per-call synthesis.
3. At least two schedules with different tiles run the same descriptor suite
   and external layout. Their operator contract and invocation descriptors are
   unchanged; outputs satisfy the same numerical profile and tolerances.
4. Invalid or out-of-range descriptors are rejected before launch. Adapter
   simulation covers backpressure, repeated launches, DMA completion ordering,
   and padding or sentinel integrity around operands.
5. Validation progresses through compiler checks, a native reference, RTL
   simulation, and FPGA execution. Each hardware candidate records its LLVM,
   RTL, and bitstream identity so shape changes cannot conceal regeneration.
   Latency and resource measurements are reported separately from correctness.
