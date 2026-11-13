# BICG and MVT: Bambu interface-dependency failure

The MINI BICG and MVT kernels produce LLVM and matching RISC-V host objects,
but the Bambu shipped in `agostini01/soda-trixie:v19.03` fails before RTL
generation. Both dense and DMA64-padded layouts reproduce the exception:

```text
absl::container_internal::raw_hash_map<>::at
```

GDB localizes it to `BuildVirtualPhi::InternalExec()`. This is not an ESP
simulation failure or a failure to fit the FPGA.

## Evidence from the live compiler

| Kernel | Function being analyzed | Owner of the referenced call | Missing basic block |
|---|---|---|---:|
| BICG | `forward_kernel_node0` | `forward_kernel_node3` | 26 |
| MVT | `forward_kernel_node0` | `forward_kernel_node4` | 32 |

In both cases, the referenced operation is a synthetic write to the external
DMA request interface:

```text
call void @P0_bambu_artificial_ParmMgr_WriteChannel(
    i1 1, i32 64, i64 <request>, i1 0)
```

The analysis is traversing a virtual-memory SSA use belonging to another
function, then looking up that statement's basic block in the current
function's control-flow graph. That lookup cannot succeed for the observed
blocks. The debugger confirms both the pass's function name and the referenced
statement's owning function; this is more specific than the exception text.

These two kernels have multiple load/compute/store groups using the same
external DMA channels. For example, BICG has load nodes 0 and 3 feeding two
different computations; both use the top's DMA request/response arguments.
The generated top makes that sharing explicit (unrelated arguments omitted):

```llvm
; %0 is the read-request stream; %1 is the read-data stream.
call void @forward_kernel_node0(ptr %1, ..., ptr %0, ...)
call void @forward_kernel_node3(ptr %1, ..., ptr %0, ...)
```

MVT has the analogous sharing between load nodes 0 and 4. Its two store nodes
also share the top's write-request and write-data streams.

## Likely source and proposed correction

In the locally available Bambu source, `InterfaceInfer.cpp` tracks the previous
access with `channel_read_vdefs` and `channel_write_vdefs`, keyed only by
interface name. `serializeInterfaceAccess` and
`preserveBlockingInterfaceAccess` add a virtual dependency on the saved
definition. When the collection includes accesses in different dataflow
functions, this permits a dependency to cross function boundaries.

The proposed correction is to track access history per **owning function and
interface**, for reads and writes. The original memory dependencies within
each function must be retained. Add a diagnostic/assertion preventing foreign
function definitions or uses from entering that function's virtual SSA graph.
Skipping a failed basic-block lookup in `BuildVirtualPhi` would hide the
inconsistent dependency graph rather than repair its construction.

This source-level diagnosis must be checked against the exact source revision
used to build the container: its binary reports `Revision unknown-unknown`.
The installed binary fingerprint is retained in the local reproducer. No Bambu
binary or generated RTL has been patched for the seven-kernel FPGA campaign.

## What remains after the compiler fix

Removing an invalid cross-function dependency is not sufficient evidence that
the synthesized design handles a shared DMA port correctly. The resulting
hardware still needs to arbitrate request/data streams and deliver read data
to the matching consumer. The intended ordering between the two LCS groups
must survive lowering.

After rebuilding Bambu, test both kernels through HLS, the backpressured DMA
adapter test, and full-system ESP simulation. Verify all outputs, repeated
invocations, request/response ownership and stalled consumers. If the shared
interface is not supported, alternatives are one explicit load/store process
that services both computations, or separate offloads sequenced by the host.

## Reproducer

The local campaign keeps original dense and padded LLVM/XML, their checksums,
Bambu logs, the binary fingerprint, and debugger transcripts in
`bambu-reproducer/`. The debugger transcripts were captured with the padded
inputs. From either
kernel's reproducer directory in the SODA container:

```sh
source /opt/panda/settings.sh
bambu accelerator.ll --architecture-xml=architecture.xml \
  --top-fname=forward_kernel --compiler=I386_CLANG19 \
  --generate-interface=INFER --generate-components=separate \
  --device=xc7vx485t-2ffg1761 --clock-period=20 --no-clean -v3
```

These inputs are copies; the generated source artifacts were preserved.
