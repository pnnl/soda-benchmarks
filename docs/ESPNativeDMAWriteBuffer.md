# Preventing an early-write deadlock on native ESP DMA

GESUMMV passed the original component test but timed out in full-system ESP
simulation. The generated host object and Bambu RTL were unchanged. A focused
VCD identifies a circular wait between the accelerator's read and write paths.

The native socket in the validation SoC services one DMA transaction at a time.
After accepting a write command, it waits for that transaction's data before
servicing another read. The generated accelerator can publish its write
descriptor before the computation has produced the output data. Producing that
data still requires input reads.

## Observed sequence

Cycles are sampled immediately before the accelerator clock edge, relative to
the start of the invocation:

| Cycle | Observed event |
|---:|---|
| 1 | First scalar read is granted. A ten-beat write command is already valid; no write data is available. |
| 25 | The first scalar read returns its 64-bit beat. |
| 27 | ESP grants the pending write command. The adapter is still consuming the padding half of the first read. |
| 28 | The second scalar read becomes valid, but its grant stays low. |
| 32 onward | ESP is in `request_data`, with `pending_dma_write=1`; the accelerator has no write data because it still needs the blocked reads. |

The original run reaches the one-million-cycle hardware timeout. The generated
LLVM contains the output-ready `peek` before the write call, but the synthesized
write-control port nevertheless becomes valid at cycle 1. Source statement
order alone therefore does not establish the required reservation timing at
the socket boundary for this design.

## Adapter correction

The 32-to-64-bit adapter now accepts the core's write descriptor locally and
collects a **complete write chunk** before offering its command to ESP. Reads
remain available while that chunk fills. Once ESP accepts the command, every
beat needed to finish it is already buffered, so draining it cannot depend on
a new read transaction.

The buffer defaults to 16 64-bit beats. Larger core requests are split into
consecutive chunks, advancing the address and reducing the remaining length.
The final short chunk is handled explicitly. This is bounded buffering; it
does not assume that all accelerator bursts fit in a fixed-size array. Core
completion is held until the final buffered write drains.

The existing restriction on writes remains: they must cover aligned complete
64-bit beats. The campaign's padded layout satisfies that requirement. No
generated RTL, LLVM, host object, page-table code or ESP socket RTL was edited
for this correction.

## Regression evidence

The component memory model now has a native-ESP mode that serializes reads
and writes. With no artificial initial stall, this model reproduces the
original GESUMMV deadlock with the old adapter and passes with the buffered
adapter. The former independent-channel model allowed the reads needed to
finish an early write. Its additional 50,000-cycle write-command stall also
allowed output computation to advance before ESP could become occupied.

All seven kernels pass with the new adapter both with and without that long
stall. GEMM and GESUMMV additionally pass with one- and two-beat buffers,
forcing request splitting and short final chunks: **18 passing cases**, each
with two invocations, numerical validation, input/padding preservation, and
backpressure. GESUMMV also passes the full-system RISC-V/ESP simulation at the
unchanged numerical threshold.

The first campaign retains the original failure under
`gesummv/system-esp-initial/`, the VCD and sampled events under
`gesummv/system-trace/`, and the before/after component runs under
`gesummv/adapter-serial-before/` and `gesummv/adapter-serial-after/`.
The complete corrected image set and performance results are described in
[ESPVC707Campaign.md](ESPVC707Campaign.md).

This is separate from the BICG/MVT compiler failure documented in
[ESPBambuMultiProcessFailure.md](ESPBambuMultiProcessFailure.md): those kernels
currently fail before RTL generation.
