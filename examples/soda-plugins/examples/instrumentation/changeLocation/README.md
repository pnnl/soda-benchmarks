This requires compilation of the mlir-opt plugin tool.

# Single/change-location counter

This example uses the `soda-instr-change-location` pass and the
`sodaInstrChangeLocation` hardware IP, instead of the concurrent-counters
flow in `../hwCounters`.

Only one counter is ever active:

- Location `0` counts cycles starting from reset (the outermost/global
  scope), with no explicit call needed.
- Entering a nested loop emits a single `@sodaInstrChangeLocation(<loc>)`
  call that switches the active location. There is no separate stop call:
  changing location implicitly "stops" the previous location and "starts"
  the new one.
- Right before the function returns, the compiler emits one extra finalize
  call with the report-sentinel location (all bits set). The hardware IP
  recognizes this sentinel and prints every location's final count once via
  `$display`, avoiding the per-iteration log flood that a naive counter
  design would produce.

This addresses two items from the instrumentation TODO list:
- "Single Counter - Tracking many locations but only one runs at a time"
- "Mechanism to not print at every loop during simulation, but only when
  simulation is done"
