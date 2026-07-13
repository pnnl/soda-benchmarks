This requires compilation of the mlir-opt plugin tool.

# Single/change-location counter (single active counter)

This recipe wires the `soda-instr-change-location` pass and the
`sodaInstrChangeLocation` hardware IP (see `../changeLocation` for the
canonical version of this recipe and a longer description).

Only one counter is active at a time:

- Location `0` counts cycles starting from reset.
- Entering a nested loop switches the active location with a single
  `@sodaInstrChangeLocation(<loc>)` call — no separate start/stop pair is
  needed, unlike `../hwCounters`'s concurrent counters.
- A finalize call (report-sentinel location, all bits set) is emitted once
  before the function returns, so the hardware IP prints every location's
  final count a single time instead of flooding the simulation log on every
  loop iteration/stop.

Previously this example's `IPs/` and `transform.mlir` were byte-identical to
`../hwCounters` (a concurrent many-counters design); they now use the
distinct single-active-counter IP described above.
