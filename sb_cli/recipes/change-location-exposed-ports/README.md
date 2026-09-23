This requires compilation of the mlir-opt plugin tool.

# Single/change-location counter, with counters exposed on the top level

This recipe is a derived version of `../change-location`. The counting
behaviour is identical -- see that README for how the single-active-counter
scheme and the report sentinel work. The difference is how the counts get
*out* of the core.

In `../change-location` the counts only ever escape as Verilog `$display`
text during simulation, so on real silicon they are unreadable without
attaching the IP to the memory bus (`../vector-dot` shows how much that
costs). Here the IP instead exposes two extra ports on the top-level
interface:

- `target_location` (IN, 32-bit) -- selects which location's counter to read.
- `final_result` (OUT, 32-bit) -- that location's current counter value.

This is the "post-tapeout" flow: an external user drives `target_location`,
runs the core, and samples `final_result`, repeating with a different
location each run to collect every counter -- with no memory-bus interface.

## How the ports reach the top level

Bambu propagates ports marked `is_global="1" is_extern="1"` in
`module_lib.xml` up to the generated top module. Both new ports carry those
attributes. The mechanism is not output-only: the upstream Bambu `pong` and
`breakout` examples declare their `sw` and `btn` *inputs* exactly this way
(`external/bambu/examples/pong/IPs.xml`).

The port names are deliberately *not* added to the `NP_functionality
LIBRARY="..."` attribute. That list names only the ports whose bit-width
Bambu has to specialize (vectors, address/data/size buses -- compare
`../vector-dot/IPs/module_lib.xml`); these two are fixed 32-bit, like `sw`
and `btn`, which are likewise unlisted.

## Why the IP keeps its original name

`transform.mlir` applies the `soda-instr-change-location` pass, which emits
calls to a C function named `sodaInstrChangeLocation`, and Bambu binds the
`module_lib.xml` cell to that name. The module, the C function, and the file
names therefore stay identical to `../change-location`; only
`module_lib.xml` and the Verilog body differ. That keeps this recipe a pure
drop-in with no change to the MLIR plugin.

## Reading the result in simulation

`final_result` is a continuous read of `counters[target_location]`, so it is
always valid and is stable once the core stops switching locations. The IP
also `$display`s the selected location alongside the usual 16-location dump:

```
[HW] sodaInstrChangeLocation: FINAL location 0 count     174239
[HW] sodaInstrChangeLocation: FINAL location 1 count         20
...
[HW] sodaInstrChangeLocation: SELECTED location 1 final_result         20
```

The SELECTED value matching the corresponding FINAL line is what proves the
exposed ports carry the right counter.
