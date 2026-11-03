# Transform schedule libraries

Schedules an experiment can be scaffolded with, for kernels whose schedule was
produced somewhere other than `sb-cli` itself.

```
schedules/<library>/<DATASET>/
  <kernel>.mlir            # the schedule on its own
  <kernel>+<recipe>.mlir   # the same schedule, instrumented
```

`sb-cli init --default_ts <library>` copies the file matching the experiment's
kernel, dataset and `--instrumentation`. The sentinel `simple` means "whatever
sb-cli would have written anyway" — the recipe's own `transform.mlir`, or the
no-op template — and is always available.

Only the schedule comes from here. The IP files an instrumented schedule needs
still come from the recipe (see [`../recipes/README.md`](../recipes/README.md)),
because a merged schedule calls exactly the same instrumentation passes.

## `alo_soda`

DSE-derived schedules for the PolyBench MINI dataset: 13 kernels x (bare + 4
instrumentation recipes) = 65 files. Requesting any other dataset, an unlisted
kernel, or the `esp` / `vector-dot` recipes fails with a message naming what is
available.

Filenames use the `benches` catalog kernel names (`twomm`, `threemm`), not the
PolyBench spellings (`2mm`, `3mm`) the sources use; the aliasing is resolved once,
when the files are generated.

`<kernel>+change-location.mlir` and `<kernel>+change-location-exposed-ports.mlir`
are byte-identical — those two recipes apply the same pass and differ only in
their IPs. Keeping both files means lookup stays a plain stem computation with no
alias layer.

### Why the instrumentation is merged in, and why it goes last

The SODA instrumentation passes walk `scf::ForOp` only (see
`examples/soda-plugins/lib/sodap/InstrPasses.cpp`). The loops they annotate are
the `scf.for` tile loops `transform.structured.tile_using_for` produces, near the
top of each schedule. Those survive to the end: `convert-linalg-to-affine-loops`
rewrites only the linalg ops, and the unroll steps match `affine.for`. So the
pass is appended just before `@__transform_main`'s `transform.yield`.

It has to go last rather than immediately after the tiling. The pass declares a
`func.func private` for the IP call, so any `match ops{["func.func"]}` running
afterwards binds two funcs instead of one, and `apply_registered_pass` then trips
`"results already set"` in `TransformResults::set` and aborts `soda-opt`.

Because appending the pass to the *wrong* place can also parse and run while
instrumenting nothing at all, the merged files are generated once and checked in
rather than assembled at scaffold time.

### Source

Generated from `examples/run-many-experiments/polybench_mini_TS/`, which is kept
for provenance. These files are the source of truth at run time.
