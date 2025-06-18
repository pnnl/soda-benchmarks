# SODA Plugins

Provide a intree way of compiling mlir-opt plugins to be used with soda-benchmarks.


## Project Structure

- `soda-plugins/` - Contains the source code for the plugins library.
- `lib/` - Contains the implementation of passes.
- `include/` - Contains the declarations of passes.


## Building

We prefer the component build approach, which allows you to build the plugins as a separate library that can be linked against the `soda-opt` or `mlir-opt` tool.

To build the plugins, run the following commands:

```sh
mkdir build && cd build

cmake -G Ninja .. \
  -DMLIR_DIR=/opt/llvm-project/lib/cmake/mlir \
  -DLLVM_EXTERNAL_LIT=/workspaces/soda/builds/llvm-project/build/bin/llvm-lit

cmake --build . --target SODAPlugin
```


## Testing

If tests are enabled and llvm-lit is available, you can run the tests with:

```sh
cmake --build . --target check-sodap
```


##  Running the Plugins

To run the plugins, you can use the `mlir-opt` tool with the `--load-pass-plugin` option to load the pass plugin library or the `--load-dialect-plugin` option to load the dialect plugin libray. We compile both in a single file which is available in the `build` directory under `lib/SODAPlugin.so`.

One of the included passes in the plugin is `soda-view-op-graph`, which generates graphviz output of the operations in the MLIR file. To use this pass, you can run the following command:

```bash
mlir-opt \
  -allow-unregistered-dialect \
  -mlir-elide-elementsattrs-if-larger=2  \
  --load-pass-plugin=/workspaces/soda-benchmarks/examples/soda-plugins/build/lib/SODAPlugin.so \
  --pass-pipeline="builtin.module(soda-view-op-graph)" \
  /workspaces/soda-benchmarks/examples/soda-plugins/test/sodap/print-op-graph.mlir
```


We also include infrastrucure to add extensions to the transform dialect. In this example we added `transform.my.change_call_target`.

```bash
mlir-opt \
  --load-pass-plugin=/workspaces/soda-benchmarks/examples/soda-plugins/build/lib/SODAPlugin.so \
  --load-dialect-plugin=/workspaces/soda-benchmarks/examples/soda-plugins/build/lib/SODAPlugin.so \
  --transform-interpreter \
  /workspaces/soda-benchmarks/examples/soda-plugins/test/sodap/my-extension.mlir 
```