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

cmake --build . --target check-sodap
```


# TODO

- [ ] Make sure library names dont mangle with soda-opt libraries, which also use `soda`