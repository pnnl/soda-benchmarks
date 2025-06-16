# SODA Plugins

Provide a intree way of compiling mlir-opt plugins to be used with soda-benchmarks.


# Project Structure

- `soda-plugins/` - Contains the source code for the plugins library.
- `lib/` - Contains the implementation of passes.
- `include/` - Contains the declarations of passes.


# TODO

- [ ] Make sure library names dont mangle with soda-opt libraries, which also use `soda`