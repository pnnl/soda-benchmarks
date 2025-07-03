# AutoEncoder for EELS data

Model architecture extracted from the following repository:

- https://github.com/hollejd1/logicalEELS/tree/dev


# License

The model architecture was developed under the following licence:

- https://github.com/hollejd1/logicalEELS/blob/main/LICENSE also included in this folder as [LICENSE](./LICENSE)


# How to run?

Run `make` in this folder using the docker image.
This will trigger the generation of `output/bambu/baseline/forward_kernel.v` which is the generated accelerator for the EELS model without any optimizations.
