# Developer Guide for Updating External Tools and Using the SODA Benchmarks

This project depends on different external tools. 

```
soda-opt
soda-translate
mlir-opt
mlir-translate
flatbuffer_translate
tf-mlir-translate
tf-opt
torch-mlir-opt - python packages available at /opt/torch-mlir/python_packages/torch_mlir
bambu
openroad
yosys
```

We recommend using the [Docker image](.devcontainer/Dockerfile#1) to ensure that all dependencies are correctly installed and configured. The tested versions of these binaries are added to the `$PATH` environment variable inside the Docker image.

However, you may want to develop new features in `soda-opt` or `bambu` HLS.

To do this, start the VS Code devcontainer, then use the scripts below to clone and build the projects locally. Add their binaries to the beginning of your `$PATH` to ensure that scripts use your local versions instead of those pre-packaged in the Docker image.
When you run the benchmarks in a shell session where your updated `$PATH` includes the locally built `soda-opt` and `bambu` binaries at the front, the benchmarks will automatically use your modified versions instead of the default ones from the Docker image.


# STEP 1: Start the VSCODE DevContainer

Clone this project if you haven't already:

```sh
git clone https://github.com/pnnl/soda-benchmarks
cd soda-benchmarks
```

To use Dev Containers, follow these steps:

1. Install Visual Studio Code: If you haven't already, download and install Visual Studio Code from [here](https://code.visualstudio.com/download).
2. Install the Container Tools extension: Open Visual Studio Code and install the "Container Tools" extension from the Extensions view.
3. Open the folder in a Dev Container: Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and select: `Dev Containers: Reopen in Container`.


# STEP 2: Downloading and Compiling Projects

## Compiling soda-opt

Comming soon...


## Compiling bambu HLS

This will clone and compile bambu HLS.

```sh
# Inside the soda-benchmarks folder
./scripts/external/setup-bambu.sh

# or...
## If you wish to use a non-default repository of bambu:
PROJ_URL=https://github.com/ferrandi/PandA-bambu.git ./scripts/external/setup-bambu.sh
```

After compilation is completed, build binaries will be available in the `builds/bambu/build/bin` folder or in the `builds/bambu/install/bin`.

Now we can add one of these folders to the beginning of our `$PATH` variable:

```sh
# In the devcontainer terminal
# Note that you may need to rerun this command every time you start a new terminal session
export PATH="/workspaces/soda-benchmarks/builds/bambu/install/bin:$PATH"
```

If you want to make further changes to bambu, edit the files under `external/bambu/`. Then, to rebuild bambu, navigate to the `builds/bambu/build` directory and run:

```sh
make -j8 && make install
```


# FAQ

## My changes to soda-opt or bambu are not being picked up, what should I do?

Check that you have added the correct path to your `$PATH` variable. You can verify this by running:

```sh
which soda-opt
which bambu
```