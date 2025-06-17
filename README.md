# SODA-BENCHMARKS

This project contains scripts and code to benchmark SODA tools in different scenarios.

## How to Use?

We depend on Docker and a pre-built [Docker image](.devcontainer/Dockerfile#1)
which contains the binaries to run the benchmarks. We recommend using VS Code
Dev Containers to orchestrate the container setup and provide the correct
environment for running the examples and benchmarks.

To use Dev Containers, follow these steps:

1. Install Visual Studio Code: If you haven't already, download and install Visual Studio Code from [here](https://code.visualstudio.com/download).
2. Install the Container Tools extension: Open Visual Studio Code and install the "Container Tools" extension from the Extensions view.
3. Open the folder in a Dev Container: Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on macOS) and select: `Dev Containers: Reopen in Container`.
4. Navigate to the relevant folder and check its README.md for instructions on executing the examples or benchmarks.


## What Next?

Start by navigating to the [examples](examples/) folder. Each example
includes a README.md with instructions on how to run it and details about the
generated artifacts.

For a step-by-step guide, see the [tutorials](tutorials/) folder. It
includes Jupyter notebooks that demonstrate the full workflow, from PyTorch
models to Verilog and GDS generation using SODA and open-source tools.

If you want to create accelerator cores for your own models, check out the
[models](models/) folder. It contains Python scripts to download or
implement machine learning models, and scripts to transform these models or
parts of them into Verilog.


## Project Structure

```
├── docs         # Documentation
├── examples     # Simple examples demonstrating the end-to-end Python to Verilog/GDS flow
├── LICENSE
├── models       # Python scripts to download or implement ML models, and scripts to transform models into Verilog
├── README.md
├── scripts      # Bash and Makefile scripts used in other folders
├── tests        # All tests (if possible)
└── tutorials    # Guided tutorials using jupyter notebooks
```


## License

This project is made available under the Apache License 2.0 with LLVM
Exceptions. See the [LICENSE](LICENSE) file for more details.


### Software from third parties included in the soda-benchmarks project

The soda-benchmarks project contains third party software which is under different
license terms. All such code will be identified clearly using at least one of
two mechanisms:

1) It will be in a separate directory tree with its own `LICENSE.txt` or
   `LICENSE` file at the top containing the specific license and restrictions
   which apply to that software, or
2) It will contain specific license and restriction terms at the top of every
   file. 


# Disclaimer Notice

This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or any
information, apparatus, product, software, or process disclosed, or represents
that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

<div align=center>
<pre style="align-text:center">
PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
</pre>
</div>
