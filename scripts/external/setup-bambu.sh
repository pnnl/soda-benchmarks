#!/bin/bash

# This script will place binaries at the same location regardless of the current working directory.

set -e -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR=$SCRIPT_DIR/../..

PROJ_URL="${PROJ_URL:-https://github.com/ferrandi/PandA-bambu.git}"

# Update and install dependencies
if ! dpkg -l | grep -q libclang-16-dev; then
    sudo apt update && sudo apt install -y gcc-11-plugin-dev g++-11-multilib autoconf \
        autoconf-archive automake libtool libbdd-dev libclang-16-dev \
        libboost-all-dev libmpc-dev libmpfr-dev libxml2-dev liblzma-dev libmpfi-dev \
        zlib1g-dev libicu-dev bison doxygen flex graphviz iverilog verilator make \
        libsuitesparse-dev libglpk-dev
else
    echo "libclang-16-dev and other dependencies are already installed."
fi

# Clone PandA-bambu if not already present
if [ ! -d "$BASE_DIR/external/bambu" ]; then
    git clone "$PROJ_URL" "$BASE_DIR/external/bambu"
    cd "$BASE_DIR/external/bambu"
    git submodule update --init --recursive
    cd "$BASE_DIR"
fi

# Build bambu
cd "$BASE_DIR/external/bambu"
make -f Makefile.init 

mkdir -p "$BASE_DIR/builds/bambu/build"
cd "$BASE_DIR/builds/bambu/build"
"$BASE_DIR/external/bambu/configure" --prefix="$BASE_DIR/builds/bambu/install" \
    --with-gcc8=/opt/gcc-8/bin/gcc \
    --with-clang16=/opt/clang-16/bin/clang \
    --enable-verilator --enable-glpk --enable-opt \
    --enable-flopoco --with-opt-level=fast --enable-release

make -j8 && make install
