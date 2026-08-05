#!/bin/bash

# This script will place binaries at the same location regardless of the current working directory.

set -e -o pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BASE_DIR=$SCRIPT_DIR/../..

PROJ_URL="${PROJ_URL:-https://github.com/ferrandi/PandA-bambu.git}"

SRC_DIR="${SRC_DIR:-$BASE_DIR/external/bambu}"
BUILD_DIR="${BUILD_DIR:-$BASE_DIR/builds/bambu/build}"
INSTALL_DIR="${INSTALL_DIR:-$BASE_DIR/builds/bambu/install}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
# BUILD_TYPE="${BUILD_TYPE:-Debug}"
JOBS="${JOBS:-$(nproc)}"


# Clone PandA-bambu if not already present
# Fetch only the latest commit of main and dev/panda
if [ ! -d "$BASE_DIR/external/bambu" ]; then
    git clone --depth 1 --branch main "$PROJ_URL" "$BASE_DIR/external/bambu"

    pushd "$BASE_DIR/external/bambu"

    # Create a local branch directly from the remote
    git fetch --depth 1 origin dev/panda:dev/panda
    git switch dev/panda

    git submodule update --init --recursive --depth 1

    popd
fi

# As root:
# source /workspaces/soda/soda-benchmarks/external/bambu/.devcontainer/library-scripts/common-debian.sh 
# As user:
# source /workspaces/soda/soda-benchmarks/external/bambu/.devcontainer/library-scripts/compiler-download.sh /tmp/bambu-compilers clang-13,clang-16,clang-19 /tmp/bambu-compilers-bak


# Compiler for bambu plugins
if ! command -v clang-19 &> /dev/null; then
    echo "clang-19 not found, installing it..."
    # must use sudo
    # source $BASE_DIR/external/bambu/.devcontainer/library-scripts/compiler-download.sh /tmp/bambu-compilers clang-19 /tmp/bambu-compilers-bak
    chmod +x $BASE_DIR/external/bambu/.devcontainer/library-scripts/compiler-download.sh
    # sudo source $BASE_DIR/external/bambu/.devcontainer/library-scripts/compiler-download.sh /tmp/bambu-compilers clang-19 /tmp/bambu-compilers-bak
    sudo bash -c "$BASE_DIR/external/bambu/.devcontainer/library-scripts/compiler-download.sh /tmp/bambu-compilers clang-19 /tmp/bambu-compilers-bak"
    pushd /usr/bin
    # BUG: compilers for plugings listed in PANDA_LIBBAMBU_COMPILER must be installed in /usr/bin
    # sudo ln -sf /tmp/bambu-compilers/clang-19/clang+llvm-19.1.7-x86_64-linux-gnu-compat/bin/clang-19 
    # sudo ln -sf /tmp/bambu-compilers/clang-19/clang+llvm-19.1.7-x86_64-linux-gnu-compat/bin/clang++-19
    # sudo ln -sf /tmp/bambu-compilers/clang-19/clang+llvm-19.1.7-x86_64-linux-gnu-compat/bin/opt-19
    sudo ln -sf /tmp/bambu-compilers/clang-19/bin/clang-19 
    sudo ln -sf /tmp/bambu-compilers/clang-19/bin/clang++-19
    sudo ln -sf /tmp/bambu-compilers/clang-19/bin/opt-19
    popd
    echo "clang-19 installed."
fi

# Update and install dependencies
if ! dpkg -s  gcc-13 &> /dev/null; then

    # These are already installed in soda docker image
    # sudo apt update && sudo apt install -y cmake ninja-build gcc-11-plugin-dev g++-11-multilib \
    #     libbdd-dev libclang-16-dev \
    #     libboost-all-dev libmpc-dev libmpfr-dev libxml2-dev liblzma-dev libmpfi-dev \
    #     zlib1g-dev libicu-dev bison doxygen flex graphviz iverilog verilator make \
    #     libsuitesparse-dev libglpk-dev wget xz-utils bc
    # Compile time
    sudo apt update && sudo apt install -y libboost-all-dev bison flex
    # Runtime
    sudo apt update && sudo apt install -y bc

    # libboost-all-dev bison flex

    cat <<'EOF' | sudo tee /etc/apt/sources.list.d/trixie.sources > /dev/null
Types: deb
URIs: http://deb.debian.org/debian
Suites: trixie
Components: main
EOF

    cat <<'EOF' | sudo tee /etc/apt/preferences.d/99debian-trixie > /dev/null
Package: *
Pin: release a=testing
Pin-Priority: 100
EOF

    sudo apt update && sudo apt install -t trixie gcc-13 g++-13 gcc-13-plugin-dev g++-13-multilib -y

else
    echo "gcc-13 and other dependencies are already installed."
fi


# TODO: As cmake user:
echo $PATH
source /tmp/bambu-compilers/settings.sh
echo $PATH

# Build bambu using CMake and ninja
# Bambu must use the gcc-13 compiler to build
# BUG: compilers for plugings listed in PANDA_LIBBAMBU_COMPILER must be installed in /usr/bin
mkdir -p "$BUILD_DIR" "$INSTALL_DIR"
cmake -S "$SRC_DIR" -B "$BUILD_DIR" -G Ninja \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DPANDA_LIBBAMBU_COMPILER=I386_CLANG19 \
    -DCMAKE_C_COMPILER=gcc-13 \
    -DCMAKE_CXX_COMPILER=g++-13 \
    -DPANDA_ENABLE_RELEASE=ON \
    -DPANDA_ENABLE_ASSERTS=OFF \
    -DPANDA_ENABLE_WERROR=ON
    

ninja -C "$BUILD_DIR"
ninja -C "$BUILD_DIR" install

cat <<EOF

bambu installed in $INSTALL_DIR

Before running bambu for synthesis or simulation, source the installed
environment script (it sets BAMBU_HLS and BAMBU_HLS_BACKEND_PATH, required by
the backend flows):

    source $INSTALL_DIR/settings.sh

To rebuild after editing sources under $SRC_DIR:

    cmake --build $BUILD_DIR -j$JOBS && cmake --install $BUILD_DIR
EOF

