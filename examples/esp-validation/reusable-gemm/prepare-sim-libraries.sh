#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Minimal actual vendor models used by the validated VC707 simulation fixture.
set -euo pipefail
dest=${1:?Usage: prepare-sim-libraries.sh NEW-directory base-modelsim.ini}
base=$(realpath "${2:?Base ESP modelsim.ini required}")
test ! -e "$dest"
mkdir -p "$dest"
dest=$(realpath "$dest")
xilinx=${XILINX_VIVADO:-/opt/xilinx/Vivado/2023.2}
cd "$dest"
vlib unisim
vlib unisims_ver
vlib secureip_ver
cat > modelsim.ini <<EOF
[Library]
others = $base
unisim = $dest/unisim
unisims_ver = $dest/unisims_ver
secureip_ver = $dest/secureip_ver
EOF
# ESP's Virtex-7 wrappers also declare legacy primitives (DCM/BUFGMUX).
# Use the vendor's complete component package, including retarget declarations.
vcom -93 -work unisim "$xilinx/data/vhdl/src/unisims/unisim_retarget_VCOMP.vhd" \
  "$xilinx/data/vhdl/src/unisims/unisim_VPKG.vhd" > compile.log
for model in BUFG IBUF IOBUF OBUF PLLE2_ADV; do
  vcom -93 -work unisim "$xilinx/data/vhdl/src/unisims/primitive/$model.vhd" >> compile.log
done
echo "ESP_MODELSIM_INI=$dest/modelsim.ini"
