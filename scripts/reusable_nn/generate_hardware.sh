#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
set -euo pipefail
repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
mkdir -p "${1:?Usage: generate_hardware.sh output-directory}"
out=$(realpath "$1")
plugin=${MLIR_PLUGIN_PATH:?Set MLIR_PLUGIN_PATH to the extended SODA plugin}
plugins=(--load-dialect-plugin="$plugin" --load-pass-plugin="$plugin")
for kind in conv add gemm; do
  mkdir -p "$out/$kind"
  cd "$out/$kind"
  printf 'module {}\n' > empty.mlir
  if [[ "$kind" == gemm ]]; then
    generator="sodap-generate-reusable-gemm{top-name=gemm_kernel tile-m=${NN_GEMM_M:-4} tile-n=${NN_GEMM_N:-4} tile-k=${NN_GEMM_K:-4}}"
  else
    generator="sodap-generate-reusable-operator{kind=$kind tile-pixels=${NN_CONV_PIXELS:-4} output-lanes=${NN_CONV_LANES:-8} tile-channels=${NN_CONV_CHANNELS:-16} block-elements=${NN_ADD_BLOCK:-128}}"
  fi
  printf '%s\n' "$generator" > schedule.txt
  mlir-opt empty.mlir "${plugins[@]}" --pass-pipeline="builtin.module($generator,canonicalize,cse)" -o hardware.mlir
  mlir-opt hardware.mlir "${plugins[@]}" --pass-pipeline="builtin.module(
    sodap-emit-bambu-architecture{top-func=${kind}_kernel file=architecture.xml},
    sodap-convert-dataflow-to-llvm,lower-affine,convert-scf-to-cf,
    convert-math-to-llvm,expand-strided-metadata,lower-affine,
    finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},
    convert-index-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" -o hardware-llvm.mlir
  mlir-translate --mlir-to-llvmir hardware-llvm.mlir \
    | llvm-link -S - ac_channel_specializations.ll \
    | opt -passes=always-inline -S -o accelerator.ll
  sha256sum accelerator.ll architecture.xml > hardware-inputs.sha256
  if [[ ${NN_SYNTHESIZE:-1} == 1 ]]; then
    BAMBU_DEVICE=${BAMBU_DEVICE:-xc7vx485t-2ffg1761} BAMBU_CLOCK_PERIOD=${BAMBU_CLOCK_PERIOD:-20} \
      "$repo/scripts/ll_to_verilog_esp_version.sh" accelerator.ll "${kind}_kernel"
  fi
done
