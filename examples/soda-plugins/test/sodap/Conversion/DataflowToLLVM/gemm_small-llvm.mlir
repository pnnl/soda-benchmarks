// The backend half of the flow: dataflow IR down to the single .ll Bambu is
// handed. Needs a panda install because ConvertDataflowToLLVM discovers the
// ac_channel ABI by compiling a generated C++ translation unit with clang.
//
// REQUIRES: panda

// The pass drops ac_channel_specializations.{cpp,ll} in the working directory,
// so run from a scratch one.
// RUN: mkdir -p %t && cd %t
// RUN: mlir-opt %S/Inputs/gemm_small.mlir \
// RUN:     --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-dataflow-to-llvm-pipeline{top-func=forward arch-file=%t/architecture.xml})" \
// RUN:     -o %t/llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t/llvm.mlir -o %t/kernel.ll
// RUN: llvm-link -S %t/kernel.ll %t/ac_channel_specializations.ll -o %t/linked.ll
// RUN: opt %t/linked.ll -passes=always-inline -S -o %t/final.ll
// RUN: FileCheck %s < %t/final.ll
// RUN: FileCheck --check-prefix=INLINED %s < %t/final.ll

// The layout is clang's, not ours -- only that it was recovered is checked.
// CHECK: %class.ac_channel = type {

// Bare-pointer calling convention: a stream operand is one ptr, not an
// exploded memref descriptor. The reads and writes are the mangled
// _bambu_internal seams, which Bambu recognizes as FIFO accesses.
// CHECK-LABEL: define void @node0(ptr noalias %0, ptr noalias %1)
// CHECK: call noundef float @_ZN10ac_channelIfE20_read_bambu_internalIfEEKT_v

// CHECK-LABEL: define void @node2(ptr noalias %0, ptr noalias %1, ptr noalias %2, float %3)
// CHECK: call noundef zeroext i1 @_ZN10ac_channelIfE21_write_bambu_internalIfEEbT_

// The top owns the channels: one alloca + ctor each, the three nodes in
// dataflow order, then the dtors.
// CHECK-LABEL: define void @forward(ptr noalias %0, ptr noalias %1, ptr noalias %2, ptr noalias %3)
// CHECK: alloca %class.ac_channel
// CHECK: call void @_ZN10ac_channelIfEC2Ev
// CHECK: alloca %class.ac_channel
// CHECK: call void @_ZN10ac_channelIfEC2Ev
// CHECK: call void @node2(
// CHECK: call void @node1(
// CHECK: call void @node0(
// CHECK: call void @_ZN10ac_channelIfED2Ev
// CHECK: call void @_ZN10ac_channelIfED2Ev

// always-inline must have consumed every __df_* wrapper call site: left
// standing, Bambu synthesises the channel accessors as modules instead of
// recognising them as FIFO ports.
// INLINED-NOT: call void @__df_ctor_f32
// INLINED-NOT: call void @__df_dtor_f32
// INLINED-NOT: call {{.*}} @__df_read_f32
// INLINED-NOT: call void @__df_write_f32
