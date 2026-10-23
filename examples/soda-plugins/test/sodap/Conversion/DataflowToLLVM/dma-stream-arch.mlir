// The DMA streams of the top become AXI4-Stream ports. With axis-top-streams
// the pipeline infers which streams of the top the design reads or writes,
// types them axis, and the XML gives them mode="axis" on the top while the
// nodes they reach keep their fifo bundles. This is the description Bambu
// takes to build the ESP DMA interface.
//
// REQUIRES: panda

// RUN: mkdir -p %t && cd %t
// RUN: mlir-opt %S/Inputs/dma_pingpong.mlir \
// RUN:     --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-dataflow-to-llvm-pipeline{top-func=top axis-top-streams=true arch-file=%t/architecture.xml})" \
// RUN:     -o %t/llvm.mlir
// RUN: FileCheck %s < %t/architecture.xml
// RUN: mlir-translate --mlir-to-llvmir %t/llvm.mlir -o %t/kernel.ll
// RUN: llvm-link -S %t/kernel.ll %t/ac_channel_specializations.ll -o %t/linked.ll
// RUN: FileCheck --check-prefix=LLVM %s < %t/linked.ll

// Without the option the top's streams stay fifo, as before.
// RUN: mlir-opt %S/Inputs/dma_pingpong.mlir \
// RUN:     --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-emit-bambu-architecture{write-to-terminal=true top-func=top})" \
// RUN:     -o /dev/null | FileCheck --check-prefix=FIFO %s

// The ports reach load (P0, P1) and store (P2, P3) as ordinary FIFOs, depth
// and all; the bank-index channels are FIFOs throughout.
// CHECK:      <function dataflow_module="1" inline="off" name="top_node0" symbol="top_node0">
// CHECK:        <bundle depth="8" mode="fifo" name="P0">
// CHECK-NEXT:   <bundle depth="16" mode="fifo" name="P1">
// CHECK-NEXT:   <bundle mode="array" name="P2">
// CHECK-NEXT:   <bundle depth="2" mode="fifo" name="P3">

// CHECK:      <function dataflow_module="1" inline="off" name="top_node1" symbol="top_node1">
// CHECK:        <bundle depth="2" mode="fifo" name="P0">
// CHECK-NEXT:   <bundle depth="2" mode="fifo" name="P1">

// CHECK:      <function dataflow_module="1" inline="off" name="top_node2" symbol="top_node2">
// CHECK:        <bundle depth="2" mode="fifo" name="P0">
// CHECK-NEXT:   <bundle mode="array" name="P1">
// CHECK-NEXT:   <bundle depth="8" mode="fifo" name="P2">
// CHECK-NEXT:   <bundle depth="16" mode="fifo" name="P3">

// On the top they are axis bundles, with no depth: the FIFO is on the far
// side of the port. The parameters still carry the ac_channel typename.
// CHECK:      <function dataflow_top="1" inline="off" name="top" symbol="top">
// CHECK:        <bundle mode="axis" name="P0"></bundle>
// CHECK-NEXT:   <bundle mode="axis" name="P1"></bundle>
// CHECK-NEXT:   <bundle mode="axis" name="P2"></bundle>
// CHECK-NEXT:   <bundle mode="axis" name="P3"></bundle>
// CHECK:        <parameter bundle="P0" includes="{{.*}}ac_channel.h" index="0" original_typename="ac_channel&lt;int64_t&gt;&amp;" port="P0" size_in_bytes="8" typename="ac_channel&lt;int64_t&gt;&amp;">
// CHECK:        <parameter bundle="P1" includes="{{.*}}ac_channel.h" index="1" original_typename="ac_channel&lt;int32_t&gt;&amp;" port="P1" size_in_bytes="4" typename="ac_channel&lt;int32_t&gt;&amp;">

// FIFO:      <function dataflow_top="1" inline="off" name="top" symbol="top">
// FIFO:        <bundle depth="8" mode="fifo" name="P0">

// The interface lives only in the XML: the IR passes every stream as a
// pointer to its ac_channel, axis or not.
// LLVM: define void @top(ptr noalias %{{.*}}, ptr noalias %{{.*}}, ptr noalias %{{.*}}, ptr noalias %{{.*}})
