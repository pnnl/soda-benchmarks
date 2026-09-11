// The Bambu --architecture-xml description, emitted while the stream types are
// still around. This is the half of the backend contract that cannot be
// recovered later: once ConvertDataflowToLLVM rewrites !dataflow.stream to
// !llvm.ptr, nothing tells a FIFO parameter apart from an array one.
//
// Input is a fixture: a gemm in the call-graph form sodap-dataflow-nodes-to-func
// produces -- func.func per node, dataflow streams across the calls.
//
// REQUIRES: panda

// RUN: mlir-opt %S/Inputs/gemm_small.mlir \
// RUN:     --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-emit-bambu-architecture{file=%t.xml top-func=forward})"
// RUN: FileCheck %s < %t.xml

// The top sees only the four memrefs of @forward -- the two channels are
// internal to the design and must not surface as ports.
// CHECK:      <function dataflow_top="1" inline="off" name="forward" symbol="forward">
// CHECK:        <bundle mode="array" name="arg0">
// CHECK-NEXT:   <bundle mode="array" name="arg1">
// CHECK-NEXT:   <bundle mode="array" name="arg2">
// CHECK-NEXT:   <bundle mode="array" name="arg3">
// CHECK:        <parameter array_dims="8,6" bundle="arg0" elem_count="48" index="0" original_typename="float (*)[6]"
// CHECK:        <parameter array_dims="6,4" bundle="arg1" elem_count="24" index="1" original_typename="float (*)[4]"

// Every node is a dataflow_module. The stream operands become fifo bundles,
// and the bundle name is the channel identity: node2 writes fifo_1, node1
// reads it and writes fifo_0, node0 reads fifo_0. Scalars stay "default".
// CHECK:      <function dataflow_module="1" inline="off" name="node2" symbol="node2">
// CHECK:        <bundle mode="array" name="arg0">
// CHECK-NEXT:   <bundle mode="array" name="arg1">
// CHECK-NEXT:   <bundle depth="32" mode="fifo" name="fifo_1">
// CHECK-NEXT:   <bundle mode="default" name="node2_P3">
// CHECK:        <parameter bundle="fifo_1" includes="{{.*}}ac_channel.h" index="2" {{.*}}port="P2"

// CHECK:      <function dataflow_module="1" inline="off" name="node1" symbol="node1">
// CHECK:        <bundle depth="32" mode="fifo" name="fifo_1">
// CHECK-NEXT:   <bundle mode="array" name="arg2">
// CHECK-NEXT:   <bundle depth="32" mode="fifo" name="fifo_0">
// CHECK-NEXT:   <bundle mode="default" name="node1_P3">
// CHECK-NEXT:   <bundle mode="default" name="node1_P4">

// CHECK:      <function dataflow_module="1" inline="off" name="node0" symbol="node0">
// CHECK:        <bundle depth="32" mode="fifo" name="fifo_0">
// CHECK-NEXT:   <bundle mode="array" name="arg3">
