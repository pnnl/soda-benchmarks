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
// RUN:     --pass-pipeline="builtin.module(sodap-emit-bambu-architecture{write-to-terminal=true top-func=forward})" \
// RUN:     -o /dev/null | FileCheck %s

// Functions come out in module order, the nodes as dataflow_module. Bundles
// are positional (P<index>): streams are fifo, memrefs array, scalars default.
// CHECK:      <function dataflow_module="1" inline="off" name="node0" symbol="node0">
// CHECK:        <bundle depth="32" mode="fifo" name="P0">
// CHECK-NEXT:   <bundle mode="array" name="P1">

// CHECK:      <function dataflow_module="1" inline="off" name="node1" symbol="node1">
// CHECK:        <bundle depth="32" mode="fifo" name="P0">
// CHECK-NEXT:   <bundle mode="array" name="P1">
// CHECK-NEXT:   <bundle depth="32" mode="fifo" name="P2">
// CHECK-NEXT:   <bundle mode="default" name="P3">
// CHECK-NEXT:   <bundle mode="default" name="P4">

// CHECK:      <function dataflow_module="1" inline="off" name="node2" symbol="node2">
// CHECK:        <bundle mode="array" name="P0">
// CHECK-NEXT:   <bundle mode="array" name="P1">
// CHECK-NEXT:   <bundle depth="32" mode="fifo" name="P2">
// CHECK-NEXT:   <bundle mode="default" name="P3">
// CHECK:        <parameter bundle="P2" includes="{{.*}}ac_channel.h" index="2" {{.*}}port="P2"

// The top sees only the four memrefs of @forward: the channels are internal.
// CHECK:      <function dataflow_top="1" inline="off" name="forward" symbol="forward">
// CHECK:        <bundle mode="array" name="P0">
// CHECK-NEXT:   <bundle mode="array" name="P1">
// CHECK-NEXT:   <bundle mode="array" name="P2">
// CHECK-NEXT:   <bundle mode="array" name="P3">
// CHECK:        <parameter array_dims="8,6" bundle="P0" elem_count="48" index="0" original_typename="float (*)[6]"
// CHECK:        <parameter array_dims="6,4" bundle="P1" elem_count="24" index="1" original_typename="float (*)[4]"
// CHECK:        <parameter array_dims="8,4" bundle="P2" elem_count="32" index="2" original_typename="float (*)[4]"
// CHECK:        <parameter array_dims="8,4" bundle="P3" elem_count="32" index="3" original_typename="float (*)[4]"

