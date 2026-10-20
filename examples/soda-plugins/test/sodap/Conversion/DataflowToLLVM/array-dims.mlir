// array_dims carries the shape, one extent per dimension, while elem_count and
// size_in_bytes stay flat: the pointer the lowering hands Bambu is linearized,
// so the extents are description, not layout. original_typename carries the
// same shape a second way, as the C declaration the parameter decays from.
//
// REQUIRES: panda

// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext \
// RUN:     --pass-pipeline="builtin.module(sodap-emit-bambu-architecture{file=%t.xml top-func=forward})"
// RUN: FileCheck %s < %t.xml

func.func @node0(%arg0: memref<2x3x4xf32>, %arg1: !dataflow.stream<f32, 8>) {
    return
}

func.func @forward(%arg0: memref<16xf32>, %arg1: memref<8x6xf32>,
                   %arg2: memref<2x3x4xf32>, %arg3: memref<2x2x2x2xi8>) {
    %0 = dataflow.stream {depth = 8 : i32} : <f32, 8>
    call @node0(%arg2, %0) : (memref<2x3x4xf32>, !dataflow.stream<f32, 8>) -> ()
    return
}

// Functions come out in module order. Every bundle is named after its
// positional port, P<index>, in the top and in every module alike.
// CHECK:      <function dataflow_module="1" inline="off" name="node0"
// CHECK:        <bundle mode="array" name="P0">
// CHECK-NEXT:   <bundle depth="8" mode="fifo" name="P1">
// CHECK:      <parameter array_dims="2,3,4" bundle="P0" elem_count="24" {{.*}}original_typename="float (*)[3][4]" port="P0" size_in_bytes="96" typename="float*"

// A rank-1 array declares itself as the plain pointer it decays to; rank 2
// and up drop only the leading extent. i8 keeps size_in_bytes apart from
// elem_count.
// CHECK:      <function dataflow_top="1" inline="off" name="forward"
// CHECK:      <parameter array_dims="16" bundle="P0" elem_count="16" {{.*}}original_typename="float*" {{.*}}size_in_bytes="64" typename="float*"
// CHECK:      <parameter array_dims="8,6" bundle="P1" elem_count="48" {{.*}}original_typename="float (*)[6]" {{.*}}size_in_bytes="192" typename="float*"
// CHECK:      <parameter array_dims="2,3,4" bundle="P2" elem_count="24" {{.*}}original_typename="float (*)[3][4]" {{.*}}size_in_bytes="96" typename="float*"
// CHECK:      <parameter array_dims="2,2,2,2" bundle="P3" elem_count="16" {{.*}}original_typename="int8_t (*)[2][2][2]" {{.*}}size_in_bytes="16" typename="int8_t*"

