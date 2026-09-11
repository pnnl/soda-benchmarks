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

// A one-dimensional array still spells its single extent, and declares itself
// as the plain pointer it decays to -- only rank two and up grow a row type.
// CHECK:      <function dataflow_top="1" inline="off" name="forward"
// CHECK:      <parameter array_dims="16" bundle="arg0" elem_count="16" {{.*}}original_typename="float*" {{.*}}size_in_bytes="64" typename="float*"

// CHECK:      <parameter array_dims="8,6" bundle="arg1" elem_count="48" {{.*}}original_typename="float (*)[6]" {{.*}}size_in_bytes="192" typename="float*"

// Rank 3 and rank 4: the extents are comma-separated in memref order,
// elem_count stays their product, and the declaration drops only the leading
// extent -- the one a C parameter loses.
// CHECK:      <parameter array_dims="2,3,4" bundle="arg2" elem_count="24" {{.*}}original_typename="float (*)[3][4]" {{.*}}size_in_bytes="96" typename="float*"

// i8 to keep size_in_bytes distinguishable from elem_count.
// CHECK:      <parameter array_dims="2,2,2,2" bundle="arg3" elem_count="16" {{.*}}original_typename="int8_t (*)[2][2][2]" {{.*}}size_in_bytes="16" typename="int8_t*"

// The same array reaches node0 as a dataflow_module parameter, under the bundle
// the top minted for it, with the shape intact.
// CHECK:      <function dataflow_module="1" inline="off" name="node0"
// CHECK:        <bundle mode="array" name="arg2">
// CHECK-NEXT:   <bundle depth="8" mode="fifo" name="fifo_0">
// CHECK:      <parameter array_dims="2,3,4" bundle="arg2" elem_count="24" {{.*}}original_typename="float (*)[3][4]" port="P0" size_in_bytes="96" typename="float*"
