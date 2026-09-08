// Round-trip every dataflow attribute. Each has a hand-written parser/printer,
// so this is the only thing standing between them and silent corruption.
//
// The booleans of pipeline_ii and fd are the delicate part: they print as the
// keywords `true`/`false`, and a printer that emitted a raw `1`/`0` instead
// would produce attributes that can be printed but never read back.
//
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext | mlir-opt --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext | FileCheck %s

// CHECK-LABEL: func.func @func_attrs
// CHECK-SAME: func_directive = #dataflow.fd<pipeline=false, targetInterval=1, dataflow=true>
// CHECK-SAME: resource = #dataflow.r<lut=100, dsp=4, bram=2>
// CHECK-SAME: timing = #dataflow.t<0 -> 8, 8, 1>
func.func @func_attrs() attributes {
    func_directive = #dataflow.fd<pipeline=false, targetInterval=1, dataflow=true>,
    resource = #dataflow.r<lut=100, dsp=4, bram=2>,
    timing = #dataflow.t<0 -> 8, 8, 1>} {
  return
}

// CHECK-LABEL: func.func @loop_attrs
func.func @loop_attrs(%arg0: memref<8xf32>) {
  affine.for %i = 0 to 8 {
    %v = affine.load %arg0[%i] : memref<8xf32>
    affine.store %v, %arg0[%i] : memref<8xf32>
  // Both booleans set, to prove `true` survives a print/parse cycle.
  // CHECK: loop_directive = #dataflow.pipeline_ii<pipeline=true, targetII=1, dataflow=true, flatten=true>
  // CHECK-SAME: loop_info = #dataflow.l<flattenTripCount=8, iterLatency=3, minII=1>
  } {loop_directive = #dataflow.pipeline_ii<pipeline=true, targetII=1, dataflow=true, flatten=true>,
     loop_info = #dataflow.l<flattenTripCount=8, iterLatency=3, minII=1>}

  affine.for %i = 0 to 8 {
  // ... and that `false` does too, rather than degrading into `0`.
  // CHECK: loop_directive = #dataflow.pipeline_ii<pipeline=false, targetII=2, dataflow=false, flatten=false>
  } {loop_directive = #dataflow.pipeline_ii<pipeline=false, targetII=2, dataflow=false, flatten=false>}
  return
}

// Negative values must survive too.
// CHECK-LABEL: func.func @negative_values
// CHECK-SAME: timing = #dataflow.t<-1 -> -1, -1, -1>
func.func @negative_values() attributes {
    timing = #dataflow.t<-1 -> -1, -1, -1>} {
  return
}
