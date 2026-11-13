// RUN: mlir-opt %s --finalize-memref-to-llvm --convert-arith-to-llvm \
// RUN:   --convert-cf-to-llvm --convert-func-to-llvm --reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%sodap_libs/libmlir_mockesp_runner_utils%shlibext | FileCheck %s
// CHECK: Called: esp_copy_to_shared_f32
// CHECK: Called: esp_copy_from_shared_f32
// CHECK: Called: esp_copy_to_shared_f32
// CHECK: Called: esp_copy_from_shared_f32
// CHECK: Called: esp_free_shared

// Verify descriptors produced by actual MLIR lowering, including rank zero.
func.func private @esp_alloc_shared(i64) -> i64
func.func private @esp_free_shared(i64)
func.func private @esp_copy_to_shared_f32(memref<*xf32>, i64, i64)
func.func private @esp_copy_from_shared_f32(i64, i64, memref<*xf32>)
func.func @main() {
  %bytes = arith.constant 32 : i64
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %off = arith.constant 2 : i64
  %scalaroff = arith.constant 7 : i64
  %value = arith.constant 4.25 : f32
  %other = arith.constant -2.5 : f32
  %h = call @esp_alloc_shared(%bytes) : (i64) -> i64
  %s = memref.alloca() : memref<f32>
  %t = memref.alloca() : memref<f32>
  memref.store %value, %s[] : memref<f32>
  %su = memref.cast %s : memref<f32> to memref<*xf32>
  %tu = memref.cast %t : memref<f32> to memref<*xf32>
  call @esp_copy_to_shared_f32(%su, %h, %scalaroff) : (memref<*xf32>, i64, i64) -> ()
  call @esp_copy_from_shared_f32(%h, %scalaroff, %tu) : (i64, i64, memref<*xf32>) -> ()
  %sout = memref.load %t[] : memref<f32>
  %ok = arith.cmpf oeq, %value, %sout : f32
  cf.assert %ok, "rank-zero copy failed"

  %storage = memref.alloca() : memref<16xf32>
  %src = memref.reinterpret_cast %storage to offset: [2], sizes: [2, 2], strides: [1, 4]
    : memref<16xf32> to memref<2x2xf32, strided<[1, 4], offset: 2>>
  memref.store %value, %src[%zero, %zero] : memref<2x2xf32, strided<[1, 4], offset: 2>>
  memref.store %other, %src[%zero, %one] : memref<2x2xf32, strided<[1, 4], offset: 2>>
  memref.store %other, %src[%one, %zero] : memref<2x2xf32, strided<[1, 4], offset: 2>>
  memref.store %value, %src[%one, %one] : memref<2x2xf32, strided<[1, 4], offset: 2>>
  %dst = memref.alloca() : memref<2x2xf32>
  %srcu = memref.cast %src : memref<2x2xf32, strided<[1, 4], offset: 2>> to memref<*xf32>
  %dstu = memref.cast %dst : memref<2x2xf32> to memref<*xf32>
  call @esp_copy_to_shared_f32(%srcu, %h, %off) : (memref<*xf32>, i64, i64) -> ()
  call @esp_copy_from_shared_f32(%h, %off, %dstu) : (i64, i64, memref<*xf32>) -> ()
  %v00 = memref.load %dst[%zero, %zero] : memref<2x2xf32>
  %v01 = memref.load %dst[%zero, %one] : memref<2x2xf32>
  %v10 = memref.load %dst[%one, %zero] : memref<2x2xf32>
  %v11 = memref.load %dst[%one, %one] : memref<2x2xf32>
  %a = arith.cmpf oeq, %v00, %value : f32
  %b = arith.cmpf oeq, %v01, %other : f32
  %c = arith.cmpf oeq, %v10, %other : f32
  %d = arith.cmpf oeq, %v11, %value : f32
  %ab = arith.andi %a, %b : i1
  %cd = arith.andi %c, %d : i1
  %all = arith.andi %ab, %cd : i1
  cf.assert %all, "strided copy failed"
  call @esp_free_shared(%h) : (i64) -> ()
  return
}
