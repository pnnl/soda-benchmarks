// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(canonicalize,func.func(sodap-dataflow-load-store),canonicalize,func.func(sodap-dataflow-schedule{reuse-buffer-budget=432}),fold-memref-alias-ops,func.func(sodap-dataflow-ping-pong),canonicalize)" | FileCheck %s --check-prefix=FIT
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(canonicalize,func.func(sodap-dataflow-load-store),canonicalize,func.func(sodap-dataflow-schedule{reuse-buffer-budget=431}),fold-memref-alias-ops,func.func(sodap-dataflow-ping-pong),canonicalize)" | FileCheck %s --check-prefix=FALLBACK
// RUN: mlir-opt %s --load-dialect-plugin=%sodap_libs/SODAPlugin%shlibext --load-pass-plugin=%sodap_libs/SODAPlugin%shlibext --pass-pipeline="builtin.module(canonicalize,func.func(sodap-dataflow-load-store),canonicalize,func.func(sodap-dataflow-schedule{reuse-buffer-budget=0}),fold-memref-alias-ops,func.func(sodap-dataflow-ping-pong),canonicalize)" | FileCheck %s --check-prefix=FALLBACK
// FIT: memref.alloc() : memref<2x2x18xf32>
// FALLBACK-NOT: memref<2x2x18xf32>
// FALLBACK: return
// The 2x18 slab requires 432 budgeted bytes: two slots plus private allowance.
// More than two row tiles exercise slot reuse in component RTL qualification.
#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module attributes {soda.bambu.container_module, soda.container_module, torch.debug_module_name = "TwoMM"} {
  func.func @forward_kernel(%arg0: memref<f32>, %arg1: memref<f32>, %arg2: memref<16x22xf32>, %arg3: memref<22x18xf32>, %arg4: memref<18x24xf32>, %arg5: memref<16x24xf32>, %arg6: memref<16x24xf32>) {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %c24 = arith.constant 24 : index
    %c22 = arith.constant 22 : index
    %c18 = arith.constant 18 : index
    %c2 = arith.constant 2 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = dataflow.dispatch : memref<16x24xf32> {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x18xf32>
      dataflow.task attributes {compute} {
        scf.for %arg7 = %c0 to %c16 step %c2 {
          scf.for %arg8 = %c0 to %c18 step %c2 {
            %subview = memref.subview %arg2[%arg7, 0] [2, 22] [1, 1] : memref<16x22xf32> to memref<2x22xf32, strided<[22, 1], offset: ?>>
            %subview_3 = memref.subview %arg3[0, %arg8] [22, 2] [1, 1] : memref<22x18xf32> to memref<22x2xf32, strided<[18, 1], offset: ?>>
            %subview_4 = memref.subview %alloc[%arg7, %arg8] [2, 2] [1, 1] : memref<16x18xf32> to memref<2x2xf32, strided<[18, 1], offset: ?>>
            linalg.fill ins(%cst : f32) outs(%subview_4 : memref<2x2xf32, strided<[18, 1], offset: ?>>)
            scf.for %arg9 = %c0 to %c22 step %c2 {
              %subview_5 = memref.subview %subview[0, %arg9] [2, 2] [1, 1] : memref<2x22xf32, strided<[22, 1], offset: ?>> to memref<2x2xf32, strided<[22, 1], offset: ?>>
              %subview_6 = memref.subview %subview_3[%arg9, 0] [2, 2] [1, 1] : memref<22x2xf32, strided<[18, 1], offset: ?>> to memref<2x2xf32, strided<[18, 1], offset: ?>>
              linalg.matmul ins(%subview_5, %subview_6 : memref<2x2xf32, strided<[22, 1], offset: ?>>, memref<2x2xf32, strided<[18, 1], offset: ?>>) outs(%subview_4 : memref<2x2xf32, strided<[18, 1], offset: ?>>)
            }
          }
        }
      }
      %alloc_0 = memref.alloc() {alignment = 64 : i64} : memref<16x18xf32>
      dataflow.task attributes {compute} {
        scf.for %arg7 = %c0 to %c16 step %c2 {
          scf.for %arg8 = %c0 to %c18 step %c2 {
            %subview = memref.subview %alloc[%arg7, %arg8] [2, 2] [1, 1] : memref<16x18xf32> to memref<2x2xf32, strided<[18, 1], offset: ?>>
            %subview_3 = memref.subview %alloc_0[%arg7, %arg8] [2, 2] [1, 1] : memref<16x18xf32> to memref<2x2xf32, strided<[18, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg0, %subview : memref<f32>, memref<2x2xf32, strided<[18, 1], offset: ?>>) outs(%subview_3 : memref<2x2xf32, strided<[18, 1], offset: ?>>) {
            ^bb0(%in: f32, %in_4: f32, %out: f32):
              %1 = arith.mulf %in, %in_4 : f32
              linalg.yield %1 : f32
            }
          }
        }
      }
      %alloc_1 = memref.alloc() {alignment = 64 : i64} : memref<16x24xf32>
      dataflow.task attributes {compute} {
        scf.for %arg7 = %c0 to %c16 step %c2 {
          scf.for %arg8 = %c0 to %c24 step %c2 {
            %subview = memref.subview %alloc_0[%arg7, 0] [2, 18] [1, 1] : memref<16x18xf32> to memref<2x18xf32, strided<[18, 1], offset: ?>>
            %subview_3 = memref.subview %arg4[0, %arg8] [18, 2] [1, 1] : memref<18x24xf32> to memref<18x2xf32, strided<[24, 1], offset: ?>>
            %subview_4 = memref.subview %alloc_1[%arg7, %arg8] [2, 2] [1, 1] : memref<16x24xf32> to memref<2x2xf32, strided<[24, 1], offset: ?>>
            linalg.fill ins(%cst : f32) outs(%subview_4 : memref<2x2xf32, strided<[24, 1], offset: ?>>)
            scf.for %arg9 = %c0 to %c18 step %c2 {
              %subview_5 = memref.subview %subview[0, %arg9] [2, 2] [1, 1] : memref<2x18xf32, strided<[18, 1], offset: ?>> to memref<2x2xf32, strided<[18, 1], offset: ?>>
              %subview_6 = memref.subview %subview_3[%arg9, 0] [2, 2] [1, 1] : memref<18x2xf32, strided<[24, 1], offset: ?>> to memref<2x2xf32, strided<[24, 1], offset: ?>>
              linalg.matmul ins(%subview_5, %subview_6 : memref<2x2xf32, strided<[18, 1], offset: ?>>, memref<2x2xf32, strided<[24, 1], offset: ?>>) outs(%subview_4 : memref<2x2xf32, strided<[24, 1], offset: ?>>)
            }
          }
        }
      }
      %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<16x24xf32>
      dataflow.task attributes {compute} {
        scf.for %arg7 = %c0 to %c16 step %c2 {
          scf.for %arg8 = %c0 to %c24 step %c2 {
            %subview = memref.subview %arg5[%arg7, %arg8] [2, 2] [1, 1] : memref<16x24xf32> to memref<2x2xf32, strided<[24, 1], offset: ?>>
            %subview_3 = memref.subview %alloc_1[%arg7, %arg8] [2, 2] [1, 1] : memref<16x24xf32> to memref<2x2xf32, strided<[24, 1], offset: ?>>
            %subview_4 = memref.subview %alloc_2[%arg7, %arg8] [2, 2] [1, 1] : memref<16x24xf32> to memref<2x2xf32, strided<[24, 1], offset: ?>>
            linalg.generic {indexing_maps = [#map, #map1, #map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%arg1, %subview, %subview_3 : memref<f32>, memref<2x2xf32, strided<[24, 1], offset: ?>>, memref<2x2xf32, strided<[24, 1], offset: ?>>) outs(%subview_4 : memref<2x2xf32, strided<[24, 1], offset: ?>>) {
            ^bb0(%in: f32, %in_5: f32, %in_6: f32, %out: f32):
              %1 = arith.mulf %in, %in_5 : f32
              %2 = arith.addf %1, %in_6 : f32
              linalg.yield %2 : f32
            }
          }
        }
      }
      dataflow.yield %alloc_2 : memref<16x24xf32>
    }
    memref.copy %0, %arg6 : memref<16x24xf32> to memref<16x24xf32>
    return
  }
}

