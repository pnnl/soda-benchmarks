#map2 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (0, 0, 0)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
module {
  memref.global "private" constant @__constant_1x1x1xf32_0 : memref<1x1x1xf32> = dense<3.000000e-01> {alignment = 64 : i64}
  memref.global "private" constant @__constant_1x1x1xf32 : memref<1x1x1xf32> = dense<0.000000e+00> {alignment = 64 : i64}
    func.func @forward(%collapse_shape: memref<1x120x16xf32>, %alt: memref<1x120x16xf32>) {
        %0 = memref.get_global @__constant_1x1x1xf32 : memref<1x1x1xf32>
        %1 = memref.get_global @__constant_1x1x1xf32_0 : memref<1x1x1xf32>
        %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x120x16xf32>
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapse_shape, %1 : memref<1x120x16xf32>, memref<1x1x1xf32>) outs(%alloc_2 : memref<1x120x16xf32>) {
        ^bb0(%in: f32, %in_36: f32, %out: f32):
          %10 = arith.mulf %in, %in_36 : f32
          linalg.yield %10 : f32
        }
        %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x120x16xi1>
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%collapse_shape, %0 : memref<1x120x16xf32>, memref<1x1x1xf32>) outs(%alloc_3 : memref<1x120x16xi1>) {
        ^bb0(%in: f32, %in_36: f32, %out: i1):
          %10 = arith.cmpf oge, %in, %in_36 : f32
          linalg.yield %10 : i1
        }
        %alloc_4 = memref.alloc() {alignment = 64 : i64} : memref<1x120x16xf32>
        linalg.generic {indexing_maps = [#map2, #map2, #map2, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%alloc_3, %collapse_shape, %alloc_2 : memref<1x120x16xi1>, memref<1x120x16xf32>, memref<1x120x16xf32>) outs(%alt : memref<1x120x16xf32>) {
        ^bb0(%in: i1, %in_36: f32, %in_37: f32, %out: f32):
          %10 = arith.select %in, %in_36, %in_37 : f32
          linalg.yield %10 : f32
        }
        return
    }
}