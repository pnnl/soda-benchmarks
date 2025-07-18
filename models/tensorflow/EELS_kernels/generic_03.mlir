module {
    func.func @forward(%mask: memref<1x120x16xi1>, %input: memref<1x120x16xf32>, %alt: memref<1x120x16xf32>, %output: memref<1x120x16xf32>) {
        linalg.generic {indexing_maps = [#map2, #map2, #map2, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%mask, %input, %alt : memref<1x120x16xi1>, memref<1x120x16xf32>, memref<1x120x16xf32>) outs(%output : memref<1x120x16xf32>) {
        ^bb0(%in: i1, %in_36: f32, %in_37: f32, %out: f32):
          %10 = arith.select %in, %in_36, %in_37 : f32
          linalg.yield %10 : f32
        }
        return
    }
}