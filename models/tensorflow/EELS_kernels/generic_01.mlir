module {
    func.func @forward(%input: memref<1x120x16xf32>, %weight: memref<1x1x1xf32>, %output: memref<1x120x16xf32>) {
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel"]} ins(%input, %weight : memref<1x120x16xf32>, memref<1x1x1xf32>) outs(%output : memref<1x120x16xf32>) {
        ^bb0(%in: f32, %in_36: f32, %out: f32):
          %10 = arith.mulf %in, %in_36 : f32
          linalg.yield %10 : f32
        }
        return
    }
}